#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <utility>
#include <string>
#include <vector>

#include "gpu/common/gpu_common.h"

extern "C" {
#include "logger.h"  // C header without extern "C" guards
}
#include "gpu/common/gpu_stream.h"

thread_local cudaStream_t gpu_active_stream = 0;

namespace {
thread_local std::vector<cudaStream_t> gpu_stream_stack;
}

namespace {
// Freed blocks stay cached in the device's default pool (the pool's default
// release threshold of 0 would hand memory back to the driver at every
// synchronization point, making each allocation as expensive as cudaMalloc).
// The threshold counts in-use memory too, so any finite default would disable
// caching as soon as a workload keeps that much data resident (e.g. a PIR
// database): the cache is unbounded by default, like PyTorch's caching
// allocator. PVDA_GPU_POOL_CACHE_MB bounds it (useful when several processes
// share the GPU), pvda_gpu_release_cached_memory() empties it, and an
// allocation that runs out of memory empties it and retries once.
cudaMemPool_t default_mempool()
{
	static cudaMemPool_t pool = [] {
		int device = 0;
		CUDA_CHECK(cudaGetDevice(&device));
		cudaMemPool_t p;
		CUDA_CHECK(cudaDeviceGetDefaultMemPool(&p, device));
		uint64_t threshold = UINT64_MAX;
		if (const char* env = std::getenv("PVDA_GPU_POOL_CACHE_MB")) threshold = std::strtoull(env, nullptr, 10) << 20;
		CUDA_CHECK(cudaMemPoolSetAttribute(p, cudaMemPoolAttrReleaseThreshold, &threshold));
		return p;
	}();
	return pool;
}
}  // namespace

namespace {
// The GPU blocks handed out by pvda_device_alloc_bytes, so that the library can tell its GPU
// memory apart from host memory without the CUDA runtime (pvda_gpu_owns): a program that never
// allocates GPU memory never initializes CUDA. Heap-allocated and never destroyed, as blocks
// may be freed by static destructors at exit.
struct BlockRegistry
{
	std::mutex mutex;
	std::map<uintptr_t, size_t> blocks;  // start -> size
};
BlockRegistry& block_registry()
{
	static BlockRegistry* registry = new BlockRegistry;
	return *registry;
}
std::atomic<size_t> live_blocks{0};

void register_block(void* ptr, size_t bytes)
{
	BlockRegistry& r = block_registry();
	std::lock_guard<std::mutex> lock(r.mutex);
	r.blocks[(uintptr_t)ptr] = bytes ? bytes : 1;
	live_blocks.fetch_add(1, std::memory_order_relaxed);
}

size_t block_size(const void* ptr)
{
	BlockRegistry& r = block_registry();
	std::lock_guard<std::mutex> lock(r.mutex);
	auto it = r.blocks.find((uintptr_t)ptr);
	return it == r.blocks.end() ? 0 : it->second;
}

void unregister_block(void* ptr)
{
	BlockRegistry& r = block_registry();
	std::lock_guard<std::mutex> lock(r.mutex);
	if (r.blocks.erase((uintptr_t)ptr)) live_blocks.fetch_sub(1, std::memory_order_relaxed);
}
}  // namespace

extern "C" int pvda_gpu_owns(const void* ptr)
{
	if (!ptr || live_blocks.load(std::memory_order_relaxed) == 0) return 0;
	BlockRegistry& r = block_registry();
	std::lock_guard<std::mutex> lock(r.mutex);
	auto it = r.blocks.upper_bound((uintptr_t)ptr);
	if (it == r.blocks.begin()) return 0;
	--it;
	return (uintptr_t)ptr < it->first + it->second;
}

namespace {
// Per-thread record of the last GPU failure (see gpu_stream.h)
thread_local std::string last_error;
thread_local bool error_pending = false;
}  // namespace

extern "C" void pvda_gpu_record_error(const char* message)
{
	last_error    = message ? message : "unknown GPU error";
	error_pending = true;
	log_message(LOG_ERROR, last_error.c_str());
}

extern "C" int pvda_gpu_take_error(void)
{
	bool had      = error_pending;
	error_pending = false;
	return had ? 1 : 0;
}

extern "C" const char* pvda_gpu_last_error(void) { return last_error.c_str(); }

namespace {
// Blocks freed by the library are kept here, per stream and exact size, instead of going back
// to the driver: the GPU operations allocate and free the same few sizes over and over (the
// scratch of every automorphism, half-product, ...) and a cudaMallocAsync costs microseconds of
// CPU time, which dominates those small operations. A block is only reused on the stream it was
// freed on, so its reuse stays ordered after the work that used it.
//
// The cache holds at most PVDA_GPU_SCRATCH_CACHE_MB (default 1024) MiB; beyond that, and on
// pvda_gpu_release_cached_memory, blocks go back to the driver.
struct ScratchCache
{
	std::mutex mutex;
	std::map<std::pair<cudaStream_t, size_t>, std::vector<void*>> blocks;
	size_t cached_bytes = 0;
	size_t limit_bytes  = [] {
        const char* env = std::getenv("PVDA_GPU_SCRATCH_CACHE_MB");
        return (size_t)(env ? std::strtoull(env, nullptr, 10) : 1024) << 20;
	}();
};
ScratchCache& scratch_cache()
{
	static ScratchCache* cache = new ScratchCache;  // never destroyed: blocks may be freed at exit
	return *cache;
}

// Takes a cached block of exactly `bytes` for the active stream, or nullptr
void* take_cached(size_t bytes)
{
	ScratchCache& c = scratch_cache();
	std::lock_guard<std::mutex> lock(c.mutex);
	auto it = c.blocks.find({gpu_active_stream, bytes});
	if (it == c.blocks.end() || it->second.empty()) return nullptr;
	void* ptr = it->second.back();
	it->second.pop_back();
	c.cached_bytes -= bytes;
	return ptr;
}

// Keeps a block for reuse on the active stream; returns false when the cache is full
bool keep_cached(void* ptr, size_t bytes)
{
	ScratchCache& c = scratch_cache();
	std::lock_guard<std::mutex> lock(c.mutex);
	if (c.cached_bytes + bytes > c.limit_bytes) return false;
	c.blocks[{gpu_active_stream, bytes}].push_back(ptr);
	c.cached_bytes += bytes;
	return true;
}

// Returns every cached block to the driver (the caller synchronizes first)
void drop_cached()
{
	ScratchCache& c = scratch_cache();
	std::lock_guard<std::mutex> lock(c.mutex);
	for (auto& entry : c.blocks)
		for (void* ptr : entry.second) (void)cudaFreeAsync(ptr, entry.first.first);
	c.blocks.clear();
	c.cached_bytes = 0;
}
}  // namespace

void* pvda_device_alloc_bytes(size_t bytes)
{
	return pvda_gpu_guard([&]() -> void* {
		if (void* cached = take_cached(bytes))
		{
			register_block(cached, bytes);
			return cached;
		}
		(void)default_mempool();  // configures the pool's release threshold on first use
		void* ptr       = nullptr;
		cudaError_t err = cudaMallocAsync(&ptr, bytes, gpu_active_stream);
		if (err == cudaErrorMemoryAllocation)
		{
			// Out of memory: give every cached block back to the driver and retry once.
			(void)cudaGetLastError();
			pvda_gpu_release_cached_memory();
			err = cudaMallocAsync(&ptr, bytes, gpu_active_stream);
		}
		CUDA_CHECK(err);
		register_block(ptr, bytes);
		return ptr;
	});
}

// The block is only reused once the work already queued on gpu_active_stream
// has completed. Every GPU backend buffer is freed on the stream that last
// used it, or on the legacy default stream, which is ordered after all
// blocking streams (the ones pvda_gpu_stream_create makes). Errors are
// ignored, like the cudaFree this replaces: static objects may be released
// after the CUDA runtime has already shut down at program exit.
void pvda_device_free_bytes(void* ptr)
{
	return pvda_gpu_guard([&]() -> void {
		if (!ptr) return;
		size_t bytes = block_size(ptr);
		unregister_block(ptr);
		if (bytes && keep_cached(ptr, bytes)) return;  // kept for reuse
		(void)cudaFreeAsync(ptr, gpu_active_stream);
	});
}

extern "C" {

void* pvda_gpu_stream_create(void)
{
	return pvda_gpu_guard([&]() -> void* {
		cudaStream_t stream = nullptr;
		CUDA_CHECK(cudaStreamCreate(&stream));
		return (void*)stream;
	});
}

void pvda_gpu_stream_destroy(void* stream)
{
	return pvda_gpu_guard([&]() -> void {
		if (!stream) return;
		CUDA_CHECK(cudaStreamDestroy((cudaStream_t)stream));
	});
}

void pvda_gpu_stream_synchronize(void* stream) { CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream)); }

void pvda_gpu_stream_push(void* stream)
{
	return pvda_gpu_guard([&]() -> void {
		gpu_stream_stack.push_back(gpu_active_stream);
		gpu_active_stream = (cudaStream_t)stream;
	});
}

void pvda_gpu_stream_pop(void)
{
	return pvda_gpu_guard([&]() -> void {
		if (gpu_stream_stack.empty())
		{
			fprintf(stderr, "pvda_gpu_stream_pop: stack underflow (pop without matching push)\n");
			abort();
		}
		gpu_active_stream = gpu_stream_stack.back();
		gpu_stream_stack.pop_back();
	});
}

void* pvda_gpu_stream_get_active(void) { return (void*)gpu_active_stream; }

void pvda_gpu_device_synchronize(void) { CUDA_CHECK(cudaDeviceSynchronize()); }

void pvda_gpu_memcpy(void* dst, const void* src, size_t bytes)
{
	return pvda_gpu_guard([&]() -> void {
		if (bytes == 0) return;
		CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, gpu_active_stream));
		cudaPointerAttributes attr{};
		CUDA_CHECK(cudaPointerGetAttributes(&attr, dst));
		if (attr.type != cudaMemoryTypeDevice) CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
	});
}

void pvda_gpu_memzero(void* ptr, size_t bytes)
{
	return pvda_gpu_guard([&]() -> void {
		if (bytes) CUDA_CHECK(cudaMemsetAsync(ptr, 0, bytes, gpu_active_stream));
	});
}

int pvda_gpu_available(void)
{
	return pvda_gpu_guard([&]() -> int {
		int count = 0;
		if (cudaGetDeviceCount(&count) != cudaSuccess)
		{
			(void)cudaGetLastError();
			return 0;
		}
		return count > 0;
	});
}

void pvda_gpu_release_cached_memory(void)
{
	return pvda_gpu_guard([&]() -> void {
		CUDA_CHECK(cudaDeviceSynchronize());
		drop_cached();
		CUDA_CHECK(cudaMemPoolTrimTo(default_mempool(), 0));
	});
}

}  // extern "C"
