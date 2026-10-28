#include "gpu/common/gpu_common.h"
#include "gpu/host/gpu_memory_host.h"
#include "gpuntt/common/modular_arith.cuh"

extern "C" {

int pvda_is_device_pointer(const void* ptr) { return is_gpu_device_pointer(ptr) ? 1 : 0; }

int64_t* pvda_gpu_upload(const int64_t* host_ptr, size_t n_elements)
{
	return pvda_gpu_guard([&]() -> int64_t* {
		Data64s* d_ptr = (Data64s*)pvda_device_alloc_bytes(n_elements * sizeof(int64_t));
		CUDA_CHECK(
		    cudaMemcpyAsync(d_ptr, host_ptr, n_elements * sizeof(int64_t), cudaMemcpyHostToDevice, gpu_active_stream));
		return (int64_t*)d_ptr;
	});
}

void pvda_gpu_download(int64_t* host_ptr, const int64_t* device_ptr, size_t n_elements)
{
	return pvda_gpu_guard([&]() -> void {
		// Downloads cross back into host memory that callers read immediately on
		// return (no further stream sync at the call site), so this primitive
		// keeps its synchronous contract despite using the active stream.
		CUDA_CHECK(cudaMemcpyAsync(host_ptr, device_ptr, n_elements * sizeof(int64_t), cudaMemcpyDeviceToHost,
		                           gpu_active_stream));
		CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
	});
}

int64_t* pvda_gpu_alloc(size_t n_elements) { return (int64_t*)pvda_device_alloc_bytes(n_elements * sizeof(int64_t)); }

void pvda_gpu_free(int64_t* device_ptr) { pvda_device_free_bytes(device_ptr); }

void pvda_gpu_copy(int64_t* dst, const int64_t* src, size_t n_elements)
{
	return pvda_gpu_guard([&]() -> void {
		CUDA_CHECK(
		    cudaMemcpyAsync(dst, src, n_elements * sizeof(int64_t), cudaMemcpyDeviceToDevice, gpu_active_stream));
	});
}

void pvda_gpu_zero(int64_t* dst, size_t n_elements)
{
	return pvda_gpu_guard(
	    [&]() -> void { CUDA_CHECK(cudaMemsetAsync(dst, 0, n_elements * sizeof(int64_t), gpu_active_stream)); });
}

int64_t* pvda_gpu_upload_limbs(const int64_t* host_ptr, int64_t host_stride, size_t limbs, size_t n)
{
	return pvda_gpu_guard([&]() -> int64_t* {
		Data64s* d_ptr = (Data64s*)pvda_device_alloc_bytes(limbs * n * sizeof(int64_t));
		CUDA_CHECK(cudaMemcpy2DAsync(d_ptr, n * sizeof(int64_t), host_ptr, (size_t)host_stride * sizeof(int64_t),
		                             n * sizeof(int64_t), limbs, cudaMemcpyHostToDevice, gpu_active_stream));
		return (int64_t*)d_ptr;
	});
}

void pvda_gpu_download_limbs(int64_t* host_ptr, int64_t host_stride, const int64_t* device_ptr, size_t limbs,
                             size_t n)
{
	return pvda_gpu_guard([&]() -> void {
		// Synchronous, like pvda_gpu_download: callers read the host data on return
		CUDA_CHECK(cudaMemcpy2DAsync(host_ptr, (size_t)host_stride * sizeof(int64_t), device_ptr, n * sizeof(int64_t),
		                             n * sizeof(int64_t), limbs, cudaMemcpyDeviceToHost, gpu_active_stream));
		CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
	});
}

}  // extern "C"
