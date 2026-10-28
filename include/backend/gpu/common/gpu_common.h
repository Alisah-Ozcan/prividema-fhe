#ifndef PVDA_GPU_COMMON_H
#define PVDA_GPU_COMMON_H

#ifdef __CUDACC__
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "gpu/common/gpu_stream.h"

// Runs f, turning any failure into a recorded GPU error (pvda_gpu_record_error) and a
// NULL / zero result: GPU failures must not escape into the C code that calls them.
template <typename F>
auto pvda_gpu_guard(F&& f) -> decltype(f())
{
	using Result = decltype(f());
	try
	{
		return f();
	} catch (const std::exception& error)
	{
		pvda_gpu_record_error(error.what());
	} catch (...)
	{
		pvda_gpu_record_error("unknown GPU error");
	}
	if constexpr (!std::is_void_v<Result>) return Result{};
}

class CudaException : public std::exception {
   public:
	CudaException(const std::string& file, int line, cudaError_t error) : file_(file), line_(line), error_(error) {}

	const char* what() const noexcept override { return m_error_string.c_str(); }

   private:
	std::string file_;
	int line_;
	cudaError_t error_;
	std::string m_error_string =
	    "CUDA Error in " + file_ + " at line " + std::to_string(line_) + ": " + cudaGetErrorString(error_);
};

#define CUDA_CHECK(err)                                       \
	do                                                        \
	{                                                         \
		cudaError_t error = err;                              \
		if (error != cudaSuccess)                             \
		{                                                     \
			throw ::CudaException(__FILE__, __LINE__, error); \
		}                                                     \
	} while (0)

// Device memory allocation used by every GPU backend buffer (pvda_gpu_alloc,
// prepared matrices, VEC_GPU scratch). Stream-ordered (cudaMallocAsync /
// cudaFreeAsync on gpu_active_stream) from the device's default memory pool,
// configured to keep freed memory cached: unlike cudaMalloc/cudaFree, neither
// call synchronizes the device, so short-lived scratch buffers are cheap.
// Defined in gpu_stream.cu.
void* pvda_device_alloc_bytes(size_t bytes);
void pvda_device_free_bytes(void* ptr);

inline bool is_gpu_device_pointer(const void* ptr)
{
	if (ptr == nullptr)
	{
		throw std::invalid_argument("Pointer is nullptr");
	}

	cudaPointerAttributes attr{};

	CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));

	switch (attr.type)
	{
		case cudaMemoryTypeUnregistered:
			return false;
		case cudaMemoryTypeDevice:
			return true;
		case cudaMemoryTypeHost:
			throw std::runtime_error("Pointer is pinned host memory, not normal CPU memory or GPU device memory");
		case cudaMemoryTypeManaged:
			throw std::runtime_error("Pointer is managed memory, not accepted as pure CPU or pure GPU pointer");
		default:
			throw std::runtime_error("Unknown CUDA pointer memory type");
	}
}

template <typename T>
class VEC_GPU {
   public:
	VEC_GPU() : d_ptr_(nullptr), size_(0) {}

	explicit VEC_GPU(size_t size) : VEC_GPU() { allocate(size); }

	VEC_GPU(const T* host_ptr, size_t size) : VEC_GPU() { copy_from_host(host_ptr, size); }

	VEC_GPU(const VEC_GPU& other) : VEC_GPU() { copy_from(other); }
	VEC_GPU& operator=(const VEC_GPU& other)
	{
		if (this != &other)
		{
			reset();
			copy_from(other);
		}
		return *this;
	}

	VEC_GPU(VEC_GPU&& other) noexcept { move_from(other); }
	VEC_GPU& operator=(VEC_GPU&& other) noexcept
	{
		if (this != &other)
		{
			reset();
			move_from(other);
		}
		return *this;
	}

	~VEC_GPU() { reset(); }

	void reset()
	{
		if (d_ptr_)
		{
			pvda_device_free_bytes(d_ptr_);
		}
		d_ptr_ = nullptr;
		size_  = 0;
	}

	void allocate(size_t size)
	{
		reset();
		if (size == 0) return;
		d_ptr_ = (T*)pvda_device_alloc_bytes(size * sizeof(T));
		size_  = size;
	}

	void copy_from_host(const T* host_ptr, size_t size)
	{
		if (size == 0)
		{
			reset();
			return;
		}
		if (!host_ptr)
		{
			fprintf(stderr, "VEC_GPU::copy_from_host: null host pointer\n");
			abort();
		}
		if (size_ != size) allocate(size);
		CUDA_CHECK(cudaMemcpyAsync(d_ptr_, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice, gpu_active_stream));
	}

	void copy_to_host(T* host_ptr, size_t size) const
	{
		if (size == 0) return;
		if (!host_ptr)
		{
			fprintf(stderr, "VEC_GPU::copy_to_host: null host pointer\n");
			abort();
		}
		if (size > size_)
		{
			fprintf(stderr, "VEC_GPU::copy_to_host: size exceeds device buffer\n");
			abort();
		}
		CUDA_CHECK(cudaMemcpyAsync(host_ptr, d_ptr_, size * sizeof(T), cudaMemcpyDeviceToHost, gpu_active_stream));
	}

	T* data() const { return d_ptr_; }
	size_t size() const { return size_; }

	explicit operator bool() const { return d_ptr_ != nullptr; }
	operator T*() { return d_ptr_; }
	operator const T*() const { return d_ptr_; }

   private:
	void copy_from(const VEC_GPU& other)
	{
		if (!other.d_ptr_ || other.size_ == 0) return;
		allocate(other.size_);
		CUDA_CHECK(cudaMemcpyAsync(d_ptr_, other.d_ptr_, other.size_ * sizeof(T), cudaMemcpyDeviceToDevice,
		                           gpu_active_stream));
	}

	void move_from(VEC_GPU& other)
	{
		d_ptr_       = other.d_ptr_;
		size_        = other.size_;
		other.d_ptr_ = nullptr;
		other.size_  = 0;
	}

	T* d_ptr_;
	size_t size_;
};

#endif  // __CUDACC__

#endif  // PVDA_GPU_COMMON_H
