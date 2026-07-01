#ifndef PVDA_GPU_COMMON_H
#define PVDA_GPU_COMMON_H

#ifdef __CUDACC__
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

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

#define CUDA_CHECK(err)                                     \
	do                                                      \
	{                                                       \
		cudaError_t error = err;                            \
		if (error != cudaSuccess)                           \
		{                                                   \
			throw CudaException(__FILE__, __LINE__, error); \
		}                                                   \
	} while (0)

bool is_gpu_device_pointer(const void* ptr)
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
class VEC_GPU
{
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
			(void)cudaFree(d_ptr_);
		}
		d_ptr_ = nullptr;
		size_ = 0;
	}

	void allocate(size_t size)
	{
		reset();
		if (size == 0)
			return;
		CUDA_CHECK(cudaMalloc((void**)&d_ptr_, size * sizeof(T)));
		size_ = size;
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
		if (size_ != size)
			allocate(size);
		CUDA_CHECK(cudaMemcpy(d_ptr_, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
	}

	void copy_to_host(T* host_ptr, size_t size) const
	{
		if (size == 0)
			return;
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
		CUDA_CHECK(cudaMemcpy(host_ptr, d_ptr_, size * sizeof(T), cudaMemcpyDeviceToHost));
	}

	T*     data() const { return d_ptr_; }
	size_t size() const { return size_; }

	explicit operator bool() const { return d_ptr_ != nullptr; }
	operator T*() { return d_ptr_; }
	operator const T*() const { return d_ptr_; }

   private:
	void copy_from(const VEC_GPU& other)
	{
		if (!other.d_ptr_ || other.size_ == 0)
			return;
		allocate(other.size_);
		CUDA_CHECK(cudaMemcpy(d_ptr_, other.d_ptr_, other.size_ * sizeof(T), cudaMemcpyDeviceToDevice));
	}

	void move_from(VEC_GPU& other)
	{
		d_ptr_       = other.d_ptr_;
		size_        = other.size_;
		other.d_ptr_ = nullptr;
		other.size_  = 0;
	}

	T*     d_ptr_;
	size_t size_;
};

#endif  // __CUDACC__

#endif  // PVDA_GPU_COMMON_H
