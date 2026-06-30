#ifndef PVDA_GPU_COMMON_H
#define PVDA_GPU_COMMON_H

#ifdef __CUDACC__
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

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

#endif  // __CUDACC__

#endif  // PVDA_GPU_COMMON_H
