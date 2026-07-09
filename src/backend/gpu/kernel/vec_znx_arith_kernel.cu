#include "gpu/kernel/vec_znx_arith_kernel.cuh"

__global__ void vec_znx_add_kernel(int64_t* res, const int64_t* a, const int64_t* b, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;
	res[idx] = a[idx] + b[idx];
}

__global__ void vec_znx_sub_kernel(int64_t* res, const int64_t* a, const int64_t* b, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;
	res[idx] = a[idx] - b[idx];
}

__global__ void vec_znx_negate_kernel(int64_t* res, const int64_t* a, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;
	res[idx] = -a[idx];
}
