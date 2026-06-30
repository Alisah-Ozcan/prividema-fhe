#include "gpu/kernel/poly_add_kernel.cuh"

__global__ void poly_add_mod_kernel(const int64_t* __restrict__ a, const int64_t* __restrict__ b,
                                    int64_t* __restrict__ c, uint64_t n, int64_t q)
{
	uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		int64_t sum = a[idx] + b[idx];
		sum         = sum % q;
		if (sum < 0) sum += q;
		c[idx] = sum;
	}
}
