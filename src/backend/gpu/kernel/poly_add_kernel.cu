#include "gpu/kernel/poly_add_kernel.cuh"

__global__ void poly_add_mod_kernel(const Data64* __restrict__ a, const Data64* __restrict__ b, Data64* __restrict__ c,
                                    uint64_t n, Modulus64 modulus)
{
	uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) c[idx] = OPERATOR_GPU_64::add(a[idx], b[idx], modulus);
}
