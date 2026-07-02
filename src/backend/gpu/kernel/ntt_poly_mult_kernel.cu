#include "gpu/kernel/ntt_poly_mult_kernel.cuh"

__global__ void pointwise_mult_kernel(Data64* a, const Data64* b, Modulus64 mod, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) a[idx] = OPERATOR_GPU_64::mult(a[idx], b[idx], mod);
}

__global__ void pointwise_mult_batch_kernel(Data64* b_batch, const Data64* a_ntt, Modulus64 mod, int n, int batch)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n * batch) b_batch[idx] = OPERATOR_GPU_64::mult(b_batch[idx], a_ntt[idx % n], mod);
}

__global__ void vmp_accumulate_kernel(Data64* c_ntt, const Data64* a_ntt, const Data64* m_ntt, Modulus64 mod, int n,
                                      int nrows, int ncols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ncols * n) return;

	int j = idx / n; // output polynomial index
	int k = idx % n; // coefficient index

	Data64 acc = 0;
	for (int i = 0; i < nrows; i++)
	{
		Data64 prod = OPERATOR_GPU_64::mult(a_ntt[i * n + k], m_ntt[(i * ncols + j) * n + k], mod);
		acc += prod;
		if (acc >= mod.value) acc -= mod.value;
	}
	c_ntt[idx] = acc;
}
