#include "gpu/common/gpu_common.h"
#include "gpu/host/normalize_host.h"
#include "gpu/kernel/normalize_kernel.cuh"
#include "gpuntt/common/modular_arith.cuh"

extern "C" {

void gpu_normalize_base2k(const int64_t* a_host, int64_t* res_host, size_t n, size_t l, uint32_t kappa)
{
	VEC_GPU<Data64s> d_a(a_host, n * l);
	VEC_GPU<Data64s> d_res(n * l);

	int threads = 256;
	int blocks  = ((int)n + threads - 1) / threads;

	normalize_base2k_kernel<<<blocks, threads>>>((int64_t*)d_res.data(), (const int64_t*)d_a.data(), (int)n, (int)l,
	                                             (int64_t)n, (int)kappa);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	d_res.copy_to_host(res_host, n * l);
}

void gpu_normalize_base2k_device(const int64_t* a_dev, int64_t* res_dev, size_t n, size_t l, int64_t stride,
                                 uint32_t kappa)
{
	int threads = 256;
	int blocks  = ((int)n + threads - 1) / threads;

	normalize_base2k_kernel<<<blocks, threads>>>(res_dev, a_dev, (int)n, (int)l, stride, (int)kappa);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

}  // extern "C"
