#include "gpu/common/gpu_common.h"
#include "gpu/host/vec_znx_arith_host.h"
#include "gpu/kernel/vec_znx_arith_kernel.cuh"

extern "C" {

void gpu_vec_znx_add(const int64_t* a_host, const int64_t* b_host, int64_t* res_host, size_t n, size_t l)
{
	size_t total = n * l;
	VEC_GPU<int64_t> d_a(a_host, total);
	VEC_GPU<int64_t> d_b(b_host, total);
	VEC_GPU<int64_t> d_res(total);

	int threads = 256;
	int blocks  = ((int)total + threads - 1) / threads;

	vec_znx_add_kernel<<<blocks, threads>>>(d_res.data(), d_a.data(), d_b.data(), (int)total);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	d_res.copy_to_host(res_host, total);
}

void gpu_vec_znx_sub(const int64_t* a_host, const int64_t* b_host, int64_t* res_host, size_t n, size_t l)
{
	size_t total = n * l;
	VEC_GPU<int64_t> d_a(a_host, total);
	VEC_GPU<int64_t> d_b(b_host, total);
	VEC_GPU<int64_t> d_res(total);

	int threads = 256;
	int blocks  = ((int)total + threads - 1) / threads;

	vec_znx_sub_kernel<<<blocks, threads>>>(d_res.data(), d_a.data(), d_b.data(), (int)total);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	d_res.copy_to_host(res_host, total);
}

void gpu_vec_znx_negate(const int64_t* a_host, int64_t* res_host, size_t n, size_t l)
{
	size_t total = n * l;
	VEC_GPU<int64_t> d_a(a_host, total);
	VEC_GPU<int64_t> d_res(total);

	int threads = 256;
	int blocks  = ((int)total + threads - 1) / threads;

	vec_znx_negate_kernel<<<blocks, threads>>>(d_res.data(), d_a.data(), (int)total);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	d_res.copy_to_host(res_host, total);
}

void gpu_vec_znx_add_device(const int64_t* a_dev, const int64_t* b_dev, int64_t* res_dev, size_t total)
{
	int threads = 256;
	int blocks  = ((int)total + threads - 1) / threads;

	vec_znx_add_kernel<<<blocks, threads>>>(res_dev, a_dev, b_dev, (int)total);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_vec_znx_sub_device(const int64_t* a_dev, const int64_t* b_dev, int64_t* res_dev, size_t total)
{
	int threads = 256;
	int blocks  = ((int)total + threads - 1) / threads;

	vec_znx_sub_kernel<<<blocks, threads>>>(res_dev, a_dev, b_dev, (int)total);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_vec_znx_negate_device(const int64_t* a_dev, int64_t* res_dev, size_t total)
{
	int threads = 256;
	int blocks  = ((int)total + threads - 1) / threads;

	vec_znx_negate_kernel<<<blocks, threads>>>(res_dev, a_dev, (int)total);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

}  // extern "C"
