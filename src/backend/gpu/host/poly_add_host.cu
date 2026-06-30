#include <cuda_runtime.h>
#include <stdint.h>

#include "gpu/common/gpu_common.h"
#include "gpu/host/poly_add_host.h"
#include "gpu/kernel/poly_add_kernel.cuh"

static int launch_poly_add(const int64_t* a, const int64_t* b, int64_t* c, uint64_t n, int64_t q, cudaEvent_t ev_start,
                           cudaEvent_t ev_stop)
{
	int64_t *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
	size_t bytes = n * sizeof(int64_t);

	try
	{
		CUDA_CHECK(cudaMalloc(&d_a, bytes));
		CUDA_CHECK(cudaMalloc(&d_b, bytes));
		CUDA_CHECK(cudaMalloc(&d_c, bytes));

		CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

		if (ev_start) cudaEventRecord(ev_start);

		poly_add_mod_kernel<<<(int)(((n) + 256 - 1) / 256), 256>>>(d_a, d_b, d_c, n, q);

		if (ev_stop)
		{
			cudaEventRecord(ev_stop);
			cudaEventSynchronize(ev_stop);
		}
		else
		{
			CUDA_CHECK(cudaDeviceSynchronize());
		}

		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));
	} catch (const CudaException& e)
	{
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		return -1;
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}

extern "C" {

int gpu_poly_add_mod(const int64_t* a, const int64_t* b, int64_t* c, uint64_t n, int64_t q)
{
	return launch_poly_add(a, b, c, n, q, nullptr, nullptr);
}

int gpu_poly_add_mod_timed(const int64_t* a, const int64_t* b, int64_t* c, uint64_t n, int64_t q, float* gpu_kernel_ms)
{
	cudaEvent_t ev_start, ev_stop;
	cudaEventCreate(&ev_start);
	cudaEventCreate(&ev_stop);

	int ret = launch_poly_add(a, b, c, n, q, ev_start, ev_stop);

	if (ret == 0 && gpu_kernel_ms) cudaEventElapsedTime(gpu_kernel_ms, ev_start, ev_stop);

	cudaEventDestroy(ev_start);
	cudaEventDestroy(ev_stop);
	return ret;
}

}  // extern "C"
