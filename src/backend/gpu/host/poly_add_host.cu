#include <cuda_runtime.h>
#include <stdint.h>

#include "gpu/common/gpu_common.h"
#include "gpu/host/poly_add_host.h"
#include "gpu/kernel/poly_add_kernel.cuh"

static int launch_poly_add(const Data64* a, const Data64* b, Data64* c, uint64_t n, Modulus64 modulus,
                           cudaEvent_t ev_start, cudaEvent_t ev_stop)
{
	try
	{
		VEC_GPU<Data64> d_a(a, n);
		VEC_GPU<Data64> d_b(b, n);
		VEC_GPU<Data64> d_c(n);

		if (ev_start) cudaEventRecord(ev_start);

		poly_add_mod_kernel<<<(int)(((n) + 256 - 1) / 256), 256>>>(d_a, d_b, d_c, n, modulus);

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
		d_c.copy_to_host(c, n);
	} catch (const CudaException& e)
	{
		return -1;
	}

	return 0;
}

extern "C" {

int gpu_poly_add_mod(const uint64_t* a, const uint64_t* b, uint64_t* c, uint64_t n, uint64_t q)
{
	return launch_poly_add(a, b, c, n, Modulus64(q), nullptr, nullptr);
}

int gpu_poly_add_mod_timed(const uint64_t* a, const uint64_t* b, uint64_t* c, uint64_t n, uint64_t q,
                           float* gpu_kernel_ms)
{
	cudaEvent_t ev_start, ev_stop;
	cudaEventCreate(&ev_start);
	cudaEventCreate(&ev_stop);

	int ret = launch_poly_add(a, b, c, n, Modulus64(q), ev_start, ev_stop);

	if (ret == 0 && gpu_kernel_ms) cudaEventElapsedTime(gpu_kernel_ms, ev_start, ev_stop);

	cudaEventDestroy(ev_start);
	cudaEventDestroy(ev_stop);
	return ret;
}

}  // extern "C"
