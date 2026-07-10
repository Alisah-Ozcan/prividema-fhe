#include <cmath>

#include "gpu/common/gpu_common.h"
#include "gpu/common/gpu_nttparameters.h"
#include "gpu/host/ntt_poly_mult_host.h"
#include "gpu/kernel/ntt_poly_mult_kernel.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"

extern "C" {

void gpu_ntt_svp(const int64_t* a_host, const int64_t* b_host, int64_t* c_host, size_t n, size_t batch)
{
	NTTParameterGenerator& gen = NTTParameterGenerator::instance();
	gen.initialize(n);

	Modulus64 mod    = gen.get_modulus();
	Root64* ntt_tab  = gen.get_ntt_table(n);
	Root64* intt_tab = gen.get_intt_table(n);
	Ninverse64 n_inv = gen.get_n_inv(n);

	int n_power = (int)std::log2((double)n);
	int threads = 256;
	int total   = (int)(n * batch);
	int blocks  = (total + threads - 1) / threads;

	gpuntt::ntt_configuration<Data64> cfg_fwd = {.n_power        = n_power,
	                                             .ntt_type       = gpuntt::FORWARD,
	                                             .ntt_layout     = gpuntt::PerPolynomial,
	                                             .reduction_poly = gpuntt::X_N_plus,
	                                             .zero_padding   = false,
	                                             .mod_inverse    = 0,
	                                             .stream         = gpu_active_stream};

	// A: single polynomial NTT
	VEC_GPU<Data64s> d_a_in(a_host, n);
	VEC_GPU<Data64> d_a(n);
	gpuntt::GPU_NTT(d_a_in.data(), d_a.data(), ntt_tab, mod, cfg_fwd, 1);

	// B: batch polynomials NTT
	VEC_GPU<Data64s> d_b_in(b_host, n * batch);
	VEC_GPU<Data64> d_b(n * batch);
	gpuntt::GPU_NTT(d_b_in.data(), d_b.data(), ntt_tab, mod, cfg_fwd, (int)batch);

	// Pointwise multiply: d_b[j*n+i] *= d_a[i]
	pointwise_mult_batch_kernel<<<blocks, threads, 0, gpu_active_stream>>>(d_b.data(), d_a.data(), mod, (int)n,
	                                                                       (int)batch);
	CUDA_CHECK(cudaGetLastError());

	// Batch inverse NTT
	gpuntt::ntt_configuration<Data64> cfg_inv = {.n_power        = n_power,
	                                             .ntt_type       = gpuntt::INVERSE,
	                                             .ntt_layout     = gpuntt::PerPolynomial,
	                                             .reduction_poly = gpuntt::X_N_plus,
	                                             .zero_padding   = false,
	                                             .mod_inverse    = n_inv,
	                                             .stream         = gpu_active_stream};
	VEC_GPU<Data64s> d_c(n * batch);
	gpuntt::GPU_INTT(d_b.data(), d_c.data(), intt_tab, mod, cfg_inv, (int)batch);

	d_c.copy_to_host(c_host, n * batch);
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

void gpu_ntt_vmp(const int64_t* a_host, const int64_t* m_host, int64_t* c_host, size_t n, size_t nrows, size_t ncols)
{
	NTTParameterGenerator& gen = NTTParameterGenerator::instance();
	gen.initialize(n);

	Modulus64 mod    = gen.get_modulus();
	Root64* ntt_tab  = gen.get_ntt_table(n);
	Root64* intt_tab = gen.get_intt_table(n);
	Ninverse64 n_inv = gen.get_n_inv(n);

	int n_power = (int)std::log2((double)n);
	int threads = 256;

	gpuntt::ntt_configuration<Data64> cfg_fwd = {.n_power        = n_power,
	                                             .ntt_type       = gpuntt::FORWARD,
	                                             .ntt_layout     = gpuntt::PerPolynomial,
	                                             .reduction_poly = gpuntt::X_N_plus,
	                                             .zero_padding   = false,
	                                             .mod_inverse    = 0,
	                                             .stream         = gpu_active_stream};

	// A vector: nrows polynomials, batched NTT
	VEC_GPU<Data64s> d_a_in(a_host, n * nrows);
	VEC_GPU<Data64> d_a(n * nrows);
	gpuntt::GPU_NTT(d_a_in.data(), d_a.data(), ntt_tab, mod, cfg_fwd, (int)nrows);

	// M matrix: nrows*ncols polynomials, batched NTT
	VEC_GPU<Data64s> d_m_in(m_host, n * nrows * ncols);
	VEC_GPU<Data64> d_m(n * nrows * ncols);
	gpuntt::GPU_NTT(d_m_in.data(), d_m.data(), ntt_tab, mod, cfg_fwd, (int)(nrows * ncols));

	// VMP accumulator: c_ntt[j*n+k] = sum_i a_ntt[i*n+k] * m_ntt[(i*ncols+j)*n+k]
	VEC_GPU<Data64> d_c_ntt(n * ncols);
	int total  = (int)(n * ncols);
	int blocks = (total + threads - 1) / threads;
	vmp_accumulate_kernel<<<blocks, threads, 0, gpu_active_stream>>>(d_c_ntt.data(), d_a.data(), d_m.data(), mod,
	                                                                 (int)n, (int)nrows, (int)ncols);
	CUDA_CHECK(cudaGetLastError());

	// Batched inverse NTT: ncols polynomials
	gpuntt::ntt_configuration<Data64> cfg_inv = {.n_power        = n_power,
	                                             .ntt_type       = gpuntt::INVERSE,
	                                             .ntt_layout     = gpuntt::PerPolynomial,
	                                             .reduction_poly = gpuntt::X_N_plus,
	                                             .zero_padding   = false,
	                                             .mod_inverse    = n_inv,
	                                             .stream         = gpu_active_stream};
	VEC_GPU<Data64s> d_c(n * ncols);
	gpuntt::GPU_INTT(d_c_ntt.data(), d_c.data(), intt_tab, mod, cfg_inv, (int)ncols);

	d_c.copy_to_host(c_host, n * ncols);
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

}  // extern "C"
