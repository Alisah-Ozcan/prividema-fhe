#include <cmath>

#include "gpu/common/gpu_common.h"
#include "gpu/common/gpu_nttparameters.h"
#include "gpu/host/vtable_ops_host.h"
#include "gpuntt/ntt_merge/ntt.cuh"

namespace {

constexpr int kThreads = 256;

inline int blocks_for(size_t total) { return (int)((total + kThreads - 1) / kThreads); }

// res[i] = a[i] + sign * b[i] limb-wise; a missing limb (i >= a_l or b_l) reads as zero
__global__ void vec_znx_addsub_kernel(int64_t* res, int res_l, int64_t res_sl, const int64_t* a, int a_l, int64_t a_sl,
                                      const int64_t* b, int b_l, int64_t b_sl, int n, int64_t sign)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= (size_t)res_l * n) return;
	int limb = (int)(idx / n);
	int coef = (int)(idx - (size_t)limb * n);

	int64_t va                = limb < a_l ? a[limb * a_sl + coef] : 0;
	int64_t vb                = limb < b_l ? b[limb * b_sl + coef] : 0;
	res[limb * res_sl + coef] = va + sign * vb;
}

__global__ void vec_znx_negate_kernel_strided(int64_t* res, int res_l, int64_t res_sl, const int64_t* a, int a_l,
                                              int64_t a_sl, int n)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= (size_t)res_l * n) return;
	int limb = (int)(idx / n);
	int coef = (int)(idx - (size_t)limb * n);

	res[limb * res_sl + coef] = limb < a_l ? -a[limb * a_sl + coef] : 0;
}

// res[j] = sum_{i < rows} a[i] * pmat[i][j] (mod p) for j < min(res_size, ncols), zero beyond
__global__ void vmp_ntt_kernel(Data64* res, int res_size, const Data64* a, int rows, const Data64* pmat, int ncols,
                               int n, Modulus64 mod)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= (size_t)res_size * n) return;
	int j = (int)(idx / n);
	int k = (int)(idx - (size_t)j * n);

	Data64 acc = 0;
	if (j < ncols)
	{
		for (int i = 0; i < rows; ++i)
		{
			acc += OPERATOR_GPU_64::mult(a[(size_t)i * n + k], pmat[((size_t)i * ncols + j) * n + k], mod);
			if (acc >= mod.value) acc -= mod.value;
		}
	}
	res[idx] = acc;
}

// res[j] = ppol * a[j] (mod p) for j < min(res_size, a_size), zero beyond
__global__ void svp_ntt_kernel(Data64* res, int res_size, const Data64* ppol, const Data64* a, int a_size, int n,
                               Modulus64 mod)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= (size_t)res_size * n) return;
	int j = (int)(idx / n);
	int k = (int)(idx - (size_t)j * n);

	res[idx] = j < a_size ? OPERATOR_GPU_64::mult(a[idx], ppol[k], mod) : 0;
}

struct NttContext
{
	Modulus64 mod;
	gpuntt::ntt_configuration<Data64> fwd;
	gpuntt::ntt_configuration<Data64> inv;
	Root64* ntt_table;
	Root64* intt_table;
};

NttContext ntt_context(size_t n)
{
	NTTParameterGenerator& gen = NTTParameterGenerator::instance();
	gen.initialize(n);
	int n_power = (int)std::log2((double)n);

	NttContext ctx;
	ctx.mod        = gen.get_modulus();
	ctx.ntt_table  = gen.get_ntt_table(n);
	ctx.intt_table = gen.get_intt_table(n);
	ctx.fwd        = {.n_power        = n_power,
	                  .ntt_type       = gpuntt::FORWARD,
	                  .ntt_layout     = gpuntt::PerPolynomial,
	                  .reduction_poly = gpuntt::X_N_plus,
	                  .zero_padding   = false,
	                  .mod_inverse    = 0,
	                  .stream         = gpu_active_stream};
	ctx.inv        = {.n_power        = n_power,
	                  .ntt_type       = gpuntt::INVERSE,
	                  .ntt_layout     = gpuntt::PerPolynomial,
	                  .reduction_poly = gpuntt::X_N_plus,
	                  .zero_padding   = false,
	                  .mod_inverse    = gen.get_n_inv(n),
	                  .stream         = gpu_active_stream};
	return ctx;
}

// Zeroes limbs [from, to) of a limb vector with stride sl
void zero_limbs(int64_t* v, size_t from, size_t to, int64_t sl, size_t n)
{
	if (to <= from) return;
	CUDA_CHECK(cudaMemset2DAsync(v + from * sl, (size_t)sl * sizeof(int64_t), 0, n * sizeof(int64_t), to - from,
	                             gpu_active_stream));
}

// Copies `limbs` limbs of n coefficients between limb vectors of strides src_sl / dst_sl
void copy_limbs(int64_t* dst, int64_t dst_sl, const int64_t* src, int64_t src_sl, size_t limbs, size_t n)
{
	if (limbs == 0) return;
	CUDA_CHECK(cudaMemcpy2DAsync(dst, (size_t)dst_sl * sizeof(int64_t), src, (size_t)src_sl * sizeof(int64_t),
	                             n * sizeof(int64_t), limbs, cudaMemcpyDeviceToDevice, gpu_active_stream));
}

}  // namespace

extern "C" {

void gpu_vt_vec_znx_add(int64_t* res, size_t res_l, int64_t res_sl, const int64_t* a, size_t a_l, int64_t a_sl,
                        const int64_t* b, size_t b_l, int64_t b_sl, size_t n)
{
	return pvda_gpu_guard([&]() -> void {
		if (res_l == 0) return;
		vec_znx_addsub_kernel<<<blocks_for(res_l * n), kThreads, 0, gpu_active_stream>>>(
		    res, (int)res_l, res_sl, a, (int)a_l, a_sl, b, (int)b_l, b_sl, (int)n, 1);
		CUDA_CHECK(cudaGetLastError());
	});
}

void gpu_vt_vec_znx_sub(int64_t* res, size_t res_l, int64_t res_sl, const int64_t* a, size_t a_l, int64_t a_sl,
                        const int64_t* b, size_t b_l, int64_t b_sl, size_t n)
{
	return pvda_gpu_guard([&]() -> void {
		if (res_l == 0) return;
		vec_znx_addsub_kernel<<<blocks_for(res_l * n), kThreads, 0, gpu_active_stream>>>(
		    res, (int)res_l, res_sl, a, (int)a_l, a_sl, b, (int)b_l, b_sl, (int)n, -1);
		CUDA_CHECK(cudaGetLastError());
	});
}

void gpu_vt_vec_znx_negate(int64_t* res, size_t res_l, int64_t res_sl, const int64_t* a, size_t a_l, int64_t a_sl,
                           size_t n)
{
	return pvda_gpu_guard([&]() -> void {
		if (res_l == 0) return;
		vec_znx_negate_kernel_strided<<<blocks_for(res_l * n), kThreads, 0, gpu_active_stream>>>(
		    res, (int)res_l, res_sl, a, (int)a_l, a_sl, (int)n);
		CUDA_CHECK(cudaGetLastError());
	});
}

void gpu_vt_vec_znx_dft(int64_t* res_ntt, size_t res_size, const int64_t* a, size_t a_l, int64_t a_sl, size_t n)
{
	return pvda_gpu_guard([&]() -> void {
		size_t rows = a_l < res_size ? a_l : res_size;
		if (rows > 0)
		{
			NttContext ctx = ntt_context(n);
			if (a_sl == (int64_t)n)
			{
				gpuntt::GPU_NTT((Data64s*)a, (Data64*)res_ntt, ctx.ntt_table, ctx.mod, ctx.fwd, (int)rows);
			}
			else
			{
				VEC_GPU<int64_t> packed(rows * n);
				copy_limbs(packed.data(), (int64_t)n, a, a_sl, rows, n);
				gpuntt::GPU_NTT((Data64s*)packed.data(), (Data64*)res_ntt, ctx.ntt_table, ctx.mod, ctx.fwd, (int)rows);
			}
		}
		zero_limbs(res_ntt, rows, res_size, (int64_t)n, n);
	});
}

void gpu_vt_vec_znx_idft(int64_t* res, size_t res_l, int64_t res_sl, const int64_t* a_ntt, size_t a_size, size_t n)
{
	return pvda_gpu_guard([&]() -> void {
		size_t rows = a_size < res_l ? a_size : res_l;
		if (rows > 0)
		{
			NttContext ctx = ntt_context(n);
			if (res_sl == (int64_t)n)
			{
				gpuntt::GPU_INTT((Data64*)a_ntt, (Data64s*)res, ctx.intt_table, ctx.mod, ctx.inv, (int)rows);
			}
			else
			{
				VEC_GPU<int64_t> packed(rows * n);
				gpuntt::GPU_INTT((Data64*)a_ntt, (Data64s*)packed.data(), ctx.intt_table, ctx.mod, ctx.inv, (int)rows);
				copy_limbs(res, res_sl, packed.data(), (int64_t)n, rows, n);
			}
		}
		zero_limbs(res, rows, res_l, res_sl, n);
	});
}

void gpu_vt_vmp_prepare(int64_t* pmat, const int64_t* mat, size_t nrows, size_t ncols, size_t n)
{
	return pvda_gpu_guard([&]() -> void {
		if (nrows * ncols == 0) return;
		NttContext ctx = ntt_context(n);
		gpuntt::GPU_NTT((Data64s*)mat, (Data64*)pmat, ctx.ntt_table, ctx.mod, ctx.fwd, (int)(nrows * ncols));
	});
}

void gpu_vt_vmp_apply_ntt(int64_t* res_ntt, size_t res_size, const int64_t* a_ntt, size_t a_rows, const int64_t* pmat,
                          size_t nrows, size_t ncols, size_t n)
{
	return pvda_gpu_guard([&]() -> void {
		if (res_size == 0) return;
		NttContext ctx = ntt_context(n);
		size_t rows    = a_rows < nrows ? a_rows : nrows;
		vmp_ntt_kernel<<<blocks_for(res_size * n), kThreads, 0, gpu_active_stream>>>(
		    (Data64*)res_ntt, (int)res_size, (const Data64*)a_ntt, (int)rows, (const Data64*)pmat, (int)ncols, (int)n,
		    ctx.mod);
		CUDA_CHECK(cudaGetLastError());
	});
}

void gpu_vt_svp_prepare(int64_t* ppol, const int64_t* pol, size_t n)
{
	return pvda_gpu_guard([&]() -> void {
		NttContext ctx = ntt_context(n);
		gpuntt::GPU_NTT((Data64s*)pol, (Data64*)ppol, ctx.ntt_table, ctx.mod, ctx.fwd, 1);
	});
}

void gpu_vt_svp_apply_ntt(int64_t* res_ntt, size_t res_size, const int64_t* ppol, const int64_t* a_ntt, size_t a_size,
                          size_t n)
{
	return pvda_gpu_guard([&]() -> void {
		if (res_size == 0) return;
		NttContext ctx = ntt_context(n);
		svp_ntt_kernel<<<blocks_for(res_size * n), kThreads, 0, gpu_active_stream>>>(
		    (Data64*)res_ntt, (int)res_size, (const Data64*)ppol, (const Data64*)a_ntt, (int)a_size, (int)n, ctx.mod);
		CUDA_CHECK(cudaGetLastError());
	});
}

}  // extern "C"
