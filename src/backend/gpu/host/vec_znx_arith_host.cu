#include "gpu/common/gpu_common.h"
#include "gpu/host/normalize_host.h"
#include "gpu/host/vec_znx_arith_host.h"
#include "gpu/host/vec_znx_rotate_automorphism_host.h"
#include "gpu/kernel/normalize_kernel.cuh"
#include "gpu/kernel/vec_znx_arith_kernel.cuh"
#include "gpu/kernel/vec_znx_rotate_automorphism_kernel.cuh"
#include "gpuntt/common/modular_arith.cuh"

namespace {

// Modular inverse of odd `p` mod `two_n` (a power of two), via Newton's
// iteration: for any odd x, x*x = 1 mod 8, and each iteration below doubles
// the number of correct low bits (3 -> 6 -> 12 -> 24 -> 48 -> 96), so 5
// iterations give the exact inverse mod 2^64; masking down to two_n (itself
// a power of two dividing 2^64) then gives the inverse mod two_n.
int64_t gpu_mod_inverse_pow2(int64_t p, int64_t two_n)
{
	uint64_t x   = (uint64_t)p;
	uint64_t inv = x;
	for (int i = 0; i < 5; ++i) inv = inv * (2 - x * inv);
	return (int64_t)(inv & (uint64_t)(two_n - 1));
}

// Element count spanning `size` limbs of stride sl (last limb's last
// coefficient index + 1) — the minimum contiguous upload needed to cover
// every limb of a (possibly strided/interleaved) PolyBiv view.
size_t span_elems(size_t size, int64_t sl, size_t n) { return size == 0 ? 0 : (size_t)((size - 1) * sl) + n; }

size_t min_sz(size_t a, size_t b) { return a < b ? a : b; }

}  // namespace

extern "C" {

void gpu_vec_znx_add(const int64_t* a_host, const int64_t* b_host, int64_t* res_host, size_t n, size_t l)
{
	size_t total = n * l;
	VEC_GPU<int64_t> d_a(a_host, total);
	VEC_GPU<int64_t> d_b(b_host, total);
	VEC_GPU<int64_t> d_res(total);

	int threads = 256;
	int blocks  = ((int)total + threads - 1) / threads;

	vec_znx_add_kernel<<<blocks, threads, 0, gpu_active_stream>>>(d_res.data(), d_a.data(), d_b.data(), (int)total);
	CUDA_CHECK(cudaGetLastError());

	// Host-facing: res_host must be valid the moment we return, and d_a/d_b/
	// d_res are about to be freed by VEC_GPU's destructors — must sync here.
	d_res.copy_to_host(res_host, total);
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

void gpu_vec_znx_sub(const int64_t* a_host, const int64_t* b_host, int64_t* res_host, size_t n, size_t l)
{
	size_t total = n * l;
	VEC_GPU<int64_t> d_a(a_host, total);
	VEC_GPU<int64_t> d_b(b_host, total);
	VEC_GPU<int64_t> d_res(total);

	int threads = 256;
	int blocks  = ((int)total + threads - 1) / threads;

	vec_znx_sub_kernel<<<blocks, threads, 0, gpu_active_stream>>>(d_res.data(), d_a.data(), d_b.data(), (int)total);
	CUDA_CHECK(cudaGetLastError());

	d_res.copy_to_host(res_host, total);
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

void gpu_vec_znx_negate(const int64_t* a_host, int64_t* res_host, size_t n, size_t l)
{
	size_t total = n * l;
	VEC_GPU<int64_t> d_a(a_host, total);
	VEC_GPU<int64_t> d_res(total);

	int threads = 256;
	int blocks  = ((int)total + threads - 1) / threads;

	vec_znx_negate_kernel<<<blocks, threads, 0, gpu_active_stream>>>(d_res.data(), d_a.data(), (int)total);
	CUDA_CHECK(cudaGetLastError());

	d_res.copy_to_host(res_host, total);
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

void gpu_vec_znx_rotate(int64_t p, const int64_t* a_host, int64_t* res_host, size_t n, size_t res_size, int64_t res_sl,
                        size_t a_size, int64_t a_sl)
{
	size_t common = min_sz(res_size, a_size);

	VEC_GPU<int64_t> d_a(a_host, span_elems(a_size, a_sl, n));
	VEC_GPU<int64_t> d_res(res_size * n);  // compact: limb i at [i*n, i*n+n)

	int total   = (int)(n * res_size);
	int threads = 256;
	int blocks  = (total + threads - 1) / threads;

	vec_znx_rotate_kernel<<<blocks, threads, 0, gpu_active_stream>>>(d_res.data(), d_a.data(), p, (int)n, (int)res_size,
	                                                                 (int)common, (int64_t)n, a_sl);
	CUDA_CHECK(cudaGetLastError());

	// Per-limb copy-out: res_sl may be > n (strided/interleaved res), and
	// the gaps between limbs belong to other polynomials — must not touch
	// them. Covers every res limb, including the zero-padded tail.
	for (size_t i = 0; i < res_size; ++i)
		CUDA_CHECK(cudaMemcpyAsync(res_host + i * res_sl, d_res.data() + i * n, n * sizeof(int64_t),
		                           cudaMemcpyDeviceToHost, gpu_active_stream));
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

void gpu_vec_znx_automorphism(int64_t p, const int64_t* a_host, int64_t* res_host, size_t n, size_t res_size,
                              int64_t res_sl, size_t a_size, int64_t a_sl)
{
	size_t common = min_sz(res_size, a_size);

	VEC_GPU<int64_t> d_a(a_host, span_elems(a_size, a_sl, n));
	VEC_GPU<int64_t> d_res(res_size * n);

	int64_t p_inv = gpu_mod_inverse_pow2(p, 2LL * (int64_t)n);

	int total   = (int)(n * res_size);
	int threads = 256;
	int blocks  = (total + threads - 1) / threads;

	vec_znx_automorphism_kernel<<<blocks, threads, 0, gpu_active_stream>>>(
	    d_res.data(), d_a.data(), p_inv, (int)n, (int)res_size, (int)common, (int64_t)n, a_sl);
	CUDA_CHECK(cudaGetLastError());

	for (size_t i = 0; i < res_size; ++i)
		CUDA_CHECK(cudaMemcpyAsync(res_host + i * res_sl, d_res.data() + i * n, n * sizeof(int64_t),
		                           cudaMemcpyDeviceToHost, gpu_active_stream));
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

// Whether the limb vectors [a, a_size limbs, stride a_sl) and [res, res_size limbs, stride
// res_sl) share memory, in which case the result goes through a scratch buffer
static bool limbs_overlap(const int64_t* a, size_t a_size, int64_t a_sl, const int64_t* res, size_t res_size,
                          int64_t res_sl, size_t n)
{
	if (a_size == 0 || res_size == 0) return false;
	const int64_t* a_end   = a + (a_size - 1) * a_sl + n;
	const int64_t* res_end = res + (res_size - 1) * res_sl + n;
	return a < res_end && res < a_end;
}

// Copies res_size limbs of n coefficients from a contiguous scratch into res (stride res_sl)
static void copy_from_scratch(int64_t* res, int64_t res_sl, const int64_t* scratch, size_t res_size, size_t n)
{
	CUDA_CHECK(cudaMemcpy2DAsync(res, (size_t)res_sl * sizeof(int64_t), scratch, n * sizeof(int64_t),
	                             n * sizeof(int64_t), res_size, cudaMemcpyDeviceToDevice, gpu_active_stream));
}

void gpu_vec_znx_rotate_device(int64_t p, const int64_t* a_dev, int64_t* res_dev, size_t n, size_t res_size,
                               int64_t res_sl, size_t a_size, int64_t a_sl)
{
	if (res_size == 0) return;
	size_t common = min_sz(res_size, a_size);
	int total     = (int)(n * res_size);
	int threads   = 256;
	int blocks    = (total + threads - 1) / threads;

	if (!limbs_overlap(a_dev, a_size, a_sl, res_dev, res_size, res_sl, n))
	{
		vec_znx_rotate_kernel<<<blocks, threads, 0, gpu_active_stream>>>(res_dev, a_dev, p, (int)n, (int)res_size,
		                                                                 (int)common, res_sl, a_sl);
		CUDA_CHECK(cudaGetLastError());
		return;
	}
	VEC_GPU<int64_t> d_scratch(res_size * n);  // released stream-ordered, no synchronization needed
	vec_znx_rotate_kernel<<<blocks, threads, 0, gpu_active_stream>>>(d_scratch.data(), a_dev, p, (int)n, (int)res_size,
	                                                                 (int)common, (int64_t)n, a_sl);
	CUDA_CHECK(cudaGetLastError());
	copy_from_scratch(res_dev, res_sl, d_scratch.data(), res_size, n);
}

void gpu_vec_znx_automorphism_device(int64_t p, const int64_t* a_dev, int64_t* res_dev, size_t n, size_t res_size,
                                     int64_t res_sl, size_t a_size, int64_t a_sl)
{
	if (res_size == 0) return;
	if (!limbs_overlap(a_dev, a_size, a_sl, res_dev, res_size, res_sl, n))
	{
		gpu_vec_znx_automorphism_device_noalias(p, a_dev, res_dev, n, res_size, res_sl, a_size, a_sl);
		return;
	}
	size_t common = min_sz(res_size, a_size);
	int64_t p_inv = gpu_mod_inverse_pow2(p, 2LL * (int64_t)n);
	int total     = (int)(n * res_size);
	int threads   = 256;
	int blocks    = (total + threads - 1) / threads;

	VEC_GPU<int64_t> d_scratch(res_size * n);  // released stream-ordered, no synchronization needed
	vec_znx_automorphism_kernel<<<blocks, threads, 0, gpu_active_stream>>>(
	    d_scratch.data(), a_dev, p_inv, (int)n, (int)res_size, (int)common, (int64_t)n, a_sl);
	CUDA_CHECK(cudaGetLastError());
	copy_from_scratch(res_dev, res_sl, d_scratch.data(), res_size, n);
}

void gpu_vec_znx_automorphism_device_noalias(int64_t p, const int64_t* a_dev, int64_t* res_dev, size_t n,
                                             size_t res_size, int64_t res_sl, size_t a_size, int64_t a_sl)
{
	size_t common = min_sz(res_size, a_size);
	int64_t p_inv = gpu_mod_inverse_pow2(p, 2LL * (int64_t)n);

	int total   = (int)(n * res_size);
	int threads = 256;
	int blocks  = (total + threads - 1) / threads;

	vec_znx_automorphism_kernel<<<blocks, threads, 0, gpu_active_stream>>>(
	    res_dev, a_dev, p_inv, (int)n, (int)res_size, (int)common, res_sl, a_sl);
	CUDA_CHECK(cudaGetLastError());
}

void gpu_normalize_base2k(const int64_t* a_host, int64_t* res_host, size_t n, size_t l, uint32_t kappa)
{
	VEC_GPU<Data64s> d_a(a_host, n * l);
	VEC_GPU<Data64s> d_res(n * l);

	int threads = 256;
	int blocks  = ((int)n + threads - 1) / threads;

	// a_size == res_size == l, same stride (n) for both — the sized kernel's
	// two extra loops that handle differing limb counts simply run zero
	// iterations in that case (see normalize_base2k_sized_kernel's doc
	// comment), so there is no separate non-sized kernel to call here.
	normalize_base2k_sized_kernel<<<blocks, threads, 0, gpu_active_stream>>>(
	    (int64_t*)d_res.data(), (int)l, (int64_t)n, (const int64_t*)d_a.data(), (int)l, (int64_t)n, (int)n,
	    (int)kappa);
	CUDA_CHECK(cudaGetLastError());

	d_res.copy_to_host(res_host, n * l);
	CUDA_CHECK(cudaStreamSynchronize(gpu_active_stream));
}

void gpu_normalize_base2k_sized_device(const int64_t* a_dev, size_t a_size, int64_t a_stride, int64_t* res_dev,
                                       size_t res_size, int64_t res_stride, size_t n, uint32_t kappa)
{
	int threads = 256;
	int blocks  = ((int)n + threads - 1) / threads;

	normalize_base2k_sized_kernel<<<blocks, threads, 0, gpu_active_stream>>>(res_dev, (int)res_size, res_stride, a_dev,
	                                                                         (int)a_size, a_stride, (int)n, (int)kappa);
	CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"
