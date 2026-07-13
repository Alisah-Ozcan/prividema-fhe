#include "gpu/kernel/normalize_kernel.cuh"
#include "gpu/kernel/vec_znx_arith_kernel.cuh"
#include "gpu/kernel/vec_znx_rotate_automorphism_kernel.cuh"

__global__ void vec_znx_add_kernel(int64_t* res, const int64_t* a, const int64_t* b, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;
	res[idx] = a[idx] + b[idx];
}

__global__ void vec_znx_sub_kernel(int64_t* res, const int64_t* a, const int64_t* b, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;
	res[idx] = a[idx] - b[idx];
}

__global__ void vec_znx_negate_kernel(int64_t* res, const int64_t* a, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;
	res[idx] = -a[idx];
}

__global__ void vec_znx_rotate_kernel(int64_t* res, const int64_t* a, int64_t p, int n, int res_size, int common_limbs,
                                      int64_t res_sl, int64_t a_sl)
{
	int idx   = blockIdx.x * blockDim.x + threadIdx.x;
	int total = n * res_size;
	if (idx >= total) return;

	int limb = idx / n;
	int j    = idx % n;

	int64_t* res_limb = res + (int64_t)limb * res_sl;

	if (limb >= common_limbs)
	{
		res_limb[j] = 0;
		return;
	}

	const int64_t* a_limb = a + (int64_t)limb * a_sl;

	int64_t two_n = 2LL * n;
	int64_t off   = (-p) & (two_n - 1);
	if (off < n)
	{
		int64_t nma = n - off;
		res_limb[j] = (j < nma) ? a_limb[j + off] : -a_limb[j - nma];
	}
	else
	{
		off -= n;
		int64_t nma = n - off;
		res_limb[j] = (j < nma) ? -a_limb[j + off] : a_limb[j - nma];
	}
}

__global__ void vec_znx_automorphism_kernel(int64_t* res, const int64_t* a, int64_t p_inv, int n, int res_size,
                                            int common_limbs, int64_t res_sl, int64_t a_sl)
{
	int idx   = blockIdx.x * blockDim.x + threadIdx.x;
	int total = n * res_size;
	if (idx >= total) return;

	int limb = idx / n;
	int j    = idx % n;

	int64_t* res_limb = res + (int64_t)limb * res_sl;

	if (limb >= common_limbs)
	{
		res_limb[j] = 0;
		return;
	}

	const int64_t* a_limb = a + (int64_t)limb * a_sl;

	int64_t two_n = 2LL * n;
	int64_t raw   = ((int64_t)j * p_inv) & (two_n - 1);
	res_limb[j]   = (raw < n) ? a_limb[raw] : -a_limb[raw - n];
}

/* Derived directly from the spqlios coeffs_arithmetic.c reference implementation.
 *
 * get_digit : returns the lowest kappa bits of x as a signed integer.
 *   (x << (64-kappa)) >> (64-kappa)  →  [-2^(k-1), 2^(k-1))
 *
 * get_carry : quotient of the remainder after subtracting the digit, divided by 2^kappa.
 *   (x - digit) >> kappa
 */
__device__ static inline int64_t get_digit(int64_t x, int kappa) { return (x << (64 - kappa)) >> (64 - kappa); }

__device__ static inline int64_t get_carry(int64_t x, int64_t digit, int kappa) { return (x - digit) >> kappa; }

__global__ void normalize_base2k_sized_kernel(int64_t* res, int res_size, int64_t res_sl, const int64_t* a, int a_size,
                                              int64_t a_sl, int n, int kappa)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= n) return;

	int64_t cin = 0;

	// Propagate carry through any of a's limbs beyond res_size — computed,
	// not written (mirrors vec_znx_normalize_base2k_ref's discard loop).
	int i = a_size - 1;
	for (; i >= res_size; i--)
	{
		int64_t x     = a[i * a_sl + k];
		int64_t digit = get_digit(x, kappa);
		int64_t carry = get_carry(x, digit, kappa);
		cin           = carry + get_carry(digit + cin, get_digit(digit + cin, kappa), kappa);
	}

	// Least-significant to most-significant within [1, min(a_size,res_size)).
	for (; i >= 1; i--)
	{
		int64_t x     = a[i * a_sl + k];
		int64_t digit = get_digit(x, kappa);
		int64_t carry = get_carry(x, digit, kappa);

		int64_t dp   = digit + cin;
		int64_t y    = get_digit(dp, kappa);
		int64_t cout = carry + get_carry(dp, y, kappa);

		res[i * res_sl + k] = y;
		cin                 = cout;
	}

	// limb 0 (most-significant): carry out is dropped.
	{
		int64_t x     = a[k];
		int64_t digit = get_digit(x, kappa);
		int64_t dp    = digit + cin;
		res[k]        = get_digit(dp, kappa);
	}

	// res's limbs beyond a_size (only possible when res_size > a_size) are zero.
	for (int j = a_size; j < res_size; j++) res[j * res_sl + k] = 0;
}
