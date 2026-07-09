#include "gpu/kernel/normalize_kernel.cuh"

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

__global__ void normalize_base2k_kernel(int64_t* res, const int64_t* a, int n, int l, int64_t stride, int kappa)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= n) return;

	int64_t cin = 0;

	// Least-significant to most-significant: limb (l-1) → limb 1
	for (int i = l - 1; i >= 1; i--)
	{
		int64_t x     = a[i * stride + k];
		int64_t digit = get_digit(x, kappa);
		int64_t carry = get_carry(x, digit, kappa);

		// Add carry from the limb below, re-normalize
		int64_t dp   = digit + cin;
		int64_t y    = get_digit(dp, kappa);
		int64_t cout = carry + get_carry(dp, y, kappa);

		res[i * stride + k] = y;
		cin                 = cout;
	}

	// limb 0 (most-significant): carry out is dropped
	{
		int64_t x     = a[k];
		int64_t digit = get_digit(x, kappa);
		int64_t dp    = digit + cin;
		res[k]        = get_digit(dp, kappa);
	}
}
