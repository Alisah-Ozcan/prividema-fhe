#pragma once
#include <stdint.h>

/* res = a * X^p mod (X^n+1), applied independently to each limb (see
 * vec_znx_rotate in spqlios/arithmetic/vec_znx.c). One thread per (limb,
 * output coefficient) pair, out-of-place only — res and a must not alias
 * (see gpu_vec_znx_rotate_device in the host wrapper for the in-place
 * case). res_sl/a_sl are the element stride between consecutive limbs of
 * the same polynomial (== n for a contiguous PolyBiv, larger for a strided
 * view such as glwe_extract_poly_view).
 *
 * Mirrors vec_znx_rotate_ref's res_size/a_size handling exactly: only the
 * first min(res_size, a_size) limbs are read from `a`; any remaining
 * res limbs (common_limbs <= limb < res_size) are zeroed rather than read
 * out of bounds. common_limbs = min(res_size, a_size), passed precomputed
 * since device code has no host-side min(). Grid covers res_size * n
 * threads. */
__global__ void vec_znx_rotate_kernel(int64_t* res, const int64_t* a, int64_t p, int n, int res_size, int common_limbs,
                                      int64_t res_sl, int64_t a_sl);

/* res = a(X^p) mod (X^n+1), i.e. the ring automorphism sigma_p (see
 * vec_znx_automorphism in spqlios/arithmetic/vec_znx.c). p must be odd.
 * p_inv is the modular inverse of p mod 2n, precomputed on the host (see
 * gpu_mod_inverse_pow2 in the host wrapper) — this turns the CPU reference's
 * sequential scatter (res[i*p mod 2n] = +-a[i]) into a race-free parallel
 * gather (res[j] = +-a[j*p_inv mod 2n]), one thread per (limb, output
 * coefficient). Same res_size/a_size/common_limbs zero-padding and
 * res_sl/a_sl convention as rotate above. */
__global__ void vec_znx_automorphism_kernel(int64_t* res, const int64_t* a, int64_t p_inv, int n, int res_size,
                                            int common_limbs, int64_t res_sl, int64_t a_sl);
