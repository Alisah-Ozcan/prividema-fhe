#pragma once
#include <stdint.h>

/* base-2^kappa normalizasyonu: her katsayı pozisyonu bağımsız işlenir.
 * Carry az-anlamlı limb'den (l-1) çok-anlamlı limb'e (0) doğru akar.
 * res[i*stride+k] ∈ [-(2^(kappa-1)), 2^(kappa-1))  for all i, k
 *
 * stride is the element distance between consecutive limbs (== n for a
 * contiguous buffer; may be larger when normalizing a strided view, e.g.
 * one component of an interleaved GLWE ciphertext). */
__global__ void normalize_base2k_kernel(int64_t* res, const int64_t* a, int n, int l, int64_t stride, int kappa);

/* Sized variant — res and a may have DIFFERENT limb counts (res_size vs
 * a_size), mirroring spqlios' vec_znx_normalize_base2k_ref: carry
 * propagates through any of a's limbs beyond res_size (computed but
 * discarded), and any of res's limbs beyond a_size are zeroed. Needed
 * wherever normalize_glwe's operands have different params (e.g. an
 * automorphism KSK's own precision vs. the GLWE ciphertext being
 * transformed, as in glwe_trace_expand) — normalize_base2k_kernel above
 * would read/write res_size==a_size==l elements from both buffers
 * regardless of their real sizes, overflowing the smaller one. */
__global__ void normalize_base2k_sized_kernel(int64_t* res, int res_size, int64_t res_sl, const int64_t* a, int a_size,
                                              int64_t a_sl, int n, int kappa);
