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
