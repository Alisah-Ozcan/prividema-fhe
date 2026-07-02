#pragma once
#include <stdint.h>

/* base-2^kappa normalizasyonu: her katsayı pozisyonu bağımsız işlenir.
 * Carry az-anlamlı limb'den (l-1) çok-anlamlı limb'e (0) doğru akar.
 * res[i*n+k] ∈ [-(2^(kappa-1)), 2^(kappa-1))  for all i, k */
__global__ void normalize_base2k_kernel(int64_t* res, const int64_t* a, int n, int l, int kappa);
