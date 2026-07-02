#ifndef PVDA_NORMALIZE_HOST_H
#define PVDA_NORMALIZE_HOST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* base-2^kappa normalization (GPU).
 *
 * Input : a_host[l * n]  — l limbs, each with n coefficients, contiguous (row-major)
 *           a_host[(i*n)+k] = coefficient k of limb i,  i ∈ [0,l), k ∈ [0,n)
 * Output: res_host[l * n] — same layout, each coefficient ∈ [-2^(kappa-1), 2^(kappa-1))
 *
 * Carry propagates from least-significant limb (i = l-1) to most-significant (i = 0).
 * Carry out of the most-significant limb is dropped. */
void gpu_normalize_base2k(const int64_t* a_host, int64_t* res_host, size_t n, size_t l, uint32_t kappa);

#ifdef __cplusplus
}
#endif

#endif // PVDA_NORMALIZE_HOST_H
