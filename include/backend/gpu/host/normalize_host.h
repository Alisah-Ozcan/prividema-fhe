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

/* Device-pointer variant — a_dev/res_dev must already reside in GPU memory
 * (see pvda_is_device_pointer). No host<->device transfer is performed.
 *
 * stride is the element distance between consecutive limbs (== n for a
 * contiguous buffer; may be larger for a strided view, e.g. one component
 * of an interleaved GLWE ciphertext — see glwe_extract_poly_view). */
void gpu_normalize_base2k_device(const int64_t* a_dev, int64_t* res_dev, size_t n, size_t l, int64_t stride,
                                 uint32_t kappa);

/* Sized device-pointer variant — a_dev and res_dev may have DIFFERENT limb
 * counts (a_size vs res_size), e.g. normalize_glwe(result, glwe) where
 * result and glwe have different params (see gpu_vec_znx_add_sized_device
 * for why this legitimately happens in glwe_trace_expand). Mirrors spqlios'
 * vec_znx_normalize_base2k_ref: carry propagates through any of a's limbs
 * beyond res_size (discarded), any of res's limbs beyond a_size are
 * zeroed. gpu_normalize_base2k_device above requires a_size == res_size —
 * using it with mismatched sizes overflows the smaller buffer. */
void gpu_normalize_base2k_sized_device(const int64_t* a_dev, size_t a_size, int64_t a_stride, int64_t* res_dev,
                                       size_t res_size, int64_t res_stride, size_t n, uint32_t kappa);

#ifdef __cplusplus
}
#endif

#endif  // PVDA_NORMALIZE_HOST_H
