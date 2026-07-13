#ifndef PVDA_NORMALIZE_HOST_H
#define PVDA_NORMALIZE_HOST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// base-2^kappa normalization (GPU).
//
// Input : a_host[l * n]  — l limbs, each with n coefficients, contiguous (row-major)
//           a_host[(i*n)+k] = coefficient k of limb i,  i ∈ [0,l), k ∈ [0,n)
// Output: res_host[l * n] — same layout, each coefficient ∈ [-2^(kappa-1), 2^(kappa-1))
//
// Carry propagates from least-significant limb (i = l-1) to most-significant (i = 0).
// Carry out of the most-significant limb is dropped.
void gpu_normalize_base2k(const int64_t* a_host, int64_t* res_host, size_t n, size_t l, uint32_t kappa);

// Device-pointer variant — a_dev and res_dev may have DIFFERENT limb counts
// (a_size vs res_size), e.g. normalize_glwe(result, glwe) where result and
// glwe have different params (see gpu_vec_znx_add_sized_device for why this
// legitimately happens in glwe_trace_expand). Mirrors spqlios'
// vec_znx_normalize_base2k_ref: carry propagates through any of a's limbs
// beyond res_size (discarded), any of res's limbs beyond a_size are zeroed.
// The common case of equal limb counts (a_size == res_size) is just this
// with both sizes set equal — see normalize_base2k_sized_kernel's doc
// comment — so there is no separate non-sized entry point.
//
// No internal scratch allocation — writes straight into the caller's
// res_dev — so this does not synchronize the stream before returning. The
// caller decides when the result actually needs to be ready
// (pvda_gpu_stream_synchronize, in gpu_stream.h): normalize_glwe
// (glwe_arithmetic.c) adds its own sync once, after every limb it needs to
// normalize, to preserve its own synchronous contract for its many callers;
// glwe_trace_expand's batched GPU path instead synchronizes once after its
// whole call sequence.
void gpu_normalize_base2k_sized_device(const int64_t* a_dev, size_t a_size, int64_t a_stride, int64_t* res_dev,
                                       size_t res_size, int64_t res_stride, size_t n, uint32_t kappa);

#ifdef __cplusplus
}
#endif

#endif  // PVDA_NORMALIZE_HOST_H
