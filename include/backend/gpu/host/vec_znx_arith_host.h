#ifndef PVDA_VEC_ZNX_ARITH_HOST_H
#define PVDA_VEC_ZNX_ARITH_HOST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Elementwise vec_znx operations (GPU).
//
// Input/Output: flattened l * n int64_t arrays, row-major
//   x[(i*n)+k] = coefficient k of limb i,  i ∈ [0,l), k ∈ [0,n)
//
// Every coefficient of every limb is processed independently — no carry
// propagation (contrast with gpu_normalize_base2k). res_host, a_host and
// b_host must all share the same (n, l), i.e. contiguous stride == n.

void gpu_vec_znx_add(const int64_t* a_host, const int64_t* b_host, int64_t* res_host, size_t n, size_t l);

void gpu_vec_znx_sub(const int64_t* a_host, const int64_t* b_host, int64_t* res_host, size_t n, size_t l);

void gpu_vec_znx_negate(const int64_t* a_host, int64_t* res_host, size_t n, size_t l);

// Device-pointer variants — a_dev/b_dev/res_dev must already reside in GPU
// memory (see pvda_is_device_pointer). No host<->device transfer is
// performed; total is the flat element count (n * l).
//
// These do NOT synchronize the stream before returning — they only launch
// the kernel into the caller's own res_dev, with no internal scratch
// allocation of their own to protect. The caller decides when the result
// actually needs to be ready (pvda_gpu_stream_synchronize, in gpu_stream.h)
// — e.g. before reading res_dev on the host, freeing a buffer it depends
// on, or handing it to code on a different stream. Every current in-tree
// caller that needs synchronous completion already adds its own sync right
// after (see add_glwe/sub_glwe in glwe_arithmetic.c and
// glwegadget_automorphism in glwegadget_arithmetic.c); batched call
// sequences (see glwe_trace_expand's CUDA path) instead synchronize once
// after the whole sequence.
void gpu_vec_znx_add_device(const int64_t* a_dev, const int64_t* b_dev, int64_t* res_dev, size_t total);

void gpu_vec_znx_sub_device(const int64_t* a_dev, const int64_t* b_dev, int64_t* res_dev, size_t total);

void gpu_vec_znx_negate_device(const int64_t* a_dev, int64_t* res_dev, size_t total);

// Sized device-pointer variants — mirror spqlios' vec_znx_add_ref /
// vec_znx_sub_ref semantics for operands of DIFFERING limb counts (unlike
// gpu_vec_znx_add_device/gpu_vec_znx_sub_device above, which require all
// three buffers to share the same flat element count). Each buffer is
// n-contiguous (stride == n): res_dev holds res_size limbs, a_dev holds
// a_size limbs, b_dev holds b_size limbs.
//
// Semantics (n-sized chunk i, 0-indexed):
//   i < min(res_size, a_size, b_size)       : res[i] = a[i] +/- b[i]
//   min(a_size,b_size) <= i < res_size,
//     bounded by max(a_size,b_size)         : res[i] = larger operand's
//                                              chunk (negated for sub when
//                                              the larger operand is b)
//   beyond both a_size and b_size           : res[i] = 0
//
// This is what glwe operand pairs with different params (e.g. an
// automorphism KSK's own precision vs. the GLWE ciphertext being
// transformed, as in glwe_trace_expand) require — reading/writing exactly
// total = res_size*n elements irrespective of a_size/b_size would silently
// over-read a smaller operand's buffer.
void gpu_vec_znx_add_sized_device(const int64_t* a_dev, size_t a_size, const int64_t* b_dev, size_t b_size,
                                  int64_t* res_dev, size_t res_size, size_t n);

void gpu_vec_znx_sub_sized_device(const int64_t* a_dev, size_t a_size, const int64_t* b_dev, size_t b_size,
                                  int64_t* res_dev, size_t res_size, size_t n);

#ifdef __cplusplus
}
#endif

#endif  // PVDA_VEC_ZNX_ARITH_HOST_H
