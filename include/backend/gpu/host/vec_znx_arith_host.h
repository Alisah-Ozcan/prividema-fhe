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

#ifdef __cplusplus
}
#endif

#endif  // PVDA_VEC_ZNX_ARITH_HOST_H
