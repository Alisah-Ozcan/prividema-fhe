#ifndef PVDA_VEC_ZNX_ARITH_HOST_H
#define PVDA_VEC_ZNX_ARITH_HOST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Elementwise vec_znx operations (GPU).
 *
 * Input/Output: flattened l * n int64_t arrays, row-major
 *   x[(i*n)+k] = coefficient k of limb i,  i ∈ [0,l), k ∈ [0,n)
 *
 * Every coefficient of every limb is processed independently — no carry
 * propagation (contrast with gpu_normalize_base2k). res_host, a_host and
 * b_host must all share the same (n, l), i.e. contiguous stride == n. */

void gpu_vec_znx_add(const int64_t* a_host, const int64_t* b_host, int64_t* res_host, size_t n, size_t l);

void gpu_vec_znx_sub(const int64_t* a_host, const int64_t* b_host, int64_t* res_host, size_t n, size_t l);

void gpu_vec_znx_negate(const int64_t* a_host, int64_t* res_host, size_t n, size_t l);

/* Device-pointer variants — a_dev/b_dev/res_dev must already reside in GPU
 * memory (see pvda_is_device_pointer). No host<->device transfer is
 * performed; total is the flat element count (n * l). */
void gpu_vec_znx_add_device(const int64_t* a_dev, const int64_t* b_dev, int64_t* res_dev, size_t total);

void gpu_vec_znx_sub_device(const int64_t* a_dev, const int64_t* b_dev, int64_t* res_dev, size_t total);

void gpu_vec_znx_negate_device(const int64_t* a_dev, int64_t* res_dev, size_t total);

#ifdef __cplusplus
}
#endif

#endif  // PVDA_VEC_ZNX_ARITH_HOST_H
