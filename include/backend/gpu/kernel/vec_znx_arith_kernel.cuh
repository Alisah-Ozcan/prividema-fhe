#pragma once
#include <stdint.h>

// Elementwise vec_znx kernels: every coefficient of every limb is independent,
// no carry propagation (unlike normalize_base2k_kernel).

__global__ void vec_znx_add_kernel(int64_t* res, const int64_t* a, const int64_t* b, int total);

__global__ void vec_znx_sub_kernel(int64_t* res, const int64_t* a, const int64_t* b, int total);

__global__ void vec_znx_negate_kernel(int64_t* res, const int64_t* a, int total);
