#ifndef POLY_ADD_KERNEL_CUH
#define POLY_ADD_KERNEL_CUH

#include "gpuntt/common/modular_arith.cuh"
#include <stdint.h>

__global__ void poly_add_mod_kernel(const Data64* __restrict__ a,
                                    const Data64* __restrict__ b,
                                    Data64* __restrict__ c, uint64_t n,
                                    Modulus64 modulus);

#endif /* POLY_ADD_KERNEL_CUH */
