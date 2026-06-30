#ifndef POLY_ADD_KERNEL_CUH
#define POLY_ADD_KERNEL_CUH

#include <stdint.h>

// Coefficient-wise polynomial addition modulo q in ZnX.
// c[idx] = (a[idx] + b[idx]) mod q, result in [0, q).
__global__ void poly_add_mod_kernel(const int64_t* __restrict__ a, const int64_t* __restrict__ b,
                                    int64_t* __restrict__ c, uint64_t n, int64_t q);

#endif  // POLY_ADD_KERNEL_CUH
