#pragma once
#include "gpuntt/common/modular_arith.cuh"

__global__ void pointwise_mult_kernel(Data64* a, const Data64* b, Modulus64 mod, int n);

// Batch: b_batch[j*n + i] *= a_ntt[i] for each j in [0, batch)
__global__ void pointwise_mult_batch_kernel(Data64* b_batch, const Data64* a_ntt, Modulus64 mod, int n, int batch);

/* VMP accumulator:
 * c_ntt[j*n + k] = sum_{i=0}^{nrows-1} a_ntt[i*n+k] * m_ntt[(i*ncols+j)*n+k]  (mod prime)
 * Output c_ntt size: ncols * n */
__global__ void vmp_accumulate_kernel(Data64* c_ntt, const Data64* a_ntt, const Data64* m_ntt, Modulus64 mod, int n,
                                      int nrows, int ncols);
