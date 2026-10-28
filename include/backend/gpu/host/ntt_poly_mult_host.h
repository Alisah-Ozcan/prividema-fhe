#ifndef PVDA_NTT_POLY_MULT_HOST_H
#define PVDA_NTT_POLY_MULT_HOST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// SVP (scalar-vector product): c[j] = a * b[j]  for j in [0, batch)
// batch=1 → single polynomial multiplication.
// b_host and c_host hold `batch` contiguous n-element polynomials.
void gpu_ntt_svp(const int64_t* a_host, const int64_t* b_host, int64_t* c_host, size_t n, size_t batch);

// VMP: c_host[j] = sum_{i=0}^{nrows-1} a_host[i] * m_host[i][j]  (mod X^n+1)
// a_host: nrows*n  (row vector, contiguous)
// m_host: nrows*ncols*n  (matrix, row-major: m[i][j] = m_host[(i*ncols+j)*n ..])
// c_host: ncols*n  (output)
void gpu_ntt_vmp(const int64_t* a_host, const int64_t* m_host, int64_t* c_host, size_t n, size_t nrows, size_t ncols);

#ifdef __cplusplus
}
#endif

#endif  // PVDA_NTT_POLY_MULT_HOST_H
