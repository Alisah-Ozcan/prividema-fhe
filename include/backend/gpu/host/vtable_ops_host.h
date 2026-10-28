#ifndef PVDA_GPU_VTABLE_OPS_HOST_H
#define PVDA_GPU_VTABLE_OPS_HOST_H

#include <stddef.h>
#include <stdint.h>

// GPU implementations of the backend virtual table operations (see
// backend_private.h), used by the CUDA backend for device-resident operands.
// They follow the spqlios semantics of the corresponding pvda_* operations:
// each operand is a vector of limbs (polynomials of n coefficients) with its
// own limb count and stride (distance between limbs, in coefficients); limbs
// of the result beyond the inputs are set to zero.
//
// The DFT domain of the GPU is the NTT domain modulo the GPU NTT prime: a DFT
// vector holds one 64-bit residue per coefficient, limbs contiguous (stride n).
// Prepared matrices (VMP) and scalars (SVP) are NTT-domain as well.
//
// All pointers are device pointers. The operations are queued on the active
// stream (gpu_stream.h) and do not synchronize.

#ifdef __cplusplus
extern "C" {
#endif

// res = a + b / a - b / -a
void gpu_vt_vec_znx_add(int64_t* res, size_t res_l, int64_t res_sl, const int64_t* a, size_t a_l, int64_t a_sl,
                        const int64_t* b, size_t b_l, int64_t b_sl, size_t n);
void gpu_vt_vec_znx_sub(int64_t* res, size_t res_l, int64_t res_sl, const int64_t* a, size_t a_l, int64_t a_sl,
                        const int64_t* b, size_t b_l, int64_t b_sl, size_t n);
void gpu_vt_vec_znx_negate(int64_t* res, size_t res_l, int64_t res_sl, const int64_t* a, size_t a_l, int64_t a_sl,
                           size_t n);

// res_ntt (res_size limbs) = NTT(a); limbs beyond a_l are zero
void gpu_vt_vec_znx_dft(int64_t* res_ntt, size_t res_size, const int64_t* a, size_t a_l, int64_t a_sl, size_t n);

// res (res_l limbs) = INTT(a_ntt); limbs beyond a_size are zero
void gpu_vt_vec_znx_idft(int64_t* res, size_t res_l, int64_t res_sl, const int64_t* a_ntt, size_t a_size, size_t n);

// pmat = NTT of each of the nrows * ncols polynomials of mat (row-major)
void gpu_vt_vmp_prepare(int64_t* pmat, const int64_t* mat, size_t nrows, size_t ncols, size_t n);

// res_ntt[j] = sum_{i < min(a_rows, nrows)} a_ntt[i] * pmat[i][j] for j < min(res_size, ncols), zero beyond
void gpu_vt_vmp_apply_ntt(int64_t* res_ntt, size_t res_size, const int64_t* a_ntt, size_t a_rows, const int64_t* pmat,
                          size_t nrows, size_t ncols, size_t n);

// ppol = NTT(pol) (one polynomial)
void gpu_vt_svp_prepare(int64_t* ppol, const int64_t* pol, size_t n);

// res_ntt[j] = ppol * a_ntt[j] for j < min(res_size, a_size), zero beyond
void gpu_vt_svp_apply_ntt(int64_t* res_ntt, size_t res_size, const int64_t* ppol, const int64_t* a_ntt, size_t a_size,
                          size_t n);

// res_ntt[j] = a_ntt[j] + b_ntt[j] for j < res_size, a limb beyond a_size / b_size reading as zero
// (so the longer input is copied past the shorter one, and limbs beyond both are zero)
void gpu_vt_vec_znx_dft_add(int64_t* res_ntt, size_t res_size, const int64_t* a_ntt, size_t a_size,
                            const int64_t* b_ntt, size_t b_size, size_t n);

#ifdef __cplusplus
}
#endif

#endif  // PVDA_GPU_VTABLE_OPS_HOST_H
