#ifndef PVDA_GGSW_EXTERNAL_PRODUCT_GPU_H
#define PVDA_GGSW_EXTERNAL_PRODUCT_GPU_H

#include <stddef.h>
#include <stdint.h>

// Forward declarations — avoids pulling in core headers here.
typedef struct glwe_ciphertext GLWECiphertext;
typedef struct glwe_ciphertext_dft GLWECiphertextDFT;
typedef struct glwe_ct_params GLWEParams;
typedef struct ggsw_ciphertext GGSWCiphertext;
typedef struct ggsw_ciphertext_prepared GGSWCiphertextPrep;
typedef struct glwegadget_ciphertext GLWEGadgetCiphertext;
typedef struct glwegadget_ciphertext_prepared GLWEGadgetCiphertextPrep;

#ifdef __cplusplus
extern "C" {
#endif

// Returns 1 if ptr resides in CUDA device memory, 0 if in ordinary host memory.
int pvda_is_device_pointer(const void* ptr);

// ---------------------------------------------------------------------------
// Raw transfer primitives (element count supplied by caller)
// ---------------------------------------------------------------------------

// Allocate a device buffer of n_elements int64_t and upload from host.
// Caller must release the returned pointer with pvda_gpu_free.
int64_t* pvda_gpu_upload(const int64_t* host_ptr, size_t n_elements);

// Download n_elements int64_t values from device to host.
void pvda_gpu_download(int64_t* host_ptr, const int64_t* device_ptr, size_t n_elements);

// Allocate an uninitialised device buffer of n_elements int64_t.
int64_t* pvda_gpu_alloc(size_t n_elements);

// Free a device buffer obtained from pvda_gpu_upload or pvda_gpu_alloc.
void pvda_gpu_free(int64_t* device_ptr);

// Device-to-device copy of n_elements int64_t (both pointers must be device pointers).
void pvda_gpu_copy(int64_t* dst, const int64_t* src, size_t n_elements);

// Zero n_elements int64_t of device memory.
void pvda_gpu_zero(int64_t* dst, size_t n_elements);

// ---------------------------------------------------------------------------
// Struct-typed transfer helpers (size computed from params automatically)
// ---------------------------------------------------------------------------

// Allocate device buffer and upload the coefficient data of a GLWECiphertext.
// Returned pointer holds  glwe->params->ciphertext_nb_limbs * nn  int64_t values.
int64_t* pvda_glwe_to_device(const GLWECiphertext* glwe);

// Download device result back into an existing GLWECiphertext (vec must be host memory).
void pvda_glwe_from_device(GLWECiphertext* result, const int64_t* d_data);

// Allocate device buffer and upload the coefficient matrix of a GGSWCiphertext.
// Returned pointer holds  nrows * ncols * nn  int64_t values (row-major).
int64_t* pvda_ggsw_to_device(const GGSWCiphertext* ggsw);

// Allocate a new GLWECiphertext whose vec lives in (uninitialised) device memory —
// the device-resident counterpart of new_glwe. Free with delete_glwe, which is
// itself device-aware (see glwe_ciphertext.c).
GLWECiphertext* pvda_new_glwe_device(const GLWEParams* params);

// ---------------------------------------------------------------------------
// GPU external product — all pointer arguments must be CUDA device pointers.
//
// d_glwe   : [a_limbs * n]        — GLWE vector, a_limbs <= nrows (its own limb count;
//                                    may be smaller than nrows when the GGSW selector's
//                                    associated GLWE params differ from the ciphertext
//                                    being multiplied — the missing rows are zero-padded,
//                                    mirroring spqlios' implicit zero-padding on the CPU path)
// d_ggsw   : [nrows * ncols * n]  — GGSW matrix, row-major
// d_result : [ncols * n]          — output (written on device, caller downloads)
//
// Computes: d_result[j] = INTT( sum_i NTT(d_glwe[i]) * NTT(d_ggsw[i][j]) )
// ---------------------------------------------------------------------------
void gpu_ggsw_external_product_device(const int64_t* d_glwe, const int64_t* d_ggsw, int64_t* d_result, size_t n,
                                      size_t nrows, size_t ncols, size_t a_limbs);

// ---------------------------------------------------------------------------
// GPU prepare — NTT of the full GGSW matrix on device.
//
// Uploads the raw int64_t GGSW matrix to device, computes NTT of every
// polynomial, and stores the result in gpu_prep->mat (cast as MatBivDFT*).
// Caller must set gpu_prep->params before calling.
// Free with pvda_gpu_free((int64_t*)gpu_prep->mat).
// ---------------------------------------------------------------------------
void ggsw_prepare_gpu(GGSWCiphertextPrep* gpu_prep, const GGSWCiphertext* ggsw);

// ---------------------------------------------------------------------------
// NTT-domain VMP — GGSW is already NTT'd, result left in NTT domain (no INTT).
//
// d_glwe      : [a_limbs * n]        — raw int64_t on device, a_limbs <= nrows (see
//                                       gpu_ggsw_external_product_device — missing rows
//                                       are zero-padded)
// d_ggsw_ntt  : [nrows * ncols * n]  — NTT-domain int64_t on device (from ggsw_prepare_gpu)
// d_result_ntt: [ncols * n]          — NTT-domain output on device
// ---------------------------------------------------------------------------
void gpu_ggsw_ext_prod_ntt_device(const int64_t* d_glwe, const int64_t* d_ggsw_ntt, int64_t* d_result_ntt, size_t n,
                                  size_t nrows, size_t ncols, size_t a_limbs);

// ---------------------------------------------------------------------------
// GPU INTT — convert NTT-domain GLWECiphertextDFT to coefficient-domain GLWECiphertext.
//
// Both d_result->vec and d_glwe_ntt->vec must be device pointers holding int64_t.
// Mirrors the CPU glwe_dft_to_coef for the GPU NTT path.
// ---------------------------------------------------------------------------
void glwe_dft_to_coef_gpu(GLWECiphertext* d_result, const GLWECiphertextDFT* d_glwe_ntt);

// ---------------------------------------------------------------------------
// GLWEGadget half-product helpers
// ---------------------------------------------------------------------------

// Allocate device buffer and upload the coefficient matrix of a GLWEGadgetCiphertext.
// Returned pointer holds  l_tilde * ciphertext_nb_limbs * nn  int64_t values (row-major).
int64_t* pvda_glwegadget_to_device(const GLWEGadgetCiphertext* glwegad);

// NTT-prepare a GLWEGadgetCiphertext on device.
// Uploads the raw int64_t gadget matrix, computes NTT of every polynomial,
// and stores the result in gpu_prep->mat (cast as MatBivDFT*).
// Caller must set gpu_prep->params before calling.
// Free with pvda_gpu_free((int64_t*)gpu_prep->mat).
void glwegadget_prepare_gpu(GLWEGadgetCiphertextPrep* gpu_prep, const GLWEGadgetCiphertext* glwegad);

// ---------------------------------------------------------------------------
// GPU half-product — gadget matrix already NTT'd (from glwegadget_prepare_gpu).
//
// d_a          : [nrows * n]           — l_tilde coef-domain int64 polys on device
// d_glwegad_ntt: [nrows * ncols * n]   — NTT-domain gadget matrix on device
// d_result     : [ncols * n]           — coef-domain output (NTT + VMP + INTT)
// ---------------------------------------------------------------------------
void gpu_glwegadget_half_prod_device(const int64_t* d_a, const int64_t* d_glwegad_ntt, int64_t* d_result, size_t n,
                                     size_t nrows, size_t ncols);

// ---------------------------------------------------------------------------
// GPU half-product leaving result in NTT domain (no INTT).
//
// d_a          : [nrows * n]           — l_tilde coef-domain int64 polys on device
// d_glwegad_ntt: [nrows * ncols * n]   — NTT-domain gadget matrix on device
// d_result_ntt : [ncols * n]           — NTT-domain output (NTT + VMP, no INTT)
// ---------------------------------------------------------------------------
void gpu_glwegadget_half_prod_ntt_device(const int64_t* d_a, const int64_t* d_glwegad_ntt, int64_t* d_result_ntt,
                                         size_t n, size_t nrows, size_t ncols);

#ifdef __cplusplus
}
#endif

#endif  // PVDA_GGSW_EXTERNAL_PRODUCT_GPU_H
