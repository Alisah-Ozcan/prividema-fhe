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
typedef struct ggsw_ct_params GGSWParams;
typedef struct glwegadget_ciphertext GLWEGadgetCiphertext;
typedef struct glwegadget_ciphertext_prepared GLWEGadgetCiphertextPrep;
typedef struct glwegadget_params GLWEGadgetParams;
typedef struct glwe_automorphism_keyswitching_key_t GLWEAutomorphismKey;

#ifdef __cplusplus
extern "C" {
#endif

// Returns 1 if ptr resides in CUDA device memory, 0 if in ordinary host memory.
int pvda_is_device_pointer(const void* ptr);

// Where a set of operand buffers lives. The GPU dispatch of the core API
// runs an operation on the GPU when all of its operands are device-resident
// and on the CPU when none is; a mix is a caller error, reported instead of
// letting either path dereference memory it cannot access.
#define PVDA_ON_HOST          0
#define PVDA_ON_DEVICE        1
#define PVDA_MIXED_RESIDENCY (-1)

static inline int pvda_residency(int n, const void* const* ptrs)
{
	int n_device = 0;
	for (int i = 0; i < n; ++i) n_device += pvda_is_device_pointer(ptrs[i]);
	return n_device == 0 ? PVDA_ON_HOST : n_device == n ? PVDA_ON_DEVICE : PVDA_MIXED_RESIDENCY;
}

#ifndef __cplusplus
// PVDA_RESIDENCY(a, b, ...) == pvda_residency over the listed pointers
#define PVDA_RESIDENCY(...)                                                                  \
	pvda_residency((int)(sizeof((const void*[]){__VA_ARGS__}) / sizeof(const void*)), \
	               (const void*[]){__VA_ARGS__})
#endif

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

// Allocate a new (raw, coefficient-domain) GLWEGadgetCiphertext whose mat lives
// in (uninitialised) device memory — the device-resident counterpart of
// new_glwegadget. Free with delete_glwegadget, which is itself device-aware.
GLWEGadgetCiphertext* pvda_new_glwegadget_device(const GLWEGadgetParams* params);

// Allocate a new (raw, coefficient-domain) GGSWCiphertext whose mat lives in
// (uninitialised) device memory — the device-resident counterpart of new_ggsw.
// Free with delete_ggsw, which is itself device-aware.
GGSWCiphertext* pvda_new_ggsw_device(const GGSWParams* params);

// Allocate a new GLWEGadgetCiphertextPrep whose mat is a device placeholder —
// the device-resident counterpart of new_glwegadget_prep. Its only purpose
// is to signal "prepare this on the GPU" to callers such as
// prepare_automorphism_key (which check pvda_is_device_pointer(...->mat) to
// decide whether to NTT-prepare on device vs DFT-prepare on host — those two
// domains are NOT interchangeable, so this placeholder must never be read
// before glwegadget_prepare(_gpu) replaces it). Free with
// delete_glwegadget_prep, which is itself device-aware.
GLWEGadgetCiphertextPrep* pvda_new_glwegadget_prep_device(const GLWEGadgetParams* params);

// Same as pvda_new_glwegadget_prep_device, for GGSWCiphertextPrep / ggsw_prepare
// (see e.g. generate_glwegad_to_ggsw_ksk). Free with delete_ggsw_prep.
GGSWCiphertextPrep* pvda_new_ggsw_prep_device(const GGSWParams* params);

// ---------------------------------------------------------------------------
// Device-resident objects from/to host objects. Every object returned here is
// freed with the regular delete_* function (they are all device-aware), and
// can be passed to the regular API, which then runs on the GPU.
// ---------------------------------------------------------------------------

// New device GLWECiphertext holding a copy of host_glwe (same params).
GLWECiphertext* pvda_glwe_upload(const GLWECiphertext* host_glwe);

// Copies device_glwe into the existing host GLWECiphertext host_dst. Both
// must have the same params. Returns once the data is on the host.
void pvda_glwe_download(GLWECiphertext* host_dst, const GLWECiphertext* device_glwe);

// New device GGSWCiphertext holding a copy of the raw (unprepared) host_ggsw.
GGSWCiphertext* pvda_ggsw_upload(const GGSWCiphertext* host_ggsw);

// New device GLWEGadgetCiphertext holding a copy of the raw host_glwegad.
GLWEGadgetCiphertext* pvda_glwegadget_upload(const GLWEGadgetCiphertext* host_glwegad);

// New GLWECiphertextDFT in the GPU (NTT) domain, the device counterpart of
// new_glwe_dft, e.g. for ggsw_external_product_to_dft or
// glwegadget_half_prod_prepared_to_dft. Free with delete_glwe_dft.
GLWECiphertextDFT* pvda_new_glwe_dft_device(const GLWEParams* params);

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
// d_a          : [nrows * n]              — l_tilde coef-domain int64 polys on device
// d_glwegad_ntt: [nrows * ncols_in * n]   — NTT-domain gadget matrix on device;
//                                            ncols_in is the matrix's OWN column
//                                            count (from the GLWEGadgetCiphertextPrep
//                                            that was NTT-prepared), independent of
//                                            the caller's result buffer size.
// d_result     : [ncols_out * n]          — coef-domain output (NTT + VMP + INTT).
//                                            Only min(ncols_in, ncols_out) columns
//                                            are ever real VMP output; if
//                                            ncols_out > ncols_in the remaining
//                                            tail is zeroed (mirrors spqlios
//                                            vmp_apply_dft's res_size vs matrix-ncols
//                                            handling on the CPU path) — ncols_in and
//                                            ncols_out must NEVER be conflated into a
//                                            single parameter here, since the gadget
//                                            matrix's own params_glwe can legitimately
//                                            differ from the result's (e.g. an
//                                            automorphism KSK's own precision vs the
//                                            GLWE ciphertext being transformed).
// ---------------------------------------------------------------------------
// Allocates internal NTT scratch buffers (d_a_ntt, d_c_ntt) that are only
// safe to free once the NTT/VMP/INTT pipeline touching them has actually
// finished, so — unlike the plain vec_znx _device functions in
// vec_znx_arith_host.h, which have no internal scratch to protect — this
// DOES synchronize internally before returning.
void gpu_glwegadget_half_prod_device(const int64_t* d_a, const int64_t* d_glwegad_ntt, int64_t* d_result, size_t n,
                                     size_t nrows, size_t ncols_in, size_t ncols_out);

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
