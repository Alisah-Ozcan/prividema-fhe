#ifndef PVDA_GGSW_EXTERNAL_PRODUCT_GPU_H
#define PVDA_GGSW_EXTERNAL_PRODUCT_GPU_H

#include <stddef.h>
#include <stdint.h>

#include "gpu/common/gpu_stream.h"

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
	// pvda_gpu_owns (a lookup in the library's block registry) rather than
	// pvda_is_device_pointer (a CUDA call): the operands of the core operations are the
	// library's own buffers, and this runs on every operation (7 ns vs 155 ns)
	int n_device = 0;
	for (int i = 0; i < n; ++i) n_device += pvda_gpu_owns(ptrs[i]);
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

// Allocate a new GLWEGadgetCiphertextPrep in GPU memory — the device-resident
// counterpart of new_glwegadget_prep (see also new_glwegadget_prep_on).
// glwegadget_prepare fills it on the GPU, in the NTT domain: GPU and CPU
// (FFT) prepared matrices are NOT interchangeable. Free with
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
// GLWEGadget half-product helpers
// ---------------------------------------------------------------------------

// Allocate device buffer and upload the coefficient matrix of a GLWEGadgetCiphertext.
// Returned pointer holds  l_tilde * ciphertext_nb_limbs * nn  int64_t values (row-major).
int64_t* pvda_glwegadget_to_device(const GLWEGadgetCiphertext* glwegad);

#ifdef __cplusplus
}
#endif

#endif  // PVDA_GGSW_EXTERNAL_PRODUCT_GPU_H
