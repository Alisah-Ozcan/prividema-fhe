#ifndef PVDA_GPU_MEMORY_HOST_H
#define PVDA_GPU_MEMORY_HOST_H

#include <stddef.h>
#include <stdint.h>

#include "gpu/common/gpu_stream.h"

// Device memory primitives of the CUDA backend (taken unchanged from the raw transfer part of
// ggsw_external_product_gpu.h on su-full-gpu)

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

// Allocate a device buffer of limbs * n int64_t and upload into it, contiguously, the limbs limbs of n
// coefficients of a host limb vector of stride host_stride (the gaps between the limbs are not copied).
// Caller must release the returned pointer with pvda_gpu_free.
int64_t* pvda_gpu_upload_limbs(const int64_t* host_ptr, int64_t host_stride, size_t limbs, size_t n);

// Download the limbs contiguous limbs of n coefficients of device_ptr into a host limb vector of stride
// host_stride, leaving the gaps between its limbs untouched. Returns once the data is on the host.
void pvda_gpu_download_limbs(int64_t* host_ptr, int64_t host_stride, const int64_t* device_ptr, size_t limbs,
                             size_t n);


#ifdef __cplusplus
}
#endif

#endif  // PVDA_GPU_MEMORY_HOST_H
