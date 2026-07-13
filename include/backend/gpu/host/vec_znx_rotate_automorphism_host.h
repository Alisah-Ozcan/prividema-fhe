#ifndef PVDA_VEC_ZNX_ROTATE_AUTOMORPHISM_HOST_H
#define PVDA_VEC_ZNX_ROTATE_AUTOMORPHISM_HOST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPU counterparts of pvda_vec_znx_rotate / pvda_vec_znx_automorphism
// (backend/spqlios_alias.c), matching the CPU reference (spqlios
// vec_znx_rotate_ref / vec_znx_automorphism_ref) exactly, including its
// res_size/a_size handling: only the first min(res_size, a_size) limbs are
// read from `a`, any remaining res limbs (up to res_size) are zeroed rather
// than read out of bounds — e.g. glwegadget_automorphism relies on this to
// zero-pad its scratch buffer when its l exceeds the source polynomial's l.
//
// n is the ring degree (coefficients per limb). res_sl/a_sl are the element
// stride between consecutive limbs of the same polynomial (== n for a
// contiguous PolyBiv, larger for a strided view such as
// glwe_extract_poly_view — see PolyBiv.stride in maths_structures.h).
// Elements outside of the limb windows (i.e. the gaps of an
// interleaved/strided view, which belong to other polynomials) are never
// written, matching the CPU reference.
//
// Safe when res and a alias (in-place, e.g. pvda_vec_znx_rotate's usage in
// glwe_trace_expand): computed into an internal scratch buffer first, so
// there's no read-after-write hazard between threads.

// Host-memory variants: uploads a_host, computes, downloads into res_host.
void gpu_vec_znx_rotate(int64_t p, const int64_t* a_host, int64_t* res_host, size_t n, size_t res_size, int64_t res_sl,
                        size_t a_size, int64_t a_sl);

void gpu_vec_znx_automorphism(int64_t p, const int64_t* a_host, int64_t* res_host, size_t n, size_t res_size,
                              int64_t res_sl, size_t a_size, int64_t a_sl);

// Device-pointer variants — a_dev/res_dev must already reside in GPU memory
// (see pvda_is_device_pointer). No host<->device transfer is performed.
//
// Both allocate an internal scratch buffer (res_dev may alias a_dev — see
// the aliasing note above — so the kernel can never write res_dev
// directly), and that scratch is only safe to free once the kernel/copies
// touching it have actually finished — so, unlike the plain add/sub/negate
// _device functions in vec_znx_arith_host.h, these DO synchronize
// internally before returning.
void gpu_vec_znx_rotate_device(int64_t p, const int64_t* a_dev, int64_t* res_dev, size_t n, size_t res_size,
                               int64_t res_sl, size_t a_size, int64_t a_sl);

void gpu_vec_znx_automorphism_device(int64_t p, const int64_t* a_dev, int64_t* res_dev, size_t n, size_t res_size,
                                     int64_t res_sl, size_t a_size, int64_t a_sl);

// Non-aliasing variant of gpu_vec_znx_automorphism_device: writes directly
// into res_dev honoring the real res_sl, instead of always routing through
// a compact scratch buffer + per-limb copy-out. Only safe when res_dev and
// a_dev do NOT alias — gpu_vec_znx_automorphism_device must support the
// aliased case (hence its scratch buffer) but glwegadget_automorphism_gpu_batched's
// (glwegadget_arithmetic.c) per-call auto_tmp scratch is always distinct
// from its source, so this variant is safe there and skips both the extra
// allocation and the copy. No internal allocation here means no internal
// wait either — does not synchronize; caller synchronizes once it actually
// needs res_dev.
void gpu_vec_znx_automorphism_device_noalias(int64_t p, const int64_t* a_dev, int64_t* res_dev, size_t n,
                                             size_t res_size, int64_t res_sl, size_t a_size, int64_t a_sl);

// Same in-place-safe scratch/copy-out strategy as gpu_vec_znx_rotate_device
// (res_dev may alias a_dev), but scratch_dev is caller-provided — sized
// res_size*n int64s — and its lifetime is the caller's responsibility, not
// freed in here. Since there is nothing left for this function itself to
// wait on, it does not synchronize; the caller synchronizes once, whenever
// it actually needs res_dev or is about to free scratch_dev.
void gpu_vec_znx_rotate_device_scratch(int64_t p, const int64_t* a_dev, int64_t* res_dev, size_t n, size_t res_size,
                                       int64_t res_sl, size_t a_size, int64_t a_sl, int64_t* scratch_dev);

#ifdef __cplusplus
}
#endif

#endif  // PVDA_VEC_ZNX_ROTATE_AUTOMORPHISM_HOST_H
