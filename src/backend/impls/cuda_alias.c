#include "impls/cuda_alias.h"

#include <stdint.h>
#include <string.h>

#include "backend_types.h"
#include "gpu/common/gpu_stream.h"
#include "gpu/host/gpu_memory_host.h"
#include "gpu/host/normalize_host.h"
#include "gpu/host/vec_znx_rotate_automorphism_host.h"
#include "gpu/host/vtable_ops_host.h"
#include "logger.h"

// Where the operands of a table operation live, see pvda_residency
#define WHERE(...) PVDA_RESIDENCY(__VA_ARGS__)

#define REJECT_MIXED(op)                                                 \
	do                                                                   \
	{                                                                    \
		log_message(LOG_ERROR, op ": operands mix host and GPU memory"); \
		return -1;                                                       \
	} while (0)

#define DISPATCH(op, residency, host_call) \
	switch (residency)                     \
	{                                      \
		case PVDA_ON_HOST:                 \
			return host_call;              \
		case PVDA_MIXED_RESIDENCY:         \
			REJECT_MIXED(op);              \
		default:                           \
			break;                         \
	}

// For an operation on backend objects and coefficient vectors: on the spqlios backend when
// everything is in host memory, on the GPU when the objects are in GPU memory (the coefficient
// vectors in host memory being uploaded for it, see znx_on_gpu), rejected otherwise
#define DISPATCH_OBJECTS(op, objects, all, host_call)      \
	do                                                     \
	{                                                      \
		if ((all) == PVDA_ON_HOST) return host_call;       \
		if ((objects) != PVDA_ON_DEVICE) REJECT_MIXED(op); \
	} while (0)

// A GPU operation records its failures (gpu_stream.h) instead of aborting: clear any stale
// one before running, and report a new one as this operation's -1
#define GPU_RUN()  pvda_gpu_take_error()
#define GPU_DONE() return pvda_gpu_take_error() ? -1 : 0

static uint64_t nn_of(const PvdaBackend* backend) { return backend->host_vt.pvda_module_extract_nn(backend); }

// --- Coefficient vectors in host memory (ZnxVec) used by GPU operations ---

// *dev = a when a is in GPU memory, else a view of a contiguous copy of the first `limbs` limbs of a
// (the ones the operation reads; all of them if a has fewer) uploaded to *upload (to release with
// pvda_gpu_free)
static int znx_on_gpu(ZnxVec* dev, int64_t** upload, const ZnxVec* a, uint64_t limbs)
{
	*dev    = *a;
	*upload = NULL;
	if (a->size == 0 || pvda_gpu_owns(a->ptr)) return 0;
	dev->size   = limbs < a->size ? limbs : a->size;
	dev->stride = (int64_t)a->nn;
	dev->ptr    = NULL;
	if (dev->size == 0) return 0;
	if (a->stride == (int64_t)a->nn)
		*upload = pvda_gpu_upload(a->ptr, dev->size * a->nn);
	else
		*upload = pvda_gpu_upload_limbs(a->ptr, a->stride, dev->size, a->nn);
	dev->ptr = *upload;
	return *upload ? 0 : -1;
}

// Downloads the res->size contiguous limbs of src (GPU memory) into res (host memory, strided)
static int znx_download(const ZnxVec* res, const int64_t* src)
{
	size_t n = res->nn;
	if (res->size == 0) return 0;
	if (res->stride == (int64_t)n)
		pvda_gpu_download(res->ptr, src, res->size * n);
	else
		pvda_gpu_download_limbs(res->ptr, res->stride, src, res->size, n);
	return 0;
}

// *dev = res when res is in GPU memory, else a view of a contiguous GPU buffer *tmp for the result
static int znx_result_on_gpu(ZnxVec* dev, int64_t** tmp, const ZnxVec* res)
{
	*dev = *res;
	*tmp = NULL;
	if (res->size == 0 || pvda_gpu_owns(res->ptr)) return 0;
	*tmp        = pvda_gpu_alloc(res->size * res->nn);
	dev->ptr    = *tmp;
	dev->stride = (int64_t)res->nn;
	return *tmp ? 0 : -1;
}

// GPU views of the operands of a coefficient-domain operation (up to two inputs and the result),
// with the GPU buffers created for those in host memory
typedef struct
{
	ZnxVec a, b, res;
	int64_t* buffers[3];
} GpuZnxOperands;

// a and b may be NULL; limbs: the number of limbs of a and b that the operation reads
static int stage(GpuZnxOperands* g, const ZnxVec* res, const ZnxVec* a, const ZnxVec* b, uint64_t limbs)
{
	memset(g, 0, sizeof(*g));
	if (a && znx_on_gpu(&g->a, &g->buffers[0], a, limbs)) return -1;
	if (b && znx_on_gpu(&g->b, &g->buffers[1], b, limbs)) return -1;
	return znx_result_on_gpu(&g->res, &g->buffers[2], res);
}

// Downloads the result into res when it was computed in a GPU buffer (if ok), and releases the
// buffers of stage
static int unstage(GpuZnxOperands* g, const ZnxVec* res, int ok)
{
	int status = ok ? 0 : -1;
	if (ok && g->buffers[2]) status = znx_download(res, g->buffers[2]);
	for (int i = 0; i < 3; ++i) pvda_gpu_free(g->buffers[i]);
	return status;
}

// --- Allocations: the backend objects are GPU buffers ---

static BE_PolyVecDFT* cuda_new_vec_znx_dft(const PvdaBackend* backend, uint64_t size)
{
	return (BE_PolyVecDFT*)pvda_gpu_alloc((size ? size : 1) * nn_of(backend));
}

static void cuda_delete_vec_znx_dft(const PvdaBackend* backend, BE_PolyVecDFT* res)
{
	if (!res) return;
	if (pvda_is_device_pointer(res))
		pvda_gpu_free((int64_t*)res);
	else
		backend->host_vt.pvda_delete_vec_znx_dft(backend, res);
}

static int64_t* cuda_new_vec_znx_big(const PvdaBackend* backend, int64_t size)
{
	return backend->host_vt.pvda_new_vec_znx_big(backend, size);
}

static void cuda_delete_vec_znx_big(const PvdaBackend* backend, int64_t* res)
{
	if (!res) return;
	if (pvda_is_device_pointer(res))
		pvda_gpu_free(res);
	else
		backend->host_vt.pvda_delete_vec_znx_big(backend, res);
}

static BE_PolySvpPrep* cuda_new_svp_ppol(const PvdaBackend* backend)
{
	return (BE_PolySvpPrep*)pvda_gpu_alloc(nn_of(backend));
}

static void cuda_delete_svp_ppol(const PvdaBackend* backend, BE_PolySvpPrep* res)
{
	if (!res) return;
	if (pvda_is_device_pointer(res))
		pvda_gpu_free((int64_t*)res);
	else
		backend->host_vt.pvda_delete_svp_ppol(backend, res);
}

// Zeroed, like the spqlios one
static BE_PolyVecVmpPrep* cuda_new_vmp_pvec(const PvdaBackend* backend, uint64_t nrows)
{
	size_t n_elements = (nrows ? nrows : 1) * nn_of(backend);
	int64_t* res      = pvda_gpu_alloc(n_elements);
	if (res) pvda_gpu_zero(res, n_elements);
	return (BE_PolyVecVmpPrep*)res;
}

static void cuda_delete_vmp_pvec(const PvdaBackend* backend, BE_PolyVecVmpPrep* pvec)
{
	if (!pvec) return;
	if (pvda_is_device_pointer(pvec))
		pvda_gpu_free((int64_t*)pvec);
	else
		backend->host_vt.pvda_delete_vmp_pvec(backend, pvec);
}

static BE_PolyMatVmpPrep* cuda_new_vmp_pmat(const PvdaBackend* backend, uint64_t nrows, uint64_t ncols)
{
	return (BE_PolyMatVmpPrep*)pvda_gpu_alloc((nrows * ncols ? nrows * ncols : 1) * nn_of(backend));
}

static void cuda_delete_vmp_pmat(const PvdaBackend* backend, BE_PolyMatVmpPrep* pmat)
{
	if (!pmat) return;
	if (pvda_is_device_pointer(pmat))
		pvda_gpu_free((int64_t*)pmat);
	else
		backend->host_vt.pvda_delete_vmp_pmat(backend, pmat);
}

// --- DFT (NTT) domain ---

static int cuda_vec_znx_dft(const PvdaBackend* backend, BE_PolyVecDFT* res, uint64_t res_size, const ZnxVec* a)
{
	DISPATCH_OBJECTS("pvda_vec_znx_dft", WHERE(res), WHERE(res, a->ptr),
	                 backend->host_vt.pvda_vec_znx_dft(backend, res, res_size, a));
	GPU_RUN();
	ZnxVec a_dev;
	int64_t* a_upload;
	if (znx_on_gpu(&a_dev, &a_upload, a, res_size)) return -1;
	gpu_vt_vec_znx_dft((int64_t*)res, res_size, a_dev.ptr, a_dev.size, a_dev.stride, a_dev.nn);
	pvda_gpu_free(a_upload);
	GPU_DONE();
}

static int cuda_vec_znx_idft(const PvdaBackend* backend, ZnxVec* res, const BE_PolyVecDFT* a_dft, uint64_t a_size)
{
	DISPATCH_OBJECTS("pvda_vec_znx_idft", WHERE(a_dft), WHERE(res->ptr, a_dft),
	                 backend->host_vt.pvda_vec_znx_idft(backend, res, a_dft, a_size));
	GPU_RUN();
	GpuZnxOperands g;
	if (stage(&g, res, NULL, NULL, 0)) return unstage(&g, res, 0);
	gpu_vt_vec_znx_idft(g.res.ptr, g.res.size, g.res.stride, (const int64_t*)a_dft, a_size, res->nn);
	if (unstage(&g, res, 1)) return -1;
	GPU_DONE();
}

static int cuda_vec_znx_dft_add(const PvdaBackend* backend, BE_PolyVecDFT* res, uint64_t res_n_limbs,
                                BE_PolyVecDFT* a, uint64_t a_n_limbs, BE_PolyVecDFT* b, uint64_t b_n_limbs)
{
	DISPATCH("pvda_vec_znx_dft_add", WHERE(res, a, b),
	         backend->host_vt.pvda_vec_znx_dft_add(backend, res, res_n_limbs, a, a_n_limbs, b, b_n_limbs));
	GPU_RUN();
	gpu_vt_vec_znx_dft_add((int64_t*)res, res_n_limbs, (const int64_t*)a, a_n_limbs, (const int64_t*)b, b_n_limbs,
	                       nn_of(backend));
	GPU_DONE();
}

static int cuda_svp_prepare(const PvdaBackend* backend, BE_PolySvpPrep* prepared_pol, const int64_t* pol)
{
	DISPATCH_OBJECTS("pvda_svp_prepare", WHERE(prepared_pol), WHERE(prepared_pol, pol),
	                 backend->host_vt.pvda_svp_prepare(backend, prepared_pol, pol));
	GPU_RUN();
	uint64_t nn       = nn_of(backend);
	ZnxVec pol_vec    = znx_vec_view(nn, 1, (int64_t)nn, (int64_t*)pol);
	ZnxVec pol_dev;
	int64_t* pol_upload;
	if (znx_on_gpu(&pol_dev, &pol_upload, &pol_vec, 1)) return -1;
	gpu_vt_svp_prepare((int64_t*)prepared_pol, pol_dev.ptr, nn);
	pvda_gpu_free(pol_upload);
	GPU_DONE();
}

static int cuda_svp_apply_dft(const PvdaBackend* backend, const BE_PolyVecDFT* res, uint64_t res_size,
                              const BE_PolySvpPrep* prepared_pol, const ZnxVec* a)
{
	DISPATCH_OBJECTS("pvda_svp_apply_dft", WHERE(res, prepared_pol), WHERE(res, prepared_pol, a->ptr),
	                 backend->host_vt.pvda_svp_apply_dft(backend, res, res_size, prepared_pol, a));
	GPU_RUN();
	ZnxVec a_dev;
	int64_t* a_upload;
	if (znx_on_gpu(&a_dev, &a_upload, a, res_size)) return -1;
	uint64_t rows  = a->size < res_size ? a->size : res_size;
	int64_t* a_ntt = pvda_gpu_alloc((rows ? rows : 1) * a->nn);
	if (!a_ntt)
	{
		pvda_gpu_free(a_upload);
		return -1;
	}
	gpu_vt_vec_znx_dft(a_ntt, rows, a_dev.ptr, a_dev.size, a_dev.stride, a_dev.nn);
	gpu_vt_svp_apply_ntt((int64_t*)res, res_size, (const int64_t*)prepared_pol, a_ntt, rows, a->nn);
	pvda_gpu_free(a_ntt);
	pvda_gpu_free(a_upload);
	GPU_DONE();
}

static int cuda_svp_apply_dft_to_dft(const PvdaBackend* backend, const BE_PolyVecDFT* res, uint64_t res_size,
                                     const BE_PolySvpPrep* ppol, const BE_PolyVecDFT* a, uint64_t a_size)
{
	DISPATCH("pvda_svp_apply_dft_to_dft", WHERE(res, ppol, a),
	         backend->host_vt.pvda_svp_apply_dft_to_dft(backend, res, res_size, ppol, a, a_size));
	GPU_RUN();
	gpu_vt_svp_apply_ntt((int64_t*)res, res_size, (const int64_t*)ppol, (const int64_t*)a, a_size, nn_of(backend));
	GPU_DONE();
}

// --- Vector-matrix products ---

static int cuda_vmp_prepare_contiguous(const PvdaBackend* backend, BE_PolyMatVmpPrep* pmat, const int64_t* mat,
                                       uint64_t nrows, uint64_t ncols)
{
	DISPATCH_OBJECTS("pvda_vmp_prepare_contiguous", WHERE(pmat), WHERE(pmat, mat),
	                 backend->host_vt.pvda_vmp_prepare_contiguous(backend, pmat, mat, nrows, ncols));
	GPU_RUN();
	uint64_t nn    = nn_of(backend);
	ZnxVec mat_vec = znx_vec_view(nn, nrows * ncols, (int64_t)nn, (int64_t*)mat);
	ZnxVec mat_dev;
	int64_t* mat_upload;
	if (znx_on_gpu(&mat_dev, &mat_upload, &mat_vec, nrows * ncols)) return -1;
	gpu_vt_vmp_prepare((int64_t*)pmat, mat_dev.ptr, nrows, ncols, nn);
	pvda_gpu_free(mat_upload);
	GPU_DONE();
}

static int cuda_vmp_apply_dft(const PvdaBackend* backend, BE_PolyVecDFT* res, uint64_t res_size, const ZnxVec* a,
                              const BE_PolyMatVmpPrep* pmat, uint64_t nrows, uint64_t ncols)
{
	DISPATCH_OBJECTS("pvda_vmp_apply_dft", WHERE(res, pmat), WHERE(res, a->ptr, pmat),
	                 backend->host_vt.pvda_vmp_apply_dft(backend, res, res_size, a, pmat, nrows, ncols));
	GPU_RUN();
	ZnxVec a_dev;
	int64_t* a_upload;
	if (znx_on_gpu(&a_dev, &a_upload, a, nrows)) return -1;
	uint64_t rows  = a->size < nrows ? a->size : nrows;
	int64_t* a_ntt = pvda_gpu_alloc((rows ? rows : 1) * a->nn);
	if (!a_ntt)
	{
		pvda_gpu_free(a_upload);
		return -1;
	}
	gpu_vt_vec_znx_dft(a_ntt, rows, a_dev.ptr, a_dev.size, a_dev.stride, a_dev.nn);
	gpu_vt_vmp_apply_ntt((int64_t*)res, res_size, a_ntt, rows, (const int64_t*)pmat, nrows, ncols, a->nn);
	pvda_gpu_free(a_ntt);
	pvda_gpu_free(a_upload);
	GPU_DONE();
}

static int cuda_vmp_apply_dft_to_dft(const PvdaBackend* backend, BE_PolyVecDFT* res, const uint64_t res_size,
                                     const BE_PolyVecDFT* a_dft, uint64_t a_size, const BE_PolyMatVmpPrep* pmat,
                                     const uint64_t nrows, const uint64_t ncols)
{
	DISPATCH("pvda_vmp_apply_dft_to_dft", WHERE(res, a_dft, pmat),
	         backend->host_vt.pvda_vmp_apply_dft_to_dft(backend, res, res_size, a_dft, a_size, pmat, nrows, ncols));
	GPU_RUN();
	gpu_vt_vmp_apply_ntt((int64_t*)res, res_size, (const int64_t*)a_dft, a_size, (const int64_t*)pmat, nrows, ncols,
	                     nn_of(backend));
	GPU_DONE();
}

// On the GPU a prepared vector (pvda_vmp_prepare_vec) is the NTT of the vector
static int cuda_vmp_apply_prepared_to_dft(const PvdaBackend* backend, BE_PolyVecDFT* res, const uint64_t res_size,
                                          const BE_PolyVecVmpPrep* a_prep, uint64_t a_size,
                                          const BE_PolyMatVmpPrep* pmat, const uint64_t nrows, const uint64_t ncols)
{
	DISPATCH(
	    "pvda_vmp_apply_prepared_to_dft", WHERE(res, a_prep, pmat),
	    backend->host_vt.pvda_vmp_apply_prepared_to_dft(backend, res, res_size, a_prep, a_size, pmat, nrows, ncols));
	GPU_RUN();
	gpu_vt_vmp_apply_ntt((int64_t*)res, res_size, (const int64_t*)a_prep, a_size, (const int64_t*)pmat, nrows, ncols,
	                     nn_of(backend));
	GPU_DONE();
}

static int cuda_vmp_prepare_vec(const PvdaBackend* backend, BE_PolyVecVmpPrep* pvec, uint64_t nrows, const ZnxVec* a)
{
	DISPATCH_OBJECTS("pvda_vmp_prepare_vec", WHERE(pvec), WHERE(pvec, a->ptr),
	                 backend->host_vt.pvda_vmp_prepare_vec(backend, pvec, nrows, a));
	GPU_RUN();
	ZnxVec a_dev;
	int64_t* a_upload;
	if (znx_on_gpu(&a_dev, &a_upload, a, nrows)) return -1;
	gpu_vt_vec_znx_dft((int64_t*)pvec, nrows, a_dev.ptr, a_dev.size, a_dev.stride, a_dev.nn);
	pvda_gpu_free(a_upload);
	GPU_DONE();
}

// --- Coefficient domain: on the GPU, host vectors being uploaded and a host result downloaded ---

static int cuda_vec_znx_normalize_base2k(const PvdaBackend* backend, uint64_t log2_base2k, ZnxVec* res,
                                         const ZnxVec* a)
{
	GPU_RUN();
	GpuZnxOperands g;
	if (stage(&g, res, a, NULL, a->size)) return unstage(&g, res, 0);
	gpu_normalize_base2k_sized_device(g.a.ptr, g.a.size, g.a.stride, g.res.ptr, g.res.size, g.res.stride, a->nn,
	                                  (uint32_t)log2_base2k);
	if (unstage(&g, res, 1)) return -1;
	GPU_DONE();
}

static int cuda_znx_small_product(const PvdaBackend* backend, int64_t* res, const int64_t* a, const int64_t* b)
{
	DISPATCH("pvda_znx_small_product", WHERE(res, a, b), backend->host_vt.pvda_znx_small_product(backend, res, a, b));
	log_message(LOG_ERROR, "pvda_znx_small_product has no GPU implementation");
	return -1;
}

static int cuda_vec_znx_negate(const PvdaBackend* backend, ZnxVec* res, const ZnxVec* a)
{
	GPU_RUN();
	GpuZnxOperands g;
	if (stage(&g, res, a, NULL, res->size)) return unstage(&g, res, 0);
	gpu_vt_vec_znx_negate(g.res.ptr, g.res.size, g.res.stride, g.a.ptr, g.a.size, g.a.stride, res->nn);
	if (unstage(&g, res, 1)) return -1;
	GPU_DONE();
}

static int cuda_vec_znx_add(const PvdaBackend* backend, ZnxVec* res, const ZnxVec* a, const ZnxVec* b)
{
	GPU_RUN();
	GpuZnxOperands g;
	if (stage(&g, res, a, b, res->size)) return unstage(&g, res, 0);
	gpu_vt_vec_znx_add(g.res.ptr, g.res.size, g.res.stride, g.a.ptr, g.a.size, g.a.stride, g.b.ptr, g.b.size,
	                   g.b.stride, res->nn);
	if (unstage(&g, res, 1)) return -1;
	GPU_DONE();
}

static int cuda_vec_znx_sub(const PvdaBackend* backend, ZnxVec* res, const ZnxVec* a, const ZnxVec* b)
{
	GPU_RUN();
	GpuZnxOperands g;
	if (stage(&g, res, a, b, res->size)) return unstage(&g, res, 0);
	gpu_vt_vec_znx_sub(g.res.ptr, g.res.size, g.res.stride, g.a.ptr, g.a.size, g.a.stride, g.b.ptr, g.b.size,
	                   g.b.stride, res->nn);
	if (unstage(&g, res, 1)) return -1;
	GPU_DONE();
}

static int cuda_vec_znx_automorphism(const PvdaBackend* backend, const int64_t p, ZnxVec* res, const ZnxVec* a)
{
	GPU_RUN();
	GpuZnxOperands g;
	if (stage(&g, res, a, NULL, res->size)) return unstage(&g, res, 0);
	// A result in a buffer of stage never aliases a, so the scratch-free variant is safe then
	if (g.buffers[2])
		gpu_vec_znx_automorphism_device_noalias(p, g.a.ptr, g.res.ptr, res->nn, g.res.size, g.res.stride, g.a.size,
		                                        g.a.stride);
	else
		gpu_vec_znx_automorphism_device(p, g.a.ptr, g.res.ptr, res->nn, g.res.size, g.res.stride, g.a.size,
		                                g.a.stride);
	if (unstage(&g, res, 1)) return -1;
	GPU_DONE();
}

static int cuda_znx_automorphism(const PvdaBackend* backend, const int64_t p, int64_t* res, const int64_t* a)
{
	uint64_t nn  = nn_of(backend);
	ZnxVec res_v = znx_vec_view(nn, 1, (int64_t)nn, res);
	ZnxVec a_v   = znx_vec_view(nn, 1, (int64_t)nn, (int64_t*)a);
	return cuda_vec_znx_automorphism(backend, p, &res_v, &a_v);
}

static int cuda_vec_znx_rotate(const PvdaBackend* backend, const int64_t p, ZnxVec* res, const ZnxVec* a)
{
	GPU_RUN();
	GpuZnxOperands g;
	if (stage(&g, res, a, NULL, res->size)) return unstage(&g, res, 0);
	gpu_vec_znx_rotate_device(p, g.a.ptr, g.res.ptr, res->nn, g.res.size, g.res.stride, g.a.size, g.a.stride);
	if (unstage(&g, res, 1)) return -1;
	GPU_DONE();
}

void pvda_fill_cuda(struct pvda_virtual_table* vt)
{
	vt->pvda_new_vec_znx_dft           = cuda_new_vec_znx_dft;
	vt->pvda_vec_znx_dft               = cuda_vec_znx_dft;
	vt->pvda_delete_vec_znx_dft        = cuda_delete_vec_znx_dft;
	vt->pvda_new_vec_znx_big           = cuda_new_vec_znx_big;
	vt->pvda_vec_znx_idft              = cuda_vec_znx_idft;
	vt->pvda_delete_vec_znx_big        = cuda_delete_vec_znx_big;
	vt->pvda_vec_znx_dft_add           = cuda_vec_znx_dft_add;
	vt->pvda_new_svp_ppol              = cuda_new_svp_ppol;
	vt->pvda_svp_prepare               = cuda_svp_prepare;
	vt->pvda_svp_apply_dft             = cuda_svp_apply_dft;
	vt->pvda_svp_apply_dft_to_dft      = cuda_svp_apply_dft_to_dft;
	vt->pvda_delete_svp_ppol           = cuda_delete_svp_ppol;
	vt->pvda_new_vmp_pvec              = cuda_new_vmp_pvec;
	vt->pvda_delete_vmp_pvec           = cuda_delete_vmp_pvec;
	vt->pvda_new_vmp_pmat              = cuda_new_vmp_pmat;
	vt->pvda_vmp_prepare_contiguous    = cuda_vmp_prepare_contiguous;
	vt->pvda_vmp_apply_dft             = cuda_vmp_apply_dft;
	vt->pvda_vmp_apply_dft_to_dft      = cuda_vmp_apply_dft_to_dft;
	vt->pvda_vmp_apply_prepared_to_dft = cuda_vmp_apply_prepared_to_dft;
	vt->pvda_delete_vmp_pmat           = cuda_delete_vmp_pmat;
	vt->pvda_vmp_prepare_vec           = cuda_vmp_prepare_vec;
	vt->pvda_vec_znx_normalize_base2k  = cuda_vec_znx_normalize_base2k;
	vt->pvda_znx_small_product         = cuda_znx_small_product;
	vt->pvda_vec_znx_negate            = cuda_vec_znx_negate;
	vt->pvda_vec_znx_add               = cuda_vec_znx_add;
	vt->pvda_vec_znx_sub               = cuda_vec_znx_sub;
	vt->pvda_znx_automorphism          = cuda_znx_automorphism;
	vt->pvda_vec_znx_automorphism      = cuda_vec_znx_automorphism;
	vt->pvda_vec_znx_rotate            = cuda_vec_znx_rotate;
	// pvda_module_extract_nn, the RNG and number-theory entries stay the spqlios/CPU ones
}
