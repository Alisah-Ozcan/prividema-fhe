#include "impls/cuda_alias.h"

#include <stdint.h>

#include "gpu/common/gpu_stream.h"
#include "gpu/host/ggsw_external_product_gpu.h"
#include "gpu/host/normalize_host.h"
#include "gpu/host/vec_znx_rotate_automorphism_host.h"
#include "gpu/host/vtable_ops_host.h"
#include "logger.h"
#include "maths_structures.h"

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

// A GPU operation records its failures (gpu_stream.h) instead of aborting: clear any stale
// one before running, and report a new one as this operation's -1
#define GPU_RUN()  pvda_gpu_take_error()
#define GPU_DONE() return pvda_gpu_take_error() ? -1 : 0

static uint64_t nn_of(const PvdaBackend* backend) { return backend->host_vt.pvda_module_extract_nn(backend); }

// --- Allocations: the table's buffers are host buffers (see cuda_alias.h) ---

static VecUnivDFT* cuda_new_vec_znx_dft(const PvdaBackend* backend, uint64_t size)
{
	return backend->host_vt.pvda_new_vec_znx_dft(backend, size);
}

static void cuda_delete_vec_znx_dft(const PvdaBackend* backend, double* res)
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

static PolyUnivDFT* cuda_new_svp_ppol(const PvdaBackend* backend)
{
	return backend->host_vt.pvda_new_svp_ppol(backend);
}

static void cuda_delete_svp_ppol(const PvdaBackend* backend, PolyUnivDFT* res)
{
	if (!res) return;
	if (pvda_is_device_pointer(res))
		pvda_gpu_free((int64_t*)res);
	else
		backend->host_vt.pvda_delete_svp_ppol(backend, res);
}

static MatBivDFT* cuda_new_vmp_pmat(const PvdaBackend* backend, uint64_t nrows, uint64_t ncols)
{
	return backend->host_vt.pvda_new_vmp_pmat(backend, nrows, ncols);
}

static void cuda_delete_vmp_pmat(const PvdaBackend* backend, MatBivDFT* pmat)
{
	if (!pmat) return;
	if (pvda_is_device_pointer(pmat))
		pvda_gpu_free((int64_t*)pmat);
	else
		backend->host_vt.pvda_delete_vmp_pmat(backend, pmat);
}

// --- DFT (NTT) domain ---

static int cuda_vec_znx_dft(const PvdaBackend* backend, PolyBivDFT* res, uint64_t res_size, const PolyBiv* a)
{
	DISPATCH("pvda_vec_znx_dft", WHERE(res, a->ptr), backend->host_vt.pvda_vec_znx_dft(backend, res, res_size, a));
	GPU_RUN();
	gpu_vt_vec_znx_dft((int64_t*)res, res_size, a->ptr, a->l, a->stride, a->nn);
	GPU_DONE();
}

static int cuda_vec_znx_idft(const PvdaBackend* backend, PolyBiv* res, const PolyBivDFT* a_dft, uint64_t a_size)
{
	DISPATCH("pvda_vec_znx_idft", WHERE(res->ptr, a_dft),
	         backend->host_vt.pvda_vec_znx_idft(backend, res, a_dft, a_size));
	GPU_RUN();
	gpu_vt_vec_znx_idft(res->ptr, res->l, res->stride, (const int64_t*)a_dft, a_size, res->nn);
	GPU_DONE();
}

static int cuda_svp_prepare(const PvdaBackend* backend, PolyUnivDFT* prepared_pol, const int64_t* pol)
{
	DISPATCH("pvda_svp_prepare", WHERE(prepared_pol, pol),
	         backend->host_vt.pvda_svp_prepare(backend, prepared_pol, pol));
	GPU_RUN();
	gpu_vt_svp_prepare((int64_t*)prepared_pol, pol, nn_of(backend));
	GPU_DONE();
}

static int cuda_svp_apply_dft(const PvdaBackend* backend, const PolyBivDFT* res, uint64_t res_size,
                              const PolyUnivDFT* prepared_pol, const PolyBiv* a)
{
	DISPATCH("pvda_svp_apply_dft", WHERE(res, prepared_pol, a->ptr),
	         backend->host_vt.pvda_svp_apply_dft(backend, res, res_size, prepared_pol, a));
	GPU_RUN();
	uint64_t rows  = a->l < res_size ? a->l : res_size;
	int64_t* a_ntt = pvda_gpu_alloc(rows * a->nn);
	if (!a_ntt) return -1;
	gpu_vt_vec_znx_dft(a_ntt, rows, a->ptr, a->l, a->stride, a->nn);
	gpu_vt_svp_apply_ntt((int64_t*)res, res_size, (const int64_t*)prepared_pol, a_ntt, rows, a->nn);
	pvda_gpu_free(a_ntt);
	GPU_DONE();
}

static int cuda_svp_apply_dft_to_dft(const PvdaBackend* backend, const PolyBivDFT* res, uint64_t res_size,
                                     const PolyUnivDFT* ppol, const PolyBivDFT* a, uint64_t a_size)
{
	DISPATCH("pvda_svp_apply_dft_to_dft", WHERE(res, ppol, a),
	         backend->host_vt.pvda_svp_apply_dft_to_dft(backend, res, res_size, ppol, a, a_size));
	GPU_RUN();
	gpu_vt_svp_apply_ntt((int64_t*)res, res_size, (const int64_t*)ppol, (const int64_t*)a, a_size, nn_of(backend));
	GPU_DONE();
}

// --- Vector-matrix products ---

static int cuda_vmp_prepare_contiguous(const PvdaBackend* backend, MatBivDFT* pmat, const int64_t* mat, uint64_t nrows,
                                       uint64_t ncols)
{
	DISPATCH("pvda_vmp_prepare_contiguous", WHERE(pmat, mat),
	         backend->host_vt.pvda_vmp_prepare_contiguous(backend, pmat, mat, nrows, ncols));
	GPU_RUN();
	gpu_vt_vmp_prepare((int64_t*)pmat, mat, nrows, ncols, nn_of(backend));
	GPU_DONE();
}

static int cuda_vmp_apply_dft(const PvdaBackend* backend, PolyBivDFT* res, uint64_t res_size, const PolyBiv* a,
                              const MatBivDFT* pmat, uint64_t nrows, uint64_t ncols)
{
	DISPATCH("pvda_vmp_apply_dft", WHERE(res, a->ptr, pmat),
	         backend->host_vt.pvda_vmp_apply_dft(backend, res, res_size, a, pmat, nrows, ncols));
	GPU_RUN();
	uint64_t rows  = a->l < nrows ? a->l : nrows;
	int64_t* a_ntt = pvda_gpu_alloc((rows ? rows : 1) * a->nn);
	if (!a_ntt) return -1;
	gpu_vt_vec_znx_dft(a_ntt, rows, a->ptr, a->l, a->stride, a->nn);
	gpu_vt_vmp_apply_ntt((int64_t*)res, res_size, a_ntt, rows, (const int64_t*)pmat, nrows, ncols, a->nn);
	pvda_gpu_free(a_ntt);
	GPU_DONE();
}

static int cuda_vmp_apply_dft_to_dft(const PvdaBackend* backend, VecBivDFT* res, const uint64_t res_size,
                                     const VecBivDFT* a_dft, uint64_t a_size, const MatBivDFT* pmat,
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
static int cuda_vmp_apply_prepared_to_dft(const PvdaBackend* backend, VecBivDFT* res, const uint64_t res_size,
                                          const VecBivDFT* a_prep, uint64_t a_size, const MatBivDFT* pmat,
                                          const uint64_t nrows, const uint64_t ncols)
{
	DISPATCH(
	    "pvda_vmp_apply_prepared_to_dft", WHERE(res, a_prep, pmat),
	    backend->host_vt.pvda_vmp_apply_prepared_to_dft(backend, res, res_size, a_prep, a_size, pmat, nrows, ncols));
	GPU_RUN();
	gpu_vt_vmp_apply_ntt((int64_t*)res, res_size, (const int64_t*)a_prep, a_size, (const int64_t*)pmat, nrows, ncols,
	                     nn_of(backend));
	GPU_DONE();
}

static int cuda_vmp_prepare_vec(const PvdaBackend* backend, PolyBivPrep* pvec, uint64_t nrows, const PolyBiv* a)
{
	DISPATCH("pvda_vmp_prepare_vec", WHERE(pvec, a->ptr),
	         backend->host_vt.pvda_vmp_prepare_vec(backend, pvec, nrows, a));
	GPU_RUN();
	gpu_vt_vec_znx_dft((int64_t*)pvec, nrows, a->ptr, a->l, a->stride, a->nn);
	GPU_DONE();
}

// --- Coefficient domain ---

static int cuda_vec_znx_normalize_base2k(const PvdaBackend* backend, uint64_t log2_base2k, PolyBiv* res,
                                         const PolyBiv* a)
{
	DISPATCH("pvda_vec_znx_normalize_base2k", WHERE(res->ptr, a->ptr),
	         backend->host_vt.pvda_vec_znx_normalize_base2k(backend, log2_base2k, res, a));
	GPU_RUN();
	gpu_normalize_base2k_sized_device(a->ptr, a->l, a->stride, res->ptr, res->l, res->stride, a->nn,
	                                  (uint32_t)log2_base2k);
	GPU_DONE();
}

static int cuda_znx_small_product(const PvdaBackend* backend, PolyUniv* res, const PolyUniv* a, const PolyUniv* b)
{
	DISPATCH("pvda_znx_small_product", WHERE(res, a, b), backend->host_vt.pvda_znx_small_product(backend, res, a, b));
	log_message(LOG_ERROR, "pvda_znx_small_product has no GPU implementation");
	return -1;
}

static int cuda_vec_znx_negate(const PvdaBackend* backend, PolyBiv* res, const PolyBiv* a)
{
	DISPATCH("pvda_vec_znx_negate", WHERE(res->ptr, a->ptr), backend->host_vt.pvda_vec_znx_negate(backend, res, a));
	GPU_RUN();
	gpu_vt_vec_znx_negate(res->ptr, res->l, res->stride, a->ptr, a->l, a->stride, res->nn);
	GPU_DONE();
}

static int cuda_vec_znx_add(const PvdaBackend* backend, PolyBiv* res, const PolyBiv* a, const PolyBiv* b)
{
	DISPATCH("pvda_vec_znx_add", WHERE(res->ptr, a->ptr, b->ptr),
	         backend->host_vt.pvda_vec_znx_add(backend, res, a, b));
	GPU_RUN();
	gpu_vt_vec_znx_add(res->ptr, res->l, res->stride, a->ptr, a->l, a->stride, b->ptr, b->l, b->stride, res->nn);
	GPU_DONE();
}

static int cuda_vec_znx_sub(const PvdaBackend* backend, PolyBiv* res, const PolyBiv* a, const PolyBiv* b)
{
	DISPATCH("pvda_vec_znx_sub", WHERE(res->ptr, a->ptr, b->ptr),
	         backend->host_vt.pvda_vec_znx_sub(backend, res, a, b));
	GPU_RUN();
	gpu_vt_vec_znx_sub(res->ptr, res->l, res->stride, a->ptr, a->l, a->stride, b->ptr, b->l, b->stride, res->nn);
	GPU_DONE();
}

static int cuda_znx_automorphism(const PvdaBackend* backend, const int64_t p, PolyUniv* res, const PolyUniv* a)
{
	DISPATCH("pvda_znx_automorphism", WHERE(res, a), backend->host_vt.pvda_znx_automorphism(backend, p, res, a));
	GPU_RUN();
	uint64_t nn = nn_of(backend);
	gpu_vec_znx_automorphism_device(p, a, res, nn, 1, (int64_t)nn, 1, (int64_t)nn);
	GPU_DONE();
}

static int cuda_vec_znx_automorphism(const PvdaBackend* backend, const int64_t p, PolyBiv* res, const PolyBiv* a)
{
	DISPATCH("pvda_vec_znx_automorphism", WHERE(res->ptr, a->ptr),
	         backend->host_vt.pvda_vec_znx_automorphism(backend, p, res, a));
	GPU_RUN();
	gpu_vec_znx_automorphism_device(p, a->ptr, res->ptr, res->nn, res->l, res->stride, a->l, a->stride);
	GPU_DONE();
}

static int cuda_vec_znx_rotate(const PvdaBackend* backend, const int64_t p, PolyBiv* res, const PolyBiv* a)
{
	DISPATCH("pvda_vec_znx_rotate", WHERE(res->ptr, a->ptr), backend->host_vt.pvda_vec_znx_rotate(backend, p, res, a));
	GPU_RUN();
	gpu_vec_znx_rotate_device(p, a->ptr, res->ptr, res->nn, res->l, res->stride, a->l, a->stride);
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
	vt->pvda_new_svp_ppol              = cuda_new_svp_ppol;
	vt->pvda_svp_prepare               = cuda_svp_prepare;
	vt->pvda_svp_apply_dft             = cuda_svp_apply_dft;
	vt->pvda_svp_apply_dft_to_dft      = cuda_svp_apply_dft_to_dft;
	vt->pvda_delete_svp_ppol           = cuda_delete_svp_ppol;
	vt->pvda_new_vmp_pmat              = cuda_new_vmp_pmat;
	vt->pvda_vmp_prepare_contiguous    = cuda_vmp_prepare_contiguous;
	vt->pvda_vmp_apply_dft             = cuda_vmp_apply_dft;
	vt->pvda_vmp_apply_dft_to_dft      = cuda_vmp_apply_dft_to_dft;
	vt->pvda_vmp_apply_prepared_to_dft = cuda_vmp_apply_prepared_to_dft;
	vt->pvda_delete_vmp_pmat           = cuda_delete_vmp_pmat;
	vt->pvda_vec_znx_normalize_base2k  = cuda_vec_znx_normalize_base2k;
	vt->pvda_znx_small_product         = cuda_znx_small_product;
	vt->pvda_vec_znx_negate            = cuda_vec_znx_negate;
	vt->pvda_vec_znx_add               = cuda_vec_znx_add;
	vt->pvda_vec_znx_sub               = cuda_vec_znx_sub;
	vt->pvda_znx_automorphism          = cuda_znx_automorphism;
	vt->pvda_vec_znx_automorphism      = cuda_vec_znx_automorphism;
	vt->pvda_vec_znx_rotate            = cuda_vec_znx_rotate;
	vt->pvda_vmp_prepare_vec           = cuda_vmp_prepare_vec;
	// pvda_module_extract_nn, the RNG and number-theory entries stay the spqlios/CPU ones
}

// --- spqlios backend in a CUDA build: reject GPU operands instead of dereferencing them ---

static int any_gpu_owned(int n, const void* const* ptrs)
{
	for (int i = 0; i < n; ++i)
		if (pvda_gpu_owns(ptrs[i])) return 1;
	return 0;
}

#define GUARD(op, ...)                                                                               \
	do                                                                                               \
	{                                                                                                \
		if (any_gpu_owned((int)(sizeof((const void*[]){__VA_ARGS__}) / sizeof(const void*)),         \
		                  (const void*[]){__VA_ARGS__}))                                             \
		{                                                                                            \
			log_message(LOG_ERROR, op ": GPU operands need a CUDA backend (pvda_new_cuda_backend)"); \
			return -1;                                                                               \
		}                                                                                            \
	} while (0)

static void guard_delete_vec_znx_dft(const PvdaBackend* backend, double* res)
{
	if (pvda_gpu_owns(res))
		pvda_gpu_free((int64_t*)res);
	else
		backend->host_vt.pvda_delete_vec_znx_dft(backend, res);
}

static void guard_delete_vec_znx_big(const PvdaBackend* backend, int64_t* res)
{
	if (pvda_gpu_owns(res))
		pvda_gpu_free(res);
	else
		backend->host_vt.pvda_delete_vec_znx_big(backend, res);
}

static void guard_delete_svp_ppol(const PvdaBackend* backend, PolyUnivDFT* res)
{
	if (pvda_gpu_owns(res))
		pvda_gpu_free((int64_t*)res);
	else
		backend->host_vt.pvda_delete_svp_ppol(backend, res);
}

static void guard_delete_vmp_pmat(const PvdaBackend* backend, MatBivDFT* pmat)
{
	if (pvda_gpu_owns(pmat))
		pvda_gpu_free((int64_t*)pmat);
	else
		backend->host_vt.pvda_delete_vmp_pmat(backend, pmat);
}

static int guard_vec_znx_dft(const PvdaBackend* backend, PolyBivDFT* res, uint64_t res_size, const PolyBiv* a)
{
	GUARD("pvda_vec_znx_dft", res, a->ptr);
	return backend->host_vt.pvda_vec_znx_dft(backend, res, res_size, a);
}

static int guard_vec_znx_idft(const PvdaBackend* backend, PolyBiv* res, const PolyBivDFT* a_dft, uint64_t a_size)
{
	GUARD("pvda_vec_znx_idft", res->ptr, a_dft);
	return backend->host_vt.pvda_vec_znx_idft(backend, res, a_dft, a_size);
}

static int guard_svp_prepare(const PvdaBackend* backend, PolyUnivDFT* prepared_pol, const int64_t* pol)
{
	GUARD("pvda_svp_prepare", prepared_pol, pol);
	return backend->host_vt.pvda_svp_prepare(backend, prepared_pol, pol);
}

static int guard_svp_apply_dft(const PvdaBackend* backend, const PolyBivDFT* res, uint64_t res_size,
                               const PolyUnivDFT* prepared_pol, const PolyBiv* a)
{
	GUARD("pvda_svp_apply_dft", res, prepared_pol, a->ptr);
	return backend->host_vt.pvda_svp_apply_dft(backend, res, res_size, prepared_pol, a);
}

static int guard_svp_apply_dft_to_dft(const PvdaBackend* backend, const PolyBivDFT* res, uint64_t res_size,
                                      const PolyUnivDFT* ppol, const PolyBivDFT* a, uint64_t a_size)
{
	GUARD("pvda_svp_apply_dft_to_dft", res, ppol, a);
	return backend->host_vt.pvda_svp_apply_dft_to_dft(backend, res, res_size, ppol, a, a_size);
}

static int guard_vmp_prepare_contiguous(const PvdaBackend* backend, MatBivDFT* pmat, const int64_t* mat, uint64_t nrows,
                                        uint64_t ncols)
{
	GUARD("pvda_vmp_prepare_contiguous", pmat, mat);
	return backend->host_vt.pvda_vmp_prepare_contiguous(backend, pmat, mat, nrows, ncols);
}

static int guard_vmp_apply_dft(const PvdaBackend* backend, PolyBivDFT* res, uint64_t res_size, const PolyBiv* a,
                               const MatBivDFT* pmat, uint64_t nrows, uint64_t ncols)
{
	GUARD("pvda_vmp_apply_dft", res, a->ptr, pmat);
	return backend->host_vt.pvda_vmp_apply_dft(backend, res, res_size, a, pmat, nrows, ncols);
}

static int guard_vmp_apply_dft_to_dft(const PvdaBackend* backend, VecBivDFT* res, const uint64_t res_size,
                                      const VecBivDFT* a_dft, uint64_t a_size, const MatBivDFT* pmat,
                                      const uint64_t nrows, const uint64_t ncols)
{
	GUARD("pvda_vmp_apply_dft_to_dft", res, a_dft, pmat);
	return backend->host_vt.pvda_vmp_apply_dft_to_dft(backend, res, res_size, a_dft, a_size, pmat, nrows, ncols);
}

static int guard_vmp_apply_prepared_to_dft(const PvdaBackend* backend, VecBivDFT* res, const uint64_t res_size,
                                           const VecBivDFT* a_prep, uint64_t a_size, const MatBivDFT* pmat,
                                           const uint64_t nrows, const uint64_t ncols)
{
	GUARD("pvda_vmp_apply_prepared_to_dft", res, a_prep, pmat);
	return backend->host_vt.pvda_vmp_apply_prepared_to_dft(backend, res, res_size, a_prep, a_size, pmat, nrows, ncols);
}

static int guard_vmp_prepare_vec(const PvdaBackend* backend, PolyBivPrep* pvec, uint64_t nrows, const PolyBiv* a)
{
	GUARD("pvda_vmp_prepare_vec", pvec, a->ptr);
	return backend->host_vt.pvda_vmp_prepare_vec(backend, pvec, nrows, a);
}

static int guard_vec_znx_normalize_base2k(const PvdaBackend* backend, uint64_t log2_base2k, PolyBiv* res,
                                          const PolyBiv* a)
{
	GUARD("pvda_vec_znx_normalize_base2k", res->ptr, a->ptr);
	return backend->host_vt.pvda_vec_znx_normalize_base2k(backend, log2_base2k, res, a);
}

static int guard_znx_small_product(const PvdaBackend* backend, PolyUniv* res, const PolyUniv* a, const PolyUniv* b)
{
	GUARD("pvda_znx_small_product", res, a, b);
	return backend->host_vt.pvda_znx_small_product(backend, res, a, b);
}

static int guard_vec_znx_negate(const PvdaBackend* backend, PolyBiv* res, const PolyBiv* a)
{
	GUARD("pvda_vec_znx_negate", res->ptr, a->ptr);
	return backend->host_vt.pvda_vec_znx_negate(backend, res, a);
}

static int guard_vec_znx_add(const PvdaBackend* backend, PolyBiv* res, const PolyBiv* a, const PolyBiv* b)
{
	GUARD("pvda_vec_znx_add", res->ptr, a->ptr, b->ptr);
	return backend->host_vt.pvda_vec_znx_add(backend, res, a, b);
}

static int guard_vec_znx_sub(const PvdaBackend* backend, PolyBiv* res, const PolyBiv* a, const PolyBiv* b)
{
	GUARD("pvda_vec_znx_sub", res->ptr, a->ptr, b->ptr);
	return backend->host_vt.pvda_vec_znx_sub(backend, res, a, b);
}

static int guard_znx_automorphism(const PvdaBackend* backend, const int64_t p, PolyUniv* res, const PolyUniv* a)
{
	GUARD("pvda_znx_automorphism", res, a);
	return backend->host_vt.pvda_znx_automorphism(backend, p, res, a);
}

static int guard_vec_znx_automorphism(const PvdaBackend* backend, const int64_t p, PolyBiv* res, const PolyBiv* a)
{
	GUARD("pvda_vec_znx_automorphism", res->ptr, a->ptr);
	return backend->host_vt.pvda_vec_znx_automorphism(backend, p, res, a);
}

static int guard_vec_znx_rotate(const PvdaBackend* backend, const int64_t p, PolyBiv* res, const PolyBiv* a)
{
	GUARD("pvda_vec_znx_rotate", res->ptr, a->ptr);
	return backend->host_vt.pvda_vec_znx_rotate(backend, p, res, a);
}

void pvda_fill_spqlios_gpu_guard(struct pvda_virtual_table* vt)
{
	vt->pvda_vec_znx_dft               = guard_vec_znx_dft;
	vt->pvda_delete_vec_znx_dft        = guard_delete_vec_znx_dft;
	vt->pvda_vec_znx_idft              = guard_vec_znx_idft;
	vt->pvda_delete_vec_znx_big        = guard_delete_vec_znx_big;
	vt->pvda_svp_prepare               = guard_svp_prepare;
	vt->pvda_svp_apply_dft             = guard_svp_apply_dft;
	vt->pvda_svp_apply_dft_to_dft      = guard_svp_apply_dft_to_dft;
	vt->pvda_delete_svp_ppol           = guard_delete_svp_ppol;
	vt->pvda_vmp_prepare_contiguous    = guard_vmp_prepare_contiguous;
	vt->pvda_vmp_apply_dft             = guard_vmp_apply_dft;
	vt->pvda_vmp_apply_dft_to_dft      = guard_vmp_apply_dft_to_dft;
	vt->pvda_vmp_apply_prepared_to_dft = guard_vmp_apply_prepared_to_dft;
	vt->pvda_delete_vmp_pmat           = guard_delete_vmp_pmat;
	vt->pvda_vec_znx_normalize_base2k  = guard_vec_znx_normalize_base2k;
	vt->pvda_znx_small_product         = guard_znx_small_product;
	vt->pvda_vec_znx_negate            = guard_vec_znx_negate;
	vt->pvda_vec_znx_add               = guard_vec_znx_add;
	vt->pvda_vec_znx_sub               = guard_vec_znx_sub;
	vt->pvda_znx_automorphism          = guard_znx_automorphism;
	vt->pvda_vec_znx_automorphism      = guard_vec_znx_automorphism;
	vt->pvda_vec_znx_rotate            = guard_vec_znx_rotate;
	vt->pvda_vmp_prepare_vec           = guard_vmp_prepare_vec;
	// allocations, pvda_module_extract_nn, the RNG and number-theory entries stay the spqlios ones
}
