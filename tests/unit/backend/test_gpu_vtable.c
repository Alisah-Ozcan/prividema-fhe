#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "backend/backend.h"
#include "backend/backend_arithmetic.h"
#include "core/glwe/bivariate_polynomial.h"
#include "gpu/host/ggsw_external_product_gpu.h"
#include "maths_structures.h"

// The CUDA backend's virtual table: every operation on GPU operands must give exactly the
// spqlios result on the same host operands, including operands of different limb counts and
// strided views. DFT-domain operations are compared after the inverse transform (the GPU DFT
// domain is an NTT domain); inputs are small enough for the CPU FFT results to be exact.

#define NN 1024

static PvdaBackend* cpu;
static PvdaBackend* gpu;

static void setup(void)
{
	srand(1234);
	cpu = pvda_new_spqlios_backend(NN);
	gpu = pvda_new_cuda_backend(NN);
	cr_assert_not_null(gpu);
}

static void teardown(void)
{
	pvda_delete_backend(cpu);
	pvda_delete_backend(gpu);
}

TestSuite(gpu_vtable, .init = setup, .fini = teardown);

static int64_t* random_vec(size_t n_elems, int bits)
{
	int64_t* v = malloc(n_elems * sizeof(int64_t));
	for (size_t i = 0; i < n_elems; ++i) v[i] = (int64_t)(rand() % (1 << bits)) - (1 << (bits - 1));
	return v;
}

static int64_t* download(const int64_t* dev, size_t n_elems)
{
	int64_t* h = malloc(n_elems * sizeof(int64_t));
	pvda_gpu_download(h, dev, n_elems);
	return h;
}

// Compares limbs [0, l) of two limb vectors of stride sl
static void assert_limbs_equal(const int64_t* exp, const int64_t* obs, uint64_t l, int64_t sl, const char* what)
{
	for (uint64_t i = 0; i < l; ++i)
		for (uint64_t j = 0; j < NN; ++j)
			cr_assert_eq(obs[i * sl + j], exp[i * sl + j], "%s: limb %lu coef %lu: expected %ld got %ld", what,
			             (unsigned long)i, (unsigned long)j, (long)exp[i * sl + j], (long)obs[i * sl + j]);
}

typedef int (*BinOp)(const PvdaBackend*, PolyBiv*, const PolyBiv*, const PolyBiv*);

// res (5 limbs, stride 2N view) = a (3 limbs) op b (4 limbs, stride 3N view)
static void check_binary(BinOp op, const char* name)
{
	const uint64_t res_l = 5, a_l = 3, b_l = 4;
	const int64_t res_sl = 2 * NN, a_sl = NN, b_sl = 3 * NN;
	int64_t* res0  = random_vec(res_l * res_sl, 8);
	int64_t* a     = random_vec(a_l * a_sl, 20);
	int64_t* b     = random_vec(b_l * b_sl, 20);
	int64_t* d_res = pvda_gpu_upload(res0, res_l * res_sl);
	int64_t* d_a   = pvda_gpu_upload(a, a_l * a_sl);
	int64_t* d_b   = pvda_gpu_upload(b, b_l * b_sl);

	PolyBiv h_res = new_biv_view(NN, res_l, res_sl, res0), h_a = new_biv_view(NN, a_l, a_sl, a),
	        h_b   = new_biv_view(NN, b_l, b_sl, b);
	PolyBiv g_res = new_biv_view(NN, res_l, res_sl, d_res), g_a = new_biv_view(NN, a_l, a_sl, d_a),
	        g_b = new_biv_view(NN, b_l, b_sl, d_b);
	cr_assert_eq(op(cpu, &h_res, &h_a, &h_b), 0);
	cr_assert_eq(op(gpu, &g_res, &g_a, &g_b), 0);

	int64_t* obs = download(d_res, res_l * res_sl);
	assert_limbs_equal(res0, obs, res_l, res_sl, name);

	free(obs), free(res0), free(a), free(b);
	pvda_gpu_free(d_res), pvda_gpu_free(d_a), pvda_gpu_free(d_b);
}

Test(gpu_vtable, add_sub_strided_sized)
{
	check_binary(pvda_vec_znx_add, "add");
	check_binary(pvda_vec_znx_sub, "sub");
}

Test(gpu_vtable, negate_normalize_automorphism_rotate)
{
	const uint64_t res_l = 6, a_l = 4;
	const int64_t res_sl = 2 * NN, a_sl = NN;
	int64_t* a     = random_vec(a_l * a_sl, 30);
	int64_t* d_a   = pvda_gpu_upload(a, a_l * a_sl);
	int64_t* h_res = calloc(res_l * res_sl, sizeof(int64_t));
	int64_t* d_res = pvda_gpu_alloc(res_l * res_sl);
	pvda_gpu_zero(d_res, res_l * res_sl);
	PolyBiv h_a = new_biv_view(NN, a_l, a_sl, a), g_a = new_biv_view(NN, a_l, a_sl, d_a);
	PolyBiv h_r = new_biv_view(NN, res_l, res_sl, h_res), g_r = new_biv_view(NN, res_l, res_sl, d_res);

	cr_assert_eq(pvda_vec_znx_negate(cpu, &h_r, &h_a), 0);
	cr_assert_eq(pvda_vec_znx_negate(gpu, &g_r, &g_a), 0);
	int64_t* obs = download(d_res, res_l * res_sl);
	assert_limbs_equal(h_res, obs, res_l, res_sl, "negate");
	free(obs);

	cr_assert_eq(pvda_vec_znx_normalize_base2k(cpu, 18, &h_r, &h_a), 0);
	cr_assert_eq(pvda_vec_znx_normalize_base2k(gpu, 18, &g_r, &g_a), 0);
	obs = download(d_res, res_l * res_sl);
	assert_limbs_equal(h_res, obs, res_l, res_sl, "normalize");
	free(obs);

	cr_assert_eq(pvda_vec_znx_automorphism(cpu, 5, &h_r, &h_a), 0);
	cr_assert_eq(pvda_vec_znx_automorphism(gpu, 5, &g_r, &g_a), 0);
	obs = download(d_res, res_l * res_sl);
	assert_limbs_equal(h_res, obs, res_l, res_sl, "automorphism");
	free(obs);

	cr_assert_eq(pvda_vec_znx_rotate(cpu, -37, &h_r, &h_a), 0);
	cr_assert_eq(pvda_vec_znx_rotate(gpu, -37, &g_r, &g_a), 0);
	obs = download(d_res, res_l * res_sl);
	assert_limbs_equal(h_res, obs, res_l, res_sl, "rotate");
	free(obs);

	free(a), free(h_res);
	pvda_gpu_free(d_a), pvda_gpu_free(d_res);
}

Test(gpu_vtable, dft_idft_round_trip)
{
	const uint64_t a_l = 3, res_l = 4;
	const int64_t a_sl = 2 * NN;
	int64_t* a         = random_vec(a_l * a_sl, 30);
	int64_t* d_a       = pvda_gpu_upload(a, a_l * a_sl);
	int64_t* d_dft     = pvda_gpu_alloc(res_l * NN);
	int64_t* d_back    = pvda_gpu_alloc(res_l * NN);
	PolyBiv g_a        = new_biv_view(NN, a_l, a_sl, d_a);
	PolyBiv g_back     = new_biv_view(NN, res_l, NN, d_back);

	cr_assert_eq(pvda_vec_znx_dft(gpu, (PolyBivDFT*)d_dft, res_l, &g_a), 0);
	cr_assert_eq(pvda_vec_znx_idft(gpu, &g_back, (PolyBivDFT*)d_dft, res_l), 0);
	int64_t* obs = download(d_back, res_l * NN);
	for (uint64_t i = 0; i < a_l; ++i)
		for (uint64_t j = 0; j < NN; ++j) cr_assert_eq(obs[i * NN + j], a[i * a_sl + j]);
	for (uint64_t j = 0; j < NN; ++j) cr_assert_eq(obs[(res_l - 1) * NN + j], 0, "limb beyond the input not zero");

	free(obs), free(a);
	pvda_gpu_free(d_a), pvda_gpu_free(d_dft), pvda_gpu_free(d_back);
}

// vmp_apply_dft, vmp_apply_dft_to_dft and vmp_prepare_vec + vmp_apply_prepared_to_dft,
// each followed by idft, against spqlios
Test(gpu_vtable, vector_matrix_products)
{
	const uint64_t nrows = 4, ncols = 3, a_l = 3, res_size = 5;
	int64_t* mat   = random_vec(nrows * ncols * NN, 8);
	int64_t* a     = random_vec(a_l * NN, 8);
	int64_t* d_mat = pvda_gpu_upload(mat, nrows * ncols * NN);
	int64_t* d_a   = pvda_gpu_upload(a, a_l * NN);
	PolyBiv h_a = new_biv_view(NN, a_l, NN, a), g_a = new_biv_view(NN, a_l, NN, d_a);

	MatBivDFT* h_pmat = pvda_new_vmp_pmat(cpu, nrows, ncols);
	int64_t* d_pmat   = pvda_gpu_alloc(nrows * ncols * NN);
	cr_assert_eq(pvda_vmp_prepare_contiguous(cpu, h_pmat, mat, nrows, ncols), 0);
	cr_assert_eq(pvda_vmp_prepare_contiguous(gpu, (MatBivDFT*)d_pmat, d_mat, nrows, ncols), 0);

	VecBivDFT* h_dft   = pvda_new_vec_znx_dft(cpu, res_size);
	VecBivDFT* h_a_dft = pvda_new_vec_znx_dft(cpu, nrows);
	int64_t* d_dft     = pvda_gpu_alloc(res_size * NN);
	int64_t* d_a_dft   = pvda_gpu_alloc(nrows * NN);
	int64_t* h_out     = calloc(res_size * NN, sizeof(int64_t));
	int64_t* d_out     = pvda_gpu_alloc(res_size * NN);
	PolyBiv h_o = new_biv_view(NN, res_size, NN, h_out), g_o = new_biv_view(NN, res_size, NN, d_out);

	for (int variant = 0; variant < 3; ++variant)
	{
		if (variant == 0)
		{
			cr_assert_eq(pvda_vmp_apply_dft(cpu, h_dft, res_size, &h_a, h_pmat, nrows, ncols), 0);
			cr_assert_eq(pvda_vmp_apply_dft(gpu, (PolyBivDFT*)d_dft, res_size, &g_a, (MatBivDFT*)d_pmat, nrows, ncols),
			             0);
		}
		else if (variant == 1)
		{
			cr_assert_eq(pvda_vec_znx_dft(cpu, h_a_dft, nrows, &h_a), 0);
			cr_assert_eq(pvda_vec_znx_dft(gpu, (PolyBivDFT*)d_a_dft, nrows, &g_a), 0);
			cr_assert_eq(pvda_vmp_apply_dft_to_dft(cpu, h_dft, res_size, h_a_dft, nrows, h_pmat, nrows, ncols), 0);
			cr_assert_eq(pvda_vmp_apply_dft_to_dft(gpu, (VecBivDFT*)d_dft, res_size, (VecBivDFT*)d_a_dft, nrows,
			                                       (MatBivDFT*)d_pmat, nrows, ncols),
			             0);
		}
		else
		{
			double* h_pvec  = calloc(nrows * NN * 2, sizeof(double));
			int64_t* d_pvec = pvda_gpu_alloc(nrows * NN);
			cr_assert_eq(pvda_vmp_prepare_vec(cpu, h_pvec, nrows, &h_a), 0);
			cr_assert_eq(pvda_vmp_prepare_vec(gpu, (PolyBivPrep*)d_pvec, nrows, &g_a), 0);
			cr_assert_eq(pvda_vmp_apply_prepared_to_dft(cpu, h_dft, res_size, h_pvec, nrows, h_pmat, nrows, ncols), 0);
			cr_assert_eq(pvda_vmp_apply_prepared_to_dft(gpu, (VecBivDFT*)d_dft, res_size, (VecBivDFT*)d_pvec, nrows,
			                                            (MatBivDFT*)d_pmat, nrows, ncols),
			             0);
			free(h_pvec);
			pvda_gpu_free(d_pvec);
		}
		cr_assert_eq(pvda_vec_znx_idft(cpu, &h_o, h_dft, res_size), 0);
		cr_assert_eq(pvda_vec_znx_idft(gpu, &g_o, (PolyBivDFT*)d_dft, res_size), 0);
		int64_t* obs = download(d_out, res_size * NN);
		assert_limbs_equal(h_out, obs, res_size, NN,
		                   variant == 0   ? "vmp_apply_dft"
		                   : variant == 1 ? "vmp_apply_dft_to_dft"
		                                  : "vmp_apply_prepared_to_dft");
		free(obs);
	}

	pvda_delete_vec_znx_dft(cpu, h_dft);
	pvda_delete_vec_znx_dft(cpu, h_a_dft);
	pvda_delete_vmp_pmat(cpu, h_pmat);
	free(h_out), free(mat), free(a);
	pvda_gpu_free(d_mat), pvda_gpu_free(d_a), pvda_gpu_free(d_pmat), pvda_gpu_free(d_dft), pvda_gpu_free(d_a_dft),
	    pvda_gpu_free(d_out);
}

Test(gpu_vtable, scalar_vector_products)
{
	const uint64_t a_l = 3, res_size = 4;
	int64_t* u   = random_vec(NN, 8);
	int64_t* a   = random_vec(a_l * NN, 8);
	int64_t* d_u = pvda_gpu_upload(u, NN);
	int64_t* d_a = pvda_gpu_upload(a, a_l * NN);
	PolyBiv h_a = new_biv_view(NN, a_l, NN, a), g_a = new_biv_view(NN, a_l, NN, d_a);

	PolyUnivDFT* h_ppol = pvda_new_svp_ppol(cpu);
	int64_t* d_ppol     = pvda_gpu_alloc(NN);
	cr_assert_eq(pvda_svp_prepare(cpu, h_ppol, u), 0);
	cr_assert_eq(pvda_svp_prepare(gpu, (PolyUnivDFT*)d_ppol, d_u), 0);

	VecBivDFT* h_dft   = pvda_new_vec_znx_dft(cpu, res_size);
	VecBivDFT* h_a_dft = pvda_new_vec_znx_dft(cpu, a_l);
	int64_t* d_dft     = pvda_gpu_alloc(res_size * NN);
	int64_t* d_a_dft   = pvda_gpu_alloc(a_l * NN);
	int64_t* h_out     = calloc(res_size * NN, sizeof(int64_t));
	int64_t* d_out     = pvda_gpu_alloc(res_size * NN);
	PolyBiv h_o = new_biv_view(NN, res_size, NN, h_out), g_o = new_biv_view(NN, res_size, NN, d_out);

	for (int variant = 0; variant < 2; ++variant)
	{
		if (variant == 0)
		{
			cr_assert_eq(pvda_svp_apply_dft(cpu, h_dft, res_size, h_ppol, &h_a), 0);
			cr_assert_eq(pvda_svp_apply_dft(gpu, (PolyBivDFT*)d_dft, res_size, (PolyUnivDFT*)d_ppol, &g_a), 0);
		}
		else
		{
			cr_assert_eq(pvda_vec_znx_dft(cpu, h_a_dft, a_l, &h_a), 0);
			cr_assert_eq(pvda_vec_znx_dft(gpu, (PolyBivDFT*)d_a_dft, a_l, &g_a), 0);
			cr_assert_eq(pvda_svp_apply_dft_to_dft(cpu, h_dft, res_size, h_ppol, h_a_dft, a_l), 0);
			cr_assert_eq(pvda_svp_apply_dft_to_dft(gpu, (PolyBivDFT*)d_dft, res_size, (PolyUnivDFT*)d_ppol,
			                                       (PolyBivDFT*)d_a_dft, a_l),
			             0);
		}
		cr_assert_eq(pvda_vec_znx_idft(cpu, &h_o, h_dft, res_size), 0);
		cr_assert_eq(pvda_vec_znx_idft(gpu, &g_o, (PolyBivDFT*)d_dft, res_size), 0);
		int64_t* obs = download(d_out, res_size * NN);
		assert_limbs_equal(h_out, obs, res_size, NN, variant == 0 ? "svp_apply_dft" : "svp_apply_dft_to_dft");
		free(obs);
	}

	pvda_delete_vec_znx_dft(cpu, h_dft);
	pvda_delete_vec_znx_dft(cpu, h_a_dft);
	pvda_delete_svp_ppol(cpu, h_ppol);
	free(h_out), free(u), free(a);
	pvda_gpu_free(d_u), pvda_gpu_free(d_a), pvda_gpu_free(d_ppol), pvda_gpu_free(d_dft), pvda_gpu_free(d_a_dft),
	    pvda_gpu_free(d_out);
}

// Host operands on the CUDA backend run on its embedded spqlios backend; mixed ones are rejected
Test(gpu_vtable, host_operands_and_mixed_operands)
{
	int64_t* a   = random_vec(2 * NN, 20);
	int64_t* b   = random_vec(2 * NN, 20);
	int64_t* r1  = calloc(2 * NN, sizeof(int64_t));
	int64_t* r2  = calloc(2 * NN, sizeof(int64_t));
	int64_t* d_b = pvda_gpu_upload(b, 2 * NN);
	PolyBiv h_a = new_biv_view(NN, 2, NN, a), h_b = new_biv_view(NN, 2, NN, b), g_b = new_biv_view(NN, 2, NN, d_b);
	PolyBiv h_r1 = new_biv_view(NN, 2, NN, r1), h_r2 = new_biv_view(NN, 2, NN, r2);

	cr_assert_eq(pvda_vec_znx_add(cpu, &h_r1, &h_a, &h_b), 0);
	cr_assert_eq(pvda_vec_znx_add(gpu, &h_r2, &h_a, &h_b), 0);
	assert_limbs_equal(r1, r2, 2, NN, "host operands on the CUDA backend");

	cr_assert_eq(pvda_vec_znx_add(gpu, &h_r2, &h_a, &g_b), -1);

	free(a), free(b), free(r1), free(r2);
	pvda_gpu_free(d_b);
}
