#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "backend/backend.h"
#include "backend/backend_arithmetic.h"
#include "gpu/host/gpu_memory_host.h"
#include "impls/cuda_alias.h"

// The CUDA backend's virtual table: every operation must give exactly the spqlios result on the
// same operands, the coefficient vectors being in host memory (the library's, uploaded for the
// operation) or in GPU memory, including operands of different limb counts, strided views (whose
// gaps must be left untouched) and in-place operations. DFT-domain operations are compared after
// the inverse transform (the GPU DFT domain is an NTT domain); inputs are small enough for the CPU
// FFT results to be exact.

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

// A copy of the n_elems values of h, in GPU memory if on_gpu, else in host memory
static int64_t* place(const int64_t* h, size_t n_elems, int on_gpu)
{
	if (on_gpu) return pvda_gpu_upload(h, n_elems);
	int64_t* c = malloc(n_elems * sizeof(int64_t));
	memcpy(c, h, n_elems * sizeof(int64_t));
	return c;
}

// The n_elems values of v (from place) in host memory
static int64_t* fetch(const int64_t* v, size_t n_elems, int on_gpu)
{
	int64_t* h = malloc(n_elems * sizeof(int64_t));
	if (on_gpu)
		pvda_gpu_download(h, v, n_elems);
	else
		memcpy(h, v, n_elems * sizeof(int64_t));
	return h;
}

static void release(int64_t* v, int on_gpu)
{
	if (on_gpu)
		pvda_gpu_free(v);
	else
		free(v);
}

static const char* where(int on_gpu) { return on_gpu ? "GPU operands" : "host operands"; }

// Compares n_elems values (the limbs and the gaps between them)
static void assert_equal(const int64_t* exp, const int64_t* obs, size_t n_elems, const char* what, int on_gpu)
{
	for (size_t i = 0; i < n_elems; ++i)
		cr_assert_eq(obs[i], exp[i], "%s (%s): element %lu (limb %lu coef %lu): expected %ld got %ld", what,
		             where(on_gpu), (unsigned long)i, (unsigned long)(i / NN), (unsigned long)(i % NN), (long)exp[i],
		             (long)obs[i]);
}

typedef int (*BinOp)(const PvdaBackend*, ZnxVec*, const ZnxVec*, const ZnxVec*);

// res (5 limbs, stride 2N view) = a (3 limbs) op b (4 limbs, stride 3N view)
static void check_binary(BinOp op, const char* name, int on_gpu)
{
	const uint64_t res_l = 5, a_l = 3, b_l = 4;
	const int64_t res_sl = 2 * NN, a_sl = NN, b_sl = 3 * NN;
	int64_t* res0  = random_vec(res_l * res_sl, 8);
	int64_t* a     = random_vec(a_l * a_sl, 20);
	int64_t* b     = random_vec(b_l * b_sl, 20);
	int64_t* g_res = place(res0, res_l * res_sl, on_gpu);
	int64_t* g_a   = place(a, a_l * a_sl, on_gpu);
	int64_t* g_b   = place(b, b_l * b_sl, on_gpu);

	ZnxVec h_r = znx_vec_view(NN, res_l, res_sl, res0), h_a = znx_vec_view(NN, a_l, a_sl, a),
	       h_b = znx_vec_view(NN, b_l, b_sl, b);
	ZnxVec g_r = znx_vec_view(NN, res_l, res_sl, g_res), g_av = znx_vec_view(NN, a_l, a_sl, g_a),
	       g_bv = znx_vec_view(NN, b_l, b_sl, g_b);
	cr_assert_eq(op(cpu, &h_r, &h_a, &h_b), 0);
	cr_assert_eq(op(gpu, &g_r, &g_av, &g_bv), 0);

	int64_t* obs = fetch(g_res, res_l * res_sl, on_gpu);
	assert_equal(res0, obs, res_l * res_sl, name, on_gpu);

	free(obs), free(res0), free(a), free(b);
	release(g_res, on_gpu), release(g_a, on_gpu), release(g_b, on_gpu);
}

Test(gpu_vtable, add_sub_strided_sized)
{
	for (int on_gpu = 0; on_gpu <= 1; ++on_gpu)
	{
		check_binary(pvda_vec_znx_add, "add", on_gpu);
		check_binary(pvda_vec_znx_sub, "sub", on_gpu);
	}
}

static const char* unary_names[] = {"negate", "normalize", "automorphism", "rotate"};

static int unary(const PvdaBackend* module, int op, ZnxVec* res, const ZnxVec* a)
{
	switch (op)
	{
		case 0:
			return pvda_vec_znx_negate(module, res, a);
		case 1:
			return pvda_vec_znx_normalize_base2k(module, 18, res, a);
		case 2:
			return pvda_vec_znx_automorphism(module, 5, res, a);
		default:
			return pvda_vec_znx_rotate(module, -37, res, a);
	}
}

// res (6 limbs, stride 2N view) = op(a) (4 limbs)
Test(gpu_vtable, negate_normalize_automorphism_rotate)
{
	const uint64_t res_l = 6, a_l = 4;
	const int64_t res_sl = 2 * NN, a_sl = NN;
	for (int on_gpu = 0; on_gpu <= 1; ++on_gpu)
		for (int op = 0; op < 4; ++op)
		{
			int64_t* res0  = random_vec(res_l * res_sl, 8);
			int64_t* a     = random_vec(a_l * a_sl, 30);
			int64_t* g_res = place(res0, res_l * res_sl, on_gpu);
			int64_t* g_a   = place(a, a_l * a_sl, on_gpu);
			ZnxVec h_r = znx_vec_view(NN, res_l, res_sl, res0), h_a = znx_vec_view(NN, a_l, a_sl, a);
			ZnxVec g_r = znx_vec_view(NN, res_l, res_sl, g_res), g_av = znx_vec_view(NN, a_l, a_sl, g_a);

			cr_assert_eq(unary(cpu, op, &h_r, &h_a), 0);
			cr_assert_eq(unary(gpu, op, &g_r, &g_av), 0);
			int64_t* obs = fetch(g_res, res_l * res_sl, on_gpu);
			assert_equal(res0, obs, res_l * res_sl, unary_names[op], on_gpu);

			free(obs), free(res0), free(a);
			release(g_res, on_gpu), release(g_a, on_gpu);
		}
}

// v = op(v), v of 4 limbs
Test(gpu_vtable, in_place)
{
	const uint64_t l = 4;
	for (int on_gpu = 0; on_gpu <= 1; ++on_gpu)
	{
		for (int op = 0; op < 4; ++op)
		{
			int64_t* v   = random_vec(l * NN, 30);
			int64_t* g_v = place(v, l * NN, on_gpu);
			ZnxVec h = znx_vec_view(NN, l, NN, v), g = znx_vec_view(NN, l, NN, g_v);

			cr_assert_eq(unary(cpu, op, &h, &h), 0);
			cr_assert_eq(unary(gpu, op, &g, &g), 0);
			int64_t* obs = fetch(g_v, l * NN, on_gpu);
			assert_equal(v, obs, l * NN, unary_names[op], on_gpu);

			free(obs), free(v);
			release(g_v, on_gpu);
		}

		int64_t* v   = random_vec(l * NN, 20);
		int64_t* w   = random_vec(l * NN, 20);
		int64_t* g_v = place(v, l * NN, on_gpu);
		int64_t* g_w = place(w, l * NN, on_gpu);
		ZnxVec h_v = znx_vec_view(NN, l, NN, v), h_w = znx_vec_view(NN, l, NN, w);
		ZnxVec g_vv = znx_vec_view(NN, l, NN, g_v), g_wv = znx_vec_view(NN, l, NN, g_w);
		cr_assert_eq(pvda_vec_znx_add(cpu, &h_v, &h_v, &h_w), 0);
		cr_assert_eq(pvda_vec_znx_add(gpu, &g_vv, &g_vv, &g_wv), 0);
		int64_t* obs = fetch(g_v, l * NN, on_gpu);
		assert_equal(v, obs, l * NN, "add in place", on_gpu);

		free(obs), free(v), free(w);
		release(g_v, on_gpu), release(g_w, on_gpu);
	}
}

Test(gpu_vtable, znx_automorphism)
{
	for (int on_gpu = 0; on_gpu <= 1; ++on_gpu)
	{
		int64_t* a     = random_vec(NN, 30);
		int64_t* res   = random_vec(NN, 8);
		int64_t* g_a   = place(a, NN, on_gpu);
		int64_t* g_res = place(res, NN, on_gpu);

		cr_assert_eq(pvda_znx_automorphism(cpu, 7, res, a), 0);
		cr_assert_eq(pvda_znx_automorphism(gpu, 7, g_res, g_a), 0);
		int64_t* obs = fetch(g_res, NN, on_gpu);
		assert_equal(res, obs, NN, "znx_automorphism", on_gpu);

		free(obs), free(a), free(res);
		release(g_a, on_gpu), release(g_res, on_gpu);
	}
}

// a (3 limbs, stride 2N) -> DFT (4 limbs) -> back (4 limbs, stride 2N view): a, then a zero limb
Test(gpu_vtable, dft_idft_round_trip)
{
	const uint64_t a_l = 3, res_l = 4;
	const int64_t a_sl = 2 * NN, back_sl = 2 * NN;
	for (int on_gpu = 0; on_gpu <= 1; ++on_gpu)
	{
		int64_t* a      = random_vec(a_l * a_sl, 30);
		int64_t* back0  = random_vec(res_l * back_sl, 8);
		int64_t* g_a    = place(a, a_l * a_sl, on_gpu);
		int64_t* g_back = place(back0, res_l * back_sl, on_gpu);
		ZnxVec g_av = znx_vec_view(NN, a_l, a_sl, g_a), g_b = znx_vec_view(NN, res_l, back_sl, g_back);

		BE_PolyVecDFT* dft = pvda_new_vec_znx_dft(gpu, res_l);
		cr_assert_not_null(dft);
		cr_assert_eq(pvda_vec_znx_dft(gpu, dft, res_l, &g_av), 0);
		cr_assert_eq(pvda_vec_znx_idft(gpu, &g_b, dft, res_l), 0);

		int64_t* expected = malloc(res_l * back_sl * sizeof(int64_t));
		memcpy(expected, back0, res_l * back_sl * sizeof(int64_t));
		for (uint64_t i = 0; i < res_l; ++i)
			for (uint64_t j = 0; j < NN; ++j) expected[i * back_sl + j] = i < a_l ? a[i * a_sl + j] : 0;
		int64_t* obs = fetch(g_back, res_l * back_sl, on_gpu);
		assert_equal(expected, obs, res_l * back_sl, "dft/idft round trip", on_gpu);

		pvda_delete_vec_znx_dft(gpu, dft);
		free(obs), free(expected), free(a), free(back0);
		release(g_a, on_gpu), release(g_back, on_gpu);
	}
}

// res = idft(dft(a) + dft(b)), with a shorter or longer than b, and res longer than both
Test(gpu_vtable, dft_add)
{
	const uint64_t res_l = 6;
	const uint64_t sizes[2][2] = {{3, 5}, {5, 3}};
	for (int s = 0; s < 2; ++s)
	{
		uint64_t a_l = sizes[s][0], b_l = sizes[s][1];
		int64_t* a     = random_vec(a_l * NN, 20);
		int64_t* b     = random_vec(b_l * NN, 20);
		int64_t* h_out = calloc(res_l * NN, sizeof(int64_t));
		int64_t* g_out = calloc(res_l * NN, sizeof(int64_t));
		ZnxVec v_a = znx_vec_view(NN, a_l, NN, a), v_b = znx_vec_view(NN, b_l, NN, b);
		ZnxVec h_o = znx_vec_view(NN, res_l, NN, h_out), g_o = znx_vec_view(NN, res_l, NN, g_out);

		PvdaBackend* backends[2] = {cpu, gpu};
		ZnxVec* outs[2]          = {&h_o, &g_o};
		for (int k = 0; k < 2; ++k)
		{
			BE_PolyVecDFT* a_dft   = pvda_new_vec_znx_dft(backends[k], a_l);
			BE_PolyVecDFT* b_dft   = pvda_new_vec_znx_dft(backends[k], b_l);
			BE_PolyVecDFT* res_dft = pvda_new_vec_znx_dft(backends[k], res_l);
			cr_assert_eq(pvda_vec_znx_dft(backends[k], a_dft, a_l, &v_a), 0);
			cr_assert_eq(pvda_vec_znx_dft(backends[k], b_dft, b_l, &v_b), 0);
			cr_assert_eq(pvda_vec_znx_dft_add(backends[k], res_dft, res_l, a_dft, a_l, b_dft, b_l), 0);
			cr_assert_eq(pvda_vec_znx_idft(backends[k], outs[k], res_dft, res_l), 0);
			pvda_delete_vec_znx_dft(backends[k], a_dft);
			pvda_delete_vec_znx_dft(backends[k], b_dft);
			pvda_delete_vec_znx_dft(backends[k], res_dft);
		}
		assert_equal(h_out, g_out, res_l * NN, "vec_znx_dft_add", 0);

		free(a), free(b), free(h_out), free(g_out);
	}
}

// vmp_apply_dft, vmp_apply_dft_to_dft and vmp_prepare_vec + vmp_apply_prepared_to_dft,
// each followed by idft, against spqlios
Test(gpu_vtable, vector_matrix_products)
{
	const uint64_t nrows = 4, ncols = 3, a_l = 3, res_size = 5;
	for (int on_gpu = 0; on_gpu <= 1; ++on_gpu)
	{
		int64_t* mat   = random_vec(nrows * ncols * NN, 8);
		int64_t* a     = random_vec(a_l * NN, 8);
		int64_t* g_mat = place(mat, nrows * ncols * NN, on_gpu);
		int64_t* g_a   = place(a, a_l * NN, on_gpu);
		ZnxVec h_a = znx_vec_view(NN, a_l, NN, a), g_av = znx_vec_view(NN, a_l, NN, g_a);

		BE_PolyMatVmpPrep* h_pmat = pvda_new_vmp_pmat(cpu, nrows, ncols);
		BE_PolyMatVmpPrep* g_pmat = pvda_new_vmp_pmat(gpu, nrows, ncols);
		cr_assert_eq(pvda_vmp_prepare_contiguous(cpu, h_pmat, mat, nrows, ncols), 0);
		cr_assert_eq(pvda_vmp_prepare_contiguous(gpu, g_pmat, g_mat, nrows, ncols), 0);

		BE_PolyVecDFT* h_dft     = pvda_new_vec_znx_dft(cpu, res_size);
		BE_PolyVecDFT* h_a_dft   = pvda_new_vec_znx_dft(cpu, nrows);
		BE_PolyVecDFT* g_dft     = pvda_new_vec_znx_dft(gpu, res_size);
		BE_PolyVecDFT* g_a_dft   = pvda_new_vec_znx_dft(gpu, nrows);
		BE_PolyVecVmpPrep* h_pvec = pvda_new_vmp_pvec(cpu, nrows);
		BE_PolyVecVmpPrep* g_pvec = pvda_new_vmp_pvec(gpu, nrows);
		int64_t* h_out            = calloc(res_size * NN, sizeof(int64_t));
		int64_t* zeros            = calloc(res_size * NN, sizeof(int64_t));
		int64_t* g_out            = place(zeros, res_size * NN, on_gpu);
		ZnxVec h_o = znx_vec_view(NN, res_size, NN, h_out), g_o = znx_vec_view(NN, res_size, NN, g_out);

		for (int variant = 0; variant < 3; ++variant)
		{
			if (variant == 0)
			{
				cr_assert_eq(pvda_vmp_apply_dft(cpu, h_dft, res_size, &h_a, h_pmat, nrows, ncols), 0);
				cr_assert_eq(pvda_vmp_apply_dft(gpu, g_dft, res_size, &g_av, g_pmat, nrows, ncols), 0);
			}
			else if (variant == 1)
			{
				cr_assert_eq(pvda_vec_znx_dft(cpu, h_a_dft, nrows, &h_a), 0);
				cr_assert_eq(pvda_vec_znx_dft(gpu, g_a_dft, nrows, &g_av), 0);
				cr_assert_eq(pvda_vmp_apply_dft_to_dft(cpu, h_dft, res_size, h_a_dft, nrows, h_pmat, nrows, ncols),
				             0);
				cr_assert_eq(pvda_vmp_apply_dft_to_dft(gpu, g_dft, res_size, g_a_dft, nrows, g_pmat, nrows, ncols),
				             0);
			}
			else
			{
				cr_assert_eq(pvda_vmp_prepare_vec(cpu, h_pvec, nrows, &h_a), 0);
				cr_assert_eq(pvda_vmp_prepare_vec(gpu, g_pvec, nrows, &g_av), 0);
				cr_assert_eq(
				    pvda_vmp_apply_prepared_to_dft(cpu, h_dft, res_size, h_pvec, nrows, h_pmat, nrows, ncols), 0);
				cr_assert_eq(
				    pvda_vmp_apply_prepared_to_dft(gpu, g_dft, res_size, g_pvec, nrows, g_pmat, nrows, ncols), 0);
			}
			cr_assert_eq(pvda_vec_znx_idft(cpu, &h_o, h_dft, res_size), 0);
			cr_assert_eq(pvda_vec_znx_idft(gpu, &g_o, g_dft, res_size), 0);
			int64_t* obs = fetch(g_out, res_size * NN, on_gpu);
			assert_equal(h_out, obs, res_size * NN,
			             variant == 0   ? "vmp_apply_dft"
			             : variant == 1 ? "vmp_apply_dft_to_dft"
			                            : "vmp_apply_prepared_to_dft",
			             on_gpu);
			free(obs);
		}

		pvda_delete_vec_znx_dft(cpu, h_dft), pvda_delete_vec_znx_dft(cpu, h_a_dft);
		pvda_delete_vec_znx_dft(gpu, g_dft), pvda_delete_vec_znx_dft(gpu, g_a_dft);
		pvda_delete_vmp_pvec(cpu, h_pvec), pvda_delete_vmp_pvec(gpu, g_pvec);
		pvda_delete_vmp_pmat(cpu, h_pmat), pvda_delete_vmp_pmat(gpu, g_pmat);
		free(h_out), free(zeros), free(mat), free(a);
		release(g_mat, on_gpu), release(g_a, on_gpu), release(g_out, on_gpu);
	}
}

Test(gpu_vtable, scalar_vector_products)
{
	const uint64_t a_l = 3, res_size = 4;
	for (int on_gpu = 0; on_gpu <= 1; ++on_gpu)
	{
		int64_t* u   = random_vec(NN, 8);
		int64_t* a   = random_vec(a_l * NN, 8);
		int64_t* g_u = place(u, NN, on_gpu);
		int64_t* g_a = place(a, a_l * NN, on_gpu);
		ZnxVec h_a = znx_vec_view(NN, a_l, NN, a), g_av = znx_vec_view(NN, a_l, NN, g_a);

		BE_PolySvpPrep* h_ppol = pvda_new_svp_ppol(cpu);
		BE_PolySvpPrep* g_ppol = pvda_new_svp_ppol(gpu);
		cr_assert_eq(pvda_svp_prepare(cpu, h_ppol, u), 0);
		cr_assert_eq(pvda_svp_prepare(gpu, g_ppol, g_u), 0);

		BE_PolyVecDFT* h_dft   = pvda_new_vec_znx_dft(cpu, res_size);
		BE_PolyVecDFT* h_a_dft = pvda_new_vec_znx_dft(cpu, a_l);
		BE_PolyVecDFT* g_dft   = pvda_new_vec_znx_dft(gpu, res_size);
		BE_PolyVecDFT* g_a_dft = pvda_new_vec_znx_dft(gpu, a_l);
		int64_t* h_out         = calloc(res_size * NN, sizeof(int64_t));
		int64_t* zeros         = calloc(res_size * NN, sizeof(int64_t));
		int64_t* g_out         = place(zeros, res_size * NN, on_gpu);
		ZnxVec h_o = znx_vec_view(NN, res_size, NN, h_out), g_o = znx_vec_view(NN, res_size, NN, g_out);

		for (int variant = 0; variant < 2; ++variant)
		{
			if (variant == 0)
			{
				cr_assert_eq(pvda_svp_apply_dft(cpu, h_dft, res_size, h_ppol, &h_a), 0);
				cr_assert_eq(pvda_svp_apply_dft(gpu, g_dft, res_size, g_ppol, &g_av), 0);
			}
			else
			{
				cr_assert_eq(pvda_vec_znx_dft(cpu, h_a_dft, a_l, &h_a), 0);
				cr_assert_eq(pvda_vec_znx_dft(gpu, g_a_dft, a_l, &g_av), 0);
				cr_assert_eq(pvda_svp_apply_dft_to_dft(cpu, h_dft, res_size, h_ppol, h_a_dft, a_l), 0);
				cr_assert_eq(pvda_svp_apply_dft_to_dft(gpu, g_dft, res_size, g_ppol, g_a_dft, a_l), 0);
			}
			cr_assert_eq(pvda_vec_znx_idft(cpu, &h_o, h_dft, res_size), 0);
			cr_assert_eq(pvda_vec_znx_idft(gpu, &g_o, g_dft, res_size), 0);
			int64_t* obs = fetch(g_out, res_size * NN, on_gpu);
			assert_equal(h_out, obs, res_size * NN, variant == 0 ? "svp_apply_dft" : "svp_apply_dft_to_dft",
			             on_gpu);
			free(obs);
		}

		pvda_delete_vec_znx_dft(cpu, h_dft), pvda_delete_vec_znx_dft(cpu, h_a_dft);
		pvda_delete_vec_znx_dft(gpu, g_dft), pvda_delete_vec_znx_dft(gpu, g_a_dft);
		pvda_delete_svp_ppol(cpu, h_ppol), pvda_delete_svp_ppol(gpu, g_ppol);
		free(h_out), free(zeros), free(u), free(a);
		release(g_u, on_gpu), release(g_a, on_gpu), release(g_out, on_gpu);
	}
}

// Coefficient vectors split between host and GPU memory are uploaded as needed
Test(gpu_vtable, mixed_operands)
{
	int64_t* a   = random_vec(2 * NN, 20);
	int64_t* b   = random_vec(2 * NN, 20);
	int64_t* r1  = calloc(2 * NN, sizeof(int64_t));
	int64_t* r2  = calloc(2 * NN, sizeof(int64_t));
	int64_t* d_b = pvda_gpu_upload(b, 2 * NN);
	ZnxVec h_a = znx_vec_view(NN, 2, NN, a), h_b = znx_vec_view(NN, 2, NN, b), g_b = znx_vec_view(NN, 2, NN, d_b);
	ZnxVec h_r1 = znx_vec_view(NN, 2, NN, r1), h_r2 = znx_vec_view(NN, 2, NN, r2);

	cr_assert_eq(pvda_vec_znx_add(cpu, &h_r1, &h_a, &h_b), 0);
	cr_assert_eq(pvda_vec_znx_add(gpu, &h_r2, &h_a, &g_b), 0);
	assert_equal(r1, r2, 2 * NN, "add of a host and a GPU vector into a host vector", 0);

	free(a), free(b), free(r1), free(r2);
	pvda_gpu_free(d_b);
}
