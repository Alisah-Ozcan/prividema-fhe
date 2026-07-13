#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backend/rng.h"
#include "backend/spqlios_alias.h"
#include "core/ggsw/ggsw_arithmetic.h"
#include "core/ggsw/ggsw_ciphertext.h"
#include "core/ggsw/ggsw_key.h"
#include "core/ggsw/ggsw_params.h"
#include "core/ggsw/glwegadget_arithmetic.h"
#include "core/ggsw/glwegadget_ciphertext.h"
#include "core/ggsw/glwegadget_key.h"
#include "core/glwe/bivariate_polynomial.h"
#include "core/glwe/glwe_arithmetic.h"
#include "core/glwe/glwe_ciphertext.h"
#include "core/glwe/glwe_key.h"
#include "core/glwe/glwe_params.h"
#include "core/glwe/glwe_transform_key.h"
#include "core/glwe/univariate_polynomial.h"
#include "ggsw_utils.h"
#include "glwegadget_utils.h"
#include "gpu/host/ggsw_external_product_gpu.h"
#include "maths_structures.h"
#include "test_utils.h"
#include "utils.h"

// glwe_trace_expand, isolated from the PIR example: exercises the SAME
// recursive doubling loop the PIR example uses (up to a full-ring-degree
// bundle, matching MATRIX_ROWS*L_TILDE_Q1 scale there), but as a small,
// fast, directly-decrypted unit test — mirrors
// tests/unit/core/ggsw/test_trace_expand.c's `trace_expand, no_noise` CPU
// test almost line for line, swapped to an all-device glwe_ct/results[]/KSK
// operand set (KSKs GPU-NTT-prepared the same way onionpir_client_phase0
// does, via pvda_new_glwegadget_prep_device + prepare_automorphism_key's
// device dispatch).

#define NN    1024
#define KAPPA 4
#define LIMBS 16

Test(gpu_trace_expand, matches_cpu_no_noise)
{
	srand(23);

	MODULE* module = pvda_new_module_info(NN);
	cr_assert_not_null(module, "MODULE allocation failed");

	GLWEParams* params_glwe             = new_glwe_params(NN, 1, KAPPA, LIMBS, 0, NOISE_UNIFORM_POWER_OF_TWO);
	params_glwe->fast_uniform_nb_bits   = 0;
	GLWEGadgetParams* params_glwegadget = new_glwegadget_params(params_glwe, KAPPA, LIMBS);

	GLWESecretKey* sk              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params_glwe);
	uniform_glwe_secret_key(module, sk, 1);
	glwe_sk_prepare(module, sk_prep, sk);

	// GPU-prepared automorphism KSK collection — same pattern as
	// onionpir_client_phase0's make_automorphism_ksk_device fix.
	GLWEAutomorphismKSKCollection* ksks = new_automorphism_ksk_collection(2 * NN);
	for (uint64_t i = 1; (1ULL << i) <= NN; ++i)
	{
		int64_t p                = (int64_t)NN / (1LL << (i - 1)) + 1;
		GLWEAutomorphismKSK* ksk = new_automorphism_ksk(params_glwegadget);
		delete_glwegadget_prep(ksk->enc_s[0]);
		ksk->enc_s[0] = pvda_new_glwegadget_prep_device(params_glwegadget);
		int pc        = prepare_automorphism_key(module, ksk, sk_prep, (int)p);
		cr_assert_eq(pc, 0, "prepare_automorphism_key (GPU) failed for p=%lld", (long long)p);
		cr_assert(pvda_is_device_pointer(ksk->enc_s[0]->mat), "expected enc_s[0]->mat device-resident for p=%lld",
		          (long long)p);
		glwegadget_ksk_collection_put_key(ksks, ksk, p);
	}

	PolyUnivRnX* m_univ_rnx     = new_univ_rnx(params_glwe);
	PolyUnivRnX* tmp_rnx        = new_univ_rnx(params_glwe);
	PolyUnivRnX* m_observed_rnx = new_univ_rnx(params_glwe);
	PolyBiv* biv_tmp            = new_biv(params_glwe);
	GLWECiphertext* glwe_ct     = new_glwe(params_glwe);

	// Same bundle sizes as the CPU reference test, plus NN itself — the
	// full-ring-degree case, matching the PIR example's
	// MATRIX_ROWS*L_TILDE_Q1=1024 scale (256*4) that isn't exercised by the
	// smaller automorphism-only tests.
	int bundled[] = {2, 4, 3, 5, 14, NN};

	for (size_t bi = 0; bi < sizeof(bundled) / sizeof(bundled[0]); ++bi)
	{
		int bund = bundled[bi];

		memset(m_univ_rnx, 0, poly_univ_bytes(params_glwe));
		rnx_random_vec(tmp_rnx, params_glwe);
		for (int i = 0; i < bund; ++i) m_univ_rnx[i] = tmp_rnx[i];

		glwe_secret_encrypt_rnx(module, glwe_ct, sk_prep, m_univ_rnx);

		int64_t* d_glwe_ct_vec     = pvda_glwe_to_device(glwe_ct);
		GLWECiphertext glwe_ct_dev = {.params = params_glwe, .vec = (VecBiv*)d_glwe_ct_vec};

		GLWECiphertext** results = calloc(bund, sizeof(GLWECiphertext*));
		for (int i = 0; i < bund; ++i) results[i] = pvda_new_glwe_device(params_glwe);

		int tr = glwe_trace_expand(module, results, bund, &glwe_ct_dev, ksks);
		cr_assert_eq(tr, 0, "glwe_trace_expand (GPU) failed for bund=%d", bund);

		int64_t factor = 1l << (next_pow2_log(bund));
		for (int i = 0; i < bund; ++i)
		{
			GLWECiphertext* result_host = new_glwe(params_glwe);
			pvda_glwe_from_device(result_host, (const int64_t*)results[i]->vec);

			glwe_secret_decrypt(module, biv_tmp, sk_prep, result_host);
			biv_to_univ_rnx(params_glwe, m_observed_rnx, biv_tmp);

			double expected = factor * m_univ_rnx[i];
			double actual   = m_observed_rnx[0];
			cr_assert(lt(dbl, rnx_torus_distance(expected, actual), 0.001), "bund=%d i=%d: expected %f got %f", bund, i,
			          expected, actual);
			for (int p = 1; p < NN; ++p)
				cr_assert(lt(dbl, rnx_torus_distance(0, m_observed_rnx[p]), 0.001),
				          "bund=%d i=%d p=%d: expected ~0 got %f", bund, i, p, m_observed_rnx[p]);

			delete_glwe(result_host);
			delete_glwe(results[i]);
		}
		free(results);
		pvda_gpu_free(d_glwe_ct_vec);
	}

	delete_automorphism_ksk_collection(ksks, 1);

	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);

	delete_biv(biv_tmp);
	delete_univ_rnx(tmp_rnx);
	delete_univ_rnx(m_univ_rnx);
	delete_univ_rnx(m_observed_rnx);
	delete_glwe(glwe_ct);

	pvda_delete_module_info(module);
	delete_glwegadget_params(params_glwegadget);
	delete_glwe_params(params_glwe);
}

// packed_glwegadget_trace_expand, isolated: this is the actual wrapper
// onionpir_server_gpu calls for the row-query path (not glwe_trace_expand
// directly) — it builds results_glwe[] views via glwegadget_extract_bivglwe
// over a caller-supplied GLWEGadgetCiphertext** array. Mirrors
// tests/unit/core/ggsw/test_trace_expand.c's `glwegad2_trace_expand, normal`
// CPU test, with results[]/packed_glwegadget device-resident and GPU-NTT-
// prepared KSKs.
Test(gpu_trace_expand, packed_glwegadget_trace_expand_matches_cpu)
{
	srand(29);

	MODULE* module = pvda_new_module_info(NN);
	cr_assert_not_null(module, "MODULE allocation failed");

	GLWEParams* params_glwe             = new_glwe_params(NN, 1, KAPPA, LIMBS, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEGadgetParams* params_glwegadget = new_glwegadget_params(params_glwe, KAPPA, LIMBS);

	double max_err_length      = 0.001;
	double critical_err_length = 0.001;

	GLWESecretKey* sk              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params_glwe);
	uniform_glwe_secret_key(module, sk, 1);
	glwe_sk_prepare(module, sk_prep, sk);

	GLWEAutomorphismKSKCollection* ksks = new_automorphism_ksk_collection(2 * NN);
	for (uint64_t i = 1; (1ULL << i) <= NN; ++i)
	{
		int64_t p                = (int64_t)NN / (1LL << (i - 1)) + 1;
		GLWEAutomorphismKSK* ksk = new_automorphism_ksk(params_glwegadget);
		delete_glwegadget_prep(ksk->enc_s[0]);
		ksk->enc_s[0] = pvda_new_glwegadget_prep_device(params_glwegadget);
		int pc        = prepare_automorphism_key(module, ksk, sk_prep, (int)p);
		cr_assert_eq(pc, 0, "prepare_automorphism_key (GPU) failed for p=%lld", (long long)p);
		glwegadget_ksk_collection_put_key(ksks, ksk, p);
	}

	PolyUniv* m_univ        = new_univ(params_glwe);
	GLWECiphertext* glwe_ct = new_glwe(params_glwe);

	// Same bundle sizes as the CPU reference test.
	int bundled[] = {2, 4, 3, 5, 14, 1};

	for (size_t bi = 0; bi < sizeof(bundled) / sizeof(bundled[0]); ++bi)
	{
		int bund = bundled[bi];

		memset(m_univ, 0, poly_univ_bytes(params_glwe));
		uniform_random_pol_znx(m_univ, bund, 1);

		glwegadget_packed_secret_encrypt(module, glwe_ct, params_glwegadget, sk_prep, m_univ, bund);

		int64_t* d_glwe_ct_vec     = pvda_glwe_to_device(glwe_ct);
		GLWECiphertext glwe_ct_dev = {.params = params_glwe, .vec = (VecBiv*)d_glwe_ct_vec};

		GLWEGadgetCiphertext** results = calloc(bund, sizeof(GLWEGadgetCiphertext*));
		for (int i = 0; i < bund; ++i) results[i] = pvda_new_glwegadget_device(params_glwegadget);

		int tr = packed_glwegadget_trace_expand(module, results, bund, params_glwegadget->l_tilde, &glwe_ct_dev, ksks);
		cr_assert_eq(tr, 0, "packed_glwegadget_trace_expand (GPU) failed for bund=%d", bund);

		size_t mat_elems     = glwegadget_coef_number(params_glwegadget);
		PolyUniv* expected_b = new_univ(params_glwe);
		for (int b = 0; b < bund; ++b)
		{
			GLWEGadgetCiphertext* result_host = new_glwegadget(params_glwegadget);
			pvda_gpu_download((int64_t*)result_host->mat, (const int64_t*)results[b]->mat, mat_elems);

			memset(expected_b, 0, poly_univ_bytes(params_glwe));
			expected_b[0] = m_univ[b];
			check_glwegadget(module, result_host, sk_prep, expected_b, max_err_length, critical_err_length);

			delete_glwegadget(result_host);
			delete_glwegadget(results[b]);
		}
		delete_univ(expected_b);
		free(results);
		pvda_gpu_free(d_glwe_ct_vec);
	}

	delete_automorphism_ksk_collection(ksks, 1);
	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_univ(m_univ);
	delete_glwe(glwe_ct);

	pvda_delete_module_info(module);
	delete_glwegadget_params(params_glwegadget);
	delete_glwe_params(params_glwe);
}

// packed_glwegadget_trace_expand_ggsw, isolated: this is what
// onionpir_server_gpu calls for the COLUMN-query path — glwe_trace_expand
// fills the k'th row of each output GGSW, then ggsw_external_product +
// normalize_glwe (both already GPU-dispatched) fill the rest using the
// GGSW(-s_i) relinearization KSKs. Mirrors test_trace_expand.c's
// `ggsw_trace_expand, no_noise` CPU test, with results[]/packed_glwegadget
// device-resident and BOTH KSK kinds GPU-NTT-prepared.
Test(gpu_trace_expand, packed_glwegadget_trace_expand_ggsw_matches_cpu)
{
	srand(31);

	MODULE* module = pvda_new_module_info(NN);
	cr_assert_not_null(module, "MODULE allocation failed");

	GLWEParams* params_glwe = new_glwe_params(NN, 1, KAPPA, LIMBS, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GGSWParams* params_ggsw = new_ggsw_params(params_glwe, 1, KAPPA, LIMBS);
	// l_tilde here MUST equal ggsw_params_l_tilde_a(params_ggsw) — glwe_ct
	// below is packed at this precision, and packed_glwegadget_trace_expand_ggsw
	// asserts results[0]->params (== params_ggsw) agrees with the l_tilde it's
	// called with (see onionpir_server_gpu's matching
	// ggsw_params_l_tilde_a(ggsw_ksk_params) usage, and
	// INIT_PVDA_PARAMS_GGSWGAD's identical (nb_limbs_tilde+1)/(k+1) formula).
	GLWEGadgetParams* params_glwegadget = new_glwegadget_params(params_glwe, KAPPA, ggsw_params_l_tilde_a(params_ggsw));

	double max_err_length      = 0.001;
	double critical_err_length = 0.001;

	uint64_t k = params_glwe->k;

	GLWESecretKey* sk              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params_glwe);
	uniform_glwe_secret_key(module, sk, 1);
	glwe_sk_prepare(module, sk_prep, sk);

	GLWEAutomorphismKSKCollection* ksks = new_automorphism_ksk_collection(2 * NN);
	for (uint64_t i = 1; (1ULL << i) <= NN; ++i)
	{
		int64_t p                = (int64_t)NN / (1LL << (i - 1)) + 1;
		GLWEAutomorphismKSK* ksk = new_automorphism_ksk(params_glwegadget);
		delete_glwegadget_prep(ksk->enc_s[0]);
		ksk->enc_s[0] = pvda_new_glwegadget_prep_device(params_glwegadget);
		int pc        = prepare_automorphism_key(module, ksk, sk_prep, (int)p);
		cr_assert_eq(pc, 0, "prepare_automorphism_key (GPU) failed for p=%lld", (long long)p);
		glwegadget_ksk_collection_put_key(ksks, ksk, p);
	}

	GGSWCiphertextPrep** ggsw_ksks = calloc(k, sizeof(GGSWCiphertextPrep*));
	for (uint64_t i = 0; i < k; ++i)
	{
		ggsw_ksks[i] = pvda_new_ggsw_prep_device(params_ggsw);
		int ggsw_pc  = generate_glwegad_to_ggsw_ksk(module, &ggsw_ksks[i], params_ggsw, sk_prep);
		cr_assert_eq(ggsw_pc, 0, "generate_glwegad_to_ggsw_ksk (GPU) failed for i=%llu", (unsigned long long)i);
		cr_assert(pvda_is_device_pointer(ggsw_ksks[i]->mat), "expected ggsw_ksks[%llu]->mat device-resident",
		          (unsigned long long)i);
	}

	PolyUniv* m_univ        = new_univ(params_glwe);
	GLWECiphertext* glwe_ct = new_glwe(params_glwe);

	int bundled[] = {2, 4, 3, 5, 14, 1};

	for (size_t bi = 0; bi < sizeof(bundled) / sizeof(bundled[0]); ++bi)
	{
		int bund = bundled[bi];

		memset(m_univ, 0, poly_univ_bytes(params_glwe));
		uniform_random_pol_znx(m_univ, bund, 1);

		glwegadget_packed_secret_encrypt(module, glwe_ct, params_glwegadget, sk_prep, m_univ, bund);

		int64_t* d_glwe_ct_vec     = pvda_glwe_to_device(glwe_ct);
		GLWECiphertext glwe_ct_dev = {.params = params_glwe, .vec = (VecBiv*)d_glwe_ct_vec};

		GGSWCiphertext** results = calloc(bund, sizeof(GGSWCiphertext*));
		for (int i = 0; i < bund; ++i) results[i] = pvda_new_ggsw_device(params_ggsw);

		int tr = packed_glwegadget_trace_expand_ggsw(module, results, bund, ggsw_params_l_tilde_a(params_ggsw),
		                                             &glwe_ct_dev, ksks, (const GGSWCiphertextPrep**)ggsw_ksks);
		cr_assert_eq(tr, 0, "packed_glwegadget_trace_expand_ggsw (GPU) failed for bund=%d", bund);

		size_t mat_elems     = ggsw_coef_number(params_ggsw);
		PolyUniv* expected_b = new_univ(params_glwe);
		for (int b = 0; b < bund; ++b)
		{
			GGSWCiphertext* result_host = new_ggsw(params_ggsw);
			pvda_gpu_download((int64_t*)result_host->mat, (const int64_t*)results[b]->mat, mat_elems);

			memset(expected_b, 0, poly_univ_bytes(params_glwe));
			expected_b[0] = m_univ[b];
			check_ggsw(module, result_host, sk_prep, expected_b, max_err_length, critical_err_length);

			delete_ggsw(result_host);
			delete_ggsw(results[b]);
		}
		delete_univ(expected_b);
		free(results);
		pvda_gpu_free(d_glwe_ct_vec);
	}

	delete_automorphism_ksk_collection(ksks, 1);
	for (uint64_t i = 0; i < k; ++i) delete_ggsw_prep(ggsw_ksks[i]);
	free(ggsw_ksks);

	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_univ(m_univ);
	delete_glwe(glwe_ct);

	pvda_delete_module_info(module);
	delete_ggsw_params(params_ggsw);
	delete_glwegadget_params(params_glwegadget);
	delete_glwe_params(params_glwe);
}

// Reproduces the REAL onionpir_server_gpu row-query path's asymmetric params:
// the packed input + automorphism KSKs live at one (16-limb, l_tilde=8)
// precision (params_in/auto_ksk_params), while the trace-expand *results*
// live at a DIFFERENT, smaller (12-limb, l_tilde=4) precision
// (params_out/row_exp_gad_params) — exactly onionpir_server_gpu's
// params_glwe_autokey vs row_exp_params split. None of the other tests in
// this file exercise this: they all reuse one shared params_glwegadget for
// input, KSKs, and results alike.
Test(gpu_trace_expand, packed_glwegadget_trace_expand_mismatched_params_matches_cpu)
{
	srand(37);

	MODULE* module = pvda_new_module_info(NN);
	cr_assert_not_null(module, "MODULE allocation failed");

	GLWEParams* params_in  = new_glwe_params(NN, 1, KAPPA, 16, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEParams* params_out = new_glwe_params(NN, 1, KAPPA, 12, 0, NOISE_UNIFORM_POWER_OF_TWO);

	GLWEGadgetParams* row_query_gad_params = new_glwegadget_params(params_in, KAPPA, 4);
	GLWEGadgetParams* auto_ksk_params      = new_glwegadget_params(params_in, KAPPA, 8);
	GLWEGadgetParams* row_exp_gad_params   = new_glwegadget_params(params_out, KAPPA, 4);

	GLWESecretKey* sk              = alloc_glwe_secret_key(params_in);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params_in);
	uniform_glwe_secret_key(module, sk, 1);
	glwe_sk_prepare(module, sk_prep, sk);

	GLWEAutomorphismKSKCollection* ksks = new_automorphism_ksk_collection(2 * NN);
	for (uint64_t i = 1; (1ULL << i) <= NN; ++i)
	{
		int64_t p                = (int64_t)NN / (1LL << (i - 1)) + 1;
		GLWEAutomorphismKSK* ksk = new_automorphism_ksk(auto_ksk_params);
		delete_glwegadget_prep(ksk->enc_s[0]);
		ksk->enc_s[0] = pvda_new_glwegadget_prep_device(auto_ksk_params);
		int pc        = prepare_automorphism_key(module, ksk, sk_prep, (int)p);
		cr_assert_eq(pc, 0, "prepare_automorphism_key (GPU) failed for p=%lld", (long long)p);
		glwegadget_ksk_collection_put_key(ksks, ksk, p);
	}

	int bund = 16;

	PolyUniv* sel_row      = new_univ(params_in);
	GLWECiphertext* row_ct = new_glwe(params_in);
	memset(sel_row, 0, poly_univ_bytes(params_in));
	int selected      = 5;
	sel_row[selected] = 1;

	glwegadget_packed_secret_encrypt(module, row_ct, row_query_gad_params, sk_prep, sel_row, bund);

	int64_t* d_row_ct_vec     = pvda_glwe_to_device(row_ct);
	GLWECiphertext row_ct_dev = {.params = params_in, .vec = (VecBiv*)d_row_ct_vec};

	GLWEGadgetCiphertext** results = calloc(bund, sizeof(GLWEGadgetCiphertext*));
	for (int i = 0; i < bund; ++i) results[i] = pvda_new_glwegadget_device(row_exp_gad_params);

	int tr = packed_glwegadget_trace_expand(module, results, bund, row_exp_gad_params->l_tilde, &row_ct_dev, ksks);
	cr_assert_eq(tr, 0, "packed_glwegadget_trace_expand (GPU) failed");

	// check_glwegadget compares against the *gadget-decomposition-scaled*
	// expected value (ldexp(expected[p], -kappa_tilde*prec_lvl) per
	// check_glwegadget's own formula), with a tolerance — a bare
	// cr_assert_eq against a hand-derived torus constant is the wrong
	// methodology here: even a mathematically-correct half-product/
	// automorphism chain has a nonzero gadget-decomposition noise floor at
	// this test's deliberately-small KAPPA=4 (see info_bits_half_prod).
	double max_err_length      = ldexp(1.0, -info_bits_half_prod(params_in, auto_ksk_params) + 2);
	double critical_err_length = max_err_length;
	PolyUniv* expected_r       = new_univ(params_out);
	size_t mat_elems           = glwegadget_coef_number(row_exp_gad_params);

	for (int r = 0; r < bund; ++r)
	{
		GLWEGadgetCiphertext* result_host = new_glwegadget(row_exp_gad_params);
		pvda_gpu_download((int64_t*)result_host->mat, (const int64_t*)results[r]->mat, mat_elems);

		memset(expected_r, 0, poly_univ_bytes(params_out));
		expected_r[0] = (r == selected) ? 1 : 0;
		check_glwegadget(module, result_host, sk_prep, expected_r, max_err_length, critical_err_length);

		delete_glwegadget(result_host);
		delete_glwegadget(results[r]);
	}
	free(results);
	delete_univ(expected_r);

	pvda_gpu_free(d_row_ct_vec);
	delete_glwe(row_ct);
	delete_univ(sel_row);

	delete_automorphism_ksk_collection(ksks, 1);
	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);

	pvda_delete_module_info(module);
	delete_glwegadget_params(row_query_gad_params);
	delete_glwegadget_params(auto_ksk_params);
	delete_glwegadget_params(row_exp_gad_params);
	delete_glwe_params(params_in);
	delete_glwe_params(params_out);
}
