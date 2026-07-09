
#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <float.h>
#include <stdlib.h>

#include "bivariate_polynomial.h"
#include "core/ggsw/ggsw_arithmetic.h"
#include "core/glwe/glwe_arithmetic.h"
#include "core/glwe/glwe_ciphertext.h"
#include "core/glwe/glwe_transform_key.h"
#include "ggsw_ciphertext.h"
#include "ggsw_params.h"
#include "glwe_params.h"
#include "rng.h"
#include "test_utils.h"
#include "tfhe.h"
#include "univariate_polynomial.h"

#ifdef ENABLE_CUDA
#include "gpu/common/gpu_nttparameters.h"
#include "gpu/host/ggsw_external_product_gpu.h"  // pvda_glwe_to_device / pvda_ggsw_to_device / pvda_glwe_from_device / pvda_gpu_alloc / pvda_gpu_free
#endif

/** The test is done without error, it is a proof of concept*/
PvdaParamTest(tfhe_cmux_unprepared, without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSW(param);

	//! Variance of the error's normal distributions
	params_glwe->fast_uniform_nb_bits = 0;
	sigma                             = 0;

	GLWESecretKey* sk_ggsw              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_glwe_prep = alloc_glwe_secret_key_prepared(params_glwe);

	GGSWCiphertext* ggsw = new_ggsw(params_ggsw);

	PolyUnivTnX* m1 = new_univ_tnx(params_glwe);
	PolyUnivTnX* m2 = new_univ_tnx(params_glwe);

	GLWECiphertext* glwe1 = new_glwe(params_glwe);
	GLWECiphertext* glwe2 = new_glwe(params_glwe);
	GLWECiphertext* res   = new_glwe(params_glwe);

	PolyBiv* res_biv     = new_biv(params_glwe);
	PolyUnivTnX* res_tnx = new_univ_tnx(params_glwe);

	PolyUniv* m_sel = new_univ(params_glwe);

	uniform_glwe_secret_key(module, sk_ggsw, 3);
	glwe_sk_prepare(module, sk_glwe_prep, sk_ggsw);

	uniform_random_pol_znx(m1, params_glwe->nn, 62);
	uniform_random_pol_znx(m2, params_glwe->nn, 62);

	glwe_secret_encrypt_tnx(module, glwe1, sk_glwe_prep, m1);
	glwe_secret_encrypt_tnx(module, glwe2, sk_glwe_prep, m2);

	//Case 0:
	memset(m_sel, 0, poly_univ_bytes(params_glwe));
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, m_sel);

	tfhe_cmux_unprepared(module, res, glwe1, glwe2, ggsw, 1);

	glwe_secret_decrypt(module, res_biv, sk_glwe_prep, res);

	biv_to_univ_tnx(params_glwe, res_tnx, res_biv);

	int decomp_noise_bits = 15;  //TODO: put a real threshold
	for (int p = 0; p < params_glwe->nn; ++p) assert_tnx_close_enough(res_tnx[p], m1[p], decomp_noise_bits);

	//Case 1:
	//
	//
	m_sel[0] = 1;
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, m_sel);

	tfhe_cmux_unprepared(module, res, glwe1, glwe2, ggsw, 1);

	glwe_secret_decrypt(module, res_biv, sk_glwe_prep, res);

	biv_to_univ_tnx(params_glwe, res_tnx, res_biv);

	for (int p = 0; p < params_glwe->nn; ++p) assert_tnx_close_enough(res_tnx[p], m2[p], decomp_noise_bits);

	delete_univ_tnx(m1);
	delete_univ_tnx(m2);

	delete_glwe(glwe1);
	delete_glwe(glwe2);
	delete_glwe(res);

	delete_ggsw(ggsw);

	delete_biv(res_biv);
	delete_univ_tnx(res_tnx);
	delete_univ(m_sel);

	delete_glwe_secret_key(sk_ggsw);
	delete_glwe_secret_key_prepared(sk_glwe_prep);

	DELETE_PVDA_PARAMS_GGSW;
}

/** The test is done without error, it is a proof of concept*/
PvdaParamTest(tfhe_cmux_prepared, without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSW(param);

	//! Variance of the error's normal distributions
	params_glwe->fast_uniform_nb_bits = 0;
	sigma                             = 0;

	GLWESecretKey* sk_ggsw              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_glwe_prep = alloc_glwe_secret_key_prepared(params_glwe);

	GGSWCiphertext* ggsw          = new_ggsw(params_ggsw);
	GGSWCiphertextPrep* ggsw_prep = new_ggsw_prep(params_ggsw);

	PolyUnivTnX* m1 = new_univ_tnx(params_glwe);
	PolyUnivTnX* m2 = new_univ_tnx(params_glwe);

	GLWECiphertext* glwe1 = new_glwe(params_glwe);
	GLWECiphertext* glwe2 = new_glwe(params_glwe);
	GLWECiphertext* res   = new_glwe(params_glwe);

	PolyBiv* res_biv     = new_biv(params_glwe);
	PolyUnivTnX* res_tnx = new_univ_tnx(params_glwe);

	PolyUniv* m_sel = new_univ(params_glwe);

	//Generate a secret key
	uniform_glwe_secret_key(module, sk_ggsw, 3);
	glwe_sk_prepare(module, sk_glwe_prep, sk_ggsw);

	// Generate random inputs for the Mux gate
	uniform_random_pol_znx(m1, params_glwe->nn, 62);
	uniform_random_pol_znx(m2, params_glwe->nn, 62);

	// Encrypt (to GLWE) the inputs
	glwe_secret_encrypt_tnx(module, glwe1, sk_glwe_prep, m1);
	glwe_secret_encrypt_tnx(module, glwe2, sk_glwe_prep, m2);

	//Case 0:
	memset(m_sel, 0, poly_univ_bytes(params_glwe));
	// Encrypt a 0 into ggsw to have a selection signal of 0
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, m_sel);
	// Prepare the ggsw
	ggsw_prepare(module, ggsw_prep, ggsw);
	// And use it as the selector in a Mux gate.
	// The output should be a ciphertext of m1
	tfhe_cmux(module, res, glwe1, glwe2, ggsw_prep, 1);

	// Check that indeed res is an encryption of m1 with at least the decomp_noise_bits MSB
	// equal to m1
	glwe_secret_decrypt(module, res_biv, sk_glwe_prep, res);
	biv_to_univ_tnx(params_glwe, res_tnx, res_biv);
	int decomp_noise_bits = 15;  //TODO: put a real threshold
	for (int p = 0; p < params_glwe->nn; ++p) assert_tnx_close_enough(res_tnx[p], m1[p], decomp_noise_bits);

	//Case 1:
	//

	// Encrypt a 1 into ggsw to have a selection signal of 2
	m_sel[0] = 1;
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, m_sel);
	// Prepare the ggsw
	ggsw_prepare(module, ggsw_prep, ggsw);

	// And use it as the selector in a Mux gate.
	// The output should now be a ciphertext of m2 due to the selection signal being 1
	tfhe_cmux(module, res, glwe1, glwe2, ggsw_prep, 1);

	// Check that indeed res is an encryption of m1 with at least the decomp_noise_bits MSB
	// equal to m2
	glwe_secret_decrypt(module, res_biv, sk_glwe_prep, res);
	biv_to_univ_tnx(params_glwe, res_tnx, res_biv);
	for (int p = 0; p < params_glwe->nn; ++p) assert_tnx_close_enough(res_tnx[p], m2[p], decomp_noise_bits);

	delete_univ_tnx(m1);
	delete_univ_tnx(m2);

	delete_glwe(glwe1);
	delete_glwe(glwe2);
	delete_glwe(res);

	delete_ggsw(ggsw);
	delete_ggsw_prep(ggsw_prep);

	delete_biv(res_biv);
	delete_univ_tnx(res_tnx);
	delete_univ(m_sel);

	delete_glwe_secret_key(sk_ggsw);
	delete_glwe_secret_key_prepared(sk_glwe_prep);

	DELETE_PVDA_PARAMS_GGSW;
}

/**
 * Tests that the TFHE CMux gate works as intended (select an input using an
 * encrypted selection signal)
 */
PvdaParamTest(tfhe_cmux_prepared, with_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSW(param);

	GLWESecretKey* sk_ggsw              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_glwe_prep = alloc_glwe_secret_key_prepared(params_glwe);

	GGSWCiphertext* ggsw          = new_ggsw(params_ggsw);
	GGSWCiphertextPrep* ggsw_prep = new_ggsw_prep(params_ggsw);

	PolyUnivTnX* m1 = new_univ_tnx(params_glwe);
	PolyUnivTnX* m2 = new_univ_tnx(params_glwe);

	GLWECiphertext* glwe1 = new_glwe(params_glwe);
	GLWECiphertext* glwe2 = new_glwe(params_glwe);
	GLWECiphertext* res   = new_glwe(params_glwe);

	PolyBiv* res_biv     = new_biv(params_glwe);
	PolyUnivTnX* res_tnx = new_univ_tnx(params_glwe);

	PolyUniv* m_sel = new_univ(params_glwe);

	//Generate a secret key
	uniform_glwe_secret_key(module, sk_ggsw, 3);
	glwe_sk_prepare(module, sk_glwe_prep, sk_ggsw);

	// Generate random inputs for the Mux gate
	uniform_random_pol_znx(m1, params_glwe->nn, 62);
	uniform_random_pol_znx(m2, params_glwe->nn, 62);

	// Encrypt (to GLWE) the inputs
	glwe_secret_encrypt_tnx(module, glwe1, sk_glwe_prep, m1);
	glwe_secret_encrypt_tnx(module, glwe2, sk_glwe_prep, m2);

	//Case 0:
	memset(m_sel, 0, poly_univ_bytes(params_glwe));
	// Encrypt a 0 into ggsw to have a selection signal of 0
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, m_sel);
	// Prepare the ggsw
	ggsw_prepare(module, ggsw_prep, ggsw);
	// And use it as the selector in a Mux gate.
	// The output should be a ciphertext of m1
	tfhe_cmux(module, res, glwe1, glwe2, ggsw_prep, 1);

	// Check that indeed res is an encryption of m1 with at least the decomp_noise_bits MSB
	// equal to m1
	glwe_secret_decrypt(module, res_biv, sk_glwe_prep, res);
	biv_to_univ_tnx(params_glwe, res_tnx, res_biv);
	int decomp_noise_bits = 5;  //TODO: put a real threshold
	for (int p = 0; p < params_glwe->nn; ++p) assert_tnx_close_enough(res_tnx[p], m1[p], decomp_noise_bits);

	//Case 1:
	//

	// Encrypt a 1 into ggsw to have a selection signal of 2
	m_sel[0] = 1;
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, m_sel);
	// Prepare the ggsw
	ggsw_prepare(module, ggsw_prep, ggsw);

	// And use it as the selector in a Mux gate.
	// The output should now be a ciphertext of m2 due to the selection signal being 1
	tfhe_cmux(module, res, glwe1, glwe2, ggsw_prep, 1);

	// Check that indeed res is an encryption of m1 with at least the decomp_noise_bits MSB
	// equal to m2
	glwe_secret_decrypt(module, res_biv, sk_glwe_prep, res);
	biv_to_univ_tnx(params_glwe, res_tnx, res_biv);
	for (int p = 0; p < params_glwe->nn; ++p) assert_tnx_close_enough(res_tnx[p], m2[p], decomp_noise_bits);

	delete_univ_tnx(m1);
	delete_univ_tnx(m2);

	delete_glwe(glwe1);
	delete_glwe(glwe2);
	delete_glwe(res);

	delete_ggsw(ggsw);
	delete_ggsw_prep(ggsw_prep);

	delete_biv(res_biv);
	delete_univ_tnx(res_tnx);
	delete_univ(m_sel);

	delete_glwe_secret_key(sk_ggsw);
	delete_glwe_secret_key_prepared(sk_glwe_prep);

	DELETE_PVDA_PARAMS_GGSW;
}

PvdaParamTest(tfhe_cmux_tree, normal, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSW(param);

	GLWESecretKey* sk_ggsw              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_glwe_prep = alloc_glwe_secret_key_prepared(params_glwe);

	GGSWCiphertext* ggsw           = new_ggsw(params_ggsw);
	GGSWCiphertextPrep* ggsw_prep0 = new_ggsw_prep(params_ggsw);
	GGSWCiphertextPrep* ggsw_prep1 = new_ggsw_prep(params_ggsw);

	// How many inputs to put in the tree
	int num_msgs = 64;
	// Different values of selection that we will try
	int selections[] = {0, 1, 2, 3, 5, 7, 8, 13, 17, 31, 32, 33, 34};
	int msgs_log2    = 6;

	// The original unencrypted inputs
	PolyUnivTnX* ms[num_msgs];

	GLWECiphertext* glwes[num_msgs];
	GLWECiphertext* res = new_glwe(params_glwe);

	PolyBiv* res_biv     = new_biv(params_glwe);
	PolyUnivTnX* res_tnx = new_univ_tnx(params_glwe);

	PolyUniv* m_sel = new_univ(params_glwe);

	uniform_glwe_secret_key(module, sk_ggsw, 2);
	glwe_sk_prepare(module, sk_glwe_prep, sk_ggsw);

	//Generate the inputs, in plaintext at ms and encrypted at glwes
	for (int i = 0; i < num_msgs; ++i)
	{
		ms[i] = new_univ_tnx(params_glwe);
		memset(ms[i], 0, poly_univ_tnx_bytes(params_glwe));
		glwes[i] = new_glwe(params_glwe);
		uniform_random_pol_znx(ms[i], params_glwe->nn, 64);
		glwe_secret_encrypt_tnx(module, glwes[i], sk_glwe_prep, ms[i]);
	}

	// Prepare a GGSW for 0 and another for 1
	memset(m_sel, 0, poly_univ_bytes(params_glwe));
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, m_sel);

	ggsw_prepare(module, ggsw_prep0, ggsw);

	m_sel[0] = 1;
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, m_sel);

	ggsw_prepare(module, ggsw_prep1, ggsw);

	GGSWCiphertextPrep* selector[msgs_log2];
	for (int i = 0; i < sizeof(selections) / sizeof(selections[0]); ++i)
	{
		int sel = selections[i];
		// Create a selector set using the GGSW for 0 and 1 that corresponds to value sel
		// Classic binary digit decomposition algorithm, but we choose which GGSW to
		// use
		for (int j = 0; j < msgs_log2; ++j)
		{
			uint64_t v = sel % 2;
			if (v)
			{
				selector[j] = ggsw_prep1;
			}
			else
			{
				selector[j] = ggsw_prep0;
			}
			sel >>= 1;
		}

		// Do not forget to restore sel to the actual value since we destroyed it while decomposing it
		sel = selections[i];

		tfhe_cmux_tree(module, res, (const GLWECiphertext**)glwes, num_msgs, (const GGSWCiphertextPrep**)selector,
		               msgs_log2, 0);
		normalize_glwe(module, res, res);

		glwe_secret_decrypt(module, res_biv, sk_glwe_prep, res);
		biv_to_univ_tnx(params_glwe, res_tnx, res_biv);

		int decomp_noise_bits = 10;
		for (int p = 0; p < params_glwe->nn; ++p) assert_tnx_close_enough(res_tnx[p], ms[sel][p], decomp_noise_bits);
	}

	delete_glwe(res);

	delete_ggsw(ggsw);
	delete_ggsw_prep(ggsw_prep0);
	delete_ggsw_prep(ggsw_prep1);

	delete_biv(res_biv);
	delete_univ_tnx(res_tnx);
	delete_univ(m_sel);

	delete_glwe_secret_key(sk_ggsw);
	delete_glwe_secret_key_prepared(sk_glwe_prep);

	DELETE_PVDA_PARAMS_GGSW;
}

#ifdef ENABLE_CUDA
/**
 * @brief tfhe_cmux has no GPU-specific code of its own: it is built from sub_glwe,
 * normalize_glwe, ggsw_external_product and add_glwe, each of which dispatches to
 * the GPU (via pvda_is_device_pointer) when given device pointers. This test drives
 * the whole CMux gate with device-resident ciphertexts and checks the result matches
 * the CPU path bit-for-bit, for both selection cases (0 -> c0, 1 -> c1).
 */
PvdaParamTest(tfhe_cmux, gpu_matches_cpu, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSW(param);

	params_glwe->fast_uniform_nb_bits = 0;
	sigma                             = 0;

	GLWESecretKey* sk_ggsw              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_glwe_prep = alloc_glwe_secret_key_prepared(params_glwe);

	GGSWCiphertext* ggsw          = new_ggsw(params_ggsw);
	GGSWCiphertextPrep* ggsw_prep = new_ggsw_prep(params_ggsw);

	PolyUnivTnX* m1 = new_univ_tnx(params_glwe);
	PolyUnivTnX* m2 = new_univ_tnx(params_glwe);

	GLWECiphertext* glwe1        = new_glwe(params_glwe);
	GLWECiphertext* glwe2        = new_glwe(params_glwe);
	GLWECiphertext* res_expected = new_glwe(params_glwe);
	GLWECiphertext* res_from_gpu = new_glwe(params_glwe);

	PolyUniv* m_sel = new_univ(params_glwe);

	uniform_glwe_secret_key(module, sk_ggsw, 3);
	glwe_sk_prepare(module, sk_glwe_prep, sk_ggsw);

	uniform_random_pol_znx(m1, params_glwe->nn, 62);
	uniform_random_pol_znx(m2, params_glwe->nn, 62);

	glwe_secret_encrypt_tnx(module, glwe1, sk_glwe_prep, m1);
	glwe_secret_encrypt_tnx(module, glwe2, sk_glwe_prep, m2);

	gpu_ntt_initialize(params_glwe->nn);
	size_t total = glwe_coef_number(params_glwe);

	// Case 0: selection signal 0 -> tfhe_cmux should select glwe1 (c0)
	// Case 1: selection signal 1 -> tfhe_cmux should select glwe2 (c1)
	for (int sel_bit = 0; sel_bit < 2; ++sel_bit)
	{
		memset(m_sel, 0, poly_univ_bytes(params_glwe));
		m_sel[0] = sel_bit;
		ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, m_sel);
		ggsw_prepare(module, ggsw_prep, ggsw);

		// CPU reference (host pointers -> CPU path throughout)
		int ret = tfhe_cmux(module, res_expected, glwe1, glwe2, ggsw_prep, 1);
		cr_assert_eq(ret, 0, "tfhe_cmux (CPU) failed");

		// GPU: upload everything, prepare the GGSW on device, run CMux entirely
		// through device pointers (same tfhe_cmux entry point).
		GLWECiphertext gpu_glwe1 = {.params = params_glwe, .vec = pvda_glwe_to_device(glwe1)};
		GLWECiphertext gpu_glwe2 = {.params = params_glwe, .vec = pvda_glwe_to_device(glwe2)};
		GLWECiphertext gpu_res   = {.params = params_glwe, .vec = pvda_gpu_alloc(total)};

		GGSWCiphertext gpu_ggsw          = {.params = params_ggsw, .mat = pvda_ggsw_to_device(ggsw)};
		GGSWCiphertextPrep gpu_ggsw_prep = {.params = params_ggsw, .mat = NULL};
		ggsw_prepare(module, &gpu_ggsw_prep, &gpu_ggsw);

		cr_assert_not_null(gpu_glwe1.vec, "pvda_glwe_to_device failed");
		cr_assert_not_null(gpu_glwe2.vec, "pvda_glwe_to_device failed");
		cr_assert_not_null(gpu_res.vec, "pvda_gpu_alloc failed");
		cr_assert_not_null(gpu_ggsw.mat, "pvda_ggsw_to_device failed");
		cr_assert_not_null(gpu_ggsw_prep.mat, "ggsw_prepare (GPU) failed");

		ret = tfhe_cmux(module, &gpu_res, &gpu_glwe1, &gpu_glwe2, &gpu_ggsw_prep, 1);
		cr_assert_eq(ret, 0, "tfhe_cmux (GPU) failed");
		pvda_glwe_from_device(res_from_gpu, gpu_res.vec);

		for (uint64_t t = 0; t < total; t++) cr_assert(eq(i64, res_from_gpu->vec[t], res_expected->vec[t]));

		pvda_gpu_free(gpu_glwe1.vec);
		pvda_gpu_free(gpu_glwe2.vec);
		pvda_gpu_free(gpu_res.vec);
		pvda_gpu_free(gpu_ggsw.mat);
		pvda_gpu_free((int64_t*)gpu_ggsw_prep.mat);
	}

	delete_univ_tnx(m1);
	delete_univ_tnx(m2);

	delete_glwe(glwe1);
	delete_glwe(glwe2);
	delete_glwe(res_expected);
	delete_glwe(res_from_gpu);

	delete_ggsw(ggsw);
	delete_ggsw_prep(ggsw_prep);

	delete_univ(m_sel);

	delete_glwe_secret_key(sk_ggsw);
	delete_glwe_secret_key_prepared(sk_glwe_prep);

	DELETE_PVDA_PARAMS_GGSW;
}

/**
 * @brief Same as tfhe_cmux::gpu_matches_cpu but for tfhe_cmux_unprepared, which takes
 * a raw (un-prepared) GGSWCiphertext and dispatches through sub_glwe / normalize_glwe /
 * ggsw_unprepared_external_product / add_glwe. ggsw_unprepared_external_product does its
 * own NTT preparation on every call, so — unlike tfhe_cmux — there is no separate
 * ggsw_prepare / GGSWCiphertextPrep step: the raw uploaded GGSW matrix is used directly.
 */
PvdaParamTest(tfhe_cmux_unprepared, gpu_matches_cpu, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSW(param);

	params_glwe->fast_uniform_nb_bits = 0;
	sigma                             = 0;

	GLWESecretKey* sk_ggsw              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_glwe_prep = alloc_glwe_secret_key_prepared(params_glwe);

	GGSWCiphertext* ggsw = new_ggsw(params_ggsw);

	PolyUnivTnX* m1 = new_univ_tnx(params_glwe);
	PolyUnivTnX* m2 = new_univ_tnx(params_glwe);

	GLWECiphertext* glwe1        = new_glwe(params_glwe);
	GLWECiphertext* glwe2        = new_glwe(params_glwe);
	GLWECiphertext* res_expected = new_glwe(params_glwe);
	GLWECiphertext* res_from_gpu = new_glwe(params_glwe);

	PolyUniv* m_sel = new_univ(params_glwe);

	uniform_glwe_secret_key(module, sk_ggsw, 3);
	glwe_sk_prepare(module, sk_glwe_prep, sk_ggsw);

	uniform_random_pol_znx(m1, params_glwe->nn, 62);
	uniform_random_pol_znx(m2, params_glwe->nn, 62);

	glwe_secret_encrypt_tnx(module, glwe1, sk_glwe_prep, m1);
	glwe_secret_encrypt_tnx(module, glwe2, sk_glwe_prep, m2);

	gpu_ntt_initialize(params_glwe->nn);
	size_t total = glwe_coef_number(params_glwe);

	// Case 0: selection signal 0 -> tfhe_cmux_unprepared should select glwe1 (c0)
	// Case 1: selection signal 1 -> tfhe_cmux_unprepared should select glwe2 (c1)
	for (int sel_bit = 0; sel_bit < 2; ++sel_bit)
	{
		memset(m_sel, 0, poly_univ_bytes(params_glwe));
		m_sel[0] = sel_bit;
		ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, m_sel);

		// CPU reference (host pointers -> CPU path throughout)
		int ret = tfhe_cmux_unprepared(module, res_expected, glwe1, glwe2, ggsw, 1);
		cr_assert_eq(ret, 0, "tfhe_cmux_unprepared (CPU) failed");

		// GPU: upload everything, run CMux entirely through device pointers
		// (same tfhe_cmux_unprepared entry point, raw GGSW — no prepare step).
		GLWECiphertext gpu_glwe1 = {.params = params_glwe, .vec = pvda_glwe_to_device(glwe1)};
		GLWECiphertext gpu_glwe2 = {.params = params_glwe, .vec = pvda_glwe_to_device(glwe2)};
		GLWECiphertext gpu_res   = {.params = params_glwe, .vec = pvda_gpu_alloc(total)};
		GGSWCiphertext gpu_ggsw  = {.params = params_ggsw, .mat = pvda_ggsw_to_device(ggsw)};

		cr_assert_not_null(gpu_glwe1.vec, "pvda_glwe_to_device failed");
		cr_assert_not_null(gpu_glwe2.vec, "pvda_glwe_to_device failed");
		cr_assert_not_null(gpu_res.vec, "pvda_gpu_alloc failed");
		cr_assert_not_null(gpu_ggsw.mat, "pvda_ggsw_to_device failed");

		ret = tfhe_cmux_unprepared(module, &gpu_res, &gpu_glwe1, &gpu_glwe2, &gpu_ggsw, 1);
		cr_assert_eq(ret, 0, "tfhe_cmux_unprepared (GPU) failed");
		pvda_glwe_from_device(res_from_gpu, gpu_res.vec);

		for (uint64_t t = 0; t < total; t++) cr_assert(eq(i64, res_from_gpu->vec[t], res_expected->vec[t]));

		pvda_gpu_free(gpu_glwe1.vec);
		pvda_gpu_free(gpu_glwe2.vec);
		pvda_gpu_free(gpu_res.vec);
		pvda_gpu_free(gpu_ggsw.mat);
	}

	delete_univ_tnx(m1);
	delete_univ_tnx(m2);

	delete_glwe(glwe1);
	delete_glwe(glwe2);
	delete_glwe(res_expected);
	delete_glwe(res_from_gpu);

	delete_ggsw(ggsw);

	delete_univ(m_sel);

	delete_glwe_secret_key(sk_ggsw);
	delete_glwe_secret_key_prepared(sk_glwe_prep);

	DELETE_PVDA_PARAMS_GGSW;
}

/**
 * @brief Same as tfhe_cmux::gpu_matches_cpu but for tfhe_cmux_tree. tfhe_cmux_tree has no
 * GPU-specific code of its own beyond allocating its intermediate nodes on the device
 * when its leaf inputs already are (see tfhe_cmux_tree_new_node / pvda_new_glwe_device in
 * tfhe.c) — the CMux gates themselves still run through tfhe_cmux's own GPU dispatch.
 * Uploads every leaf GLWE and both selector GGSWs once, then runs the whole binary tree
 * for a handful of selection values, checking the GPU result matches the CPU path
 * bit-for-bit.
 */
PvdaParamTest(tfhe_cmux_tree, gpu_matches_cpu, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSW(param);

	GLWESecretKey* sk_ggsw              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_glwe_prep = alloc_glwe_secret_key_prepared(params_glwe);

	GGSWCiphertext* ggsw           = new_ggsw(params_ggsw);
	GGSWCiphertextPrep* ggsw_prep0 = new_ggsw_prep(params_ggsw);
	GGSWCiphertextPrep* ggsw_prep1 = new_ggsw_prep(params_ggsw);

	int num_msgs      = 16;
	int selections[]  = {0, 1, 5, 9, 15};
	int msgs_log2     = 4;
	int num_selctions = sizeof(selections) / sizeof(selections[0]);

	PolyUnivTnX* ms[16];
	GLWECiphertext* glwes[16];
	GLWECiphertext* res_expected = new_glwe(params_glwe);
	GLWECiphertext* res_from_gpu = new_glwe(params_glwe);

	PolyUniv* m_sel = new_univ(params_glwe);

	uniform_glwe_secret_key(module, sk_ggsw, 2);
	glwe_sk_prepare(module, sk_glwe_prep, sk_ggsw);

	for (int i = 0; i < num_msgs; ++i)
	{
		ms[i] = new_univ_tnx(params_glwe);
		memset(ms[i], 0, poly_univ_tnx_bytes(params_glwe));
		glwes[i] = new_glwe(params_glwe);
		uniform_random_pol_znx(ms[i], params_glwe->nn, 64);
		glwe_secret_encrypt_tnx(module, glwes[i], sk_glwe_prep, ms[i]);
	}

	gpu_ntt_initialize(params_glwe->nn);
	size_t total = glwe_coef_number(params_glwe);

	// Selector for bit 0: prepare on CPU (ggsw_prep0) and upload + prepare on GPU.
	memset(m_sel, 0, poly_univ_bytes(params_glwe));
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, m_sel);
	ggsw_prepare(module, ggsw_prep0, ggsw);
	GGSWCiphertext gpu_ggsw0          = {.params = params_ggsw, .mat = pvda_ggsw_to_device(ggsw)};
	GGSWCiphertextPrep gpu_ggsw_prep0 = {.params = params_ggsw, .mat = NULL};
	ggsw_prepare(module, &gpu_ggsw_prep0, &gpu_ggsw0);
	cr_assert_not_null(gpu_ggsw0.mat, "pvda_ggsw_to_device failed");
	cr_assert_not_null(gpu_ggsw_prep0.mat, "ggsw_prepare (GPU) failed");

	// Selector for bit 1: same, reusing the (now overwritten) ggsw ciphertext.
	m_sel[0] = 1;
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, m_sel);
	ggsw_prepare(module, ggsw_prep1, ggsw);
	GGSWCiphertext gpu_ggsw1          = {.params = params_ggsw, .mat = pvda_ggsw_to_device(ggsw)};
	GGSWCiphertextPrep gpu_ggsw_prep1 = {.params = params_ggsw, .mat = NULL};
	ggsw_prepare(module, &gpu_ggsw_prep1, &gpu_ggsw1);
	cr_assert_not_null(gpu_ggsw1.mat, "pvda_ggsw_to_device failed");
	cr_assert_not_null(gpu_ggsw_prep1.mat, "ggsw_prepare (GPU) failed");

	// Upload every leaf GLWE once — reused across every selection below.
	GLWECiphertext* gpu_glwes[16];
	for (int i = 0; i < num_msgs; ++i)
	{
		gpu_glwes[i] = (GLWECiphertext*)malloc(sizeof(GLWECiphertext));
		cr_assert_not_null(gpu_glwes[i], "malloc failed");
		gpu_glwes[i]->params = params_glwe;
		gpu_glwes[i]->vec    = (VecBiv*)pvda_glwe_to_device(glwes[i]);
		cr_assert_not_null(gpu_glwes[i]->vec, "pvda_glwe_to_device failed");
	}

	GGSWCiphertextPrep* selector[4];
	GGSWCiphertextPrep* selector_gpu[4];
	for (int i = 0; i < num_selctions; ++i)
	{
		int sel = selections[i];
		for (int j = 0; j < msgs_log2; ++j)
		{
			uint64_t v      = sel % 2;
			selector[j]     = v ? ggsw_prep1 : ggsw_prep0;
			selector_gpu[j] = v ? &gpu_ggsw_prep1 : &gpu_ggsw_prep0;
			sel >>= 1;
		}
		sel = selections[i];

		// CPU reference (host pointers -> CPU path throughout).
		int ret = tfhe_cmux_tree(module, res_expected, (const GLWECiphertext**)glwes, num_msgs,
		                         (const GGSWCiphertextPrep**)selector, msgs_log2, 0);
		cr_assert_eq(ret, 0, "tfhe_cmux_tree (CPU) failed");
		normalize_glwe(module, res_expected, res_expected);

		// GPU: same tfhe_cmux_tree entry point, device-resident leaves/selectors/result —
		// intermediate tree nodes are allocated on the device by tfhe_cmux_tree itself.
		GLWECiphertext gpu_res = {.params = params_glwe, .vec = pvda_gpu_alloc(total)};
		cr_assert_not_null(gpu_res.vec, "pvda_gpu_alloc failed");

		ret = tfhe_cmux_tree(module, &gpu_res, (const GLWECiphertext**)gpu_glwes, num_msgs,
		                     (const GGSWCiphertextPrep**)selector_gpu, msgs_log2, 0);
		cr_assert_eq(ret, 0, "tfhe_cmux_tree (GPU) failed");
		normalize_glwe(module, &gpu_res, &gpu_res);
		pvda_glwe_from_device(res_from_gpu, gpu_res.vec);

		for (uint64_t t = 0; t < total; t++)
			cr_assert(eq(i64, res_from_gpu->vec[t], res_expected->vec[t]), "sel=%d mismatch at t=%llu", sel,
			          (unsigned long long)t);

		pvda_gpu_free(gpu_res.vec);
	}

	for (int i = 0; i < num_msgs; ++i)
	{
		delete_univ_tnx(ms[i]);
		delete_glwe(glwes[i]);
		delete_glwe(gpu_glwes[i]);
	}

	pvda_gpu_free(gpu_ggsw0.mat);
	pvda_gpu_free((int64_t*)gpu_ggsw_prep0.mat);
	pvda_gpu_free(gpu_ggsw1.mat);
	pvda_gpu_free((int64_t*)gpu_ggsw_prep1.mat);

	delete_glwe(res_expected);
	delete_glwe(res_from_gpu);

	delete_ggsw(ggsw);
	delete_ggsw_prep(ggsw_prep0);
	delete_ggsw_prep(ggsw_prep1);

	delete_univ(m_sel);

	delete_glwe_secret_key(sk_ggsw);
	delete_glwe_secret_key_prepared(sk_glwe_prep);

	DELETE_PVDA_PARAMS_GGSW;
}
#endif  // ENABLE_CUDA
