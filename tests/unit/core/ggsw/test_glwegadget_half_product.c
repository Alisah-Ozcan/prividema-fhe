#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <float.h>
#include <stdint.h>

#include "bivariate_polynomial.h"
#include "core/glwe/glwe_arithmetic.h"
#include "core/glwe/glwe_ciphertext.h"
#include "core/glwe/glwe_transform_key.h"
#include "glwe_key.h"
#include "glwe_params.h"
#include "glwegadget_arithmetic.h"
#include "glwegadget_ciphertext.h"
#include "rng.h"
#include "test_utils.h"
#include "univariate_polynomial.h"
#include "utils.h"

#ifdef ENABLE_CUDA
#include "gpu/common/gpu_nttparameters.h"
#include "gpu/host/ggsw_external_product_gpu.h"
#endif

#define TIMING
// ---------------------------------------------------------------------------
// Timing helpers — enabled with -DTIMING (cmake -DTIMING=ON)
// ---------------------------------------------------------------------------
#ifdef TIMING
#include <stdio.h>
#include <time.h>

#define PVDA_TIME_START(label)         \
	struct timespec _ts_start_##label; \
	clock_gettime(CLOCK_MONOTONIC, &_ts_start_##label)

#define PVDA_TIME_END(label, pglwe, pggsw)                                                                           \
	do                                                                                                               \
	{                                                                                                                \
		struct timespec _ts_end_##label;                                                                             \
		clock_gettime(CLOCK_MONOTONIC, &_ts_end_##label);                                                            \
		double _ms_##label = (_ts_end_##label.tv_sec - _ts_start_##label.tv_sec) * 1e3 +                             \
		                     (_ts_end_##label.tv_nsec - _ts_start_##label.tv_nsec) * 1e-6;                           \
		fprintf(stderr, "[TIMING] " #label " (n=%llu, k=%llu, kappa=%llu, limbs=%llu, limbs_tilde=%llu): %.3f ms\n", \
		        (unsigned long long)(pglwe)->nn, (unsigned long long)(pglwe)->k, (unsigned long long)(pglwe)->kappa, \
		        (unsigned long long)(pglwe)->ciphertext_nb_limbs,                                                    \
		        (unsigned long long)(pggsw)->ciphertext_nb_limbs_tilde, _ms_##label);                                \
	} while (0)
#else
#define PVDA_TIME_START(label)             ((void)0)
#define PVDA_TIME_END(label, pglwe, pggsw) ((void)0)
#endif

PvdaParamTest(glwegadget_half_product, without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSWGAD(param);

	sigma                             = 0;
	params_glwe->fast_uniform_nb_bits = 0;

	double biv_epsilon = glwe_bivariate_epsilon(params_glwe);
	double err_length =
	    params_glwe->nn * (2 * DBL_EPSILON + biv_epsilon) + 2 * glwe_params_l_a(params_glwe) * biv_epsilon;
	double critical_err_length =
	    params_glwe->nn * (3 * DBL_EPSILON + biv_epsilon) + 2 * glwe_params_l_a(params_glwe) * biv_epsilon;

	GLWESecretKey* sk                      = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_prep         = alloc_glwe_secret_key_prepared(params_glwe);
	GLWECiphertext* glwe                   = new_glwe(params_glwe);
	GLWEGadgetCiphertext* glwegad          = new_glwegadget(params_glwegadget);
	GLWEGadgetCiphertextPrep* glwegad_prep = new_glwegadget_prep(params_glwegadget);
	PolyUniv* u_univ                       = new_univ(params_glwe);
	PolyUnivTnX* m_univ_tnx                = new_univ_tnx(params_glwe);
	PolyUnivRnX* m_univ_rnx                = new_univ_rnx(params_glwe);
	PolyUnivTnX* um_expected_tnx           = new_univ_tnx(params_glwe);
	PolyUnivRnX* um_expected_rnx           = new_univ_rnx(params_glwe);
	PolyBiv* um_observed                   = new_biv(params_glwe);
	PolyUnivRnX* um_observed_rnx           = new_univ_rnx(params_glwe);
	PolyBiv* m                             = new_biv(params_glwe);

	int nn    = params_glwe->nn;
	int k     = params_glwe->k;
	int kappa = params_glwe->kappa;

	// Generate secret key
	uniform_glwe_secret_key(module, sk, 3);
	glwe_sk_prepare(module, sk_prep, sk);

	//Generate the public vector m and the data that will be in the gadget u
	uniform_random_pol_znx(u_univ, params_glwe->nn, 3);
	uniform_random_pol_znx(m_univ_tnx, params_glwe->nn, 62);
	univ_tnx_to_biv(params_glwe, m, m_univ_tnx, 0);

	//Compute the (negacyclic) polynomial product of u*m
	memset(um_expected_tnx, 0, poly_univ_rnx_bytes(params_glwe));
	for (int i = 0; i < params_glwe->nn; ++i)
		for (int j = 0; j < params_glwe->nn; ++j)
		{
			if (i + j < params_glwe->nn)
				um_expected_tnx[(i + j) % params_glwe->nn] += (uint64_t)u_univ[i] * m_univ_tnx[j];
			else
				um_expected_tnx[(i + j) % params_glwe->nn] -= (uint64_t)u_univ[i] * m_univ_tnx[j];
		}
	// to rnx
	univ_tnx_to_rnx(params_glwe, um_expected_rnx, um_expected_tnx);

	// Generate and prepare a GLWEGadget for u
	glwegadget_secret_encrypt(module, glwegad, sk_prep, u_univ);
	glwegadget_prepare(module, glwegad_prep, glwegad);

	//Half product of GLWEGadget(u) with m
	PVDA_TIME_START(cpu_half_prod);
	glwegadget_half_prod(module, glwe, glwegad_prep, m);
	PVDA_TIME_END(cpu_half_prod, params_glwe, params_ggsw);
	normalize_glwe(module, glwe, glwe);

	// Decrypt the result into um_observed, which should be u*m
	glwe_secret_decrypt(module, um_observed, sk_prep, glwe);
	biv_to_univ_rnx(params_glwe, um_observed_rnx, um_observed);

	//Assert that the observed and expected values for u*m are close enough
	pvda_assert_polynomial_distance(params_glwe, um_observed_rnx, um_expected_rnx, err_length, critical_err_length);

	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_glwe(glwe);
	delete_glwegadget(glwegad);
	delete_glwegadget_prep(glwegad_prep);
	delete_univ(u_univ);
	delete_univ_tnx(m_univ_tnx);
	delete_univ_rnx(m_univ_rnx);
	delete_univ_tnx(um_expected_tnx);
	delete_univ_rnx(um_expected_rnx);
	delete_biv(um_observed);
	delete_univ_rnx(um_observed_rnx);
	delete_biv(m);
	DELETE_PVDA_PARAMS_GGSWGAD;
}

PvdaParamTest(glwegadget_half_product_dft_to_dft, without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSWGAD(param);

	sigma                             = 0;
	params_glwe->fast_uniform_nb_bits = 0;

	double biv_epsilon = glwe_bivariate_epsilon(params_glwe);
	double err_length =
	    params_glwe->nn * (2 * DBL_EPSILON + biv_epsilon) + 2 * glwe_params_l_a(params_glwe) * biv_epsilon;
	double critical_err_length =
	    params_glwe->nn * (3 * DBL_EPSILON + biv_epsilon) + 2 * glwe_params_l_a(params_glwe) * biv_epsilon;

	GLWESecretKey* sk                      = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_prep         = alloc_glwe_secret_key_prepared(params_glwe);
	GLWECiphertextDFT* glwe_dft            = new_glwe_dft(params_glwe);
	GLWECiphertext* glwe                   = new_glwe(params_glwe);
	GLWEGadgetCiphertext* glwegad          = new_glwegadget(params_glwegadget);
	GLWEGadgetCiphertextPrep* glwegad_prep = new_glwegadget_prep(params_glwegadget);
	PolyUniv* u_univ                       = new_univ(params_glwe);
	PolyUnivTnX* m_univ_tnx                = new_univ_tnx(params_glwe);
	PolyUnivRnX* m_univ_rnx                = new_univ_rnx(params_glwe);
	PolyUnivTnX* um_expected_tnx           = new_univ_tnx(params_glwe);
	PolyUnivRnX* um_expected_rnx           = new_univ_rnx(params_glwe);
	PolyBiv* um_observed                   = new_biv(params_glwe);
	PolyUnivRnX* um_observed_rnx           = new_univ_rnx(params_glwe);
	PolyBiv* m                             = new_biv(params_glwe);
	PolyBivDFT* m_dft                      = new_biv_dft(params_glwe);

	int nn    = params_glwe->nn;
	int k     = params_glwe->k;
	int kappa = params_glwe->kappa;

	// Generate secret key
	uniform_glwe_secret_key(module, sk, 3);
	glwe_sk_prepare(module, sk_prep, sk);

	//Generate the public vector m and the data that will be in the gadget u
	uniform_random_pol_znx(u_univ, params_glwe->nn, 3);
	uniform_random_pol_znx(m_univ_tnx, params_glwe->nn, 62);
	univ_tnx_to_biv(params_glwe, m, m_univ_tnx, 0);

	//Compute the (negacyclic) polynomial product of u*m
	memset(um_expected_tnx, 0, poly_univ_rnx_bytes(params_glwe));
	for (int i = 0; i < params_glwe->nn; ++i)
		for (int j = 0; j < params_glwe->nn; ++j)
		{
			if (i + j < params_glwe->nn)
				um_expected_tnx[(i + j) % params_glwe->nn] += (uint64_t)u_univ[i] * m_univ_tnx[j];
			else
				um_expected_tnx[(i + j) % params_glwe->nn] -= (uint64_t)u_univ[i] * m_univ_tnx[j];
		}
	// to rnx
	univ_tnx_to_rnx(params_glwe, um_expected_rnx, um_expected_tnx);

	// Generate and prepare a GLWEGadget for u
	glwegadget_secret_encrypt(module, glwegad, sk_prep, u_univ);
	glwegadget_prepare(module, glwegad_prep, glwegad);

	//Prepare m to be in dft space
	biv_coefs_to_dft(module, params_glwe, m_dft, m);

	//Half product of GLWEGadget(u) with m with prepared and dft inputs
	PVDA_TIME_START(cpu_half_prod_dft_to_dft);
	glwegadget_half_prod_dft_to_dft(module, glwe_dft, glwegad_prep, m_dft);
	PVDA_TIME_END(cpu_half_prod_dft_to_dft, params_glwe, params_ggsw);
	glwe_dft_to_coef(module, glwe, glwe_dft);  //Return the result GLWE in DFT domain to coefficient domain
	normalize_glwe(module, glwe, glwe);        // And normalize it

	// Decrypt the result into um_observed, which should be u*m
	glwe_secret_decrypt(module, um_observed, sk_prep, glwe);
	biv_to_univ_rnx(params_glwe, um_observed_rnx, um_observed);

	//Assert that the observed and expected values for u*m are close enough
	pvda_assert_polynomial_distance(params_glwe, um_observed_rnx, um_expected_rnx, err_length, critical_err_length);

	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_glwe(glwe);
	delete_glwe_dft(glwe_dft);
	delete_glwegadget(glwegad);
	delete_glwegadget_prep(glwegad_prep);
	delete_univ(u_univ);
	delete_univ_tnx(m_univ_tnx);
	delete_univ_rnx(m_univ_rnx);
	delete_univ_tnx(um_expected_tnx);
	delete_univ_rnx(um_expected_rnx);
	delete_biv(um_observed);
	delete_univ_rnx(um_observed_rnx);
	delete_biv(m);
	free(m_dft);
	DELETE_PVDA_PARAMS_GGSWGAD;
}

PvdaParamTest(glwegadget_half_product_prepared_to_dft, without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSWGAD(param);

	sigma                             = 0;
	params_glwe->fast_uniform_nb_bits = 0;

	double biv_epsilon = glwe_bivariate_epsilon(params_glwe);
	double err_length =
	    params_glwe->nn * (2 * DBL_EPSILON + biv_epsilon) + 2 * glwe_params_l_a(params_glwe) * biv_epsilon;
	double critical_err_length =
	    params_glwe->nn * (3 * DBL_EPSILON + biv_epsilon) + 2 * glwe_params_l_a(params_glwe) * biv_epsilon;

	GLWESecretKey* sk                      = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_prep         = alloc_glwe_secret_key_prepared(params_glwe);
	GLWECiphertextDFT* glwe_dft            = new_glwe_dft(params_glwe);
	GLWECiphertext* glwe                   = new_glwe(params_glwe);
	GLWEGadgetCiphertext* glwegad          = new_glwegadget(params_glwegadget);
	GLWEGadgetCiphertextPrep* glwegad_prep = new_glwegadget_prep(params_glwegadget);
	PolyUniv* u_univ                       = new_univ(params_glwe);
	PolyUnivTnX* m_univ_tnx                = new_univ_tnx(params_glwe);
	PolyUnivRnX* m_univ_rnx                = new_univ_rnx(params_glwe);
	PolyUnivTnX* um_expected_tnx           = new_univ_tnx(params_glwe);
	PolyUnivRnX* um_expected_rnx           = new_univ_rnx(params_glwe);
	PolyBiv* um_observed                   = new_biv(params_glwe);
	PolyUnivRnX* um_observed_rnx           = new_univ_rnx(params_glwe);
	PolyBiv* m                             = new_biv(params_glwe);
	PolyBivDFT* m_prep                     = new_biv_dft(params_glwe);

	int nn    = params_glwe->nn;
	int k     = params_glwe->k;
	int kappa = params_glwe->kappa;

	// Generate secret key
	uniform_glwe_secret_key(module, sk, 3);
	glwe_sk_prepare(module, sk_prep, sk);

	//Generate the public vector m and the data that will be in the gadget u
	uniform_random_pol_znx(u_univ, params_glwe->nn, 3);
	uniform_random_pol_znx(m_univ_tnx, params_glwe->nn, 62);
	univ_tnx_to_biv(params_glwe, m, m_univ_tnx, 0);

	//Compute the (negacyclic) polynomial product of u*m
	memset(um_expected_tnx, 0, poly_univ_rnx_bytes(params_glwe));
	for (int i = 0; i < params_glwe->nn; ++i)
		for (int j = 0; j < params_glwe->nn; ++j)
		{
			if (i + j < params_glwe->nn)
				um_expected_tnx[(i + j) % params_glwe->nn] += (uint64_t)u_univ[i] * m_univ_tnx[j];
			else
				um_expected_tnx[(i + j) % params_glwe->nn] -= (uint64_t)u_univ[i] * m_univ_tnx[j];
		}
	// to rnx
	univ_tnx_to_rnx(params_glwe, um_expected_rnx, um_expected_tnx);

	// Generate and prepare a GLWEGadget for u
	glwegadget_secret_encrypt(module, glwegad, sk_prep, u_univ);
	glwegadget_prepare(module, glwegad_prep, glwegad);

	// Get the public vector into a "prepared" for half-product format (different from DFT!)
	biv_coefs_to_prep(module, params_glwe, m_prep, m);

	//Half product of GLWEGadget(u) with m, with prepared gadget and prepared public vector, which is different than DFT
	PVDA_TIME_START(cpu_half_prod_prepared_to_dft);
	glwegadget_half_prod_prepared_to_dft(module, glwe_dft, glwegad_prep, m_prep);
	PVDA_TIME_END(cpu_half_prod_prepared_to_dft, params_glwe, params_ggsw);
	glwe_dft_to_coef(module, glwe, glwe_dft);
	normalize_glwe(module, glwe, glwe);

	// Decrypt the result into um_observed, which should be u*m
	glwe_secret_decrypt(module, um_observed, sk_prep, glwe);
	biv_to_univ_rnx(params_glwe, um_observed_rnx, um_observed);

	//Assert that the observed and expected values for u*m are close enough
	pvda_assert_polynomial_distance(params_glwe, um_observed_rnx, um_expected_rnx, err_length, critical_err_length);

	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_glwe(glwe);
	delete_glwe_dft(glwe_dft);
	delete_glwegadget(glwegad);
	delete_glwegadget_prep(glwegad_prep);
	delete_univ(u_univ);
	delete_univ_tnx(m_univ_tnx);
	delete_univ_rnx(m_univ_rnx);
	delete_univ_tnx(um_expected_tnx);
	delete_univ_rnx(um_expected_rnx);
	delete_biv(um_observed);
	delete_univ_rnx(um_observed_rnx);
	delete_biv(m);
	free(m_prep);
	DELETE_PVDA_PARAMS_GGSWGAD;
}

#ifdef ENABLE_CUDA
/** GPU variant: glwegadget_half_prod runs on GPU, normalize + decrypt on CPU */
PvdaParamTest(glwegadget_half_product, gpu_without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSWGAD(param);

	sigma                             = 0;
	params_glwe->fast_uniform_nb_bits = 0;

	double biv_epsilon = glwe_bivariate_epsilon(params_glwe);
	double err_length =
	    params_glwe->nn * (2 * DBL_EPSILON + biv_epsilon) + 2 * glwe_params_l_a(params_glwe) * biv_epsilon;
	double critical_err_length =
	    params_glwe->nn * (3 * DBL_EPSILON + biv_epsilon) + 2 * glwe_params_l_a(params_glwe) * biv_epsilon;

	GLWESecretKey* sk              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params_glwe);
	GLWECiphertext* glwe           = new_glwe(params_glwe);
	GLWEGadgetCiphertext* glwegad  = new_glwegadget(params_glwegadget);
	PolyUniv* u_univ               = new_univ(params_glwe);
	PolyUnivTnX* m_univ_tnx        = new_univ_tnx(params_glwe);
	PolyUnivRnX* um_expected_rnx   = new_univ_rnx(params_glwe);
	PolyUnivTnX* um_expected_tnx   = new_univ_tnx(params_glwe);
	PolyBiv* um_observed           = new_biv(params_glwe);
	PolyUnivRnX* um_observed_rnx   = new_univ_rnx(params_glwe);
	PolyBiv* m                     = new_biv(params_glwe);

	uniform_glwe_secret_key(module, sk, 3);
	glwe_sk_prepare(module, sk_prep, sk);

	uniform_random_pol_znx(u_univ, params_glwe->nn, 3);
	uniform_random_pol_znx(m_univ_tnx, params_glwe->nn, 62);
	univ_tnx_to_biv(params_glwe, m, m_univ_tnx, 0);

	memset(um_expected_tnx, 0, poly_univ_rnx_bytes(params_glwe));
	for (int i = 0; i < params_glwe->nn; ++i)
		for (int j = 0; j < params_glwe->nn; ++j)
		{
			if (i + j < params_glwe->nn)
				um_expected_tnx[(i + j) % params_glwe->nn] += (uint64_t)u_univ[i] * m_univ_tnx[j];
			else
				um_expected_tnx[(i + j) % params_glwe->nn] -= (uint64_t)u_univ[i] * m_univ_tnx[j];
		}
	univ_tnx_to_rnx(params_glwe, um_expected_rnx, um_expected_tnx);

	glwegadget_secret_encrypt(module, glwegad, sk_prep, u_univ);

	gpu_ntt_initialize(params_glwe->nn);
	size_t result_elems = (size_t)(glwe_params_n_limbs(params_glwe) * params_glwe->nn);

	GLWEGadgetCiphertext gpu_glwegad          = {.params = params_glwegadget,
	                                             .mat    = (MatBiv*)pvda_glwegadget_to_device(glwegad)};
	GLWEGadgetCiphertextPrep gpu_glwegad_prep = {.params = params_glwegadget, .mat = NULL};
	glwegadget_prepare(module, &gpu_glwegad_prep, &gpu_glwegad);
	GLWECiphertext gpu_result = {.params = params_glwe, .vec = pvda_gpu_alloc(result_elems)};

	cr_assert_not_null(gpu_glwegad.mat, "pvda_glwegadget_to_device failed");
	cr_assert_not_null(gpu_glwegad_prep.mat, "glwegadget_prepare failed");
	cr_assert_not_null(gpu_result.vec, "pvda_gpu_alloc for result failed");

	PVDA_TIME_START(gpu_half_prod);
	glwegadget_half_prod(module, &gpu_result, &gpu_glwegad_prep, m);
	PVDA_TIME_END(gpu_half_prod, params_glwe, params_ggsw);

	pvda_glwe_from_device(glwe, gpu_result.vec);
	normalize_glwe(module, glwe, glwe);
	glwe_secret_decrypt(module, um_observed, sk_prep, glwe);
	biv_to_univ_rnx(params_glwe, um_observed_rnx, um_observed);

	pvda_assert_polynomial_distance(params_glwe, um_observed_rnx, um_expected_rnx, err_length, critical_err_length);

	pvda_gpu_free(gpu_glwegad.mat);
	pvda_gpu_free((int64_t*)gpu_glwegad_prep.mat);
	pvda_gpu_free(gpu_result.vec);

	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_glwe(glwe);
	delete_glwegadget(glwegad);
	delete_univ(u_univ);
	delete_univ_tnx(m_univ_tnx);
	delete_univ_rnx(um_expected_rnx);
	delete_univ_tnx(um_expected_tnx);
	delete_biv(um_observed);
	delete_univ_rnx(um_observed_rnx);
	delete_biv(m);
	DELETE_PVDA_PARAMS_GGSWGAD;
}

/** GPU variant: glwegadget_half_prod_dft_to_dft — input as NTT-domain device ptr, result in NTT domain then INTT */
PvdaParamTest(glwegadget_half_product_dft_to_dft, gpu_without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSWGAD(param);

	sigma                             = 0;
	params_glwe->fast_uniform_nb_bits = 0;

	double biv_epsilon = glwe_bivariate_epsilon(params_glwe);
	double err_length =
	    params_glwe->nn * (2 * DBL_EPSILON + biv_epsilon) + 2 * glwe_params_l_a(params_glwe) * biv_epsilon;
	double critical_err_length =
	    params_glwe->nn * (3 * DBL_EPSILON + biv_epsilon) + 2 * glwe_params_l_a(params_glwe) * biv_epsilon;

	GLWESecretKey* sk              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params_glwe);
	GLWECiphertext* glwe           = new_glwe(params_glwe);
	GLWEGadgetCiphertext* glwegad  = new_glwegadget(params_glwegadget);
	PolyUniv* u_univ               = new_univ(params_glwe);
	PolyUnivTnX* m_univ_tnx        = new_univ_tnx(params_glwe);
	PolyUnivRnX* um_expected_rnx   = new_univ_rnx(params_glwe);
	PolyUnivTnX* um_expected_tnx   = new_univ_tnx(params_glwe);
	PolyBiv* um_observed           = new_biv(params_glwe);
	PolyUnivRnX* um_observed_rnx   = new_univ_rnx(params_glwe);
	PolyBiv* m                     = new_biv(params_glwe);

	uniform_glwe_secret_key(module, sk, 3);
	glwe_sk_prepare(module, sk_prep, sk);

	uniform_random_pol_znx(u_univ, params_glwe->nn, 3);
	uniform_random_pol_znx(m_univ_tnx, params_glwe->nn, 62);
	univ_tnx_to_biv(params_glwe, m, m_univ_tnx, 0);

	memset(um_expected_tnx, 0, poly_univ_rnx_bytes(params_glwe));
	for (int i = 0; i < params_glwe->nn; ++i)
		for (int j = 0; j < params_glwe->nn; ++j)
		{
			if (i + j < params_glwe->nn)
				um_expected_tnx[(i + j) % params_glwe->nn] += (uint64_t)u_univ[i] * m_univ_tnx[j];
			else
				um_expected_tnx[(i + j) % params_glwe->nn] -= (uint64_t)u_univ[i] * m_univ_tnx[j];
		}
	univ_tnx_to_rnx(params_glwe, um_expected_rnx, um_expected_tnx);

	glwegadget_secret_encrypt(module, glwegad, sk_prep, u_univ);

	gpu_ntt_initialize(params_glwe->nn);
	size_t l_tilde      = (size_t)params_glwegadget->l_tilde;
	size_t nn           = (size_t)params_glwe->nn;
	size_t result_elems = (size_t)(glwe_params_n_limbs(params_glwe) * params_glwe->nn);

	GLWEGadgetCiphertext gpu_glwegad          = {.params = params_glwegadget,
	                                             .mat    = (MatBiv*)pvda_glwegadget_to_device(glwegad)};
	GLWEGadgetCiphertextPrep gpu_glwegad_prep = {.params = params_glwegadget, .mat = NULL};
	glwegadget_prepare(module, &gpu_glwegad_prep, &gpu_glwegad);

	// Upload first l_tilde coef-domain limb polynomials of m (treated as NTT input by GPU bypass)
	int64_t* d_m_coef = pvda_gpu_upload(m->ptr, l_tilde * nn);

	GLWECiphertextDFT gpu_result_dft = {.params = params_glwe, .vec = (VecBivDFT*)pvda_gpu_alloc(result_elems)};
	GLWECiphertext gpu_result_coef   = {.params = params_glwe, .vec = pvda_gpu_alloc(result_elems)};

	cr_assert_not_null(gpu_glwegad.mat, "pvda_glwegadget_to_device failed");
	cr_assert_not_null(gpu_glwegad_prep.mat, "glwegadget_prepare failed");
	cr_assert_not_null(d_m_coef, "pvda_gpu_upload for m failed");
	cr_assert_not_null(gpu_result_dft.vec, "pvda_gpu_alloc for NTT result failed");
	cr_assert_not_null(gpu_result_coef.vec, "pvda_gpu_alloc for coef result failed");

	PVDA_TIME_START(gpu_half_prod_dft_to_dft);
	glwegadget_half_prod_dft_to_dft(module, &gpu_result_dft, &gpu_glwegad_prep, (PolyBivDFT*)d_m_coef);
	PVDA_TIME_END(gpu_half_prod_dft_to_dft, params_glwe, params_ggsw);

	glwe_dft_to_coef(module, &gpu_result_coef, &gpu_result_dft);
	pvda_glwe_from_device(glwe, gpu_result_coef.vec);
	normalize_glwe(module, glwe, glwe);
	glwe_secret_decrypt(module, um_observed, sk_prep, glwe);
	biv_to_univ_rnx(params_glwe, um_observed_rnx, um_observed);

	pvda_assert_polynomial_distance(params_glwe, um_observed_rnx, um_expected_rnx, err_length, critical_err_length);

	pvda_gpu_free(gpu_glwegad.mat);
	pvda_gpu_free((int64_t*)gpu_glwegad_prep.mat);
	pvda_gpu_free(d_m_coef);
	pvda_gpu_free((int64_t*)gpu_result_dft.vec);
	pvda_gpu_free(gpu_result_coef.vec);

	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_glwe(glwe);
	delete_glwegadget(glwegad);
	delete_univ(u_univ);
	delete_univ_tnx(m_univ_tnx);
	delete_univ_rnx(um_expected_rnx);
	delete_univ_tnx(um_expected_tnx);
	delete_biv(um_observed);
	delete_univ_rnx(um_observed_rnx);
	delete_biv(m);
	DELETE_PVDA_PARAMS_GGSWGAD;
}

/** GPU variant: glwegadget_half_prod_prepared_to_dft — same NTT pipeline as dft_to_dft on GPU */
PvdaParamTest(glwegadget_half_product_prepared_to_dft, gpu_without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSWGAD(param);

	sigma                             = 0;
	params_glwe->fast_uniform_nb_bits = 0;

	double biv_epsilon = glwe_bivariate_epsilon(params_glwe);
	double err_length =
	    params_glwe->nn * (2 * DBL_EPSILON + biv_epsilon) + 2 * glwe_params_l_a(params_glwe) * biv_epsilon;
	double critical_err_length =
	    params_glwe->nn * (3 * DBL_EPSILON + biv_epsilon) + 2 * glwe_params_l_a(params_glwe) * biv_epsilon;

	GLWESecretKey* sk              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params_glwe);
	GLWECiphertext* glwe           = new_glwe(params_glwe);
	GLWEGadgetCiphertext* glwegad  = new_glwegadget(params_glwegadget);
	PolyUniv* u_univ               = new_univ(params_glwe);
	PolyUnivTnX* m_univ_tnx        = new_univ_tnx(params_glwe);
	PolyUnivRnX* um_expected_rnx   = new_univ_rnx(params_glwe);
	PolyUnivTnX* um_expected_tnx   = new_univ_tnx(params_glwe);
	PolyBiv* um_observed           = new_biv(params_glwe);
	PolyUnivRnX* um_observed_rnx   = new_univ_rnx(params_glwe);
	PolyBiv* m                     = new_biv(params_glwe);

	uniform_glwe_secret_key(module, sk, 3);
	glwe_sk_prepare(module, sk_prep, sk);

	uniform_random_pol_znx(u_univ, params_glwe->nn, 3);
	uniform_random_pol_znx(m_univ_tnx, params_glwe->nn, 62);
	univ_tnx_to_biv(params_glwe, m, m_univ_tnx, 0);

	memset(um_expected_tnx, 0, poly_univ_rnx_bytes(params_glwe));
	for (int i = 0; i < params_glwe->nn; ++i)
		for (int j = 0; j < params_glwe->nn; ++j)
		{
			if (i + j < params_glwe->nn)
				um_expected_tnx[(i + j) % params_glwe->nn] += (uint64_t)u_univ[i] * m_univ_tnx[j];
			else
				um_expected_tnx[(i + j) % params_glwe->nn] -= (uint64_t)u_univ[i] * m_univ_tnx[j];
		}
	univ_tnx_to_rnx(params_glwe, um_expected_rnx, um_expected_tnx);

	glwegadget_secret_encrypt(module, glwegad, sk_prep, u_univ);

	gpu_ntt_initialize(params_glwe->nn);
	size_t l_tilde      = (size_t)params_glwegadget->l_tilde;
	size_t nn           = (size_t)params_glwe->nn;
	size_t result_elems = (size_t)(glwe_params_n_limbs(params_glwe) * params_glwe->nn);

	GLWEGadgetCiphertext gpu_glwegad          = {.params = params_glwegadget,
	                                             .mat    = (MatBiv*)pvda_glwegadget_to_device(glwegad)};
	GLWEGadgetCiphertextPrep gpu_glwegad_prep = {.params = params_glwegadget, .mat = NULL};
	glwegadget_prepare(module, &gpu_glwegad_prep, &gpu_glwegad);

	// Upload first l_tilde coef-domain limb polynomials of m (treated as NTT input by GPU bypass)
	int64_t* d_m_coef = pvda_gpu_upload(m->ptr, l_tilde * nn);

	GLWECiphertextDFT gpu_result_dft = {.params = params_glwe, .vec = (VecBivDFT*)pvda_gpu_alloc(result_elems)};
	GLWECiphertext gpu_result_coef   = {.params = params_glwe, .vec = pvda_gpu_alloc(result_elems)};

	cr_assert_not_null(gpu_glwegad.mat, "pvda_glwegadget_to_device failed");
	cr_assert_not_null(gpu_glwegad_prep.mat, "glwegadget_prepare failed");
	cr_assert_not_null(d_m_coef, "pvda_gpu_upload for m failed");
	cr_assert_not_null(gpu_result_dft.vec, "pvda_gpu_alloc for NTT result failed");
	cr_assert_not_null(gpu_result_coef.vec, "pvda_gpu_alloc for coef result failed");

	PVDA_TIME_START(gpu_half_prod_prepared_to_dft);
	glwegadget_half_prod_prepared_to_dft(module, &gpu_result_dft, &gpu_glwegad_prep, (PolyBivPrep*)d_m_coef);
	PVDA_TIME_END(gpu_half_prod_prepared_to_dft, params_glwe, params_ggsw);

	glwe_dft_to_coef(module, &gpu_result_coef, &gpu_result_dft);
	pvda_glwe_from_device(glwe, gpu_result_coef.vec);
	normalize_glwe(module, glwe, glwe);
	glwe_secret_decrypt(module, um_observed, sk_prep, glwe);
	biv_to_univ_rnx(params_glwe, um_observed_rnx, um_observed);

	pvda_assert_polynomial_distance(params_glwe, um_observed_rnx, um_expected_rnx, err_length, critical_err_length);

	pvda_gpu_free(gpu_glwegad.mat);
	pvda_gpu_free((int64_t*)gpu_glwegad_prep.mat);
	pvda_gpu_free(d_m_coef);
	pvda_gpu_free((int64_t*)gpu_result_dft.vec);
	pvda_gpu_free(gpu_result_coef.vec);

	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_glwe(glwe);
	delete_glwegadget(glwegad);
	delete_univ(u_univ);
	delete_univ_tnx(m_univ_tnx);
	delete_univ_rnx(um_expected_rnx);
	delete_univ_tnx(um_expected_tnx);
	delete_biv(um_observed);
	delete_univ_rnx(um_observed_rnx);
	delete_biv(m);
	DELETE_PVDA_PARAMS_GGSWGAD;
}
#endif
