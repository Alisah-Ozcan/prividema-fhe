#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <float.h>

#include "bivariate_polynomial.h"
#include "core/ggsw/ggsw_arithmetic.h"
#include "core/glwe/glwe_arithmetic.h"
#include "core/glwe/glwe_ciphertext.h"
#include "core/glwe/glwe_transform_key.h"
#include "ggsw_params.h"
#include "glwe_params.h"
#include "rng.h"
#include "test_utils.h"
#include "univariate_polynomial.h"

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

#define PVDA_TIME_START(label) \
    struct timespec _ts_start_##label; \
    clock_gettime(CLOCK_MONOTONIC, &_ts_start_##label)

#define PVDA_TIME_END(label, pglwe, pggsw) \
    do { \
        struct timespec _ts_end_##label; \
        clock_gettime(CLOCK_MONOTONIC, &_ts_end_##label); \
        double _ms_##label = (_ts_end_##label.tv_sec  - _ts_start_##label.tv_sec)  * 1e3 \
                           + (_ts_end_##label.tv_nsec - _ts_start_##label.tv_nsec) * 1e-6; \
        fprintf(stderr, \
                "[TIMING] " #label \
                " (n=%llu, k=%llu, kappa=%llu, limbs=%llu, limbs_tilde=%llu): %.3f ms\n", \
                (unsigned long long)(pglwe)->nn, \
                (unsigned long long)(pglwe)->k, \
                (unsigned long long)(pglwe)->kappa, \
                (unsigned long long)(pglwe)->ciphertext_nb_limbs, \
                (unsigned long long)(pggsw)->ciphertext_nb_limbs_tilde, \
                _ms_##label); \
    } while (0)
#else
#define PVDA_TIME_START(label)              ((void)0)
#define PVDA_TIME_END(label, pglwe, pggsw)  ((void)0)
#endif

/** The test is done without error, it is a proof of concept*/
PvdaParamTest(ggsw_external_product, without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSW(param);

	//! Variance of the error's normal distributions
	params_glwe->fast_uniform_nb_bits = 0;
	sigma                             = 0;
	double err_length                 = glwe_bivariate_epsilon(params_glwe) + 3 * sigma + 3 * DBL_EPSILON;
	double critical_err_length        = glwe_bivariate_epsilon(params_glwe) + 5 * sigma + 5 * DBL_EPSILON;

	GLWESecretKey* sk_ggsw              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_glwe_prep = alloc_glwe_secret_key_prepared(params_glwe);
	GGSWCiphertext* ggsw                = new_ggsw(params_ggsw);
	GLWECiphertext* glwe_tilde          = new_glwe(params_glwe);
	GLWECiphertext* ext_prod_observed   = new_glwe(params_glwe);
	PolyUniv* u_univ                    = new_univ(params_glwe);
	PolyBiv* m                          = new_biv(params_glwe);

	PolyBiv* phase_observed           = new_biv(params_glwe);
	PolyUnivRnX* um_observed_univ_rnx = new_univ_rnx(params_glwe);
	PolyUnivDFT* u_univ_dft           = new_univ_dft(module);
	PolyBivDFT* um_dft                = new_biv_dft(params_glwe);
	PolyBiv* um                       = new_biv(params_glwe);
	PolyUnivRnX* um_univ_rnx          = new_univ_rnx(params_glwe);

	uniform_glwe_secret_key(module, sk_ggsw, 3);
	glwe_sk_prepare(module, sk_glwe_prep, sk_ggsw);

	// Draws uniformly both messages
	uniform_random_pol_znx(u_univ, params_glwe->nn, params_glwe->kappa);
	uniform_random_biv_poly(params_glwe, m, 1);

	// Computation with function
	glwe_secret_encrypt_phase(module, glwe_tilde, sk_glwe_prep, m);
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, u_univ);

	// Computes the external product of glwe_tilde and ggsw
	// It should result in a bivGLWE(u*m) using the base-2Kappa decomposition
	PVDA_TIME_START(cpu_unprepared_external_product);
	ggsw_unprepared_external_product(module, ext_prod_observed, glwe_tilde, ggsw);
	PVDA_TIME_END(cpu_unprepared_external_product, params_glwe, params_ggsw);
	normalize_glwe(module, ext_prod_observed, ext_prod_observed);
	glwe_secret_decrypt(module, phase_observed, sk_glwe_prep, ext_prod_observed);
	biv_to_univ_rnx(params_glwe, um_observed_univ_rnx, phase_observed);

	//Computes u*m manually
	univ_coefs_to_dft(module, u_univ_dft, u_univ);
	pvda_svp_apply_dft(module, um_dft, ggsw_params_l_tilde_a(params_ggsw), u_univ_dft, m);
	biv_dft_to_coefs(module, params_glwe, um, um_dft);
	pvda_vec_znx_normalize_base2k(module, params_glwe->kappa, um, um);
	biv_to_univ_rnx(params_glwe, um_univ_rnx, um);

	//! Asserts um_computed_univ(X) = u * m_univ
	pvda_assert_polynomial_distance(params_glwe, um_observed_univ_rnx, um_univ_rnx, err_length, critical_err_length);

	// Clean up
	delete_biv(m);
	delete_univ(u_univ);
	delete_univ_dft(u_univ_dft);
	delete_biv(phase_observed);
	delete_biv(um);
	delete_univ_rnx(um_univ_rnx);
	free(um_dft);
	delete_univ_rnx(um_observed_univ_rnx);

	delete_glwe(ext_prod_observed);
	delete_glwe(glwe_tilde);
	delete_ggsw(ggsw);

	delete_glwe_secret_key(sk_ggsw);
	delete_glwe_secret_key_prepared(sk_glwe_prep);

	DELETE_PVDA_PARAMS_GGSW;
}

/** Prepared GGSW external product — result in DFT domain, then converted back */
PvdaParamTest(ggsw_external_product, to_dft_without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSW(param);

	params_glwe->fast_uniform_nb_bits = 0;
	sigma                             = 0;
	double err_length                 = glwe_bivariate_epsilon(params_glwe) + 3 * sigma + 3 * DBL_EPSILON;
	double critical_err_length        = glwe_bivariate_epsilon(params_glwe) + 5 * sigma + 5 * DBL_EPSILON;

	GLWESecretKey* sk_ggsw              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_glwe_prep = alloc_glwe_secret_key_prepared(params_glwe);
	GGSWCiphertext* ggsw                = new_ggsw(params_ggsw);
	GGSWCiphertextPrep* ggsw_prep       = new_ggsw_prep(params_ggsw);
	GLWECiphertext* glwe_tilde          = new_glwe(params_glwe);
	GLWECiphertextDFT* result_dft       = new_glwe_dft(params_glwe);
	GLWECiphertext* ext_prod_observed   = new_glwe(params_glwe);
	PolyUniv* u_univ                    = new_univ(params_glwe);
	PolyBiv* m                          = new_biv(params_glwe);

	PolyBiv* phase_observed           = new_biv(params_glwe);
	PolyUnivRnX* um_observed_univ_rnx = new_univ_rnx(params_glwe);
	PolyUnivDFT* u_univ_dft           = new_univ_dft(module);
	PolyBivDFT* um_dft                = new_biv_dft(params_glwe);
	PolyBiv* um                       = new_biv(params_glwe);
	PolyUnivRnX* um_univ_rnx          = new_univ_rnx(params_glwe);

	uniform_glwe_secret_key(module, sk_ggsw, 3);
	glwe_sk_prepare(module, sk_glwe_prep, sk_ggsw);

	uniform_random_pol_znx(u_univ, params_glwe->nn, params_glwe->kappa);
	uniform_random_biv_poly(params_glwe, m, 1);

	glwe_secret_encrypt_phase(module, glwe_tilde, sk_glwe_prep, m);
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, u_univ);
	ggsw_prepare(module, ggsw_prep, ggsw);

	PVDA_TIME_START(cpu_external_product_to_dft);
	ggsw_external_product_to_dft(module, result_dft, glwe_tilde, ggsw_prep);
	PVDA_TIME_END(cpu_external_product_to_dft, params_glwe, params_ggsw);

	glwe_dft_to_coef(module, ext_prod_observed, result_dft);
	normalize_glwe(module, ext_prod_observed, ext_prod_observed);
	glwe_secret_decrypt(module, phase_observed, sk_glwe_prep, ext_prod_observed);
	biv_to_univ_rnx(params_glwe, um_observed_univ_rnx, phase_observed);

	//Computes u*m manually
	univ_coefs_to_dft(module, u_univ_dft, u_univ);
	pvda_svp_apply_dft(module, um_dft, ggsw_params_l_tilde_a(params_ggsw), u_univ_dft, m);
	biv_dft_to_coefs(module, params_glwe, um, um_dft);
	pvda_vec_znx_normalize_base2k(module, params_glwe->kappa, um, um);
	biv_to_univ_rnx(params_glwe, um_univ_rnx, um);

	pvda_assert_polynomial_distance(params_glwe, um_observed_univ_rnx, um_univ_rnx, err_length, critical_err_length);

	// Clean up
	delete_biv(m);
	delete_univ(u_univ);
	delete_univ_dft(u_univ_dft);
	delete_biv(phase_observed);
	delete_biv(um);
	delete_univ_rnx(um_univ_rnx);
	free(um_dft);
	delete_univ_rnx(um_observed_univ_rnx);

	delete_glwe(ext_prod_observed);
	delete_glwe_dft(result_dft);
	delete_glwe(glwe_tilde);
	delete_ggsw(ggsw);
	delete_ggsw_prep(ggsw_prep);

	delete_glwe_secret_key(sk_ggsw);
	delete_glwe_secret_key_prepared(sk_glwe_prep);

	DELETE_PVDA_PARAMS_GGSW;
}

/** Prepared GGSW external product — result directly in coefficient domain */
PvdaParamTest(ggsw_external_product, prepared_without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSW(param);

	params_glwe->fast_uniform_nb_bits = 0;
	sigma                             = 0;
	double err_length                 = glwe_bivariate_epsilon(params_glwe) + 3 * sigma + 3 * DBL_EPSILON;
	double critical_err_length        = glwe_bivariate_epsilon(params_glwe) + 5 * sigma + 5 * DBL_EPSILON;

	GLWESecretKey* sk_ggsw              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_glwe_prep = alloc_glwe_secret_key_prepared(params_glwe);
	GGSWCiphertext* ggsw                = new_ggsw(params_ggsw);
	GGSWCiphertextPrep* ggsw_prep       = new_ggsw_prep(params_ggsw);
	GLWECiphertext* glwe_tilde          = new_glwe(params_glwe);
	GLWECiphertext* ext_prod_observed   = new_glwe(params_glwe);
	PolyUniv* u_univ                    = new_univ(params_glwe);
	PolyBiv* m                          = new_biv(params_glwe);

	PolyBiv* phase_observed           = new_biv(params_glwe);
	PolyUnivRnX* um_observed_univ_rnx = new_univ_rnx(params_glwe);
	PolyUnivDFT* u_univ_dft           = new_univ_dft(module);
	PolyBivDFT* um_dft                = new_biv_dft(params_glwe);
	PolyBiv* um                       = new_biv(params_glwe);
	PolyUnivRnX* um_univ_rnx          = new_univ_rnx(params_glwe);

	uniform_glwe_secret_key(module, sk_ggsw, 3);
	glwe_sk_prepare(module, sk_glwe_prep, sk_ggsw);

	uniform_random_pol_znx(u_univ, params_glwe->nn, params_glwe->kappa);
	uniform_random_biv_poly(params_glwe, m, 1);

	glwe_secret_encrypt_phase(module, glwe_tilde, sk_glwe_prep, m);
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, u_univ);
	ggsw_prepare(module, ggsw_prep, ggsw);

	PVDA_TIME_START(cpu_external_product);
	ggsw_external_product(module, ext_prod_observed, glwe_tilde, ggsw_prep);
	PVDA_TIME_END(cpu_external_product, params_glwe, params_ggsw);

	normalize_glwe(module, ext_prod_observed, ext_prod_observed);
	glwe_secret_decrypt(module, phase_observed, sk_glwe_prep, ext_prod_observed);
	biv_to_univ_rnx(params_glwe, um_observed_univ_rnx, phase_observed);

	//Computes u*m manually
	univ_coefs_to_dft(module, u_univ_dft, u_univ);
	pvda_svp_apply_dft(module, um_dft, ggsw_params_l_tilde_a(params_ggsw), u_univ_dft, m);
	biv_dft_to_coefs(module, params_glwe, um, um_dft);
	pvda_vec_znx_normalize_base2k(module, params_glwe->kappa, um, um);
	biv_to_univ_rnx(params_glwe, um_univ_rnx, um);

	pvda_assert_polynomial_distance(params_glwe, um_observed_univ_rnx, um_univ_rnx, err_length, critical_err_length);

	// Clean up
	delete_biv(m);
	delete_univ(u_univ);
	delete_univ_dft(u_univ_dft);
	delete_biv(phase_observed);
	delete_biv(um);
	delete_univ_rnx(um_univ_rnx);
	free(um_dft);
	delete_univ_rnx(um_observed_univ_rnx);

	delete_glwe(ext_prod_observed);
	delete_glwe(glwe_tilde);
	delete_ggsw(ggsw);
	delete_ggsw_prep(ggsw_prep);

	delete_glwe_secret_key(sk_ggsw);
	delete_glwe_secret_key_prepared(sk_glwe_prep);

	DELETE_PVDA_PARAMS_GGSW;
}

#ifdef ENABLE_CUDA
/** GPU variant: external product runs on GPU, normalize + decrypt on CPU */
PvdaParamTest(ggsw_external_product, gpu_without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSW(param);

	params_glwe->fast_uniform_nb_bits = 0;
	sigma                             = 0;
	double err_length                 = glwe_bivariate_epsilon(params_glwe) + 3 * sigma + 3 * DBL_EPSILON;
	double critical_err_length        = glwe_bivariate_epsilon(params_glwe) + 5 * sigma + 5 * DBL_EPSILON;

	GLWESecretKey* sk_ggsw              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_glwe_prep = alloc_glwe_secret_key_prepared(params_glwe);
	GGSWCiphertext* ggsw                = new_ggsw(params_ggsw);
	GLWECiphertext* glwe_tilde          = new_glwe(params_glwe);
	GLWECiphertext* ext_prod_observed   = new_glwe(params_glwe);
	PolyUniv* u_univ                    = new_univ(params_glwe);
	PolyBiv* m                          = new_biv(params_glwe);

	PolyBiv* phase_observed           = new_biv(params_glwe);
	PolyUnivRnX* um_observed_univ_rnx = new_univ_rnx(params_glwe);
	PolyUnivDFT* u_univ_dft           = new_univ_dft(module);
	PolyBivDFT* um_dft                = new_biv_dft(params_glwe);
	PolyBiv* um                       = new_biv(params_glwe);
	PolyUnivRnX* um_univ_rnx          = new_univ_rnx(params_glwe);

	uniform_glwe_secret_key(module, sk_ggsw, 3);
	glwe_sk_prepare(module, sk_glwe_prep, sk_ggsw);

	uniform_random_pol_znx(u_univ, params_glwe->nn, params_glwe->kappa);
	uniform_random_biv_poly(params_glwe, m, 1);

	glwe_secret_encrypt_phase(module, glwe_tilde, sk_glwe_prep, m);
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, u_univ);

	// GPU external product:
	// Upload ciphertexts, run NTT-based VMP on device, download result.
	gpu_ntt_initialize(params_glwe->nn);
	//PVDA_TIME_START(gpu_unprepared_external_product);
	size_t result_elems              = (size_t)(glwe_params_n_limbs(params_glwe) * params_glwe->nn);
	GLWECiphertext gpu_glwe_tilde    = {.params = params_glwe, .vec = pvda_glwe_to_device(glwe_tilde)};
	GGSWCiphertext gpu_ggsw          = {.params = params_ggsw, .mat = pvda_ggsw_to_device(ggsw)};
	GLWECiphertext gpu_result        = {.params = params_glwe, .vec = pvda_gpu_alloc(result_elems)};

	cr_assert_not_null(gpu_glwe_tilde.vec, "pvda_glwe_to_device failed");
	cr_assert_not_null(gpu_ggsw.mat,       "pvda_ggsw_to_device failed");
	cr_assert_not_null(gpu_result.vec,     "pvda_gpu_alloc for result failed");

	PVDA_TIME_START(gpu_unprepared_external_product);
	ggsw_unprepared_external_product(module, &gpu_result, &gpu_glwe_tilde, &gpu_ggsw);
	PVDA_TIME_END(gpu_unprepared_external_product, params_glwe, params_ggsw);

	// Download result then normalize and decrypt on CPU
	pvda_glwe_from_device(ext_prod_observed, gpu_result.vec);
	//PVDA_TIME_END(gpu_unprepared_external_product, params_glwe, params_ggsw);
	normalize_glwe(module, ext_prod_observed, ext_prod_observed);
	glwe_secret_decrypt(module, phase_observed, sk_glwe_prep, ext_prod_observed);
	biv_to_univ_rnx(params_glwe, um_observed_univ_rnx, phase_observed);

	// Reference: u*m computed manually
	univ_coefs_to_dft(module, u_univ_dft, u_univ);
	pvda_svp_apply_dft(module, um_dft, ggsw_params_l_tilde_a(params_ggsw), u_univ_dft, m);
	biv_dft_to_coefs(module, params_glwe, um, um_dft);
	pvda_vec_znx_normalize_base2k(module, params_glwe->kappa, um, um);
	biv_to_univ_rnx(params_glwe, um_univ_rnx, um);

	pvda_assert_polynomial_distance(params_glwe, um_observed_univ_rnx, um_univ_rnx, err_length, critical_err_length);

	// Cleanup GPU buffers
	pvda_gpu_free(gpu_glwe_tilde.vec);
	pvda_gpu_free(gpu_ggsw.mat);
	pvda_gpu_free(gpu_result.vec);

	// Cleanup CPU allocations
	delete_biv(m);
	delete_univ(u_univ);
	delete_univ_dft(u_univ_dft);
	delete_biv(phase_observed);
	delete_biv(um);
	delete_univ_rnx(um_univ_rnx);
	free(um_dft);
	delete_univ_rnx(um_observed_univ_rnx);

	delete_glwe(ext_prod_observed);
	delete_glwe(glwe_tilde);
	delete_ggsw(ggsw);

	delete_glwe_secret_key(sk_ggsw);
	delete_glwe_secret_key_prepared(sk_glwe_prep);

	DELETE_PVDA_PARAMS_GGSW;
}

/** GPU variant: ggsw_external_product_to_dft runs on GPU — result (int64_t in coef domain) downloaded directly */
PvdaParamTest(ggsw_external_product, gpu_to_dft_without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSW(param);

	params_glwe->fast_uniform_nb_bits = 0;
	sigma                             = 0;
	double err_length          = glwe_bivariate_epsilon(params_glwe) + 3 * sigma + 3 * DBL_EPSILON;
	double critical_err_length = glwe_bivariate_epsilon(params_glwe) + 5 * sigma + 5 * DBL_EPSILON;

	GLWESecretKey* sk_ggsw              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_glwe_prep = alloc_glwe_secret_key_prepared(params_glwe);
	GGSWCiphertext* ggsw                = new_ggsw(params_ggsw);
	GLWECiphertext* glwe_tilde          = new_glwe(params_glwe);
	GLWECiphertext* ext_prod_observed   = new_glwe(params_glwe);
	PolyUniv* u_univ                    = new_univ(params_glwe);
	PolyBiv* m                          = new_biv(params_glwe);

	PolyBiv* phase_observed           = new_biv(params_glwe);
	PolyUnivRnX* um_observed_univ_rnx = new_univ_rnx(params_glwe);
	PolyUnivDFT* u_univ_dft           = new_univ_dft(module);
	PolyBivDFT* um_dft                = new_biv_dft(params_glwe);
	PolyBiv* um                       = new_biv(params_glwe);
	PolyUnivRnX* um_univ_rnx          = new_univ_rnx(params_glwe);

	uniform_glwe_secret_key(module, sk_ggsw, 3);
	glwe_sk_prepare(module, sk_glwe_prep, sk_ggsw);

	uniform_random_pol_znx(u_univ, params_glwe->nn, params_glwe->kappa);
	uniform_random_biv_poly(params_glwe, m, 1);

	glwe_secret_encrypt_phase(module, glwe_tilde, sk_glwe_prep, m);
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, u_univ);

	gpu_ntt_initialize(params_glwe->nn);
	size_t result_elems = (size_t)(glwe_params_n_limbs(params_glwe) * params_glwe->nn);

	// GPU-backed structs: GLWE on device; GGSW NTT-prepared on device via ggsw_prepare_gpu
	GLWECiphertext gpu_glwe_tilde    = {.params = params_glwe, .vec = pvda_glwe_to_device(glwe_tilde)};
	GGSWCiphertextPrep gpu_ggsw_prep = {.params = params_ggsw, .mat = NULL};
	ggsw_prepare_gpu(&gpu_ggsw_prep, ggsw);
	// NTT-domain result buffer (int64_t on device, cast as VecBivDFT* for the struct)
	GLWECiphertextDFT gpu_result_dft = {.params = params_glwe, .vec = (VecBivDFT*)pvda_gpu_alloc(result_elems)};
	// Coefficient-domain output buffer (filled by glwe_dft_to_coef_gpu)
	GLWECiphertext gpu_result_coef   = {.params = params_glwe, .vec = pvda_gpu_alloc(result_elems)};

	cr_assert_not_null(gpu_glwe_tilde.vec,  "pvda_glwe_to_device failed");
	cr_assert_not_null(gpu_ggsw_prep.mat,   "ggsw_prepare_gpu failed");
	cr_assert_not_null(gpu_result_dft.vec,  "pvda_gpu_alloc for NTT result failed");
	cr_assert_not_null(gpu_result_coef.vec, "pvda_gpu_alloc for coef result failed");

	PVDA_TIME_START(gpu_external_product_to_dft);
	ggsw_external_product_to_dft(module, &gpu_result_dft, &gpu_glwe_tilde, &gpu_ggsw_prep);
	glwe_dft_to_coef_gpu(&gpu_result_coef, &gpu_result_dft);
	PVDA_TIME_END(gpu_external_product_to_dft, params_glwe, params_ggsw);
	pvda_glwe_from_device(ext_prod_observed, gpu_result_coef.vec);

	normalize_glwe(module, ext_prod_observed, ext_prod_observed);
	glwe_secret_decrypt(module, phase_observed, sk_glwe_prep, ext_prod_observed);
	biv_to_univ_rnx(params_glwe, um_observed_univ_rnx, phase_observed);

	univ_coefs_to_dft(module, u_univ_dft, u_univ);
	pvda_svp_apply_dft(module, um_dft, ggsw_params_l_tilde_a(params_ggsw), u_univ_dft, m);
	biv_dft_to_coefs(module, params_glwe, um, um_dft);
	pvda_vec_znx_normalize_base2k(module, params_glwe->kappa, um, um);
	biv_to_univ_rnx(params_glwe, um_univ_rnx, um);

	pvda_assert_polynomial_distance(params_glwe, um_observed_univ_rnx, um_univ_rnx, err_length, critical_err_length);

	pvda_gpu_free(gpu_glwe_tilde.vec);
	pvda_gpu_free((int64_t*)gpu_ggsw_prep.mat);
	pvda_gpu_free((int64_t*)gpu_result_dft.vec);
	pvda_gpu_free(gpu_result_coef.vec);

	delete_biv(m);
	delete_univ(u_univ);
	delete_univ_dft(u_univ_dft);
	delete_biv(phase_observed);
	delete_biv(um);
	delete_univ_rnx(um_univ_rnx);
	free(um_dft);
	delete_univ_rnx(um_observed_univ_rnx);

	delete_glwe(ext_prod_observed);
	delete_glwe(glwe_tilde);
	delete_ggsw(ggsw);

	delete_glwe_secret_key(sk_ggsw);
	delete_glwe_secret_key_prepared(sk_glwe_prep);

	DELETE_PVDA_PARAMS_GGSW;
}

/** GPU variant: ggsw_external_product (prepared path) runs on GPU */
PvdaParamTest(ggsw_external_product, gpu_prepared_without_error, default_params_fn)
{
	INIT_PVDA_PARAMS_GGSW(param);

	params_glwe->fast_uniform_nb_bits = 0;
	sigma                             = 0;
	double err_length          = glwe_bivariate_epsilon(params_glwe) + 3 * sigma + 3 * DBL_EPSILON;
	double critical_err_length = glwe_bivariate_epsilon(params_glwe) + 5 * sigma + 5 * DBL_EPSILON;

	GLWESecretKey* sk_ggsw              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_glwe_prep = alloc_glwe_secret_key_prepared(params_glwe);
	GGSWCiphertext* ggsw                = new_ggsw(params_ggsw);
	GLWECiphertext* glwe_tilde          = new_glwe(params_glwe);
	GLWECiphertext* ext_prod_observed   = new_glwe(params_glwe);
	PolyUniv* u_univ                    = new_univ(params_glwe);
	PolyBiv* m                          = new_biv(params_glwe);

	PolyBiv* phase_observed           = new_biv(params_glwe);
	PolyUnivRnX* um_observed_univ_rnx = new_univ_rnx(params_glwe);
	PolyUnivDFT* u_univ_dft           = new_univ_dft(module);
	PolyBivDFT* um_dft                = new_biv_dft(params_glwe);
	PolyBiv* um                       = new_biv(params_glwe);
	PolyUnivRnX* um_univ_rnx          = new_univ_rnx(params_glwe);

	uniform_glwe_secret_key(module, sk_ggsw, 3);
	glwe_sk_prepare(module, sk_glwe_prep, sk_ggsw);

	uniform_random_pol_znx(u_univ, params_glwe->nn, params_glwe->kappa);
	uniform_random_biv_poly(params_glwe, m, 1);

	glwe_secret_encrypt_phase(module, glwe_tilde, sk_glwe_prep, m);
	ggsw_secret_encrypt(module, ggsw, sk_glwe_prep, u_univ);

	gpu_ntt_initialize(params_glwe->nn);
	size_t result_elems = (size_t)(glwe_params_n_limbs(params_glwe) * params_glwe->nn);

	GLWECiphertext gpu_glwe_tilde    = {.params = params_glwe, .vec = pvda_glwe_to_device(glwe_tilde)};
	GGSWCiphertextPrep gpu_ggsw_prep = {.params = params_ggsw, .mat = NULL};
	ggsw_prepare_gpu(&gpu_ggsw_prep, ggsw);
	GLWECiphertext gpu_result = {.params = params_glwe, .vec = pvda_gpu_alloc(result_elems)};

	cr_assert_not_null(gpu_glwe_tilde.vec, "pvda_glwe_to_device failed");
	cr_assert_not_null(gpu_ggsw_prep.mat,  "ggsw_prepare_gpu failed");
	cr_assert_not_null(gpu_result.vec,     "pvda_gpu_alloc for result failed");

	PVDA_TIME_START(gpu_external_product);
	ggsw_external_product(module, &gpu_result, &gpu_glwe_tilde, &gpu_ggsw_prep);
	PVDA_TIME_END(gpu_external_product, params_glwe, params_ggsw);

	pvda_glwe_from_device(ext_prod_observed, gpu_result.vec);
	normalize_glwe(module, ext_prod_observed, ext_prod_observed);
	glwe_secret_decrypt(module, phase_observed, sk_glwe_prep, ext_prod_observed);
	biv_to_univ_rnx(params_glwe, um_observed_univ_rnx, phase_observed);

	univ_coefs_to_dft(module, u_univ_dft, u_univ);
	pvda_svp_apply_dft(module, um_dft, ggsw_params_l_tilde_a(params_ggsw), u_univ_dft, m);
	biv_dft_to_coefs(module, params_glwe, um, um_dft);
	pvda_vec_znx_normalize_base2k(module, params_glwe->kappa, um, um);
	biv_to_univ_rnx(params_glwe, um_univ_rnx, um);

	pvda_assert_polynomial_distance(params_glwe, um_observed_univ_rnx, um_univ_rnx, err_length, critical_err_length);

	pvda_gpu_free(gpu_glwe_tilde.vec);
	pvda_gpu_free((int64_t*)gpu_ggsw_prep.mat);
	pvda_gpu_free(gpu_result.vec);

	delete_biv(m);
	delete_univ(u_univ);
	delete_univ_dft(u_univ_dft);
	delete_biv(phase_observed);
	delete_biv(um);
	delete_univ_rnx(um_univ_rnx);
	free(um_dft);
	delete_univ_rnx(um_observed_univ_rnx);

	delete_glwe(ext_prod_observed);
	delete_glwe(glwe_tilde);
	delete_ggsw(ggsw);

	delete_glwe_secret_key(sk_ggsw);
	delete_glwe_secret_key_prepared(sk_glwe_prep);

	DELETE_PVDA_PARAMS_GGSW;
}
#endif
