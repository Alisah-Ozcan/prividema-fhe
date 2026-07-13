#include <criterion/criterion.h>
#include <stdint.h>
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
#include "core/glwe/glwe_arithmetic.h"
#include "core/glwe/glwe_ciphertext.h"
#include "core/glwe/glwe_key.h"
#include "core/glwe/glwe_params.h"
#include "core/glwe/glwe_transform_key.h"
#include "core/glwe/univariate_polynomial.h"
#include "glwegadget_utils.h"
#include "gpu/common/gpu_stream.h"
#include "gpu/host/ggsw_external_product_gpu.h"
#include "gpu/host/vec_znx_rotate_automorphism_host.h"
#include "maths_structures.h"
#include "test_utils.h"

// glwegadget_automorphism, isolated: does the #ifdef ENABLE_CUDA GPU dispatch
// (glwegadget_automorphism / glwe_trace_expand's device branches, wired
// around the vec_znx_rotate/automorphism GPU kernels, and
// prepare_automorphism_key's GPU KSK preparation) match the CPU reference,
// given an all-device glwe/result/KSK operand set?
//
// Key gotcha this file guards against: an automorphism KSK entry's ->mat
// must be *NTT-prepared on the GPU* (glwegadget_prepare's device dispatch,
// fed a device-resident RAW gadget ciphertext) — never produced by
// byte-copying an already CPU-DFT-prepared matrix to device. CPU
// glwegadget_prepare computes a spqlios DFT-domain (double) representation;
// GPU glwegadget_prepare_gpu computes an NTT-domain (int64 residue)
// representation. Same footprint, completely different content — a naive
// byte-copy silently produces a "valid-looking" but cryptographically wrong
// KSK (see ksk_gpu_prepared_from_raw_matches_cpu below).

#define NN    512
#define KAPPA 18

// Prepares a k=1 automorphism KSK entirely on the GPU: pre-allocates enc_s[0]
// as a device Prep placeholder (pvda_new_glwegadget_prep_device) so
// prepare_automorphism_key's device dispatch NTT-prepares it on the GPU from
// a raw upload, instead of DFT-preparing on the CPU.
static GLWEAutomorphismKSK* new_automorphism_ksk_gpu(const MODULE* module, GLWEGadgetParams* params_glwegadget,
                                                     const GLWESecretKeyPrepared* sk_prep, int auto_p)
{
	GLWEAutomorphismKSK* ksk = new_automorphism_ksk(params_glwegadget);
	delete_glwegadget_prep(ksk->enc_s[0]);
	ksk->enc_s[0] = pvda_new_glwegadget_prep_device(params_glwegadget);
	int pc        = prepare_automorphism_key(module, ksk, sk_prep, auto_p);
	cr_assert_eq(pc, 0, "prepare_automorphism_key (GPU) failed");
	cr_assert(pvda_is_device_pointer(ksk->enc_s[0]->mat), "expected enc_s[0]->mat to end up device-resident");
	return ksk;
}

Test(gpu_glwegadget_automorphism, cpu_baseline_sanity)
{
	srand(7);

	MODULE* module = pvda_new_module_info(NN);
	cr_assert_not_null(module, "MODULE allocation failed");

	GLWEParams* params_glwe             = new_glwe_params(NN, 1, KAPPA, 8, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEGadgetParams* params_glwegadget = new_glwegadget_params(params_glwe, KAPPA, 8);

	GLWESecretKey* sk              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params_glwe);
	uniform_glwe_secret_key(module, sk, 1);
	glwe_sk_prepare(module, sk_prep, sk);

	GLWEAutomorphismKSK* ksk = new_automorphism_ksk(params_glwegadget);
	int auto_p               = 5;
	int pc                   = prepare_automorphism_key(module, ksk, sk_prep, auto_p);
	cr_assert_eq(pc, 0, "prepare_automorphism_key failed");

	PolyUnivTnX* m_tnx = new_univ_tnx(params_glwe);
	uniform_random_pol_znx((PolyUniv*)m_tnx, NN, 64);

	GLWECiphertext* glwe_ct = new_glwe(params_glwe);
	glwe_secret_encrypt_tnx(module, glwe_ct, sk_prep, m_tnx);

	GLWECiphertext* glwe_res_cpu  = new_glwe(params_glwe);
	GLWECiphertext* glwe_norm_cpu = new_glwe(params_glwe);
	glwegadget_automorphism(module, glwe_res_cpu, ksk, glwe_ct);
	normalize_glwe(module, glwe_norm_cpu, glwe_res_cpu);

	PolyBiv* m_auto_cpu         = new_biv(params_glwe);
	PolyUnivTnX* m_observed_cpu = new_univ_tnx(params_glwe);
	glwe_secret_decrypt(module, m_auto_cpu, sk_prep, glwe_norm_cpu);
	biv_to_univ_tnx(params_glwe, m_observed_cpu, m_auto_cpu);

	PolyUnivTnX* m_expected = new_univ_tnx(params_glwe);
	pvda_znx_automorphism(module, auto_p, m_expected, m_tnx);

	int64_t decomp_noise_bits = info_bits_half_prod(params_glwe, params_glwegadget);
	for (int p = 0; p < NN; ++p) assert_tnx_close_enough(m_observed_cpu[p], m_expected[p], decomp_noise_bits);

	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_automorphism_ksk(ksk);

	delete_univ_tnx(m_tnx);
	delete_univ_tnx(m_expected);
	delete_univ_tnx(m_observed_cpu);

	delete_glwe(glwe_ct);
	delete_glwe(glwe_res_cpu);
	delete_glwe(glwe_norm_cpu);

	delete_biv(m_auto_cpu);

	pvda_delete_module_info(module);
	delete_glwegadget_params(params_glwegadget);
	delete_glwe_params(params_glwe);
}

// The main end-to-end regression: glwegadget_automorphism's GPU dispatch,
// fed an all-device glwe/result and a GPU-NTT-prepared KSK (via
// new_automorphism_ksk_gpu), must decrypt to the same message as the CPU
// reference.
Test(gpu_glwegadget_automorphism, matches_cpu_no_noise)
{
	srand(7);

	MODULE* module = pvda_new_module_info(NN);
	cr_assert_not_null(module, "MODULE allocation failed");

	GLWEParams* params_glwe             = new_glwe_params(NN, 1, KAPPA, 8, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEGadgetParams* params_glwegadget = new_glwegadget_params(params_glwe, KAPPA, 8);

	GLWESecretKey* sk              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params_glwe);
	uniform_glwe_secret_key(module, sk, 1);
	glwe_sk_prepare(module, sk_prep, sk);

	int auto_p = 5;

	PolyUnivTnX* m_tnx = new_univ_tnx(params_glwe);
	uniform_random_pol_znx((PolyUniv*)m_tnx, NN, 64);

	GLWECiphertext* glwe_ct = new_glwe(params_glwe);
	glwe_secret_encrypt_tnx(module, glwe_ct, sk_prep, m_tnx);

	PolyUnivTnX* m_expected = new_univ_tnx(params_glwe);
	pvda_znx_automorphism(module, auto_p, m_expected, m_tnx);
	int64_t decomp_noise_bits = info_bits_half_prod(params_glwe, params_glwegadget);

	// GPU-prepared KSK + device glwe_ct/result — see new_automorphism_ksk_gpu.
	GLWEAutomorphismKSK* ksk = new_automorphism_ksk_gpu(module, params_glwegadget, sk_prep, auto_p);

	int64_t* d_glwe_ct_vec           = pvda_glwe_to_device(glwe_ct);
	GLWECiphertext glwe_ct_dev       = {.params = params_glwe, .vec = (VecBiv*)d_glwe_ct_vec};
	GLWECiphertext* glwe_res_gpu_dev = pvda_new_glwe_device(params_glwe);

	glwegadget_automorphism(module, glwe_res_gpu_dev, ksk, &glwe_ct_dev);

	GLWECiphertext* glwe_res_gpu = new_glwe(params_glwe);
	pvda_glwe_from_device(glwe_res_gpu, (const int64_t*)glwe_res_gpu_dev->vec);

	GLWECiphertext* glwe_norm_gpu = new_glwe(params_glwe);
	normalize_glwe(module, glwe_norm_gpu, glwe_res_gpu);

	PolyBiv* m_auto_gpu         = new_biv(params_glwe);
	PolyUnivTnX* m_observed_gpu = new_univ_tnx(params_glwe);
	glwe_secret_decrypt(module, m_auto_gpu, sk_prep, glwe_norm_gpu);
	biv_to_univ_tnx(params_glwe, m_observed_gpu, m_auto_gpu);

	for (int p = 0; p < NN; ++p) assert_tnx_close_enough(m_observed_gpu[p], m_expected[p], decomp_noise_bits);

	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_automorphism_ksk(ksk);

	delete_univ_tnx(m_tnx);
	delete_univ_tnx(m_expected);
	delete_univ_tnx(m_observed_gpu);

	delete_glwe(glwe_ct);
	delete_glwe(glwe_res_gpu);
	delete_glwe(glwe_norm_gpu);
	delete_glwe(glwe_res_gpu_dev);

	delete_biv(m_auto_gpu);

	pvda_delete_module_info(module);
	delete_glwegadget_params(params_glwegadget);
	delete_glwe_params(params_glwe);
}

// Regression test for the actual bug found during development: byte-copying
// an already CPU-DFT-prepared KSK matrix to device produces a matrix that
// LOOKS valid (right size, right pointer) but is cryptographically garbage
// when read back by the GPU's NTT-domain VMP. Preparing from a raw upload
// (glwegadget_prepare's own device dispatch) is the only correct way.
Test(gpu_glwegadget_automorphism, ksk_gpu_prepared_from_raw_matches_cpu)
{
	srand(11);

	MODULE* module = pvda_new_module_info(NN);
	cr_assert_not_null(module, "MODULE allocation failed");

	GLWEParams* params_glwe             = new_glwe_params(NN, 1, KAPPA, 8, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEGadgetParams* params_glwegadget = new_glwegadget_params(params_glwe, KAPPA, 8);

	GLWESecretKey* sk              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params_glwe);
	uniform_glwe_secret_key(module, sk, 1);
	glwe_sk_prepare(module, sk_prep, sk);

	int auto_p = 5;

	GLWEAutomorphismKSK* ksk_cpu = new_automorphism_ksk(params_glwegadget);
	cr_assert_eq(prepare_automorphism_key(module, ksk_cpu, sk_prep, auto_p), 0, "CPU KSK prep failed");

	GLWEAutomorphismKSK* ksk_gpu = new_automorphism_ksk_gpu(module, params_glwegadget, sk_prep, auto_p);

	PolyUnivTnX* m_tnx = new_univ_tnx(params_glwe);
	uniform_random_pol_znx((PolyUniv*)m_tnx, NN, 64);
	GLWECiphertext* glwe_ct = new_glwe(params_glwe);
	glwe_secret_encrypt_tnx(module, glwe_ct, sk_prep, m_tnx);

	GLWECiphertext* result_cpu = new_glwe(params_glwe);
	glwegadget_automorphism(module, result_cpu, ksk_cpu, glwe_ct);

	int64_t* d_glwe_ct_vec         = pvda_glwe_to_device(glwe_ct);
	GLWECiphertext glwe_ct_dev     = {.params = params_glwe, .vec = (VecBiv*)d_glwe_ct_vec};
	GLWECiphertext* result_gpu_dev = pvda_new_glwe_device(params_glwe);
	glwegadget_automorphism(module, result_gpu_dev, ksk_gpu, &glwe_ct_dev);

	GLWECiphertext* result_gpu = new_glwe(params_glwe);
	pvda_glwe_from_device(result_gpu, (const int64_t*)result_gpu_dev->vec);

	// glwegadget_automorphism's CPU (spqlios DFT half-product) and GPU (NTT
	// half-product) paths are different algorithms for the same operation —
	// not expected to be bit-exact, but must be decryption-close.
	GLWECiphertext* glwe_norm_cpu = new_glwe(params_glwe);
	GLWECiphertext* glwe_norm_gpu = new_glwe(params_glwe);
	normalize_glwe(module, glwe_norm_cpu, result_cpu);
	normalize_glwe(module, glwe_norm_gpu, result_gpu);

	PolyBiv* m_cpu              = new_biv(params_glwe);
	PolyBiv* m_gpu              = new_biv(params_glwe);
	PolyUnivTnX* m_observed_cpu = new_univ_tnx(params_glwe);
	PolyUnivTnX* m_observed_gpu = new_univ_tnx(params_glwe);
	glwe_secret_decrypt(module, m_cpu, sk_prep, glwe_norm_cpu);
	glwe_secret_decrypt(module, m_gpu, sk_prep, glwe_norm_gpu);
	biv_to_univ_tnx(params_glwe, m_observed_cpu, m_cpu);
	biv_to_univ_tnx(params_glwe, m_observed_gpu, m_gpu);

	int64_t decomp_noise_bits = info_bits_half_prod(params_glwe, params_glwegadget);
	for (int p = 0; p < NN; ++p) assert_tnx_close_enough(m_observed_cpu[p], m_observed_gpu[p], decomp_noise_bits);

	delete_biv(m_cpu);
	delete_biv(m_gpu);
	delete_univ_tnx(m_observed_cpu);
	delete_univ_tnx(m_observed_gpu);
	delete_glwe(glwe_norm_cpu);
	delete_glwe(glwe_norm_gpu);
	delete_glwe(result_cpu);
	delete_glwe(result_gpu);
	delete_glwe(result_gpu_dev);
	delete_glwe(glwe_ct);
	delete_univ_tnx(m_tnx);
	delete_automorphism_ksk(ksk_cpu);
	delete_automorphism_ksk(ksk_gpu);
	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	pvda_delete_module_info(module);
	delete_glwegadget_params(params_glwegadget);
	delete_glwe_params(params_glwe);
}

// gpu_active_stream override (pvda_gpu_stream_push/pop) must not change the
// result — same invariant checked for the raw kernels in
// test_gpu_vec_znx_rotate_automorphism.c, checked here at the
// glwegadget_automorphism integration level too.
Test(gpu_glwegadget_automorphism, matches_cpu_on_pushed_stream)
{
	srand(13);

	MODULE* module = pvda_new_module_info(NN);
	cr_assert_not_null(module, "MODULE allocation failed");

	GLWEParams* params_glwe             = new_glwe_params(NN, 1, KAPPA, 8, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEGadgetParams* params_glwegadget = new_glwegadget_params(params_glwe, KAPPA, 8);

	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params_glwe);
	GLWESecretKey* sk              = alloc_glwe_secret_key(params_glwe);
	uniform_glwe_secret_key(module, sk, 1);
	glwe_sk_prepare(module, sk_prep, sk);

	int auto_p = 5;

	PolyUnivTnX* m_tnx = new_univ_tnx(params_glwe);
	uniform_random_pol_znx((PolyUniv*)m_tnx, NN, 64);
	GLWECiphertext* glwe_ct = new_glwe(params_glwe);
	glwe_secret_encrypt_tnx(module, glwe_ct, sk_prep, m_tnx);

	PolyUnivTnX* m_expected = new_univ_tnx(params_glwe);
	pvda_znx_automorphism(module, auto_p, m_expected, m_tnx);
	int64_t decomp_noise_bits = info_bits_half_prod(params_glwe, params_glwegadget);

	void* stream = pvda_gpu_stream_create();
	cr_assert_not_null(stream, "pvda_gpu_stream_create failed");
	pvda_gpu_stream_push(stream);

	GLWEAutomorphismKSK* ksk = new_automorphism_ksk_gpu(module, params_glwegadget, sk_prep, auto_p);

	int64_t* d_glwe_ct_vec           = pvda_glwe_to_device(glwe_ct);
	GLWECiphertext glwe_ct_dev       = {.params = params_glwe, .vec = (VecBiv*)d_glwe_ct_vec};
	GLWECiphertext* glwe_res_gpu_dev = pvda_new_glwe_device(params_glwe);

	glwegadget_automorphism(module, glwe_res_gpu_dev, ksk, &glwe_ct_dev);

	GLWECiphertext* glwe_res_gpu = new_glwe(params_glwe);
	pvda_glwe_from_device(glwe_res_gpu, (const int64_t*)glwe_res_gpu_dev->vec);

	pvda_gpu_stream_pop();
	cr_assert_null(pvda_gpu_stream_get_active(), "pop did not restore the default stream");
	pvda_gpu_stream_destroy(stream);

	GLWECiphertext* glwe_norm_gpu = new_glwe(params_glwe);
	normalize_glwe(module, glwe_norm_gpu, glwe_res_gpu);

	PolyBiv* m_auto_gpu         = new_biv(params_glwe);
	PolyUnivTnX* m_observed_gpu = new_univ_tnx(params_glwe);
	glwe_secret_decrypt(module, m_auto_gpu, sk_prep, glwe_norm_gpu);
	biv_to_univ_tnx(params_glwe, m_observed_gpu, m_auto_gpu);

	for (int p = 0; p < NN; ++p) assert_tnx_close_enough(m_observed_gpu[p], m_expected[p], decomp_noise_bits);

	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_automorphism_ksk(ksk);
	delete_univ_tnx(m_tnx);
	delete_univ_tnx(m_expected);
	delete_univ_tnx(m_observed_gpu);
	delete_glwe(glwe_ct);
	delete_glwe(glwe_res_gpu);
	delete_glwe(glwe_norm_gpu);
	delete_glwe(glwe_res_gpu_dev);
	delete_biv(m_auto_gpu);
	pvda_delete_module_info(module);
	delete_glwegadget_params(params_glwegadget);
	delete_glwe_params(params_glwe);
}

// Same DFT-vs-NTT regression as ksk_gpu_prepared_from_raw_matches_cpu, but
// for the OTHER KSK kind used by the PIR examples: GGSW KSKs prepared via
// generate_glwegad_to_ggsw_ksk + pvda_new_ggsw_prep_device, consumed by
// ggsw_external_product.
Test(gpu_glwegadget_automorphism, ggsw_ksk_gpu_prepared_from_raw_matches_cpu)
{
	srand(17);

	MODULE* module = pvda_new_module_info(NN);
	cr_assert_not_null(module, "MODULE allocation failed");

	GLWEParams* params_glwe = new_glwe_params(NN, 1, KAPPA, 8, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GGSWParams* params_ggsw = new_ggsw_params(params_glwe, 1, KAPPA, 8);

	GLWESecretKey* sk              = alloc_glwe_secret_key(params_glwe);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params_glwe);
	uniform_glwe_secret_key(module, sk, 1);
	glwe_sk_prepare(module, sk_prep, sk);

	GGSWCiphertextPrep* ggsw_ksks_cpu[1] = {NULL};
	cr_assert_eq(generate_glwegad_to_ggsw_ksk(module, ggsw_ksks_cpu, params_ggsw, sk_prep), 0,
	             "CPU GGSW KSK gen failed");

	GGSWCiphertextPrep* ggsw_ksks_gpu[1] = {pvda_new_ggsw_prep_device(params_ggsw)};
	cr_assert_eq(generate_glwegad_to_ggsw_ksk(module, ggsw_ksks_gpu, params_ggsw, sk_prep), 0,
	             "GPU GGSW KSK gen failed");
	cr_assert(pvda_is_device_pointer(ggsw_ksks_gpu[0]->mat),
	          "expected ggsw_ksks_gpu[0]->mat to end up device-resident");

	PolyUnivTnX* m_tnx = new_univ_tnx(params_glwe);
	uniform_random_pol_znx((PolyUniv*)m_tnx, NN, 64);
	GLWECiphertext* glwe_ct = new_glwe(params_glwe);
	glwe_secret_encrypt_tnx(module, glwe_ct, sk_prep, m_tnx);

	GLWECiphertext* result_cpu = new_glwe(params_glwe);
	cr_assert_eq(ggsw_external_product(module, result_cpu, glwe_ct, ggsw_ksks_cpu[0]), 0,
	             "CPU ggsw_external_product failed");

	int64_t* d_glwe_ct_vec         = pvda_glwe_to_device(glwe_ct);
	GLWECiphertext glwe_ct_dev     = {.params = params_glwe, .vec = (VecBiv*)d_glwe_ct_vec};
	GLWECiphertext* result_gpu_dev = pvda_new_glwe_device(params_glwe);
	cr_assert_eq(ggsw_external_product(module, result_gpu_dev, &glwe_ct_dev, ggsw_ksks_gpu[0]), 0,
	             "GPU ggsw_external_product failed");

	GLWECiphertext* result_gpu = new_glwe(params_glwe);
	pvda_glwe_from_device(result_gpu, (const int64_t*)result_gpu_dev->vec);

	GLWECiphertext* glwe_norm_cpu = new_glwe(params_glwe);
	GLWECiphertext* glwe_norm_gpu = new_glwe(params_glwe);
	normalize_glwe(module, glwe_norm_cpu, result_cpu);
	normalize_glwe(module, glwe_norm_gpu, result_gpu);

	PolyBiv* m_cpu               = new_biv(params_glwe);
	PolyBiv* m_gpu               = new_biv(params_glwe);
	PolyUnivTnX* m_observed_cpu2 = new_univ_tnx(params_glwe);
	PolyUnivTnX* m_observed_gpu2 = new_univ_tnx(params_glwe);
	glwe_secret_decrypt(module, m_cpu, sk_prep, glwe_norm_cpu);
	glwe_secret_decrypt(module, m_gpu, sk_prep, glwe_norm_gpu);
	biv_to_univ_tnx(params_glwe, m_observed_cpu2, m_cpu);
	biv_to_univ_tnx(params_glwe, m_observed_gpu2, m_gpu);

	// Generous fixed tolerance — this is a CPU-vs-GPU cross-check, not a
	// tight noise-budget assertion (no ready-made info_bits helper for a
	// bare GGSW KSK's decomposition here).
	int64_t decomp_noise_bits = 20;
	for (int p = 0; p < NN; ++p) assert_tnx_close_enough(m_observed_cpu2[p], m_observed_gpu2[p], decomp_noise_bits);

	delete_biv(m_cpu);
	delete_biv(m_gpu);
	delete_univ_tnx(m_observed_cpu2);
	delete_univ_tnx(m_observed_gpu2);
	delete_glwe(glwe_norm_cpu);
	delete_glwe(glwe_norm_gpu);
	delete_glwe(result_cpu);
	delete_glwe(result_gpu);
	delete_glwe(result_gpu_dev);
	delete_glwe(glwe_ct);
	delete_univ_tnx(m_tnx);
	delete_ggsw_prep(ggsw_ksks_cpu[0]);
	delete_ggsw_prep(ggsw_ksks_gpu[0]);
	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	pvda_delete_module_info(module);
	delete_ggsw_params(params_ggsw);
	delete_glwe_params(params_glwe);
}

// Regression test for a hypothesised remaining PIR bug: glwegadget_half_prod's
// GPU path (gpu_glwegadget_half_prod_device) takes a SINGLE `ncols` used both
// as the KSK/gadget matrix's own column count AND as the output write size —
// unlike the CPU path (pvda_vmp_apply_dft -> spqlios vmp_apply_dft), which
// keeps "res_size" (caller's buffer capacity) and the matrix's own "ncols"
// as two independent parameters. The real PIR example's automorphism KSKs
// are built from a GLWEGadgetParams whose own params_glwe (16 limbs) differs
// from the target GLWE's params_glwe (12 limbs) — exactly the case none of
// this file's other tests exercise (they all use one shared params_glwe for
// everything). This test allocates a canary buffer immediately after the
// (smaller) result buffer and checks it is untouched after the GPU call —
// if glwegadget_half_prod's GPU path writes glwegadget_prep_ct->params's
// column count (larger) instead of result's own column count (smaller),
// this canary gets clobbered.
Test(gpu_glwegadget_automorphism, half_prod_device_does_not_overflow_smaller_result)
{
	srand(19);

	MODULE* module = pvda_new_module_info(NN);
	cr_assert_not_null(module, "MODULE allocation failed");

	// KSK's own params_glwe: MORE limbs (16) than the result's (12) — mirrors
	// exemples_PIR_gpu.c's auto_ksk_params (params_glwe_autokey, 16 limbs)
	// vs row_exp_gad_params (row_exp_params, 12 limbs).
	GLWEParams* ksk_params_glwe             = new_glwe_params(NN, 1, KAPPA, 16, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEParams* result_params_glwe          = new_glwe_params(NN, 1, KAPPA, 12, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEGadgetParams* ksk_glwegadget_params = new_glwegadget_params(ksk_params_glwe, KAPPA, 8);

	GLWESecretKey* sk              = alloc_glwe_secret_key(ksk_params_glwe);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(ksk_params_glwe);
	uniform_glwe_secret_key(module, sk, 1);
	glwe_sk_prepare(module, sk_prep, sk);

	int auto_p = 5;

	GLWEAutomorphismKSK* ksk = new_automorphism_ksk(ksk_glwegadget_params);
	delete_glwegadget_prep(ksk->enc_s[0]);
	ksk->enc_s[0] = pvda_new_glwegadget_prep_device(ksk_glwegadget_params);
	int pc        = prepare_automorphism_key(module, ksk, sk_prep, auto_p);
	cr_assert_eq(pc, 0, "prepare_automorphism_key (GPU) failed");

	// auto_tmp: contiguous device buffer of biv_l limbs, matching what
	// glwegadget_automorphism allocates internally (biv_l = max(l_b_result, nrows)).
	uint64_t nrows      = ksk_glwegadget_params->l_tilde;           // 8
	uint64_t l_b_result = glwe_params_l_b(result_params_glwe);      // 6 (12 limbs, k=1)
	uint64_t biv_l      = l_b_result > nrows ? l_b_result : nrows;  // 8
	int64_t* d_auto_tmp = pvda_gpu_alloc(biv_l * NN);
	pvda_gpu_zero(d_auto_tmp, biv_l * NN);

	// result: sized for result_params_glwe (12 limbs) — SMALLER than the
	// KSK's own 16-limb column count.
	GLWECiphertext* result = pvda_new_glwe_device(result_params_glwe);
	size_t result_elems    = 12 * NN;

	// Canary: a second device allocation, written with a known sentinel and
	// checked for corruption after the half_prod call. cudaMalloc doesn't
	// guarantee adjacency, so this is a best-effort/opportunistic check —
	// the authoritative check is the compute-sanitizer run this test is
	// meant to be exercised under.
	int64_t* canary = pvda_gpu_alloc(4 * NN);
	static int64_t canary_pattern[4 * NN];
	for (size_t i = 0; i < 4 * NN; ++i) canary_pattern[i] = 0x4141414141414141LL;
	{
		int64_t* d_tmp = pvda_gpu_upload(canary_pattern, 4 * NN);
		pvda_gpu_copy(canary, d_tmp, 4 * NN);
		pvda_gpu_free(d_tmp);
	}

	PolyBiv auto_tmp_view = new_biv_view(NN, biv_l, (int64_t)NN, (PolyBivUnderlying*)d_auto_tmp);
	int hr                = glwegadget_half_prod(module, result, ksk->enc_s[0], &auto_tmp_view);
	cr_assert_eq(hr, 0, "glwegadget_half_prod (GPU, mismatched params) failed");

	int64_t canary_out[4 * NN];
	pvda_gpu_download(canary_out, canary, 4 * NN);
	for (size_t i = 0; i < 4 * NN; ++i)
		cr_assert_eq(canary_out[i], (int64_t)0x4141414141414141LL,
		             "canary clobbered at index %zu (0x%llx) — glwegadget_half_prod wrote past result's %zu-limb "
		             "buffer using the KSK's own (larger) column count instead",
		             i, (unsigned long long)canary_out[i], result_elems / NN);

	pvda_gpu_free(canary);
	pvda_gpu_free(d_auto_tmp);
	delete_glwe(result);
	delete_automorphism_ksk(ksk);
	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	pvda_delete_module_info(module);
	delete_glwegadget_params(ksk_glwegadget_params);
	delete_glwe_params(result_params_glwe);
	delete_glwe_params(ksk_params_glwe);
}

// Full numeric-correctness counterpart to
// half_prod_device_does_not_overflow_smaller_result above: that test only
// checked for a canary/overflow, not whether the RESULT VALUE itself is
// correct. Here the SOURCE glwe being transformed (not just the result) has
// FEWER limbs than the KSK's own precision — exactly
// onionpir_server_gpu's row_trace_unprep (12-limb, row_exp_params) being fed
// through automorphism KSKs built on params_glwe_autokey (16-limb) inside
// glwe_trace_expand. None of the other tests in this file give
// glwegadget_automorphism a source glwe smaller than the KSK/result.
Test(gpu_glwegadget_automorphism, source_smaller_than_ksk_matches_cpu)
{
	srand(23);

	MODULE* module = pvda_new_module_info(NN);
	cr_assert_not_null(module, "MODULE allocation failed");

	// KSK + result: 16-limb precision. Source glwe being transformed: 12-limb
	// (smaller) — mirrors glwe_trace_expand's tmp_glwe/tmp_glwe2 (16-limb,
	// matching auto_ksk_params) vs results[b] (12-limb, row_exp_params).
	GLWEParams* ksk_params_glwe             = new_glwe_params(NN, 1, KAPPA, 16, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEParams* src_params_glwe             = new_glwe_params(NN, 1, KAPPA, 12, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GLWEGadgetParams* ksk_glwegadget_params = new_glwegadget_params(ksk_params_glwe, KAPPA, 8);

	GLWESecretKey* sk              = alloc_glwe_secret_key(ksk_params_glwe);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(ksk_params_glwe);
	uniform_glwe_secret_key(module, sk, 1);
	glwe_sk_prepare(module, sk_prep, sk);

	int auto_p = 5;

	PolyUnivTnX* m_tnx = new_univ_tnx(src_params_glwe);
	uniform_random_pol_znx((PolyUniv*)m_tnx, NN, 64);

	GLWECiphertext* glwe_ct = new_glwe(src_params_glwe);
	glwe_secret_encrypt_tnx(module, glwe_ct, sk_prep, m_tnx);

	PolyUnivTnX* m_expected = new_univ_tnx(src_params_glwe);
	pvda_znx_automorphism(module, auto_p, m_expected, m_tnx);

	GLWEAutomorphismKSK* ksk = new_automorphism_ksk_gpu(module, ksk_glwegadget_params, sk_prep, auto_p);

	int64_t* d_glwe_ct_vec           = pvda_glwe_to_device(glwe_ct);
	GLWECiphertext glwe_ct_dev       = {.params = src_params_glwe, .vec = (VecBiv*)d_glwe_ct_vec};
	GLWECiphertext* glwe_res_gpu_dev = pvda_new_glwe_device(ksk_params_glwe);

	int ar = glwegadget_automorphism(module, glwe_res_gpu_dev, ksk, &glwe_ct_dev);
	cr_assert_eq(ar, 0, "glwegadget_automorphism (GPU, mismatched source) failed");

	GLWECiphertext* glwe_res_gpu = new_glwe(ksk_params_glwe);
	pvda_glwe_from_device(glwe_res_gpu, (const int64_t*)glwe_res_gpu_dev->vec);

	GLWECiphertext* glwe_norm_gpu = new_glwe(ksk_params_glwe);
	normalize_glwe(module, glwe_norm_gpu, glwe_res_gpu);

	PolyBiv* m_auto_gpu         = new_biv(ksk_params_glwe);
	PolyUnivTnX* m_observed_gpu = new_univ_tnx(ksk_params_glwe);
	glwe_secret_decrypt(module, m_auto_gpu, sk_prep, glwe_norm_gpu);
	biv_to_univ_tnx(ksk_params_glwe, m_observed_gpu, m_auto_gpu);

	// Decomposition noise floor governed by the KSK's own precision
	// (ksk_glwegadget_params) — the source's smaller precision only limits
	// how many of the KSK's limbs carry real (non-zero-padded) information,
	// not the per-limb error introduced by the gadget decomposition itself.
	int64_t decomp_noise_bits = info_bits_half_prod(ksk_params_glwe, ksk_glwegadget_params);
	for (int p = 0; p < NN; ++p) assert_tnx_close_enough(m_observed_gpu[p], m_expected[p], decomp_noise_bits);

	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_automorphism_ksk(ksk);

	delete_univ_tnx(m_tnx);
	delete_univ_tnx(m_expected);
	delete_univ_tnx(m_observed_gpu);

	delete_glwe(glwe_ct);
	delete_glwe(glwe_res_gpu);
	delete_glwe(glwe_norm_gpu);
	delete_glwe(glwe_res_gpu_dev);

	delete_biv(m_auto_gpu);

	pvda_delete_module_info(module);
	delete_glwegadget_params(ksk_glwegadget_params);
	delete_glwe_params(src_params_glwe);
	delete_glwe_params(ksk_params_glwe);
}
