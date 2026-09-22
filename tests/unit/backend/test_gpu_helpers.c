#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <stdint.h>
#include <string.h>

#include "backend/backend.h"
#include "core/ggsw/ggsw_ciphertext.h"
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
#include "gpu/host/ggsw_external_product_gpu.h"
#include "maths_structures.h"

// The device object helpers of ggsw_external_product_gpu.h: upload/download
// round trips, device DFT buffers, and automorphism keys prepared on the GPU.

#define NN    1024
#define KAPPA 18
#define LIMBS 8

Test(gpu_helpers, glwe_upload_download_round_trip)
{
	PvdaBackend* module = pvda_new_spqlios_backend(NN);
	GLWEParams* params  = new_glwe_params(NN, 1, KAPPA, LIMBS, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GLWECiphertext* src = new_glwe(params);
	GLWECiphertext* dst = new_glwe(params);
	size_t n_elems      = glwe_coef_number(params);
	for (size_t i = 0; i < n_elems; ++i) src->vec[i] = (int64_t)(i * 2654435761u) - (1ll << 31);

	GLWECiphertext* dev = pvda_glwe_upload(src);
	cr_assert_not_null(dev);
	cr_assert(pvda_is_device_pointer(dev->vec), "uploaded GLWE should be device-resident");
	cr_assert_eq(dev->params, params);

	pvda_glwe_download(dst, dev);
	cr_assert(eq(int, memcmp(src->vec, dst->vec, n_elems * sizeof(int64_t)), 0), "round trip changed the data");

	delete_glwe(dev);
	delete_glwe(src);
	delete_glwe(dst);
	delete_glwe_params(params);
	pvda_delete_backend(module);
}

Test(gpu_helpers, ggsw_and_glwegadget_upload)
{
	PvdaBackend* module          = pvda_new_spqlios_backend(NN);
	GLWEParams* params           = new_glwe_params(NN, 1, KAPPA, LIMBS, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GGSWParams* ggsw_params      = new_ggsw_params(params, 1, KAPPA, LIMBS);
	GLWEGadgetParams* gad_params = new_glwegadget_params(params, KAPPA, LIMBS / 2);

	GGSWCiphertext* ggsw = new_ggsw(ggsw_params);
	GGSWCiphertext* dev  = pvda_ggsw_upload(ggsw);
	cr_assert_not_null(dev);
	cr_assert(pvda_is_device_pointer(dev->mat), "uploaded GGSW should be device-resident");
	cr_assert_eq(dev->params, ggsw_params);

	GLWEGadgetCiphertext* gad     = new_glwegadget(gad_params);
	GLWEGadgetCiphertext* gad_dev = pvda_glwegadget_upload(gad);
	cr_assert_not_null(gad_dev);
	cr_assert(pvda_is_device_pointer(gad_dev->mat), "uploaded GLWEGadget should be device-resident");
	cr_assert_eq(gad_dev->params, gad_params);

	delete_ggsw(dev);
	delete_ggsw(ggsw);
	delete_glwegadget(gad_dev);
	delete_glwegadget(gad);
	delete_glwegadget_params(gad_params);
	delete_ggsw_params(ggsw_params);
	delete_glwe_params(params);
	pvda_delete_backend(module);
}

Test(gpu_helpers, glwe_dft_device_alloc_and_delete)
{
	GLWEParams* params          = new_glwe_params(NN, 1, KAPPA, LIMBS, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GLWECiphertextDFT* glwe_dft = pvda_new_glwe_dft_device(params);
	cr_assert_not_null(glwe_dft);
	cr_assert(pvda_is_device_pointer(glwe_dft->vec), "device GLWE DFT should be device-resident");
	cr_assert_eq(glwe_dft->params, params);
	delete_glwe_dft(glwe_dft);  // must release device memory, not call free() on it
	delete_glwe_params(params);
}

// An automorphism with a key from pvda_new_automorphism_key_device on device
// operands must decrypt to the same phase as the CPU automorphism.
Test(gpu_helpers, automorphism_key_device_matches_cpu)
{
	srand(7);
	PvdaBackend* module          = pvda_new_spqlios_backend(NN);
	GLWEParams* params           = new_glwe_params(NN, 1, KAPPA, LIMBS, 0, NOISE_UNIFORM_POWER_OF_TWO);
	params->fast_uniform_nb_bits = 0;
	GLWEGadgetParams* gad_params = new_glwegadget_params(params, KAPPA, LIMBS / 2);

	GLWESecretKey* sk              = alloc_glwe_secret_key(params);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params);
	uniform_glwe_secret_key(module, sk, 1);
	glwe_sk_prepare(module, sk_prep, sk);

	const int p                  = 5;
	GLWEAutomorphismKey* key_cpu = new_automorphism_key(gad_params);
	GLWEAutomorphismKey* key_gpu = pvda_new_automorphism_key_device(gad_params);
	cr_assert_not_null(key_gpu);
	cr_assert_eq(compute_automorphism_key(module, key_cpu, sk_prep, p), 0);
	cr_assert_eq(compute_automorphism_key(module, key_gpu, sk_prep, p), 0);
	cr_assert(pvda_is_device_pointer(key_gpu->glwe_ksk->enc_s[0]->mat), "key should be prepared on the GPU");

	PolyBiv* m          = new_biv(params);
	GLWECiphertext* ct  = new_glwe(params);
	GLWECiphertext* cpu = new_glwe(params);
	GLWECiphertext* gpu = new_glwe(params);
	uniform_random_biv_poly(module, params, m, 1);
	glwe_secret_encrypt_phase(module, ct, sk_prep, m);

	cr_assert_eq(glwegadget_automorphism(module, cpu, key_cpu, ct), 0);

	GLWECiphertext* ct_dev  = pvda_glwe_upload(ct);
	GLWECiphertext* res_dev = pvda_new_glwe_device(params);
	cr_assert_eq(glwegadget_automorphism(module, res_dev, key_gpu, ct_dev), 0);
	pvda_glwe_download(gpu, res_dev);

	PolyBiv* phase_cpu = new_biv(params);
	PolyBiv* phase_gpu = new_biv(params);
	normalize_glwe(module, cpu, cpu);
	normalize_glwe(module, gpu, gpu);
	glwe_secret_decrypt(module, phase_cpu, sk_prep, cpu);
	glwe_secret_decrypt(module, phase_gpu, sk_prep, gpu);
	PolyUnivTnX* tnx_cpu = new_univ_tnx(params);
	PolyUnivTnX* tnx_gpu = new_univ_tnx(params);
	biv_to_univ_tnx(params, tnx_cpu, phase_cpu, 0);
	biv_to_univ_tnx(params, tnx_gpu, phase_gpu, 0);
	// Both keys encrypt the same key material with independent noise: compare
	// the phases up to the key-switching noise, far below the message bits.
	for (int i = 0; i < NN; ++i)
	{
		int64_t diff = (int64_t)(tnx_cpu[i] - tnx_gpu[i]);
		cr_assert(lt(i64, llabs(diff), 1ll << 40), "coefficient %d differs by %lld", i, (long long)diff);
	}

	delete_univ_tnx(tnx_cpu);
	delete_univ_tnx(tnx_gpu);
	delete_biv(phase_cpu);
	delete_biv(phase_gpu);
	delete_glwe(ct_dev);
	delete_glwe(res_dev);
	delete_glwe(ct);
	delete_glwe(cpu);
	delete_glwe(gpu);
	delete_biv(m);
	delete_automorphism_key(key_cpu);
	delete_automorphism_key(key_gpu);
	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_glwegadget_params(gad_params);
	delete_glwe_params(params);
	pvda_delete_backend(module);
}
