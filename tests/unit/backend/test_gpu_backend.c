#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "backend/backend.h"
#include "backend/backend_arithmetic.h"
#include "core/ggsw/ggsw_arithmetic.h"
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
#include "rng.h"

// Backend selection: the CUDA backend, pvda_new_backend's configuration, and the
// backend-placed (*_on / *_clone_on) objects.

Test(gpu_backend, cuda_backend_creation)
{
	PvdaBackend* backend = pvda_new_cuda_backend(4096);
	cr_assert_not_null(backend, "pvda_new_cuda_backend failed on a CUDA build with a GPU");
	cr_assert_eq(pvda_backend_type(backend), PVDA_BACKEND_CUDA);
	cr_assert_eq(pvda_module_extract_nn(backend), 4096);
	pvda_delete_backend(backend);
}

Test(gpu_backend, new_backend_from_config)
{
	PvdaBackendConfig cpu_config = {.nn = 1024};
	PvdaBackend* cpu             = pvda_new_backend(cpu_config);
	cr_assert_not_null(cpu);
	cr_assert_eq(pvda_backend_type(cpu), PVDA_BACKEND_SPQLIOS, "zero-initialized type should select spqlios");
	cr_assert_eq(pvda_module_extract_nn(cpu), 1024);

	PvdaBackendConfig gpu_config = {.nn = 1024, .type = PVDA_BACKEND_CUDA};
	PvdaBackend* gpu             = pvda_new_backend(gpu_config);
	cr_assert_not_null(gpu);
	cr_assert_eq(pvda_backend_type(gpu), PVDA_BACKEND_CUDA);
	cr_assert_eq(pvda_module_extract_nn(gpu), 1024);

	pvda_delete_backend(cpu);
	pvda_delete_backend(gpu);
}

Test(gpu_backend, spqlios_backend_type)
{
	PvdaBackend* backend = pvda_new_spqlios_backend(2048);
	cr_assert_eq(pvda_backend_type(backend), PVDA_BACKEND_SPQLIOS);
	pvda_delete_backend(backend);
}

#define NN    1024
#define KAPPA 18
#define LIMBS 8

// Every *_on constructor places its object on the backend's side
Test(gpu_backend, constructors_follow_the_backend)
{
	PvdaBackend* backends[2] = {pvda_new_spqlios_backend(NN), pvda_new_cuda_backend(NN)};
	GLWEParams* params       = new_glwe_params(NN, 1, KAPPA, LIMBS, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GGSWParams* ggsw_params  = new_ggsw_params(params, 1, KAPPA, LIMBS);
	GLWEGadgetParams* gadp   = new_glwegadget_params(params, KAPPA, LIMBS / 2);

	for (int b = 0; b < 2; ++b)
	{
		int on_device               = pvda_backend_type(backends[b]) == PVDA_BACKEND_CUDA;
		GLWECiphertext* glwe        = new_glwe_on(backends[b], params);
		GLWECiphertextDFT* glwe_dft = new_glwe_dft_on(backends[b], params);
		GGSWCiphertext* ggsw        = new_ggsw_on(backends[b], ggsw_params);
		GGSWCiphertextPrep* prep    = new_ggsw_prep_on(backends[b], ggsw_params);
		GLWEGadgetCiphertext* gad   = new_glwegadget_on(backends[b], gadp);
		GLWEGadgetCiphertextPrep* g = new_glwegadget_prep_on(backends[b], gadp);
		GLWEAutomorphismKey* key    = new_automorphism_key_on(backends[b], gadp);

		cr_assert_eq(pvda_is_device_pointer(glwe->vec), on_device);
		cr_assert_eq(pvda_is_device_pointer(glwe_dft->vec), on_device);
		cr_assert_eq(pvda_is_device_pointer(ggsw->mat), on_device);
		cr_assert_eq(pvda_is_device_pointer(prep->mat), on_device);
		cr_assert_eq(pvda_is_device_pointer(gad->mat), on_device);
		cr_assert_eq(pvda_is_device_pointer(g->mat), on_device);
		cr_assert_eq(pvda_is_device_pointer(key->glwe_ksk->enc_s[0]->mat), on_device);

		// zero-initialized like new_glwe
		GLWECiphertext* host = new_glwe(params);
		host->vec[0]         = 1;
		glwe_copy_to_host(host, glwe);
		for (uint64_t i = 0; i < glwe_coef_number(params); ++i) cr_assert_eq(host->vec[i], 0);

		delete_glwe(host);
		delete_automorphism_key(key);
		delete_glwegadget_prep(g);
		delete_glwegadget(gad);
		delete_ggsw_prep(prep);
		delete_ggsw(ggsw);
		delete_glwe_dft(glwe_dft);
		delete_glwe(glwe);
	}
	delete_glwegadget_params(gadp);
	delete_ggsw_params(ggsw_params);
	delete_glwe_params(params);
	pvda_delete_backend(backends[0]);
	pvda_delete_backend(backends[1]);
}

// Host -> GPU -> GPU -> host clones keep the data
Test(gpu_backend, clone_round_trips)
{
	PvdaBackend* cpu        = pvda_new_spqlios_backend(NN);
	PvdaBackend* gpu        = pvda_new_cuda_backend(NN);
	GLWEParams* params      = new_glwe_params(NN, 1, KAPPA, LIMBS, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GGSWParams* ggsw_params = new_ggsw_params(params, 1, KAPPA, LIMBS);

	GLWECiphertext* src = new_glwe(params);
	for (uint64_t i = 0; i < glwe_coef_number(params); ++i) src->vec[i] = (int64_t)(i * 7919) - 12345;
	GLWECiphertext* on_gpu   = glwe_clone_on(gpu, src);
	GLWECiphertext* on_gpu_2 = glwe_clone_on(gpu, on_gpu);
	GLWECiphertext* back     = glwe_clone_on(cpu, on_gpu_2);
	cr_assert(pvda_is_device_pointer(on_gpu->vec) && pvda_is_device_pointer(on_gpu_2->vec));
	cr_assert_eq(pvda_is_device_pointer(back->vec), 0);
	cr_assert(eq(int, memcmp(src->vec, back->vec, glwe_coef_number(params) * sizeof(int64_t)), 0));

	GGSWCiphertext* ggsw = new_ggsw(ggsw_params);
	for (uint64_t i = 0; i < ggsw_coef_number(ggsw_params); ++i) ggsw->mat[i] = (int64_t)i - 777;
	GGSWCiphertext* ggsw_gpu  = ggsw_clone_on(gpu, ggsw);
	GGSWCiphertext* ggsw_back = ggsw_clone_on(cpu, ggsw_gpu);
	cr_assert(eq(int, memcmp(ggsw->mat, ggsw_back->mat, ggsw_coef_number(ggsw_params) * sizeof(int64_t)), 0));

	delete_ggsw(ggsw_back);
	delete_ggsw(ggsw_gpu);
	delete_ggsw(ggsw);
	delete_glwe(back);
	delete_glwe(on_gpu_2);
	delete_glwe(on_gpu);
	delete_glwe(src);
	delete_ggsw_params(ggsw_params);
	delete_glwe_params(params);
	pvda_delete_backend(cpu);
	pvda_delete_backend(gpu);
}

// Written once against the backend-placed API: a GGSW external product followed by an
// automorphism. Encryption happens on the host; the result is brought back to the host.
static void external_product_then_automorphism(const PvdaBackend* backend, GLWECiphertext* host_result,
                                               const GLWESecretKeyPrepared* sk_prep, const GLWECiphertext* host_ct,
                                               const GGSWCiphertext* host_ggsw, GLWEGadgetParams* gadp)
{
	const GLWEParams* params = host_ct->params;

	GLWECiphertext* ct            = glwe_clone_on(backend, host_ct);
	GGSWCiphertext* ggsw          = ggsw_clone_on(backend, host_ggsw);
	GGSWCiphertextPrep* ggsw_prep = new_ggsw_prep_on(backend, host_ggsw->params);
	cr_assert_eq(ggsw_prepare(backend, ggsw_prep, ggsw), 0);

	GLWEAutomorphismKey* key = new_automorphism_key_on(backend, gadp);
	cr_assert_eq(compute_automorphism_key(backend, key, sk_prep, 5), 0);

	GLWECiphertext* prod = new_glwe_on(backend, params);
	GLWECiphertext* res  = new_glwe_on(backend, params);
	cr_assert_eq(ggsw_external_product(backend, prod, ct, ggsw_prep), 0);
	cr_assert_eq(normalize_glwe(backend, prod, prod), 0);
	cr_assert_eq(glwegadget_automorphism(backend, res, key, prod), 0);
	cr_assert_eq(normalize_glwe(backend, res, res), 0);
	glwe_copy_to_host(host_result, res);

	delete_glwe(res);
	delete_glwe(prod);
	delete_automorphism_key(key);
	delete_ggsw_prep(ggsw_prep);
	delete_ggsw(ggsw);
	delete_glwe(ct);
}

Test(gpu_backend, same_code_on_both_backends)
{
	srand(11);
	PvdaBackend* cpu             = pvda_new_spqlios_backend(NN);
	PvdaBackend* gpu             = pvda_new_cuda_backend(NN);
	GLWEParams* params           = new_glwe_params(NN, 1, KAPPA, LIMBS, 0, NOISE_UNIFORM_POWER_OF_TWO);
	params->fast_uniform_nb_bits = 0;
	GGSWParams* ggsw_params      = new_ggsw_params(params, 1, KAPPA, LIMBS);
	GLWEGadgetParams* gadp       = new_glwegadget_params(params, KAPPA, LIMBS / 2);

	// Host-side setup (key generation, encryption) on the CPU backend
	GLWESecretKey* sk              = alloc_glwe_secret_key(params);
	GLWESecretKeyPrepared* sk_prep = alloc_glwe_secret_key_prepared(params);
	uniform_glwe_secret_key(cpu, sk, 1);
	glwe_sk_prepare(cpu, sk_prep, sk);

	PolyBiv* m  = new_biv(params);
	PolyUniv* u = new_univ(params);
	uniform_random_biv_poly(cpu, params, m, 1);
	uniform_pow2_random_pol_znx(cpu, u, NN, 1);
	GLWECiphertext* ct   = new_glwe(params);
	GGSWCiphertext* ggsw = new_ggsw(ggsw_params);
	glwe_secret_encrypt_phase(cpu, ct, sk_prep, m);
	ggsw_secret_encrypt(cpu, ggsw, sk_prep, u);

	GLWECiphertext* res_cpu = new_glwe(params);
	GLWECiphertext* res_gpu = new_glwe(params);
	external_product_then_automorphism(cpu, res_cpu, sk_prep, ct, ggsw, gadp);
	external_product_then_automorphism(gpu, res_gpu, sk_prep, ct, ggsw, gadp);

	PolyBiv* phase_cpu   = new_biv(params);
	PolyBiv* phase_gpu   = new_biv(params);
	PolyUnivTnX* tnx_cpu = new_univ_tnx(params);
	PolyUnivTnX* tnx_gpu = new_univ_tnx(params);
	glwe_secret_decrypt(cpu, phase_cpu, sk_prep, res_cpu);
	glwe_secret_decrypt(cpu, phase_gpu, sk_prep, res_gpu);
	biv_to_univ_tnx(params, tnx_cpu, phase_cpu, 0);
	biv_to_univ_tnx(params, tnx_gpu, phase_gpu, 0);
	// Independent key-switching keys: equal up to noise far below the message bits
	for (int i = 0; i < NN; ++i)
	{
		int64_t diff = (int64_t)(tnx_cpu[i] - tnx_gpu[i]);
		cr_assert(lt(i64, llabs(diff), 1ll << 40), "coefficient %d differs by %lld", i, (long long)diff);
	}

	delete_univ_tnx(tnx_cpu);
	delete_univ_tnx(tnx_gpu);
	delete_biv(phase_cpu);
	delete_biv(phase_gpu);
	delete_glwe(res_cpu);
	delete_glwe(res_gpu);
	delete_ggsw(ggsw);
	delete_glwe(ct);
	delete_univ(u);
	delete_biv(m);
	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(sk_prep);
	delete_glwegadget_params(gadp);
	delete_ggsw_params(ggsw_params);
	delete_glwe_params(params);
	pvda_delete_backend(cpu);
	pvda_delete_backend(gpu);
}

// A spqlios backend given GPU objects reports them instead of dereferencing them
Test(gpu_backend, spqlios_backend_rejects_gpu_operands)
{
	PvdaBackend* cpu        = pvda_new_spqlios_backend(NN);
	PvdaBackend* gpu        = pvda_new_cuda_backend(NN);
	GLWEParams* params      = new_glwe_params(NN, 1, KAPPA, LIMBS, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GGSWParams* ggsw_params = new_ggsw_params(params, 1, KAPPA, LIMBS);

	GLWECiphertext* host     = new_glwe(params);
	GLWECiphertext* dev      = new_glwe_on(gpu, params);
	GLWECiphertext* dev_res  = new_glwe_on(gpu, params);
	GGSWCiphertext* ggsw     = new_ggsw_on(gpu, ggsw_params);
	GGSWCiphertextPrep* prep = new_ggsw_prep_on(gpu, ggsw_params);
	cr_assert_eq(ggsw_prepare(gpu, prep, ggsw), 0);

	// Operations returning a status fail with -1
	cr_assert_eq(normalize_glwe(cpu, dev_res, dev), -1);
	cr_assert_eq(ggsw_external_product(cpu, dev_res, dev, prep), -1);
	PolyBiv v = glwe_flattened_biv(dev);
	cr_assert_eq(pvda_vec_znx_add(cpu, &v, &v, &v), -1);

	// Void operations leave their GPU result untouched
	for (uint64_t i = 0; i < glwe_coef_number(params); ++i) host->vec[i] = (int64_t)i;
	GLWECiphertext* marked = glwe_clone_on(gpu, host);
	add_glwe(cpu, marked, dev, dev);
	GLWECiphertext* back = new_glwe(params);
	glwe_copy_to_host(back, marked);
	cr_assert(eq(int, memcmp(back->vec, host->vec, glwe_coef_number(params) * sizeof(int64_t)), 0));

	delete_glwe(back);
	delete_glwe(marked);
	delete_ggsw_prep(prep);
	delete_ggsw(ggsw);
	delete_glwe(dev_res);
	delete_glwe(dev);
	delete_glwe(host);
	delete_ggsw_params(ggsw_params);
	delete_glwe_params(params);
	pvda_delete_backend(cpu);
	pvda_delete_backend(gpu);
}

// A host-only program in a CUDA build does not initialize CUDA (no context on the GPU)
Test(gpu_backend, host_only_use_does_not_initialize_cuda)
{
	PvdaBackend* cpu         = pvda_new_spqlios_backend(NN);
	GLWEParams* params       = new_glwe_params(NN, 1, KAPPA, LIMBS, 0, NOISE_UNIFORM_POWER_OF_TWO);
	GLWESecretKey* sk        = alloc_glwe_secret_key(params);
	GLWESecretKeyPrepared* p = alloc_glwe_secret_key_prepared(params);
	uniform_glwe_secret_key(cpu, sk, 1);
	glwe_sk_prepare(cpu, p, sk);

	PolyBiv* m          = new_biv(params);
	GLWECiphertext* ct  = new_glwe(params);
	GLWECiphertext* ct2 = new_glwe(params);
	uniform_random_biv_poly(cpu, params, m, 1);
	glwe_secret_encrypt_phase(cpu, ct, p, m);
	add_glwe(cpu, ct2, ct, ct);
	normalize_glwe(cpu, ct2, ct2);
	glwe_copy(ct, ct2);
	glwe_copy_to_host(ct, ct2);
	GLWECiphertext* ct3 = glwe_clone_on(cpu, ct);
	delete_glwe(ct3);
	delete_glwe(ct2);
	delete_glwe(ct);
	delete_biv(m);
	delete_glwe_secret_key(sk);
	delete_glwe_secret_key_prepared(p);
	delete_glwe_params(params);
	pvda_delete_backend(cpu);

	cr_assert_eq(cuInit(0), CUDA_SUCCESS);
	CUdevice dev;
	cr_assert_eq(cuDeviceGet(&dev, 0), CUDA_SUCCESS);
	unsigned int flags = 0;
	int active         = 1;
	cr_assert_eq(cuDevicePrimaryCtxGetState(dev, &flags, &active), CUDA_SUCCESS);
	cr_assert_eq(active, 0, "host-only use created a CUDA context");
}
