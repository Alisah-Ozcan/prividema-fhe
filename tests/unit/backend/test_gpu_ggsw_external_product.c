#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>

#include "core/ggsw/ggsw_arithmetic.h"
#include "core/ggsw/ggsw_ciphertext.h"
#include "core/ggsw/ggsw_params.h"
#include "core/glwe/glwe_ciphertext.h"
#include "core/glwe/glwe_params.h"
#include "gpu/common/gpu_nttparameters.h"
#include "gpu/host/ggsw_external_product_gpu.h"

/*
 {.nn                        = (1 << 14),
         .k                         = 1,
         .kappa                     = 19,
         .ciphertext_nb_limbs       = 15l * 2,
         .ciphertext_nb_limbs_tilde = 15l * 2,
*/

// Small parameters for fast testing
//#define NN          512
//#define K           1
//#define KAPPA       8
//#define N_LIMBS     2   // ciphertext_nb_limbs (GLWE)
//#define NB_LIMBS_T  2   // ciphertext_nb_limbs_tilde (GGSW rows)
//#define COEF_MOD    16  // coefficients in [-8, 8)

#define NN         (1 << 14)
#define K          1
#define KAPPA      19
#define N_LIMBS    15l * 2  // ciphertext_nb_limbs (GLWE)
#define NB_LIMBS_T 15l * 2  // ciphertext_nb_limbs_tilde (GGSW rows)
#define COEF_MOD   16       // coefficients in [-8, 8)

// nrows = NB_LIMBS_T, ncols = N_LIMBS
// GLWE data  : nrows * NN  elements
// GGSW data  : nrows * ncols * NN elements
// result data: ncols * NN elements

// ---------------------------------------------------------------------------
// Test 1: GPU external product matches CPU external product
// ---------------------------------------------------------------------------

Test(gpu_ggsw_ep, gpu_matches_cpu)
{
	srand(42);

	GLWEParams* params_glwe = new_glwe_params(NN, K, KAPPA, N_LIMBS, 0.0, NOISE_UNIFORM_POWER_OF_TWO);
	cr_assert_not_null(params_glwe, "GLWEParams allocation failed");

	GGSWParams* params_ggsw = new_ggsw_params(params_glwe, K, KAPPA, NB_LIMBS_T);
	cr_assert_not_null(params_ggsw, "GGSWParams allocation failed");

	gpu_ntt_initialize(NN);

	uint64_t nrows      = ggsw_num_rows(params_ggsw);
	uint64_t ncols      = glwe_params_n_limbs(params_glwe);
	size_t result_elems = (size_t)(ncols * NN);

	// Allocate and fill CPU GLWE and GGSW with small random coefficients
	GLWECiphertext* cpu_glwe = new_glwe(params_glwe);
	cr_assert_not_null(cpu_glwe, "cpu_glwe allocation failed");
	for (size_t i = 0; i < nrows * NN; i++) cpu_glwe->vec[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;

	GGSWCiphertext* cpu_ggsw = new_ggsw(params_ggsw);
	cr_assert_not_null(cpu_ggsw, "cpu_ggsw allocation failed");
	for (size_t i = 0; i < nrows * ncols * NN; i++) cpu_ggsw->mat[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;

	// ---------------------------------------------------------------------------
	// CPU external product (reference)
	// ---------------------------------------------------------------------------

	MODULE* module = pvda_new_module_info(NN);
	cr_assert_not_null(module, "MODULE allocation failed");

	GLWECiphertext* cpu_result = new_glwe(params_glwe);
	cr_assert_not_null(cpu_result, "cpu_result allocation failed");

	int ret = ggsw_unprepared_external_product(module, cpu_result, cpu_glwe, cpu_ggsw);
	cr_assert_eq(ret, 0, "CPU ggsw_unprepared_external_product failed");

	// ---------------------------------------------------------------------------
	// GPU external product
	//
	// pvda_glwe_to_device / pvda_ggsw_to_device read the element count from the
	// struct params and upload in one call.  pvda_glwe_from_device downloads back.
	//
	// Stack structs hold params on host; vec/mat point to device memory so that
	// pvda_is_device_pointer(glwe->vec) == 1 inside ggsw_unprepared_external_product
	// and the GPU path is taken automatically.
	// ---------------------------------------------------------------------------

	// GPU-backed structs: params live on host, vec/mat point to device memory.
	// pvda_is_device_pointer(glwe->vec) == 1 triggers the GPU path automatically.
	GLWECiphertext gpu_glwe   = {.params = params_glwe, .vec = pvda_glwe_to_device(cpu_glwe)};
	GGSWCiphertext gpu_ggsw   = {.params = params_ggsw, .mat = pvda_ggsw_to_device(cpu_ggsw)};
	GLWECiphertext gpu_result = {.params = params_glwe, .vec = pvda_gpu_alloc(result_elems)};

	cr_assert_not_null(gpu_glwe.vec, "pvda_glwe_to_device failed");
	cr_assert_not_null(gpu_ggsw.mat, "pvda_ggsw_to_device failed");
	cr_assert_not_null(gpu_result.vec, "pvda_gpu_alloc for result failed");

	ret = ggsw_unprepared_external_product(module, &gpu_result, &gpu_glwe, &gpu_ggsw);
	cr_assert_eq(ret, 0, "GPU ggsw_unprepared_external_product failed");

	// Download GPU result directly into an existing GLWECiphertext
	GLWECiphertext* download_target = new_glwe(params_glwe);
	cr_assert_not_null(download_target, "download_target allocation failed");
	pvda_glwe_from_device(download_target, gpu_result.vec);

	// ---------------------------------------------------------------------------
	// Compare GPU result with CPU result coefficient by coefficient
	// ---------------------------------------------------------------------------

	for (size_t i = 0; i < result_elems; i++)
	{
		cr_assert_eq(download_target->vec[i], cpu_result->vec[i], "Mismatch at element %zu: GPU=%lld  CPU=%lld", i,
		             (long long)download_target->vec[i], (long long)cpu_result->vec[i]);
	}

	// Cleanup
	pvda_gpu_free(gpu_glwe.vec);
	pvda_gpu_free(gpu_ggsw.mat);
	pvda_gpu_free(gpu_result.vec);

	delete_glwe(cpu_glwe);
	delete_glwe(cpu_result);
	delete_glwe(download_target);
	delete_ggsw(cpu_ggsw);
	delete_glwe_params(params_glwe);
	delete_ggsw_params(params_ggsw);
	pvda_delete_module_info(module);
}