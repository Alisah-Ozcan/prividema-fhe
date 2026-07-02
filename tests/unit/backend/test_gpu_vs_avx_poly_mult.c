#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>

#include "backend/spqlios_alias.h"
#include "core/glwe/bivariate_polynomial.h"
#include "gpu/common/gpu_nttparameters.h"
#include "gpu/host/ntt_poly_mult_host.h"

#define POLY_SIZE 4096
#define COEF_MOD  64  // coefficients in [-32, 32)
#define BATCH     4

// ---------------------------------------------------------------------------
// Test 1: GPU NTT and AVX FFT polynomial multiplication produce equal results
// ---------------------------------------------------------------------------

Test(gpu_vs_avx, poly_mult_results_match)
{
	srand(42);

	static int64_t A[POLY_SIZE], B[POLY_SIZE];
	for (int i = 0; i < POLY_SIZE; i++)
	{
		A[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;
		B[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;
	}

	// GPU NTT multiplication
	gpu_ntt_initialize(POLY_SIZE);

	static int64_t C_gpu[POLY_SIZE];
	gpu_ntt_svp(A, B, C_gpu, POLY_SIZE, 1);

	// AVX / spqlios FFT multiplication
	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	PolyUnivDFT* a_ppol = pvda_new_svp_ppol(module);
	cr_assert_not_null(a_ppol, "svp_ppol allocation failed");
	pvda_svp_prepare(module, a_ppol, A);

	static int64_t B_buf[POLY_SIZE];
	for (int i = 0; i < POLY_SIZE; i++) B_buf[i] = B[i];
	PolyBiv b_biv = new_biv_view(POLY_SIZE, 1, POLY_SIZE, B_buf);

	VecUnivDFT* res_dft = pvda_new_vec_znx_dft(module, 1);
	cr_assert_not_null(res_dft, "res_dft allocation failed");
	pvda_svp_apply_dft(module, res_dft, 1, a_ppol, &b_biv);

	static int64_t C_avx[POLY_SIZE];
	PolyBiv c_biv = new_biv_view(POLY_SIZE, 1, POLY_SIZE, C_avx);
	pvda_vec_znx_idft(module, &c_biv, res_dft, 1);

	for (int i = 0; i < POLY_SIZE; i++)
	{
		cr_assert_eq(C_gpu[i], C_avx[i], "Mismatch at [%d]: GPU-NTT=%lld  AVX-FFT=%lld", i, (long long)C_gpu[i],
		             (long long)C_avx[i]);
	}

	pvda_delete_svp_ppol(a_ppol);
	pvda_delete_vec_znx_dft(res_dft);
	pvda_delete_module_info(module);
}

// ---------------------------------------------------------------------------
// Test 2: GPU NTT batch and AVX FFT batch produce equal results
// ---------------------------------------------------------------------------

Test(gpu_vs_avx, poly_mult_batch_results_match)
{
	srand(42);

	static int64_t A[POLY_SIZE];
	for (int i = 0; i < POLY_SIZE; i++) A[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;

	static int64_t B_buf[BATCH * POLY_SIZE];
	for (int i = 0; i < BATCH * POLY_SIZE; i++) B_buf[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;

	// GPU NTT batch multiplication
	gpu_ntt_initialize(POLY_SIZE);

	static int64_t C_gpu[BATCH * POLY_SIZE];
	gpu_ntt_svp(A, B_buf, C_gpu, POLY_SIZE, BATCH);

	// AVX / spqlios FFT batch multiplication
	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	PolyUnivDFT* a_ppol = pvda_new_svp_ppol(module);
	cr_assert_not_null(a_ppol, "svp_ppol allocation failed");
	pvda_svp_prepare(module, a_ppol, A);

	PolyBiv b_biv = new_biv_view(POLY_SIZE, BATCH, POLY_SIZE, B_buf);

	VecUnivDFT* res_dft = pvda_new_vec_znx_dft(module, BATCH);
	cr_assert_not_null(res_dft, "res_dft allocation failed");
	pvda_svp_apply_dft(module, res_dft, BATCH, a_ppol, &b_biv);

	static int64_t C_avx[BATCH * POLY_SIZE];
	PolyBiv c_biv = new_biv_view(POLY_SIZE, BATCH, POLY_SIZE, C_avx);
	pvda_vec_znx_idft(module, &c_biv, res_dft, BATCH);

	for (int b = 0; b < BATCH; b++)
	{
		for (int i = 0; i < POLY_SIZE; i++)
		{
			int64_t gpu_val = C_gpu[b * POLY_SIZE + i];
			int64_t avx_val = C_avx[b * POLY_SIZE + i];
			cr_assert_eq(gpu_val, avx_val, "Mismatch at batch=%d coef=%d: GPU=%lld  AVX=%lld", b, i,
			             (long long)gpu_val, (long long)avx_val);
		}
	}

	pvda_delete_svp_ppol(a_ppol);
	pvda_delete_vec_znx_dft(res_dft);
	pvda_delete_module_info(module);
}
