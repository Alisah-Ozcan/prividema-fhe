#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>

#include "backend/spqlios_alias.h"
#include "core/glwe/bivariate_polynomial.h"
#include "gpu/common/gpu_nttparameters.h"
#include "gpu/host/ntt_poly_mult_host.h"

// VMP (Vector-Matrix Product):
//   res[j] = sum_{i=0}^{NROWS-1} A[i] * M[i][j]  (mod X^n+1)
//
// Matrix layout (row-major):
//   M_flat[(i*NCOLS + j)*POLY_SIZE + k]  =  coefficient k of M[i][j]

#define POLY_SIZE 4096
#define COEF_MOD  64  // coefficients in [-32, 32)
#define NROWS     2
#define NCOLS     3

// ---------------------------------------------------------------------------
// Test 1: GPU NTT VMP and AVX VMP produce equal results
// ---------------------------------------------------------------------------

Test(gpu_vs_avx_vmp, vmp_results_match)
{
	srand(42);

	// A: input vector — NROWS polynomials, contiguous
	static int64_t A_flat[NROWS * POLY_SIZE];
	for (int i = 0; i < NROWS * POLY_SIZE; i++) A_flat[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;

	// M: NROWS x NCOLS polynomials, row-major
	static int64_t M_flat[NROWS * NCOLS * POLY_SIZE];
	for (int i = 0; i < NROWS * NCOLS * POLY_SIZE; i++) M_flat[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;

	// GPU NTT VMP
	gpu_ntt_initialize(POLY_SIZE);

	static int64_t C_gpu[NCOLS * POLY_SIZE];
	gpu_ntt_vmp(A_flat, M_flat, C_gpu, POLY_SIZE, NROWS, NCOLS);

	// AVX / spqlios VMP
	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	MatBivDFT* pmat = (MatBivDFT*)pvda_new_vmp_pmat(module, NROWS, NCOLS);
	cr_assert_not_null(pmat, "pmat allocation failed");
	int ret = pvda_vmp_prepare_contiguous(module, (double*)pmat, M_flat, NROWS, NCOLS);
	cr_assert_eq(ret, 0, "vmp_prepare_contiguous failed");

	PolyBiv a_biv = new_biv_view(POLY_SIZE, NROWS, POLY_SIZE, A_flat);

	VecUnivDFT* res_dft = pvda_new_vec_znx_dft(module, NCOLS);
	cr_assert_not_null(res_dft, "res_dft allocation failed");
	ret = pvda_vmp_apply_dft(module, res_dft, NCOLS, &a_biv, pmat, NROWS, NCOLS);
	cr_assert_eq(ret, 0, "vmp_apply_dft failed");

	static int64_t C_avx[NCOLS * POLY_SIZE];
	PolyBiv c_biv = new_biv_view(POLY_SIZE, NCOLS, POLY_SIZE, C_avx);
	pvda_vec_znx_idft(module, &c_biv, res_dft, NCOLS);

	// Compare coefficient by coefficient for each output polynomial
	for (int j = 0; j < NCOLS; j++)
	{
		for (int k = 0; k < POLY_SIZE; k++)
		{
			int64_t gpu_val = C_gpu[j * POLY_SIZE + k];
			int64_t avx_val = C_avx[j * POLY_SIZE + k];
			cr_assert_eq(gpu_val, avx_val, "Mismatch at col=%d coef=%d: GPU=%lld  AVX=%lld", j, k, (long long)gpu_val,
			             (long long)avx_val);
		}
	}

	pvda_delete_vmp_pmat((double*)pmat);
	pvda_delete_vec_znx_dft(res_dft);
	pvda_delete_module_info(module);
}
