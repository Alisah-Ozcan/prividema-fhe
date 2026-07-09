#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>

#include "backend/spqlios_alias.h"
#include "core/glwe/bivariate_polynomial.h"

#define POLY_SIZE 32
#define COEF_MOD  64  // coefficients in [-32, 32)
#define BATCH     4   // number of B polynomials

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void schoolbook(const int64_t* a, const int64_t* b, int64_t* c)
{
	int64_t tmp[2 * POLY_SIZE] = {0};
	for (int i = 0; i < POLY_SIZE; i++)
		for (int j = 0; j < POLY_SIZE; j++) tmp[i + j] += a[i] * b[j];

	// X^n ≡ -1 reduction
	for (int i = 0; i < POLY_SIZE; i++) c[i] = tmp[i] - tmp[i + POLY_SIZE];
}

// ---------------------------------------------------------------------------
// Test 1: AVX FFT (SVP) polynomial multiplication matches schoolbook
// ---------------------------------------------------------------------------

Test(avx_poly_mult, svp_matches_schoolbook)
{
	srand(42);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	int64_t A[POLY_SIZE], B[POLY_SIZE];
	for (int i = 0; i < POLY_SIZE; i++)
	{
		A[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;
		B[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;
	}

	// Prepare A in DFT domain
	PolyUnivDFT* a_ppol = pvda_new_svp_ppol(module);
	cr_assert_not_null(a_ppol, "svp_ppol allocation failed");
	pvda_svp_prepare(module, a_ppol, A);

	// Wrap B as a single-limb PolyBiv
	int64_t B_buf[POLY_SIZE];
	for (int i = 0; i < POLY_SIZE; i++) B_buf[i] = B[i];
	PolyBiv b_biv = new_biv_view(POLY_SIZE, 1, POLY_SIZE, B_buf);

	// SVP: A_dft * B → result in DFT domain
	VecUnivDFT* res_dft = pvda_new_vec_znx_dft(module, 1);
	cr_assert_not_null(res_dft, "res_dft allocation failed");
	pvda_svp_apply_dft(module, res_dft, 1, a_ppol, &b_biv);

	// IDFT back to coefficient domain
	int64_t C_avx[POLY_SIZE] = {0};
	PolyBiv c_biv            = new_biv_view(POLY_SIZE, 1, POLY_SIZE, C_avx);
	pvda_vec_znx_idft(module, &c_biv, res_dft, 1);

	// Schoolbook reference
	int64_t C_ref[POLY_SIZE];
	schoolbook(A, B, C_ref);

	for (int i = 0; i < POLY_SIZE; i++)
	{
		cr_assert_eq(C_avx[i], C_ref[i], "Mismatch at [%d]: AVX=%lld  Schoolbook=%lld", i, (long long)C_avx[i],
		             (long long)C_ref[i]);
	}

	pvda_delete_svp_ppol(a_ppol);
	pvda_delete_vec_znx_dft(res_dft);
	pvda_delete_module_info(module);
}

// ---------------------------------------------------------------------------
// Test 2: Batch SVP — pvda_svp_apply_dft with BATCH polynomials vs schoolbook
//         res_dft[j] = A * B[j]  for j in [0, BATCH)
// ---------------------------------------------------------------------------

Test(avx_poly_mult, svp_batch_matches_schoolbook)
{
	srand(42);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	// A: single scalar polynomial
	static int64_t A[POLY_SIZE];
	for (int i = 0; i < POLY_SIZE; i++) A[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;

	// B: BATCH polynomials, contiguous [B0 | B1 | ... | B_{BATCH-1}]
	static int64_t B_buf[BATCH * POLY_SIZE];
	for (int i = 0; i < BATCH * POLY_SIZE; i++) B_buf[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;

	// Prepare A in DFT domain
	PolyUnivDFT* a_ppol = pvda_new_svp_ppol(module);
	cr_assert_not_null(a_ppol, "svp_ppol allocation failed");
	pvda_svp_prepare(module, a_ppol, A);

	// Wrap B_buf as BATCH-limb PolyBiv (stride = POLY_SIZE)
	PolyBiv b_biv = new_biv_view(POLY_SIZE, BATCH, POLY_SIZE, B_buf);

	// Output DFT buffer for BATCH polynomials
	VecUnivDFT* res_dft = pvda_new_vec_znx_dft(module, BATCH);
	cr_assert_not_null(res_dft, "res_dft allocation failed");
	pvda_svp_apply_dft(module, res_dft, BATCH, a_ppol, &b_biv);

	// IDFT back to coefficient domain
	static int64_t C_avx[BATCH * POLY_SIZE];
	PolyBiv c_biv = new_biv_view(POLY_SIZE, BATCH, POLY_SIZE, C_avx);
	pvda_vec_znx_idft(module, &c_biv, res_dft, BATCH);

	// Schoolbook reference: C_ref[j] = A * B[j]
	static int64_t C_ref[BATCH * POLY_SIZE];
	for (int b = 0; b < BATCH; b++) schoolbook(A, B_buf + b * POLY_SIZE, C_ref + b * POLY_SIZE);

	for (int b = 0; b < BATCH; b++)
	{
		for (int i = 0; i < POLY_SIZE; i++)
		{
			cr_assert_eq(C_avx[b * POLY_SIZE + i], C_ref[b * POLY_SIZE + i],
			             "Mismatch at batch=%d coef=%d: AVX=%lld  Schoolbook=%lld", b, i,
			             (long long)C_avx[b * POLY_SIZE + i], (long long)C_ref[b * POLY_SIZE + i]);
		}
	}

	pvda_delete_svp_ppol(a_ppol);
	pvda_delete_vec_znx_dft(res_dft);
	pvda_delete_module_info(module);
}
