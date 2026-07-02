#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "backend/spqlios_alias.h"
#include "core/glwe/bivariate_polynomial.h"

// VMP (Vector-Matrix Product):
//   res[j] = sum_{i=0}^{NROWS-1} A[i] * M[i][j]  (mod X^n+1)
//
// Matrix layout (row-major):
//   M_flat[(i*NCOLS + j)*POLY_SIZE + k]  =  coefficient k of M[i][j]

#define POLY_SIZE 32
#define COEF_MOD  64
#define NROWS     2
#define NCOLS     3

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void poly_mult_schoolbook(const int64_t* a, const int64_t* b, int64_t* c)
{
	int64_t tmp[2 * POLY_SIZE] = {0};
	for (int i = 0; i < POLY_SIZE; i++)
		for (int j = 0; j < POLY_SIZE; j++) tmp[i + j] += a[i] * b[j];

	// X^n ≡ -1 reduction
	for (int i = 0; i < POLY_SIZE; i++) c[i] = tmp[i] - tmp[i + POLY_SIZE];
}

// ---------------------------------------------------------------------------
// Test 1: AVX VMP (pvda_vmp_apply_dft) result matches schoolbook
// ---------------------------------------------------------------------------

Test(avx_vmp, vmp_apply_dft_vs_schoolbook)
{
	srand(42);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	// A: input vector — NROWS polynomials, contiguous
	static int64_t A_flat[NROWS * POLY_SIZE];
	for (int i = 0; i < NROWS * POLY_SIZE; i++) A_flat[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;

	// M: NROWS x NCOLS polynomials, row-major
	static int64_t M_flat[NROWS * NCOLS * POLY_SIZE];
	for (int i = 0; i < NROWS * NCOLS * POLY_SIZE; i++) M_flat[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;

	// Prepare the matrix in DFT domain
	MatBivDFT* pmat = (MatBivDFT*)pvda_new_vmp_pmat(module, NROWS, NCOLS);
	cr_assert_not_null(pmat, "pmat allocation failed");
	int ret = pvda_vmp_prepare_contiguous(module, (double*)pmat, M_flat, NROWS, NCOLS);
	cr_assert_eq(ret, 0, "vmp_prepare_contiguous failed");

	// Wrap A as NROWS-limb PolyBiv (stride = POLY_SIZE)
	PolyBiv a_biv = new_biv_view(POLY_SIZE, NROWS, POLY_SIZE, A_flat);

	// Output DFT buffer for NCOLS polynomials
	VecUnivDFT* res_dft = pvda_new_vec_znx_dft(module, NCOLS);
	cr_assert_not_null(res_dft, "res_dft allocation failed");
	ret = pvda_vmp_apply_dft(module, res_dft, NCOLS, &a_biv, pmat, NROWS, NCOLS);
	cr_assert_eq(ret, 0, "vmp_apply_dft failed");

	// IDFT back to coefficient domain
	static int64_t C_vmp[NCOLS * POLY_SIZE];
	PolyBiv c_biv = new_biv_view(POLY_SIZE, NCOLS, POLY_SIZE, C_vmp);
	pvda_vec_znx_idft(module, &c_biv, res_dft, NCOLS);

	// Schoolbook reference: C[j] = sum_{i} A[i] * M[i][j]
	static int64_t C_ref[NCOLS * POLY_SIZE];
	memset(C_ref, 0, sizeof(C_ref));

	for (int j = 0; j < NCOLS; j++)
	{
		int64_t prod[POLY_SIZE];
		for (int i = 0; i < NROWS; i++)
		{
			const int64_t* a_i  = A_flat + i * POLY_SIZE;
			const int64_t* m_ij = M_flat + (i * NCOLS + j) * POLY_SIZE;
			int64_t* c_j        = C_ref + j * POLY_SIZE;

			poly_mult_schoolbook(a_i, m_ij, prod);
			for (int k = 0; k < POLY_SIZE; k++) c_j[k] += prod[k];
		}
	}

	for (int j = 0; j < NCOLS; j++)
	{
		for (int k = 0; k < POLY_SIZE; k++)
		{
			cr_assert_eq(C_vmp[j * POLY_SIZE + k], C_ref[j * POLY_SIZE + k],
			             "Mismatch at col=%d coef=%d: VMP=%lld  Schoolbook=%lld", j, k,
			             (long long)C_vmp[j * POLY_SIZE + k], (long long)C_ref[j * POLY_SIZE + k]);
		}
	}

	pvda_delete_vmp_pmat((double*)pmat);
	pvda_delete_vec_znx_dft(res_dft);
	pvda_delete_module_info(module);
}
