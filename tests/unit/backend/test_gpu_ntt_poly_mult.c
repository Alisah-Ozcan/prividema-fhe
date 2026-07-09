#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>

#include "gpu/common/gpu_nttparameters.h"
#include "gpu/host/ntt_poly_mult_host.h"

#define POLY_SIZE 4096
#define COEF_MOD  64  // coefficients in [-32, 32)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void schoolbook(const int64_t* a, const int64_t* b, int64_t* c)
{
	static int64_t tmp[2 * POLY_SIZE];
	for (int i = 0; i < 2 * POLY_SIZE; i++) tmp[i] = 0;

	for (int i = 0; i < POLY_SIZE; i++)
		for (int j = 0; j < POLY_SIZE; j++) tmp[i + j] += a[i] * b[j];

	// X^n ≡ -1 reduction
	for (int i = 0; i < POLY_SIZE; i++) c[i] = tmp[i] - tmp[i + POLY_SIZE];
}

// ---------------------------------------------------------------------------
// Test 1: GPU NTT polynomial multiplication matches schoolbook
// ---------------------------------------------------------------------------

Test(gpu_ntt_poly_mult, ntt_vs_schoolbook)
{
	gpu_ntt_initialize(POLY_SIZE);

	srand(42);
	static int64_t A[POLY_SIZE], B[POLY_SIZE];
	for (int i = 0; i < POLY_SIZE; i++)
	{
		A[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;
		B[i] = (int64_t)(rand() % COEF_MOD) - COEF_MOD / 2;
	}

	// GPU NTT multiplication
	static int64_t C_ntt[POLY_SIZE];
	gpu_ntt_svp(A, B, C_ntt, POLY_SIZE, 1);

	// Schoolbook reference
	static int64_t C_ref[POLY_SIZE];
	schoolbook(A, B, C_ref);

	for (int i = 0; i < POLY_SIZE; i++)
	{
		cr_assert_eq(C_ntt[i], C_ref[i], "Mismatch at [%d]: GPU-NTT=%lld  Schoolbook=%lld", i, (long long)C_ntt[i],
		             (long long)C_ref[i]);
	}
}
