#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "backend/spqlios_alias.h"
#include "core/glwe/bivariate_polynomial.h"
#include "gpu/host/normalize_host.h"

// base-2^kappa normalization:
//   Each coefficient position is processed independently.
//   Carry flows from the least-significant limb (i = L-1) to the
//   most-significant limb (i = 0).
//   Output: every coefficient in [-2^(kappa-1), 2^(kappa-1))
//
// CPU reference: pvda_vec_znx_normalize_base2k  (spqlios AVX)
// GPU:           gpu_normalize_base2k            (CUDA kernel)

#define POLY_SIZE 32
#define KAPPA     4  // base = 2^4 = 16, range = [-8, 7]
#define L         4  // number of limbs

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void assert_in_range(const int64_t* data, int n, int l, int kappa, const char* label)
{
	int64_t half = (int64_t)1 << (kappa - 1);
	for (int i = 0; i < l; i++)
	{
		for (int k = 0; k < n; k++)
		{
			int64_t v = data[i * n + k];
			cr_assert(v >= -half && v < half, "%s limb=%d coef=%d: %lld not in [-%lld, %lld)", label, i, k,
			          (long long)v, (long long)half, (long long)half);
		}
	}
}

// ---------------------------------------------------------------------------
// Test 1: GPU result matches CPU (spqlios reference)
// ---------------------------------------------------------------------------

Test(gpu_normalize, base2k_gpu_matches_cpu)
{
	srand(42);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	// Input coefficients in [-100, 100] — normalization required
	static int64_t a_flat[L * POLY_SIZE];
	for (int i = 0; i < L * POLY_SIZE; i++) a_flat[i] = (int64_t)(rand() % 201) - 100;

	// CPU (spqlios)
	static int64_t C_cpu[L * POLY_SIZE];
	PolyBiv a_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, a_flat);
	PolyBiv c_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, C_cpu);
	int ret       = pvda_vec_znx_normalize_base2k(module, KAPPA, &c_biv, &a_biv);
	cr_assert_eq(ret, 0, "pvda_vec_znx_normalize_base2k failed");

	// GPU
	static int64_t C_gpu[L * POLY_SIZE];
	gpu_normalize_base2k(a_flat, C_gpu, POLY_SIZE, L, KAPPA);

	// Compare limb by limb, coefficient by coefficient
	for (int i = 0; i < L; i++)
	{
		for (int k = 0; k < POLY_SIZE; k++)
		{
			int64_t cpu_val = C_cpu[i * POLY_SIZE + k];
			int64_t gpu_val = C_gpu[i * POLY_SIZE + k];
			cr_assert_eq(cpu_val, gpu_val, "Mismatch at limb=%d coef=%d: CPU=%lld  GPU=%lld", i, k, (long long)cpu_val,
			             (long long)gpu_val);
		}
	}

	pvda_delete_module_info(module);
}

// ---------------------------------------------------------------------------
// Test 2: GPU output coefficients lie within [-2^(kappa-1), 2^(kappa-1))
// ---------------------------------------------------------------------------

Test(gpu_normalize, base2k_output_in_range)
{
	srand(123);

	// Larger input values to stress carry propagation
	static int64_t a_flat[L * POLY_SIZE];
	for (int i = 0; i < L * POLY_SIZE; i++) a_flat[i] = (int64_t)(rand() % 1001) - 500;

	static int64_t C_gpu[L * POLY_SIZE];
	gpu_normalize_base2k(a_flat, C_gpu, POLY_SIZE, L, KAPPA);

	assert_in_range(C_gpu, POLY_SIZE, L, KAPPA, "GPU");
}

// ---------------------------------------------------------------------------
// Test 3: Zero input produces zero output
// ---------------------------------------------------------------------------

Test(gpu_normalize, base2k_zero_input)
{
	static int64_t a_zero[L * POLY_SIZE];
	memset(a_zero, 0, sizeof(a_zero));

	static int64_t C_gpu[L * POLY_SIZE];
	gpu_normalize_base2k(a_zero, C_gpu, POLY_SIZE, L, KAPPA);

	for (int i = 0; i < L * POLY_SIZE; i++)
		cr_assert_eq(C_gpu[i], 0LL, "Zero input must produce zero output at coef=%d", i);
}
