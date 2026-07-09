#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "backend/spqlios_alias.h"
#include "core/glwe/bivariate_polynomial.h"
#include "gpu/host/vec_znx_arith_host.h"

// Elementwise vec_znx operations:
//   Every coefficient of every limb is processed independently, no carry
//   propagation (contrast with base2k normalization).
//
// CPU reference: pvda_vec_znx_add / pvda_vec_znx_sub / pvda_vec_znx_negate (spqlios AVX)
// GPU:           gpu_vec_znx_add / gpu_vec_znx_sub / gpu_vec_znx_negate     (CUDA kernels)

#define POLY_SIZE 32
#define L         4  // number of limbs

// ---------------------------------------------------------------------------
// Test 1: GPU add matches CPU (spqlios reference)
// ---------------------------------------------------------------------------

Test(gpu_vec_znx_arith, add_gpu_matches_cpu)
{
	srand(42);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	static int64_t a_flat[L * POLY_SIZE];
	static int64_t b_flat[L * POLY_SIZE];
	for (int i = 0; i < L * POLY_SIZE; i++)
	{
		a_flat[i] = (int64_t)(rand() % 2001) - 1000;
		b_flat[i] = (int64_t)(rand() % 2001) - 1000;
	}

	// CPU (spqlios)
	static int64_t C_cpu[L * POLY_SIZE];
	PolyBiv a_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, a_flat);
	PolyBiv b_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, b_flat);
	PolyBiv c_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, C_cpu);
	int ret       = pvda_vec_znx_add(module, &c_biv, &a_biv, &b_biv);
	cr_assert_eq(ret, 0, "pvda_vec_znx_add failed");

	// GPU
	static int64_t C_gpu[L * POLY_SIZE];
	gpu_vec_znx_add(a_flat, b_flat, C_gpu, POLY_SIZE, L);

	for (int i = 0; i < L * POLY_SIZE; i++)
		cr_assert_eq(C_cpu[i], C_gpu[i], "Mismatch at index %d: CPU=%lld GPU=%lld", i, (long long)C_cpu[i],
		             (long long)C_gpu[i]);

	pvda_delete_module_info(module);
}

// ---------------------------------------------------------------------------
// Test 2: GPU sub matches CPU (spqlios reference)
// ---------------------------------------------------------------------------

Test(gpu_vec_znx_arith, sub_gpu_matches_cpu)
{
	srand(43);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	static int64_t a_flat[L * POLY_SIZE];
	static int64_t b_flat[L * POLY_SIZE];
	for (int i = 0; i < L * POLY_SIZE; i++)
	{
		a_flat[i] = (int64_t)(rand() % 2001) - 1000;
		b_flat[i] = (int64_t)(rand() % 2001) - 1000;
	}

	// CPU (spqlios)
	static int64_t C_cpu[L * POLY_SIZE];
	PolyBiv a_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, a_flat);
	PolyBiv b_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, b_flat);
	PolyBiv c_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, C_cpu);
	int ret       = pvda_vec_znx_sub(module, &c_biv, &a_biv, &b_biv);
	cr_assert_eq(ret, 0, "pvda_vec_znx_sub failed");

	// GPU
	static int64_t C_gpu[L * POLY_SIZE];
	gpu_vec_znx_sub(a_flat, b_flat, C_gpu, POLY_SIZE, L);

	for (int i = 0; i < L * POLY_SIZE; i++)
		cr_assert_eq(C_cpu[i], C_gpu[i], "Mismatch at index %d: CPU=%lld GPU=%lld", i, (long long)C_cpu[i],
		             (long long)C_gpu[i]);

	pvda_delete_module_info(module);
}

// ---------------------------------------------------------------------------
// Test 3: GPU negate matches CPU (spqlios reference)
// ---------------------------------------------------------------------------

Test(gpu_vec_znx_arith, negate_gpu_matches_cpu)
{
	srand(44);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	static int64_t a_flat[L * POLY_SIZE];
	for (int i = 0; i < L * POLY_SIZE; i++) a_flat[i] = (int64_t)(rand() % 2001) - 1000;

	// CPU (spqlios)
	static int64_t C_cpu[L * POLY_SIZE];
	PolyBiv a_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, a_flat);
	PolyBiv c_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, C_cpu);
	int ret       = pvda_vec_znx_negate(module, &c_biv, &a_biv);
	cr_assert_eq(ret, 0, "pvda_vec_znx_negate failed");

	// GPU
	static int64_t C_gpu[L * POLY_SIZE];
	gpu_vec_znx_negate(a_flat, C_gpu, POLY_SIZE, L);

	for (int i = 0; i < L * POLY_SIZE; i++)
		cr_assert_eq(C_cpu[i], C_gpu[i], "Mismatch at index %d: CPU=%lld GPU=%lld", i, (long long)C_cpu[i],
		             (long long)C_gpu[i]);

	pvda_delete_module_info(module);
}

// ---------------------------------------------------------------------------
// Test 4: Zero input edge cases
// ---------------------------------------------------------------------------

Test(gpu_vec_znx_arith, zero_input)
{
	static int64_t zero[L * POLY_SIZE];
	memset(zero, 0, sizeof(zero));

	static int64_t C_add[L * POLY_SIZE];
	gpu_vec_znx_add(zero, zero, C_add, POLY_SIZE, L);
	for (int i = 0; i < L * POLY_SIZE; i++) cr_assert_eq(C_add[i], 0LL, "add: zero+zero must be zero at %d", i);

	static int64_t C_sub[L * POLY_SIZE];
	gpu_vec_znx_sub(zero, zero, C_sub, POLY_SIZE, L);
	for (int i = 0; i < L * POLY_SIZE; i++) cr_assert_eq(C_sub[i], 0LL, "sub: zero-zero must be zero at %d", i);

	static int64_t C_neg[L * POLY_SIZE];
	gpu_vec_znx_negate(zero, C_neg, POLY_SIZE, L);
	for (int i = 0; i < L * POLY_SIZE; i++) cr_assert_eq(C_neg[i], 0LL, "negate: -zero must be zero at %d", i);
}
