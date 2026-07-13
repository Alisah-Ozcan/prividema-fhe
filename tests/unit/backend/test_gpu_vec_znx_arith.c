#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "backend/spqlios_alias.h"
#include "core/glwe/bivariate_polynomial.h"
#include "gpu/common/gpu_stream.h"
#include "gpu/host/ggsw_external_product_gpu.h"
#include "gpu/host/normalize_host.h"
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

// ---------------------------------------------------------------------------
// Test 5: gpu_active_stream override (pvda_gpu_stream_push/pop)
//
// Every kernel launch and memcpy in the GPU backend runs on the thread-local
// gpu_active_stream instead of the implicit default stream (see
// gpu/common/gpu_stream.h). Pushing a non-default stream must not change the
// result of a GPU op, must be visible via pvda_gpu_stream_get_active while
// pushed, and must be fully undone by the matching pop.
// ---------------------------------------------------------------------------

Test(gpu_vec_znx_arith, add_gpu_matches_cpu_on_pushed_stream)
{
	srand(45);

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

	// Default stream is active until we push our own.
	cr_assert_null(pvda_gpu_stream_get_active(), "active stream should default to NULL");

	void* stream = pvda_gpu_stream_create();
	cr_assert_not_null(stream, "pvda_gpu_stream_create failed");

	pvda_gpu_stream_push(stream);
	cr_assert_eq(pvda_gpu_stream_get_active(), stream, "active stream not switched by push");

	// GPU add runs entirely on the pushed stream: kernel launch + both
	// host<->device memcpys inside gpu_vec_znx_add pick up gpu_active_stream.
	static int64_t C_gpu[L * POLY_SIZE];
	gpu_vec_znx_add(a_flat, b_flat, C_gpu, POLY_SIZE, L);

	for (int i = 0; i < L * POLY_SIZE; i++)
		cr_assert_eq(C_cpu[i], C_gpu[i], "Mismatch at index %d on pushed stream: CPU=%lld GPU=%lld", i,
		             (long long)C_cpu[i], (long long)C_gpu[i]);

	pvda_gpu_stream_pop();
	cr_assert_null(pvda_gpu_stream_get_active(), "pop did not restore the default stream");

	pvda_gpu_stream_destroy(stream);

	pvda_delete_module_info(module);
}

// ---------------------------------------------------------------------------
// Test: gpu_vec_znx_add_sized_device / gpu_vec_znx_sub_sized_device match
// spqlios' vec_znx_add_ref / vec_znx_sub_ref for operands of DIFFERING limb
// counts (a_size != b_size != res_size) — isolates the exact bug class found
// in add_glwe/sub_glwe's GPU dispatch (glwe_trace_expand calls them with
// operands whose params legitimately differ), independent of any GLWE/FHE
// semantics.
// ---------------------------------------------------------------------------

Test(gpu_vec_znx_arith, add_sub_sized_device_matches_cpu_mismatched_sizes)
{
	srand(51);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	size_t a_size   = 5;
	size_t b_size   = 3;
	size_t res_size = 6;  // > max(a_size, b_size): exercises the zero-pad tail too

	int64_t* a_flat = malloc(a_size * POLY_SIZE * sizeof(int64_t));
	int64_t* b_flat = malloc(b_size * POLY_SIZE * sizeof(int64_t));
	for (size_t i = 0; i < a_size * POLY_SIZE; i++) a_flat[i] = (int64_t)(rand() % 2001) - 1000;
	for (size_t i = 0; i < b_size * POLY_SIZE; i++) b_flat[i] = (int64_t)(rand() % 2001) - 1000;

	int64_t* add_cpu = malloc(res_size * POLY_SIZE * sizeof(int64_t));
	int64_t* sub_cpu = malloc(res_size * POLY_SIZE * sizeof(int64_t));
	PolyBiv a_biv    = new_biv_view(POLY_SIZE, a_size, POLY_SIZE, a_flat);
	PolyBiv b_biv    = new_biv_view(POLY_SIZE, b_size, POLY_SIZE, b_flat);
	PolyBiv add_biv  = new_biv_view(POLY_SIZE, res_size, POLY_SIZE, add_cpu);
	PolyBiv sub_biv  = new_biv_view(POLY_SIZE, res_size, POLY_SIZE, sub_cpu);
	cr_assert_eq(pvda_vec_znx_add(module, &add_biv, &a_biv, &b_biv), 0, "pvda_vec_znx_add failed");
	cr_assert_eq(pvda_vec_znx_sub(module, &sub_biv, &a_biv, &b_biv), 0, "pvda_vec_znx_sub failed");

	int64_t* d_a       = pvda_gpu_upload(a_flat, a_size * POLY_SIZE);
	int64_t* d_b       = pvda_gpu_upload(b_flat, b_size * POLY_SIZE);
	int64_t* d_add_res = pvda_gpu_alloc(res_size * POLY_SIZE);
	int64_t* d_sub_res = pvda_gpu_alloc(res_size * POLY_SIZE);

	gpu_vec_znx_add_sized_device(d_a, a_size, d_b, b_size, d_add_res, res_size, POLY_SIZE);
	gpu_vec_znx_sub_sized_device(d_a, a_size, d_b, b_size, d_sub_res, res_size, POLY_SIZE);

	int64_t* add_gpu = malloc(res_size * POLY_SIZE * sizeof(int64_t));
	int64_t* sub_gpu = malloc(res_size * POLY_SIZE * sizeof(int64_t));
	pvda_gpu_download(add_gpu, d_add_res, res_size * POLY_SIZE);
	pvda_gpu_download(sub_gpu, d_sub_res, res_size * POLY_SIZE);

	for (size_t i = 0; i < res_size * POLY_SIZE; i++)
	{
		cr_assert_eq(add_cpu[i], add_gpu[i], "add mismatch at %zu: CPU=%lld GPU=%lld", i, (long long)add_cpu[i],
		             (long long)add_gpu[i]);
		cr_assert_eq(sub_cpu[i], sub_gpu[i], "sub mismatch at %zu: CPU=%lld GPU=%lld", i, (long long)sub_cpu[i],
		             (long long)sub_gpu[i]);
	}

	pvda_gpu_free(d_a);
	pvda_gpu_free(d_b);
	pvda_gpu_free(d_add_res);
	pvda_gpu_free(d_sub_res);
	free(a_flat);
	free(b_flat);
	free(add_cpu);
	free(sub_cpu);
	free(add_gpu);
	free(sub_gpu);

	pvda_delete_module_info(module);
}

// ---------------------------------------------------------------------------
// Test: gpu_normalize_base2k_sized_device matches spqlios'
// vec_znx_normalize_base2k_ref for DIFFERING a_size/res_size (both
// truncating: a_size > res_size, and zero-padding: res_size > a_size) —
// isolates the second bug class found in normalize_glwe's GPU dispatch.
// ---------------------------------------------------------------------------

Test(gpu_vec_znx_arith, normalize_sized_device_matches_cpu_mismatched_sizes)
{
	srand(53);
	uint64_t kappa = 6;

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	// Case 1: a_size(8) > res_size(5) — truncating (carry still propagates
	// through the discarded top limbs).
	{
		size_t a_size = 8, res_size = 5;
		int64_t* a_flat = malloc(a_size * POLY_SIZE * sizeof(int64_t));
		for (size_t i = 0; i < a_size * POLY_SIZE; i++) a_flat[i] = (int64_t)(rand() % 2000001) - 1000000;

		int64_t* res_cpu = malloc(res_size * POLY_SIZE * sizeof(int64_t));
		PolyBiv a_biv    = new_biv_view(POLY_SIZE, a_size, POLY_SIZE, a_flat);
		PolyBiv res_biv  = new_biv_view(POLY_SIZE, res_size, POLY_SIZE, res_cpu);
		cr_assert_eq(pvda_vec_znx_normalize_base2k(module, kappa, &res_biv, &a_biv), 0,
		             "pvda_vec_znx_normalize_base2k failed");

		int64_t* d_a   = pvda_gpu_upload(a_flat, a_size * POLY_SIZE);
		int64_t* d_res = pvda_gpu_alloc(res_size * POLY_SIZE);
		gpu_normalize_base2k_sized_device(d_a, a_size, POLY_SIZE, d_res, res_size, POLY_SIZE, POLY_SIZE,
		                                  (uint32_t)kappa);
		int64_t* res_gpu = malloc(res_size * POLY_SIZE * sizeof(int64_t));
		pvda_gpu_download(res_gpu, d_res, res_size * POLY_SIZE);

		for (size_t i = 0; i < res_size * POLY_SIZE; i++)
			cr_assert_eq(res_cpu[i], res_gpu[i], "truncating case mismatch at %zu: CPU=%lld GPU=%lld", i,
			             (long long)res_cpu[i], (long long)res_gpu[i]);

		pvda_gpu_free(d_a);
		pvda_gpu_free(d_res);
		free(a_flat);
		free(res_cpu);
		free(res_gpu);
	}

	// Case 2: a_size(4) < res_size(7) — zero-padding the extra output limbs.
	{
		size_t a_size = 4, res_size = 7;
		int64_t* a_flat = malloc(a_size * POLY_SIZE * sizeof(int64_t));
		for (size_t i = 0; i < a_size * POLY_SIZE; i++) a_flat[i] = (int64_t)(rand() % 2000001) - 1000000;

		int64_t* res_cpu = malloc(res_size * POLY_SIZE * sizeof(int64_t));
		PolyBiv a_biv    = new_biv_view(POLY_SIZE, a_size, POLY_SIZE, a_flat);
		PolyBiv res_biv  = new_biv_view(POLY_SIZE, res_size, POLY_SIZE, res_cpu);
		cr_assert_eq(pvda_vec_znx_normalize_base2k(module, kappa, &res_biv, &a_biv), 0,
		             "pvda_vec_znx_normalize_base2k failed");

		int64_t* d_a   = pvda_gpu_upload(a_flat, a_size * POLY_SIZE);
		int64_t* d_res = pvda_gpu_alloc(res_size * POLY_SIZE);
		gpu_normalize_base2k_sized_device(d_a, a_size, POLY_SIZE, d_res, res_size, POLY_SIZE, POLY_SIZE,
		                                  (uint32_t)kappa);
		int64_t* res_gpu = malloc(res_size * POLY_SIZE * sizeof(int64_t));
		pvda_gpu_download(res_gpu, d_res, res_size * POLY_SIZE);

		for (size_t i = 0; i < res_size * POLY_SIZE; i++)
			cr_assert_eq(res_cpu[i], res_gpu[i], "zero-pad case mismatch at %zu: CPU=%lld GPU=%lld", i,
			             (long long)res_cpu[i], (long long)res_gpu[i]);

		pvda_gpu_free(d_a);
		pvda_gpu_free(d_res);
		free(a_flat);
		free(res_cpu);
		free(res_gpu);
	}

	pvda_delete_module_info(module);
}
