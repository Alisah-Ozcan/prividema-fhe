#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "backend/spqlios_alias.h"
#include "core/glwe/bivariate_polynomial.h"
#include "gpu/common/gpu_stream.h"
#include "gpu/host/vec_znx_rotate_automorphism_host.h"

// vec_znx_rotate / vec_znx_automorphism:
//   Applied independently, limb by limb (no carry propagation, contrast with
//   base2k normalization). res = a . X^p (rotate) or res = a(X^p) (automorphism).
//
// CPU reference: pvda_vec_znx_rotate / pvda_vec_znx_automorphism (spqlios)
// GPU:           gpu_vec_znx_rotate / gpu_vec_znx_automorphism    (CUDA kernels)

#define POLY_SIZE 32
#define L         4  // number of limbs

// ---------------------------------------------------------------------------
// Test 1: GPU rotate matches CPU (spqlios reference), contiguous layout
// ---------------------------------------------------------------------------

Test(gpu_vec_znx_rotate, rotate_gpu_matches_cpu)
{
	srand(42);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	static int64_t a_flat[L * POLY_SIZE];
	for (int i = 0; i < L * POLY_SIZE; i++) a_flat[i] = (int64_t)(rand() % 2001) - 1000;

	int64_t test_p[] = {0, 1, -1, 3, -3, (int64_t)POLY_SIZE, -(int64_t)POLY_SIZE, 5, -17};

	for (size_t t = 0; t < sizeof(test_p) / sizeof(test_p[0]); t++)
	{
		int64_t p = test_p[t];

		// CPU (spqlios)
		static int64_t C_cpu[L * POLY_SIZE];
		PolyBiv a_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, a_flat);
		PolyBiv c_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, C_cpu);
		int ret       = pvda_vec_znx_rotate(module, p, &c_biv, &a_biv);
		cr_assert_eq(ret, 0, "pvda_vec_znx_rotate failed for p=%lld", (long long)p);

		// GPU
		static int64_t C_gpu[L * POLY_SIZE];
		gpu_vec_znx_rotate(p, a_flat, C_gpu, POLY_SIZE, L, POLY_SIZE, L, POLY_SIZE);

		for (int i = 0; i < L * POLY_SIZE; i++)
			cr_assert_eq(C_cpu[i], C_gpu[i], "p=%lld: mismatch at index %d: CPU=%lld GPU=%lld", (long long)p, i,
			             (long long)C_cpu[i], (long long)C_gpu[i]);
	}

	pvda_delete_module_info(module);
}

// ---------------------------------------------------------------------------
// Test 2: GPU automorphism matches CPU (spqlios reference), contiguous layout
// ---------------------------------------------------------------------------

Test(gpu_vec_znx_rotate, automorphism_gpu_matches_cpu)
{
	srand(43);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	static int64_t a_flat[L * POLY_SIZE];
	for (int i = 0; i < L * POLY_SIZE; i++) a_flat[i] = (int64_t)(rand() % 2001) - 1000;

	// p must be odd, 0 < p < 2n
	int64_t test_p[] = {1, 3, 5, 2 * POLY_SIZE - 1, POLY_SIZE + 1, 17};

	for (size_t t = 0; t < sizeof(test_p) / sizeof(test_p[0]); t++)
	{
		int64_t p = test_p[t];

		// CPU (spqlios)
		static int64_t C_cpu[L * POLY_SIZE];
		PolyBiv a_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, a_flat);
		PolyBiv c_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, C_cpu);
		int ret       = pvda_vec_znx_automorphism(module, p, &c_biv, &a_biv);
		cr_assert_eq(ret, 0, "pvda_vec_znx_automorphism failed for p=%lld", (long long)p);

		// GPU
		static int64_t C_gpu[L * POLY_SIZE];
		gpu_vec_znx_automorphism(p, a_flat, C_gpu, POLY_SIZE, L, POLY_SIZE, L, POLY_SIZE);

		for (int i = 0; i < L * POLY_SIZE; i++)
			cr_assert_eq(C_cpu[i], C_gpu[i], "p=%lld: mismatch at index %d: CPU=%lld GPU=%lld", (long long)p, i,
			             (long long)C_cpu[i], (long long)C_gpu[i]);
	}

	pvda_delete_module_info(module);
}

// ---------------------------------------------------------------------------
// Test 3: strided SOURCE (mirrors glwe_extract_poly_view: one logical
// polynomial's limbs interleaved among PACKS-1 others, stride = PACKS*n).
// Exercises the a_sl != n path used by glwegadget_automorphism.
// ---------------------------------------------------------------------------

#define PACKS 3  // e.g. k+1 = 3 for a k=2 GLWE (a_0, a_1, b interleaved)

Test(gpu_vec_znx_rotate, automorphism_gpu_matches_cpu_strided_source)
{
	srand(44);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	// Interleaved buffer: limb i, poly j lives at [i*PACKS*n + j*n, +n)
	static int64_t interleaved[L * PACKS * POLY_SIZE];
	for (size_t i = 0; i < L * PACKS * POLY_SIZE; i++) interleaved[i] = (int64_t)(rand() % 2001) - 1000;

	int64_t stride     = PACKS * POLY_SIZE;
	int poly_idx       = 1;  // pick the middle logical polynomial, like glwe_extract_poly_view
	int64_t* a_strided = interleaved + poly_idx * POLY_SIZE;

	int64_t p = 5;

	// CPU (spqlios) — strided source, contiguous dest (matches glwegadget_automorphism's
	// pvda_vec_znx_automorphism(module, p, auto_tmp /*contiguous*/, &a /*strided*/) usage)
	static int64_t C_cpu[L * POLY_SIZE];
	PolyBiv a_biv = new_biv_view(POLY_SIZE, L, stride, a_strided);
	PolyBiv c_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, C_cpu);
	int ret       = pvda_vec_znx_automorphism(module, p, &c_biv, &a_biv);
	cr_assert_eq(ret, 0, "pvda_vec_znx_automorphism failed");

	// GPU
	static int64_t C_gpu[L * POLY_SIZE];
	gpu_vec_znx_automorphism(p, a_strided, C_gpu, POLY_SIZE, L, POLY_SIZE, L, stride);

	for (int i = 0; i < L * POLY_SIZE; i++)
		cr_assert_eq(C_cpu[i], C_gpu[i], "mismatch at index %d: CPU=%lld GPU=%lld", i, (long long)C_cpu[i],
		             (long long)C_gpu[i]);

	pvda_delete_module_info(module);
}

// ---------------------------------------------------------------------------
// Test 4: strided DEST — verifies the GPU per-limb copy-out never touches
// the "gap" elements between limbs (which belong to other interleaved
// polynomials), matching the CPU reference's behaviour exactly.
// ---------------------------------------------------------------------------

Test(gpu_vec_znx_rotate, rotate_gpu_matches_cpu_strided_dest)
{
	srand(45);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	static int64_t a_flat[L * POLY_SIZE];
	for (int i = 0; i < L * POLY_SIZE; i++) a_flat[i] = (int64_t)(rand() % 2001) - 1000;

	int64_t stride = PACKS * POLY_SIZE;
	int poly_idx   = 1;
	int64_t p      = -7;

	// CPU: write into a strided view of a sentinel-filled interleaved buffer.
	static int64_t cpu_interleaved[L * PACKS * POLY_SIZE];
	for (size_t i = 0; i < L * PACKS * POLY_SIZE; i++) cpu_interleaved[i] = 424242;
	PolyBiv a_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, a_flat);
	PolyBiv c_biv = new_biv_view(POLY_SIZE, L, stride, cpu_interleaved + poly_idx * POLY_SIZE);
	int ret       = pvda_vec_znx_rotate(module, p, &c_biv, &a_biv);
	cr_assert_eq(ret, 0, "pvda_vec_znx_rotate failed");

	// GPU: same sentinel-filled interleaved buffer.
	static int64_t gpu_interleaved[L * PACKS * POLY_SIZE];
	for (size_t i = 0; i < L * PACKS * POLY_SIZE; i++) gpu_interleaved[i] = 424242;
	int64_t* res_strided = gpu_interleaved + poly_idx * POLY_SIZE;
	gpu_vec_znx_rotate(p, a_flat, res_strided, POLY_SIZE, L, stride, L, POLY_SIZE);

	// Every element (touched limb coefficients AND untouched gap sentinels)
	// must match between the two interleaved buffers.
	for (size_t i = 0; i < L * PACKS * POLY_SIZE; i++)
		cr_assert_eq(cpu_interleaved[i], gpu_interleaved[i], "mismatch at flat index %zu: CPU=%lld GPU=%lld", i,
		             (long long)cpu_interleaved[i], (long long)gpu_interleaved[i]);

	// Sanity: the gaps really were left untouched (still the sentinel), i.e.
	// this test is actually exercising the strided-write path.
	for (int i = 0; i < L; i++)
		for (int other = 0; other < PACKS; other++)
		{
			if (other == poly_idx) continue;
			int64_t v = gpu_interleaved[i * stride + other * POLY_SIZE];
			cr_assert_eq(v, 424242, "gap at limb=%d poly=%d was clobbered: %lld", i, other, (long long)v);
		}

	pvda_delete_module_info(module);
}

// ---------------------------------------------------------------------------
// Test 5: gpu_active_stream override (pvda_gpu_stream_push/pop) must not
// change the result — same invariant as test_gpu_vec_znx_arith.c.
// ---------------------------------------------------------------------------

Test(gpu_vec_znx_rotate, automorphism_gpu_matches_cpu_on_pushed_stream)
{
	srand(46);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	static int64_t a_flat[L * POLY_SIZE];
	for (int i = 0; i < L * POLY_SIZE; i++) a_flat[i] = (int64_t)(rand() % 2001) - 1000;

	int64_t p = 9;

	static int64_t C_cpu[L * POLY_SIZE];
	PolyBiv a_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, a_flat);
	PolyBiv c_biv = new_biv_view(POLY_SIZE, L, POLY_SIZE, C_cpu);
	int ret       = pvda_vec_znx_automorphism(module, p, &c_biv, &a_biv);
	cr_assert_eq(ret, 0, "pvda_vec_znx_automorphism failed");

	void* stream = pvda_gpu_stream_create();
	cr_assert_not_null(stream, "pvda_gpu_stream_create failed");
	pvda_gpu_stream_push(stream);
	cr_assert_eq(pvda_gpu_stream_get_active(), stream, "active stream not switched by push");

	static int64_t C_gpu[L * POLY_SIZE];
	gpu_vec_znx_automorphism(p, a_flat, C_gpu, POLY_SIZE, L, POLY_SIZE, L, POLY_SIZE);

	for (int i = 0; i < L * POLY_SIZE; i++)
		cr_assert_eq(C_cpu[i], C_gpu[i], "mismatch at index %d on pushed stream: CPU=%lld GPU=%lld", i,
		             (long long)C_cpu[i], (long long)C_gpu[i]);

	pvda_gpu_stream_pop();
	cr_assert_null(pvda_gpu_stream_get_active(), "pop did not restore the default stream");
	pvda_gpu_stream_destroy(stream);

	pvda_delete_module_info(module);
}

// ---------------------------------------------------------------------------
// Test 6: zero input produces zero output for both ops
// ---------------------------------------------------------------------------

Test(gpu_vec_znx_rotate, zero_input)
{
	static int64_t zero[L * POLY_SIZE];
	memset(zero, 0, sizeof(zero));

	static int64_t C_rot[L * POLY_SIZE];
	gpu_vec_znx_rotate(3, zero, C_rot, POLY_SIZE, L, POLY_SIZE, L, POLY_SIZE);
	for (int i = 0; i < L * POLY_SIZE; i++) cr_assert_eq(C_rot[i], 0LL, "rotate: zero input must give zero at %d", i);

	static int64_t C_auto[L * POLY_SIZE];
	gpu_vec_znx_automorphism(3, zero, C_auto, POLY_SIZE, L, POLY_SIZE, L, POLY_SIZE);
	for (int i = 0; i < L * POLY_SIZE; i++)
		cr_assert_eq(C_auto[i], 0LL, "automorphism: zero input must give zero at %d", i);
}

// ---------------------------------------------------------------------------
// Test 7: res_size > a_size — CPU reference (vec_znx_automorphism_ref /
// vec_znx_rotate_ref) zero-pads res limbs beyond a_size instead of reading
// out of bounds. glwegadget_automorphism relies on exactly this: its
// scratch buffer's l (biv_l = max(l_b_result, nrows)) can exceed the source
// polynomial's l, and glwegadget_half_prod later reads all `nrows` limbs of
// that scratch buffer regardless.
// ---------------------------------------------------------------------------

#define L_SMALL 2  // a_size: fewer limbs than res_size below
#define L_BIG   5  // res_size

Test(gpu_vec_znx_rotate, automorphism_gpu_matches_cpu_zero_pads_tail)
{
	srand(47);

	MODULE* module = pvda_new_module_info(POLY_SIZE);
	cr_assert_not_null(module, "MODULE allocation failed");

	static int64_t a_flat[L_SMALL * POLY_SIZE];
	for (int i = 0; i < L_SMALL * POLY_SIZE; i++) a_flat[i] = (int64_t)(rand() % 2001) - 1000;

	int64_t p = 5;

	// CPU (spqlios): res has more limbs (L_BIG) than a (L_SMALL) — the tail
	// must come out zeroed, not garbage.
	static int64_t C_cpu[L_BIG * POLY_SIZE];
	memset(C_cpu, 0xAB, sizeof(C_cpu));  // poison, so a missed zero-pad would be visible
	PolyBiv a_biv = new_biv_view(POLY_SIZE, L_SMALL, POLY_SIZE, a_flat);
	PolyBiv c_biv = new_biv_view(POLY_SIZE, L_BIG, POLY_SIZE, C_cpu);
	int ret       = pvda_vec_znx_automorphism(module, p, &c_biv, &a_biv);
	cr_assert_eq(ret, 0, "pvda_vec_znx_automorphism failed");

	// GPU
	static int64_t C_gpu[L_BIG * POLY_SIZE];
	memset(C_gpu, 0xAB, sizeof(C_gpu));
	gpu_vec_znx_automorphism(p, a_flat, C_gpu, POLY_SIZE, L_BIG, POLY_SIZE, L_SMALL, POLY_SIZE);

	for (int i = 0; i < L_BIG * POLY_SIZE; i++)
		cr_assert_eq(C_cpu[i], C_gpu[i], "mismatch at index %d: CPU=%lld GPU=%lld", i, (long long)C_cpu[i],
		             (long long)C_gpu[i]);

	// Sanity: the tail really is zero (not just "matches CPU which happened
	// to also be poisoned"), i.e. this test actually exercises the zero-pad.
	for (int i = L_SMALL * POLY_SIZE; i < L_BIG * POLY_SIZE; i++)
		cr_assert_eq(C_gpu[i], 0LL, "tail limb at index %d must be zero-padded, got %lld", i, (long long)C_gpu[i]);

	pvda_delete_module_info(module);
}
