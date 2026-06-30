#include <criterion/criterion.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gpu/host/poly_add_host.h"

#define TEST_MODULUS ((int64_t)0x7FFFFFFF)

static void cpu_poly_add_mod(const int64_t* a, const int64_t* b, int64_t* c, uint64_t n, int64_t q)
{
	for (uint64_t i = 0; i < n; i++)
	{
		int64_t sum = a[i] + b[i];
		sum         = sum % q;
		if (sum < 0) sum += q;
		c[i] = sum;
	}
}

static void fill_random(int64_t* buf, uint64_t n, int64_t q)
{
	for (uint64_t i = 0; i < n; i++) buf[i] = (int64_t)((unsigned)rand() % (unsigned)q);
}

static double elapsed_ms(struct timespec t0, struct timespec t1)
{
	return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
}

// ---------------------------------------------------------------------------
// Test 1: Correctness — GPU output must match CPU output exactly
// ---------------------------------------------------------------------------

Test(gpu_poly_add, correctness_vs_cpu)
{
	const uint64_t n = 1u << 16;
	const int64_t q  = TEST_MODULUS;

	int64_t* a     = malloc(n * sizeof(int64_t));
	int64_t* b     = malloc(n * sizeof(int64_t));
	int64_t* cpu_c = malloc(n * sizeof(int64_t));
	int64_t* gpu_c = malloc(n * sizeof(int64_t));

	cr_assert(a && b && cpu_c && gpu_c, "Memory allocation failed");

	srand(42);
	fill_random(a, n, q);
	fill_random(b, n, q);

	cpu_poly_add_mod(a, b, cpu_c, n, q);

	int ret = gpu_poly_add_mod(a, b, gpu_c, n, q);
	cr_assert_eq(ret, 0, "gpu_poly_add_mod returned %d (CUDA error)", ret);

	for (uint64_t i = 0; i < n; i++)
		cr_assert_eq(gpu_c[i], cpu_c[i], "Mismatch at index %llu: cpu=%lld  gpu=%lld", (unsigned long long)i,
		             (long long)cpu_c[i], (long long)gpu_c[i]);

	free(a);
	free(b);
	free(cpu_c);
	free(gpu_c);
}

// ---------------------------------------------------------------------------
// Test 2: Timing comparison — CPU vs GPU throughput
// ---------------------------------------------------------------------------
/*
Test(gpu_poly_add, timing_comparison)
{
    const uint64_t n = 1u << 20;
    const int64_t  q = TEST_MODULUS;

    int64_t* a     = malloc(n * sizeof(int64_t));
    int64_t* b     = malloc(n * sizeof(int64_t));
    int64_t* cpu_c = malloc(n * sizeof(int64_t));
    int64_t* gpu_c = malloc(n * sizeof(int64_t));

    cr_assert(a && b && cpu_c && gpu_c, "Memory allocation failed");

    srand(123);
    fill_random(a, n, q);
    fill_random(b, n, q);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cpu_poly_add_mod(a, b, cpu_c, n, q);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_ms = elapsed_ms(t0, t1);

    float gpu_kernel_ms = 0.0f;
    int   ret           = gpu_poly_add_mod_timed(a, b, gpu_c, n, q, &gpu_kernel_ms);
    cr_assert_eq(ret, 0, "gpu_poly_add_mod_timed returned %d (CUDA error)", ret);

    printf("\n");
    printf("  poly_add_mod, n = %u coefficients, q = %lld\n", (unsigned)n, (long long)q);
    printf("  CPU time       : %.3f ms\n", cpu_ms);
    printf("  GPU kernel     : %.3f ms  (excludes H2D/D2H)\n", (double)gpu_kernel_ms);
    if (gpu_kernel_ms > 0.0f)
        printf("  Kernel speedup : %.2fx\n", cpu_ms / (double)gpu_kernel_ms);
    printf("\n");

    for (uint64_t i = 0; i < n; i++)
        cr_assert_eq(gpu_c[i], cpu_c[i],
                     "Mismatch at index %llu", (unsigned long long)i);

    free(a);
    free(b);
    free(cpu_c);
    free(gpu_c);
}
*/