#include <criterion/criterion.h>
#include <stddef.h>
#include <stdint.h>

#include "gpu/common/gpu_nttparameters.h"
#include "gpu/host/ggsw_external_product_gpu.h"

#define CARRIER_PRIME 1152921504634109953ULL

// ---------------------------------------------------------------------------
// Test 1: Singleton initializes tables — pointers non-null, modulus correct
// ---------------------------------------------------------------------------

Test(gpu_nttparameters, init_returns_valid_tables)
{
	const size_t n = 1u << 12;

	gpu_ntt_initialize(n);

	const uint64_t* ntt  = gpu_ntt_get_table(n);
	const uint64_t* intt = gpu_ntt_get_intt_table(n);

	cr_assert_not_null(ntt, "NTT table pointer must not be null for n=%zu", n);
	cr_assert_not_null(intt, "INTT table pointer must not be null for n=%zu", n);

	cr_assert_eq(gpu_ntt_get_modulus(), CARRIER_PRIME, "Modulus must be the carrier prime %llu",
	             (unsigned long long)CARRIER_PRIME);
}

// ---------------------------------------------------------------------------
// Test 2: Cache — repeated initialize returns the same pointer (no realloc)
// ---------------------------------------------------------------------------

Test(gpu_nttparameters, cache_hit_same_pointer)
{
	const size_t n = 1u << 14;

	gpu_ntt_initialize(n);
	const uint64_t* ptr_first = gpu_ntt_get_table(n);

	gpu_ntt_initialize(n);
	const uint64_t* ptr_second = gpu_ntt_get_table(n);

	cr_assert_eq(ptr_first, ptr_second, "Second initialize must return cached pointer, not reallocate");
}

// ---------------------------------------------------------------------------
// Test 3: Different ring sizes produce distinct table pointers
// ---------------------------------------------------------------------------

Test(gpu_nttparameters, different_sizes_distinct_tables)
{
	const size_t n1 = 1u << 12;
	const size_t n2 = 1u << 14;

	gpu_ntt_initialize(n1);
	gpu_ntt_initialize(n2);

	const uint64_t* ntt1 = gpu_ntt_get_table(n1);
	const uint64_t* ntt2 = gpu_ntt_get_table(n2);

	cr_assert_not_null(ntt1, "NTT table for n=%zu must not be null", n1);
	cr_assert_not_null(ntt2, "NTT table for n=%zu must not be null", n2);
	cr_assert_neq(ntt1, ntt2, "Tables for different ring sizes must be at different addresses");
}

// ---------------------------------------------------------------------------
// Test 4: NTT and INTT tables are at different addresses for the same size
// ---------------------------------------------------------------------------

Test(gpu_nttparameters, ntt_and_intt_distinct)
{
	const size_t n = 1u << 16;

	gpu_ntt_initialize(n);

	const uint64_t* ntt  = gpu_ntt_get_table(n);
	const uint64_t* intt = gpu_ntt_get_intt_table(n);

	cr_assert_neq(ntt, intt, "NTT and INTT tables must occupy separate device buffers");
}

// ---------------------------------------------------------------------------
// Test 5: n_inv precomputed — n * n_inv ≡ 1 (mod prime)
// ---------------------------------------------------------------------------

Test(gpu_nttparameters, n_inv_is_correct)
{
	const size_t n = 1u << 12;

	gpu_ntt_initialize(n);

	uint64_t prime = gpu_ntt_get_modulus();
	uint64_t n_inv = gpu_ntt_get_n_inv(n);

	// n * n_inv mod prime == 1
	__uint128_t product = (__uint128_t)n * (__uint128_t)n_inv;
	uint64_t result     = (uint64_t)(product % (__uint128_t)prime);

	cr_assert_eq(result, 1ULL, "n * n_inv mod prime must equal 1, got %llu", (unsigned long long)result);
}

// ---------------------------------------------------------------------------
// Test 6: n_inv cache — different ring sizes yield different n_inv values
// ---------------------------------------------------------------------------

Test(gpu_nttparameters, n_inv_differs_per_size)
{
	const size_t n1 = 1u << 12;
	const size_t n2 = 1u << 14;

	gpu_ntt_initialize(n1);
	gpu_ntt_initialize(n2);

	uint64_t inv1 = gpu_ntt_get_n_inv(n1);
	uint64_t inv2 = gpu_ntt_get_n_inv(n2);

	cr_assert_neq(inv1, inv2, "Different ring sizes must yield different n_inv values");
}

// ---------------------------------------------------------------------------
// Test 7: pvda_is_device_pointer correctly distinguishes host from device
// ---------------------------------------------------------------------------

Test(gpu_nttparameters, device_pointer_detection)
{
	static int64_t host_buf[64];

	cr_assert_eq(pvda_is_device_pointer(host_buf), 0, "Stack buffer must be identified as host pointer");

	int64_t* d_buf = pvda_gpu_alloc(64);
	cr_assert_not_null(d_buf, "GPU alloc failed");
	cr_assert_eq(pvda_is_device_pointer(d_buf), 1, "cudaMalloc buffer must be identified as device pointer");

	pvda_gpu_free(d_buf);
}
