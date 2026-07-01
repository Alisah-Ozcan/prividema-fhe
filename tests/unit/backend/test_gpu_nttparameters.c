#include <criterion/criterion.h>
#include <stddef.h>
#include <stdint.h>

#include "gpu/common/gpu_nttparameters.h"

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
