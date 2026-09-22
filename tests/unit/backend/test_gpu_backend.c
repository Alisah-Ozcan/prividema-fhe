#include <criterion/criterion.h>
#include <stdint.h>

#include "backend/backend.h"

// Backend selection: the CUDA backend and pvda_new_backend's configuration.

Test(gpu_backend, cuda_backend_creation)
{
	PvdaBackend* backend = pvda_new_cuda_backend(4096);
	cr_assert_not_null(backend, "pvda_new_cuda_backend failed on a CUDA build with a GPU");
	cr_assert_eq(pvda_backend_type(backend), PVDA_BACKEND_CUDA);
	cr_assert_eq(pvda_module_extract_nn(backend), 4096);
	pvda_delete_backend(backend);
}

Test(gpu_backend, new_backend_from_config)
{
	PvdaBackendConfig cpu_config = {.nn = 1024};
	PvdaBackend* cpu             = pvda_new_backend(cpu_config);
	cr_assert_not_null(cpu);
	cr_assert_eq(pvda_backend_type(cpu), PVDA_BACKEND_SPQLIOS, "zero-initialized type should select spqlios");
	cr_assert_eq(pvda_module_extract_nn(cpu), 1024);

	PvdaBackendConfig gpu_config = {.nn = 1024, .type = PVDA_BACKEND_CUDA};
	PvdaBackend* gpu             = pvda_new_backend(gpu_config);
	cr_assert_not_null(gpu);
	cr_assert_eq(pvda_backend_type(gpu), PVDA_BACKEND_CUDA);
	cr_assert_eq(pvda_module_extract_nn(gpu), 1024);

	pvda_delete_backend(cpu);
	pvda_delete_backend(gpu);
}

Test(gpu_backend, spqlios_backend_type)
{
	PvdaBackend* backend = pvda_new_spqlios_backend(2048);
	cr_assert_eq(pvda_backend_type(backend), PVDA_BACKEND_SPQLIOS);
	pvda_delete_backend(backend);
}
