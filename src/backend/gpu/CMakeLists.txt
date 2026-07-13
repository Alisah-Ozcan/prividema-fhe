# ----------------------------------------
# Build GPU backend library (CUDA)
#
# Structure:
#   kernel/ — __global__ device kernels
#   host/   — extern "C" host wrappers that launch kernels
#   common/ — header-only shared constants and macros
# ----------------------------------------

add_library(PVDA_GPU STATIC
    common/gpu_stream.cu
    common/gpu_nttparameters.cu
    kernel/ntt_poly_mult_kernel.cu
    kernel/vec_znx_arith_kernel.cu
    host/ntt_poly_mult_host.cu
    host/vec_znx_arith_host.cu
    host/ggsw_external_product_gpu.cu
)

set_target_properties(PVDA_GPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME pvdagpu
)

target_include_directories(PVDA_GPU
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include/backend   # consumers: #include "gpu/host/poly_add_host.h"
    PRIVATE
        # Core struct headers needed by ggsw_external_product_gpu.cu
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/common
        ${PROJECT_SOURCE_DIR}/include/core/glwe
        ${PROJECT_SOURCE_DIR}/include/core/ggsw
)

target_link_libraries(PVDA_GPU PUBLIC CUDA::cudart GPUNTT::ntt)
