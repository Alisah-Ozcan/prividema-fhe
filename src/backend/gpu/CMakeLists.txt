# ----------------------------------------
# Build GPU backend library (CUDA)
#
# Structure:
#   kernel/ — __global__ device kernels
#   host/   — extern "C" host wrappers that launch kernels
#   common/ — header-only shared constants and macros
# ----------------------------------------

add_library(PVDA_GPU STATIC
    common/gpu_nttparameters.cu
    kernel/poly_add_kernel.cu
    host/poly_add_host.cu
)

set_target_properties(PVDA_GPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME pvdagpu
)

target_include_directories(PVDA_GPU
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include/backend   # consumers: #include "gpu/host/poly_add_host.h"
)

target_link_libraries(PVDA_GPU PUBLIC CUDA::cudart GPUNTT::ntt)
