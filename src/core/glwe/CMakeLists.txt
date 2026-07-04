# ----------------------------------------
# Build GLWE shared library
# ----------------------------------------

# Source files for GLWE
set(GLWE_SRCS
    glwe_arithmetic.c
    glwe_params.c
    glwe_ciphertext.c
    glwe_key.c
    glwe_transform_key.c
    univariate_polynomial.c
    bivariate_polynomial.c
)

# Create a shared library named GLWE
add_library(PVDA_GLWE ${GLWE_SRCS})

# Set the output library name explicitly
set_target_properties(PVDA_GLWE PROPERTIES OUTPUT_NAME pvdaglwe)

# Link dependencies
target_link_libraries(PVDA_GLWE PRIVATE libspqlios PVDA_COMMON PVDA_BACKEND)

# Include directories
target_include_directories(PVDA_GLWE
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include/core/glwe/   # Public headers for consumers
)

if(ENABLE_CUDA)
    target_link_libraries(PVDA_GLWE PRIVATE PVDA_GPU)
    target_include_directories(PVDA_GLWE PRIVATE
        ${PROJECT_SOURCE_DIR}/include/backend
    )
    target_compile_definitions(PVDA_GLWE PRIVATE ENABLE_CUDA)
endif()
