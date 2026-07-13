# ------------------------------------------------------------
# Criterion (portable, pkg-config only)
# ------------------------------------------------------------

find_package(PkgConfig REQUIRED)

# Creates imported target: PkgConfig::CRITERION
pkg_check_modules(CRITERION REQUIRED IMPORTED_TARGET criterion)

# ------------------------------------------------------------
# Platform-specific configuration
# ------------------------------------------------------------

if(APPLE)
  set(NEED_MATH_LIB FALSE)
else()
  set(NEED_MATH_LIB TRUE)
endif()

# ------------------------------------------------------------
# Collect all test source files
# ------------------------------------------------------------

file(GLOB TEST_BACKEND_FILES backend/*.c)
file(GLOB TEST_AVX_BACKEND_FILES backend/test_avx*.c)
file(GLOB TEST_COMMON_FILES  common/*.c)
file(GLOB TEST_GGSW_FILES core/ggsw/*.c)
file(GLOB TEST_GLWE_FILES core/glwe/*.c)
file(GLOB TEST_SCHEMES_FILES schemes/*.c)

add_subdirectory(utils)
# ------------------------------------------------------------
# Function to build unit tests
# ------------------------------------------------------------

function(BuildUnitTests testfiles libname)

  foreach(test_file IN LISTS ${testfiles})
    get_filename_component(test_name ${test_file} NAME_WE)

    add_executable(${test_name} ${test_file})

    # Project includes only
    target_include_directories(${test_name} PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            utils/include
        )

    # Link libraries
    target_link_libraries(${test_name} PRIVATE
            ${libname}
            libspqlios
            PVDA_UTEST_UTILS
            PVDA_CORE
            # Criterion via pkg-config
            PkgConfig::CRITERION
        )

    # Math lib (Linux only)
    if(NEED_MATH_LIB)
      target_link_libraries(${test_name} PRIVATE m)
    endif()

    # Register test
    add_test(
            NAME ${test_name}
            COMMAND $<TARGET_FILE:${test_name}>
        )

  endforeach()

endfunction()

# ------------------------------------------------------------
# Build all unit tests
# ------------------------------------------------------------

BuildUnitTests(TEST_COMMON_FILES      PVDA_COMMON)
BuildUnitTests(TEST_AVX_BACKEND_FILES PVDA_COMMON)
BuildUnitTests(TEST_GGSW_FILES        PVDA_CORE)
BuildUnitTests(TEST_GLWE_FILES        PVDA_CORE)
BuildUnitTests(TEST_SCHEMES_FILES     PVDA_SCHEMES)

# ------------------------------------------------------------
# CUDA GPU tests (only when ENABLE_CUDA=ON)
# ------------------------------------------------------------
if(ENABLE_CUDA)

  file(GLOB TEST_GPU_FILES backend/test_gpu*.c)

  function(BuildCudaTests testfiles)
    foreach(test_file IN LISTS ${testfiles})
      get_filename_component(test_name ${test_file} NAME_WE)

      add_executable(${test_name} ${test_file})

      set_target_properties(${test_name} PROPERTIES
          CUDA_RESOLVE_DEVICE_SYMBOLS ON
      )

      target_include_directories(${test_name} PRIVATE
          ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/include/backend
          utils/include
      )

      target_link_libraries(${test_name} PRIVATE
          PVDA_GPU
          PVDA_CORE
          PVDA_BACKEND
          PVDA_COMMON
          libspqlios
          PkgConfig::CRITERION
          CUDA::cudart
      )

      if(NEED_MATH_LIB)
        target_link_libraries(${test_name} PRIVATE m)
      endif()

      add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    endforeach()
  endfunction()

  BuildCudaTests(TEST_GPU_FILES)

  # test_gpu_glwegadget_automorphism / test_gpu_trace_expand need the shared
  # test-utils helpers (assert_tnx_close_enough / info_bits_half_prod /
  # rnx_random_vec) that plain BuildCudaTests targets don't link by default.
  target_link_libraries(test_gpu_glwegadget_automorphism PRIVATE PVDA_UTEST_UTILS)
  target_link_libraries(test_gpu_trace_expand PRIVATE PVDA_UTEST_UTILS)

  # test_ggsw_external_product also has an #ifdef ENABLE_CUDA GPU variant —
  # add GPU linking to that specific target (already created by BuildUnitTests above).
  target_link_libraries(test_ggsw_external_product PRIVATE PVDA_GPU CUDA::cudart)
  target_compile_definitions(test_ggsw_external_product PRIVATE ENABLE_CUDA)
  target_include_directories(test_ggsw_external_product PRIVATE
      ${PROJECT_SOURCE_DIR}/include/backend
  )

  # test_glwegadget_half_product also has an #ifdef ENABLE_CUDA GPU variant —
  # add GPU linking to that specific target (already created by BuildUnitTests above).
  target_link_libraries(test_glwegadget_half_product PRIVATE PVDA_GPU CUDA::cudart)
  target_compile_definitions(test_glwegadget_half_product PRIVATE ENABLE_CUDA)
  target_include_directories(test_glwegadget_half_product PRIVATE
      ${PROJECT_SOURCE_DIR}/include/backend
  )

  # test_glwe_ciphertext also has an #ifdef ENABLE_CUDA GPU variant (add_glwe/sub_glwe/normalize_glwe
  # dispatch) — add GPU linking to that specific target (already created by BuildUnitTests above).
  target_link_libraries(test_glwe_ciphertext PRIVATE PVDA_GPU CUDA::cudart)
  target_compile_definitions(test_glwe_ciphertext PRIVATE ENABLE_CUDA)
  target_include_directories(test_glwe_ciphertext PRIVATE
      ${PROJECT_SOURCE_DIR}/include/backend
  )

  # test_tfhe also has an #ifdef ENABLE_CUDA GPU variant (tfhe_cmux, inherited from the
  # GPU dispatch of sub_glwe/normalize_glwe/ggsw_external_product/add_glwe) — add GPU
  # linking to that specific target (already created by BuildUnitTests above).
  target_link_libraries(test_tfhe PRIVATE PVDA_GPU CUDA::cudart)
  target_compile_definitions(test_tfhe PRIVATE ENABLE_CUDA)
  target_include_directories(test_tfhe PRIVATE
      ${PROJECT_SOURCE_DIR}/include/backend
  )

endif()
