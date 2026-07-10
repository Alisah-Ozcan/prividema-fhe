cmake_minimum_required(VERSION 3.10)

# Collect all example source files — GPU-specific ones (*_gpu*.c, e.g.
# exemples_PIR_gpu.c / exemples_PIR_gpu_stream.c) are built separately below,
# only when ENABLE_CUDA is set, since they call GPU functions directly.
file(GLOB TEST_FILES *.c)
list(FILTER TEST_FILES EXCLUDE REGEX ".*_gpu.*\\.c$")

# Create test executables for each test file
foreach(test_file ${TEST_FILES})
  # Get filename without extension
  get_filename_component(test_name ${test_file} NAME_WE)

  # Create executable
  add_executable(${test_name} ${test_file})

  # Link with the main project library
  target_link_libraries(${test_name} PVDA_CORE PVDA_SCHEMES)

  # Add include path for headers
  target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)

endforeach()

if(ENABLE_CUDA)
  file(GLOB GPU_TEST_FILES *_gpu*.c)

  foreach(test_file ${GPU_TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)

    add_executable(${test_name} ${test_file})

    target_link_libraries(${test_name} PVDA_CORE PVDA_SCHEMES PVDA_GPU CUDA::cudart)

    target_include_directories(${test_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/backend
    )
    target_compile_definitions(${test_name} PRIVATE ENABLE_CUDA)
  endforeach()
endif()
