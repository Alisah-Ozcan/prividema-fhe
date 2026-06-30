cmake_minimum_required(VERSION 3.10)
project(prividema-fhe)

# ----------------------------------------
# Include FetchContent for external dependencies
# Enable testing framework
# ----------------------------------------
include(FetchContent)
include(CTest)
enable_testing()

# ----------------------------------------
# Set C standard
# ----------------------------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ----------------------------------------
# Build options
# ----------------------------------------
# ----------------------------------------
# Build Options
# ----------------------------------------
# Note on CMake cache:
# When you first configure the project with specific options (e.g., -DBUILD_TESTS=ON),
# the values are stored in CMake's cache at build/CMakeCache.txt.
# If you later re-run CMake without specifying the option, it will reuse the cached value,
# which may not be what you expect.
#
# Example:
#   cmake .. -DBUILD_TESTS=ON   # BUILD_TESTS is set to ON
#   cmake ..                    # BUILD_TESTS will still be ON due to cache
#
# To reset, either:
#   - Use -U option to remove cached variables
#   - Delete the CMakeCache.txt file manually

option(BUILD_TESTS "Build the Tests" ON)
option(BUILD_EXAMPLES "Build the examples" ON)
option(BUILD_BENCHMARKS "Build the benchmarks" OFF)
option(PROFILING_OPTIONS "Add options to allow for proper profiling even in realease-like builds" OFF)
option(BUILD_NATIVE "Enable march=native optimisations" OFF)
option(ENABLE_DEBUG "Enable additional prints for debugging purpose" OFF)
option(BUILD_DOCS "Build the documentation using Doxygen" OFF)
option(ENABLE_CUDA "Enable CUDA GPU backend" OFF)

# ----------------------------------------
# Debug print configuration
# ----------------------------------------
if(ENABLE_DEBUG)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
  endif()
  message(STATUS "Debug Prints Enabled")
  add_compile_definitions(DEBUG=1)
else()
  add_compile_definitions(DEBUG=0)
endif()

if(PROFILING_OPTIONS)
  add_compile_options(-fno-omit-frame-pointer -g)
endif()

if(BUILD_NATIVE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

    if(COMPILER_SUPPORTS_MARCH_NATIVE)
      add_compile_options(-march=native)
      message(STATUS "Native optimization (-march=native) enabled.")
    else()
      message(WARNING "-march=native is not supported by this compiler.")
    endif()

  elseif(MSVC)
    message(STATUS "MSVC detected. -march=native skipped (manual /arch flag required).")
  endif()

endif()

# ----------------------------------------
# CUDA configuration
# ----------------------------------------
if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA enabled — toolkit version: ${CUDAToolkit_VERSION}")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Cover Maxwell (52) through Hopper (90); override with -DCMAKE_CUDA_ARCHITECTURES=<sm>
    set(CMAKE_CUDA_ARCHITECTURES 52 60 70 75 80 86 89 90)
  endif()
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported OUTPUT lto_error)

if(lto_supported)
  message(STATUS "LTO/IPO enabled for optimized builds")
else()
  message(STATUS "LTO not supported: ${lto_error}")
endif()

# ----------------------------------------
# Import spqlios library
#
# ----------------------------------------
# Disable spqlios internal tests
set(ENABLE_TESTING OFF CACHE BOOL "Disable Tests" FORCE)
add_subdirectory(spqlios-arithmetic)

# Enable LTO after external dependencies.
# Disabled when ENABLE_CUDA=ON: CUDA fat-binary symbols conflict with GCC LTO.
if(lto_supported AND NOT ENABLE_CUDA)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  add_compile_options(-flto)
  add_link_options(-flto)
elseif(ENABLE_CUDA)
  message(STATUS "LTO disabled — incompatible with CUDA fat-binary linking")
endif()

# ----------------------------------------
# Add main source directory
# ----------------------------------------
add_subdirectory(src)


# ----------------------------------------
# Build unit tests if enabled
# ----------------------------------------
if(BUILD_TESTS)
  message(STATUS "Building Test Files")
  if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING
    "Upstream Criterion 2.4.3 (and possibly older versions) have a bug where Release (optimized) builds"
    "can have their parameterized tests be optimized away (deleted).\n"
    "PATCH YOUR LOCAL CRITERION INSTALLATION OR COMPILE IN Debug MODE IF YOU WANT TO RUN THE TESTS!"
    )
  endif()
  add_subdirectory(tests)
endif()

# ----------------------------------------
# Build examples if enabled
# ----------------------------------------
if(BUILD_EXAMPLES)
  message(STATUS "Building Example Files")
  add_subdirectory(examples)
endif()

if(BUILD_BENCHMARKS)
  message(STATUS "Building Benchmark Files")
  add_subdirectory(benchmarks)
endif()

# ----------------------------------------
# Doxygen documentation
# ----------------------------------------
# ----------------------------------------
# clang-format target
# ----------------------------------------
find_program(CLANG_FORMAT_EXE clang-format)
if(CLANG_FORMAT_EXE)
  file(GLOB_RECURSE ALL_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/*.c
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.cu
    ${PROJECT_SOURCE_DIR}/include/*.h
    ${PROJECT_SOURCE_DIR}/include/*.hpp
    ${PROJECT_SOURCE_DIR}/include/*.cuh
    ${PROJECT_SOURCE_DIR}/tests/*.c
    ${PROJECT_SOURCE_DIR}/tests/*.cpp
    ${PROJECT_SOURCE_DIR}/tests/*.cu
    ${PROJECT_SOURCE_DIR}/examples/*.c
    ${PROJECT_SOURCE_DIR}/examples/*.cpp
  )
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format on all source files"
    VERBATIM
  )
  add_custom_target(format-check
    COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror ${ALL_SOURCE_FILES}
    COMMENT "Checking clang-format compliance (no files modified)"
    VERBATIM
  )
else()
  message(STATUS "clang-format not found — format/format-check targets unavailable")
endif()

if(BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(doc_html ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating HTML documentation from Markdown"
            VERBATIM
        )
  else()
    message(WARNING "Doxygen needs to be installed to generate the documentation")
  endif()
endif()
