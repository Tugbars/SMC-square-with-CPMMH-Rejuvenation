# =============================================================================
# SMC² RBPF - CUDA Implementation
# =============================================================================

cmake_minimum_required(VERSION 3.20)

# Allow standalone build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(smc2_rbpf_cuda VERSION 1.0.0 LANGUAGES CXX CUDA)
    
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
    
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
else()
    # Enable CUDA when included from parent
    enable_language(CUDA)
endif()

# =============================================================================
# Find CUDA Toolkit
# =============================================================================

find_package(CUDAToolkit REQUIRED)
message(STATUS "[CUDA] Toolkit Version: ${CUDAToolkit_VERSION}")

# =============================================================================
# CUDA Configuration
# =============================================================================

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xcompiler -O3" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0 -Xcompiler -Od" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -lineinfo" CACHE STRING "" FORCE)

# GPU architectures (auto-detect or use common ones)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Common architectures: Volta(70), Turing(75), Ampere(80,86), Ada(89), Hopper(90)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures" FORCE)
endif()
message(STATUS "[CUDA] Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# =============================================================================
# Library: smc2_rbpf_cuda
# =============================================================================

add_library(smc2_rbpf_cuda STATIC
    src/smc2_rbpf_cuda.cu
)

target_include_directories(smc2_rbpf_cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(smc2_rbpf_cuda
    PUBLIC
        CUDA::cudart
        CUDA::curand
)

set_target_properties(smc2_rbpf_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# =============================================================================
# Test Executable
# =============================================================================

add_executable(test_smc2_cuda
    test/test_smc2_cuda.cu
)

target_link_libraries(test_smc2_cuda PRIVATE smc2_rbpf_cuda)

set_target_properties(test_smc2_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "[CUDA] Targets: smc2_rbpf_cuda (lib), test_smc2_cuda (exe)")
