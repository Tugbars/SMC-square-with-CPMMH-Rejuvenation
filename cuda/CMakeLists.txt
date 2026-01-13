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
    get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT "CUDA" IN_LIST languages)
        enable_language(CUDA)
    endif()
endif()

# =============================================================================
# Find CUDA Toolkit
# =============================================================================

if(NOT DEFINED CUDAToolkit_FOUND OR NOT CUDAToolkit_FOUND)
    find_package(CUDAToolkit REQUIRED)
endif()
message(STATUS "[CUDA] Toolkit Version: ${CUDAToolkit_VERSION}")

# =============================================================================
# CUDA Configuration - Blackwell (RTX 5080)
# =============================================================================

set(CMAKE_CUDA_ARCHITECTURES 120)

# Aggressive optimization flags
set(CUDA_COMMON_OPTIONS
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --fmad=true
        --restrict
        -Xptxas -O3
        -Xptxas -dlcm=ca
        --extra-device-vectorization
        -lineinfo
        --expt-relaxed-constexpr
    >
)

# Windows-specific options
set(CUDA_WIN_OPTIONS
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=/W3
        -Xcompiler=/wd4244
        -Xcompiler=/wd4267
    >
)

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

target_compile_options(smc2_rbpf_cuda PRIVATE ${CUDA_COMMON_OPTIONS})

target_link_libraries(smc2_rbpf_cuda
    PUBLIC
        CUDA::cudart
        CUDA::curand
)

set_target_properties(smc2_rbpf_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES 120
)

if(WIN32)
    target_compile_options(smc2_rbpf_cuda PRIVATE ${CUDA_WIN_OPTIONS})
endif()

# =============================================================================
# Test Executable
# =============================================================================

add_executable(test_smc2_cuda
    test/test_smc2_cuda.cu
)

target_compile_options(test_smc2_cuda PRIVATE ${CUDA_COMMON_OPTIONS})

target_link_libraries(test_smc2_cuda 
    PRIVATE 
        smc2_rbpf_cuda
        CUDA::cudart
        CUDA::curand
)

set_target_properties(test_smc2_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES 120
)

if(WIN32)
    target_compile_options(test_smc2_cuda PRIVATE ${CUDA_WIN_OPTIONS})
else()
    target_link_libraries(test_smc2_cuda PRIVATE m)
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "  ┌─────────────────────────────────────────────────────────────┐")
message(STATUS "  │ SMC² RBPF CUDA                                              │")
message(STATUS "  │                                                             │")
message(STATUS "  │   Library:  smc2_rbpf_cuda                                  │")
message(STATUS "  │   Test:     test_smc2_cuda                                  │")
message(STATUS "  │   Target:   sm_120 (RTX 5080 Blackwell)                     │")
message(STATUS "  │   CUDA:     ${CUDAToolkit_VERSION}")
message(STATUS "  └─────────────────────────────────────────────────────────────┘")
message(STATUS "")