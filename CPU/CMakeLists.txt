# =============================================================================
# SMC² RBPF - CPU Implementation
# =============================================================================

cmake_minimum_required(VERSION 3.20)

# Allow standalone build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(smc2_rbpf_cpu VERSION 1.0.0 LANGUAGES C)
    
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
    
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

# =============================================================================
# C Standard
# =============================================================================

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# =============================================================================
# Compiler Flags
# =============================================================================

message(STATUS "[CPU] C Compiler: ${CMAKE_C_COMPILER_ID}")

if(CMAKE_C_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_C_COMPILER_ID MATCHES "Intel")
    message(STATUS "[CPU] Configuring for Intel ICX")
    
    if(WIN32)
        set(CMAKE_C_FLAGS_RELEASE "/O3 /Qipo /QxHost /fp:fast" CACHE STRING "" FORCE)
        set(CMAKE_C_FLAGS_DEBUG "/Od /Zi /DEBUG" CACHE STRING "" FORCE)
        set(OPENMP_FLAG "/Qopenmp")
    else()
        set(CMAKE_C_FLAGS_RELEASE "-O3 -ipo -xHost -fp-model=fast" CACHE STRING "" FORCE)
        set(CMAKE_C_FLAGS_DEBUG "-O0 -g" CACHE STRING "" FORCE)
        set(OPENMP_FLAG "-qopenmp")
    endif()
    
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    message(STATUS "[CPU] Configuring for MSVC")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /GL /fp:fast" CACHE STRING "" FORCE)
    set(CMAKE_C_FLAGS_DEBUG "/Od /Zi" CACHE STRING "" FORCE)
    set(OPENMP_FLAG "/openmp")
    
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(STATUS "[CPU] Configuring for GCC/Clang")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -ffast-math" CACHE STRING "" FORCE)
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g" CACHE STRING "" FORCE)
    set(OPENMP_FLAG "-fopenmp")
endif()

# =============================================================================
# Find OpenMP
# =============================================================================

find_package(OpenMP QUIET)
if(OpenMP_C_FOUND)
    message(STATUS "[CPU] OpenMP: Found (${OpenMP_C_VERSION})")
else()
    message(STATUS "[CPU] OpenMP: Not found")
endif()

# =============================================================================
# Library: smc2_rbpf_cpu
# =============================================================================

add_library(smc2_rbpf_cpu STATIC
    src/smc2_rbpf.c
)

target_include_directories(smc2_rbpf_cpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# OpenMP
if(OpenMP_C_FOUND)
    target_link_libraries(smc2_rbpf_cpu PUBLIC OpenMP::OpenMP_C)
elseif(DEFINED OPENMP_FLAG)
    target_compile_options(smc2_rbpf_cpu PRIVATE ${OPENMP_FLAG})
    target_link_options(smc2_rbpf_cpu PRIVATE ${OPENMP_FLAG})
endif()

# Math library (Linux/Unix)
if(UNIX)
    target_link_libraries(smc2_rbpf_cpu PUBLIC m)
endif()

# =============================================================================
# Test Executable
# =============================================================================

add_executable(test_smc2_cpu
    test/test_smc2.c
)

target_link_libraries(test_smc2_cpu PRIVATE smc2_rbpf_cpu)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "[CPU] Targets: smc2_rbpf_cpu (lib), test_smc2_cpu (exe)")