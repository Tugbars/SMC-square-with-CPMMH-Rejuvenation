cmake_minimum_required(VERSION 3.20)

project(smc2_rbpf
    VERSION 1.0.0
    DESCRIPTION "SMC² with RBPF Inner Filter"
)

# =============================================================================
# Options
# =============================================================================

option(SMC2_BUILD_CPU  "Build CPU implementation"  OFF)
option(SMC2_BUILD_CUDA "Build CUDA implementation" ON)

# =============================================================================
# Build type
# =============================================================================

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =============================================================================
# Output directories
# =============================================================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "╔═══════════════════════════════════════════════════════════╗")
message(STATUS "║  SMC² RBPF - Configuration                                ║")
message(STATUS "╠═══════════════════════════════════════════════════════════╣")
message(STATUS "║  Build Type:  ${CMAKE_BUILD_TYPE}")
message(STATUS "║  Build CPU:   ${SMC2_BUILD_CPU}")
message(STATUS "║  Build CUDA:  ${SMC2_BUILD_CUDA}")
message(STATUS "╚═══════════════════════════════════════════════════════════╝")
message(STATUS "")

# =============================================================================
# Subdirectories
# =============================================================================

if(SMC2_BUILD_CPU)
    add_subdirectory(cpu)
endif()

if(SMC2_BUILD_CUDA)
    add_subdirectory(cuda)
endif()