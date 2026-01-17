# =============================================================================
# SMC² RBPF - Test Suite
# =============================================================================

# =============================================================================
# Test Options - Each test can be individually enabled/disabled
# =============================================================================

option(SMC2_TEST_CUDA_MAIN "Build main CUDA filter test (test_smc2_cuda)" ON)
option(SMC2_TEST_CONVERGENCE "Build convergence diagnostics test (test_smc2_convergence)" ON)
option(SMC2_TEST_CALIBRATION_GPU "Build GPU prior calibration test (test_prior_calibration_gpu)" ON)

# =============================================================================
# Test 1: Main CUDA Test (test_smc2_cuda)
# =============================================================================

if(SMC2_TEST_CUDA_MAIN)
    add_executable(test_smc2_cuda
        test_smc2_cuda.cu
    )

    target_compile_options(test_smc2_cuda PRIVATE ${CUDA_COMMON_OPTIONS})

    if(SMC2_USE_CUB_SORT)
        target_compile_definitions(test_smc2_cuda PRIVATE SMC2_USE_CUB_SORT)
    endif()

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

    list(APPEND SMC2_TEST_TARGETS test_smc2_cuda)
endif()

# =============================================================================
# Test 2: Convergence Diagnostics Test (test_smc2_convergence)
# =============================================================================

if(SMC2_TEST_CONVERGENCE)
    add_executable(test_smc2_convergence
        test_smc2_convergence.cu
    )

    target_compile_options(test_smc2_convergence PRIVATE ${CUDA_COMMON_OPTIONS})

    target_include_directories(test_smc2_convergence
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )

    target_link_libraries(test_smc2_convergence
        PRIVATE
            CUDA::cudart
    )

    set_target_properties(test_smc2_convergence PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 17
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES 120
    )

    if(WIN32)
        target_compile_options(test_smc2_convergence PRIVATE ${CUDA_WIN_OPTIONS})
    else()
        target_link_libraries(test_smc2_convergence PRIVATE m)
    endif()

    list(APPEND SMC2_TEST_TARGETS test_smc2_convergence)
endif()

# =============================================================================
# Test 3: GPU Prior Calibration Test (test_prior_calibration_gpu)
# =============================================================================

if(SMC2_TEST_CALIBRATION_GPU)
    add_executable(test_prior_calibration_gpu
        test_prior_calibration_gpu.cu
    )

    target_compile_options(test_prior_calibration_gpu PRIVATE ${CUDA_COMMON_OPTIONS})

    target_include_directories(test_prior_calibration_gpu
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )

    target_link_libraries(test_prior_calibration_gpu
        PRIVATE
            CUDA::cudart
    )

    set_target_properties(test_prior_calibration_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 17
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES 120
    )

    if(WIN32)
        target_compile_options(test_prior_calibration_gpu PRIVATE ${CUDA_WIN_OPTIONS})
    else()
        target_link_libraries(test_prior_calibration_gpu PRIVATE m)
    endif()

    list(APPEND SMC2_TEST_TARGETS test_prior_calibration_gpu)
endif()

# =============================================================================
# CTest Integration (optional)
# =============================================================================

if(BUILD_TESTING AND SMC2_TEST_TARGETS)
    include(CTest)
    
    if(SMC2_TEST_CUDA_MAIN)
        add_test(NAME smc2_cuda_test 
                 COMMAND test_smc2_cuda)
    endif()
    
    if(SMC2_TEST_CONVERGENCE)
        add_test(NAME smc2_convergence_test 
                 COMMAND test_smc2_convergence)
    endif()
    
    if(SMC2_TEST_CALIBRATION_GPU)
        add_test(NAME prior_calibration_gpu_test 
                 COMMAND test_prior_calibration_gpu)
    endif()
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "  [Tests]")
if(SMC2_TEST_CUDA_MAIN)
    message(STATUS "    ✓ test_smc2_cuda             (CUDA, main filter test)")
else()
    message(STATUS "    ✗ test_smc2_cuda             (disabled)")
endif()
if(SMC2_TEST_CONVERGENCE)
    message(STATUS "    ✓ test_smc2_convergence      (CUDA, convergence diagnostics)")
else()
    message(STATUS "    ✗ test_smc2_convergence      (disabled)")
endif()
if(SMC2_TEST_CALIBRATION_GPU)
    message(STATUS "    ✓ test_prior_calibration_gpu (CUDA, GPU prior calibration)")
else()
    message(STATUS "    ✗ test_prior_calibration_gpu (disabled)")
endif()

# =============================================================================
# Tools (optional, expensive validation)
# =============================================================================

option(SMC2_BUILD_TOOLS "Build validation tools (SBC)" OFF)

if(SMC2_BUILD_TOOLS)
    # SBC Validation Tool
    add_executable(sbc_validation
        ../tools/sbc_validation.cu
    )

    target_compile_options(sbc_validation PRIVATE ${CUDA_COMMON_OPTIONS})

    target_include_directories(sbc_validation
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )

    target_link_libraries(sbc_validation
        PRIVATE
            smc2_rbpf_cuda
            CUDA::cudart
            CUDA::curand
    )

    set_target_properties(sbc_validation PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 17
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES 120
    )

    if(WIN32)
        target_compile_options(sbc_validation PRIVATE ${CUDA_WIN_OPTIONS})
    else()
        target_link_libraries(sbc_validation PRIVATE m)
    endif()

    message(STATUS "  [Tools]")
    message(STATUS "    ✓ sbc_validation             (CUDA, Simulation-Based Calibration)")
endif()