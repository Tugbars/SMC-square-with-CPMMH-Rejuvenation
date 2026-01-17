# =============================================================================
# SMC² RBPF - Test Suite
# =============================================================================

# =============================================================================
# Test Options - Each test can be individually enabled/disabled
# =============================================================================

option(SMC2_TEST_CUDA_MAIN "Build main CUDA filter test (test_smc2_cuda)" ON)
option(SMC2_TEST_CONVERGENCE "Build convergence diagnostics test (test_smc2_convergence)" ON)
option(SMC2_TEST_CALIBRATION "Build prior calibration test (test_prior_calibration)" ON)

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
# Test 3: Prior Calibration Test (test_prior_calibration) - CPU only
# =============================================================================

if(SMC2_TEST_CALIBRATION)
    add_executable(test_prior_calibration
        test_prior_calibration.cpp
    )

    target_include_directories(test_prior_calibration
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )

    set_target_properties(test_prior_calibration PROPERTIES
        CXX_STANDARD 17
    )

    if(WIN32)
        target_compile_options(test_prior_calibration PRIVATE /W3 /wd4244 /wd4267)
    else()
        target_link_libraries(test_prior_calibration PRIVATE m)
        target_compile_options(test_prior_calibration PRIVATE -Wall -Wextra)
    endif()

    list(APPEND SMC2_TEST_TARGETS test_prior_calibration)
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
    
    if(SMC2_TEST_CALIBRATION)
        add_test(NAME prior_calibration_test 
                 COMMAND test_prior_calibration)
    endif()
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "  [Tests]")
if(SMC2_TEST_CUDA_MAIN)
    message(STATUS "    ✓ test_smc2_cuda          (CUDA, main filter test)")
else()
    message(STATUS "    ✗ test_smc2_cuda          (disabled)")
endif()
if(SMC2_TEST_CONVERGENCE)
    message(STATUS "    ✓ test_smc2_convergence   (CUDA, convergence diagnostics)")
else()
    message(STATUS "    ✗ test_smc2_convergence   (disabled)")
endif()
if(SMC2_TEST_CALIBRATION)
    message(STATUS "    ✓ test_prior_calibration  (CPU, prior calibration)")
else()
    message(STATUS "    ✗ test_prior_calibration  (disabled)")
endif()
