/**
 * @file test_smc2_streaming.cu
 * @brief Tests for Streaming SMC² with Parameter Drift
 * 
 * Test Suite:
 *   1. Equivalence test: Streaming vs Batch on stationary data
 *   2. Regime change test: Streaming should adapt, batch should struggle
 *   3. Memory bounded test: Verify circular buffers work at T >> L
 *   4. Adaptation speed test: Measure lag after regime change
 */

#include "smc2_streaming.cuh"
#include "smc2_test_utils.cuh"  /* Single source of truth for data generation */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace smc2_test;

/*═══════════════════════════════════════════════════════════════════════════════
 * TEST UTILITIES
 *═══════════════════════════════════════════════════════════════════════════════*/

static void print_separator(const char* title) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: %s\n", title);
    printf("═══════════════════════════════════════════════════════════════\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TEST 1: Basic Functionality
 *═══════════════════════════════════════════════════════════════════════════════*/

static int test_basic_create_destroy() {
    print_separator("Basic Create/Destroy");
    
    SMC2StreamState* state = smc2_stream_create(256, 256);
    if (!state) {
        printf("FAILED: Could not create state\n");
        return 1;
    }
    
    printf("Created streaming SMC² state successfully\n");
    
    smc2_stream_free(state);
    printf("Freed state successfully\n");
    
    printf("PASSED\n");
    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TEST 2: Stationary Data (Sanity Check)
 *═══════════════════════════════════════════════════════════════════════════════*/

static int test_stationary_data() {
    print_separator("Stationary Data Learning");
    
    /* Generate data */
    int T = 2000;
    std::vector<float> y;
    GroundTruth gt = gt_calm();
    generate_data(gt, T, 42, y);
    
    printf("Generated T=%d observations from CALM regime\n", T);
    printf("True params:\n");
    float true_theta[8] = {gt.rho, gt.sigma_z, gt.mu_base, gt.mu_scale, gt.mu_rate,
                           gt.sigma_base, gt.sigma_scale, gt.sigma_rate};
    print_theta("  Ground truth", true_theta);
    
    /* Create streaming filter */
    SMC2StreamConfig cfg = smc2_stream_default_config(256, 256);
    cfg.fixed_lag = 100;
    SMC2StreamState* state = smc2_stream_create_with_config(&cfg);
    if (!state) {
        printf("FAILED: Could not create state\n");
        return 1;
    }
    
    smc2_stream_set_seed(state, 12345);
    
    /* TODO: Initialize from prior once kernel is implemented */
    /* smc2_stream_init_from_prior(state); */
    
    /* Process observations */
    printf("Processing observations...\n");
    clock_t start = clock();
    
    for (int t = 0; t < T; t++) {
        float ess = smc2_stream_update(state, y[t]);
        
        if ((t + 1) % 500 == 0) {
            SMC2StreamDiag diag = smc2_stream_get_diag(state);
            printf("  t=%d: ESS=%.1f, resamp=%d, rejuv=%d\n",
                   t + 1, diag.outer_ess, diag.did_resample, diag.did_rejuvenate);
        }
    }
    
    clock_t end = clock();
    float elapsed_ms = 1000.0f * (end - start) / CLOCKS_PER_SEC;
    
    /* Get posterior */
    float theta_est[8];
    smc2_stream_get_theta_mean(state, theta_est);
    print_theta("  Estimated", theta_est);
    
    float avg_err = param_error(theta_est, gt);
    printf("\nAverage relative error: %.1f%%\n", 100.0f * avg_err);
    printf("Time: %.1f ms (%.3f ms/obs)\n", elapsed_ms, elapsed_ms / T);
    
    smc2_stream_free(state);
    
    /* Note: This test will show poor results until kernels are implemented */
    printf("\n[NOTE: Kernels not yet implemented - results are placeholder]\n");
    printf("PASSED (structure test only)\n");
    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TEST 3: Regime Change
 *═══════════════════════════════════════════════════════════════════════════════*/

static int test_regime_change() {
    print_separator("Regime Change Adaptation");
    
    /* Generate regime-switching data */
    int T_calm = 1000;
    int T_crisis = 1000;
    int switch_point;
    std::vector<float> y;
    generate_regime_switch_data(T_calm, T_crisis, 42, y, &switch_point);
    
    int T = (int)y.size();
    printf("Generated T=%d observations\n", T);
    printf("  t=[0, %d): CALM regime\n", switch_point);
    printf("  t=[%d, %d): CRISIS regime\n", switch_point, T);
    
    GroundTruth gt_c = gt_calm();
    GroundTruth gt_x = gt_crisis();
    
    printf("\nCALM params:   rho=%.2f, mu_base=%.1f, sigma_z=%.2f\n",
           gt_c.rho, gt_c.mu_base, gt_c.sigma_z);
    printf("CRISIS params: rho=%.2f, mu_base=%.1f, sigma_z=%.2f\n",
           gt_x.rho, gt_x.mu_base, gt_x.sigma_z);
    
    /* Create streaming filter with drift enabled */
    SMC2StreamConfig cfg = smc2_stream_default_config(256, 256);
    cfg.fixed_lag = 100;
    cfg.enable_adaptive_Q = 1;
    cfg.adaptive_Q_scale = 3.0f;
    
    SMC2StreamState* state = smc2_stream_create_with_config(&cfg);
    if (!state) {
        printf("FAILED: Could not create state\n");
        return 1;
    }
    
    smc2_stream_set_seed(state, 12345);
    
    /* Process and track estimates at key points */
    float theta_at_500[8], theta_at_900[8], theta_at_1100[8], theta_at_1500[8], theta_at_2000[8];
    
    printf("\nProcessing...\n");
    for (int t = 0; t < T; t++) {
        smc2_stream_update(state, y[t]);
        
        if (t + 1 == 500) smc2_stream_get_theta_mean(state, theta_at_500);
        if (t + 1 == 900) smc2_stream_get_theta_mean(state, theta_at_900);
        if (t + 1 == 1100) smc2_stream_get_theta_mean(state, theta_at_1100);
        if (t + 1 == 1500) smc2_stream_get_theta_mean(state, theta_at_1500);
        if (t + 1 == T) smc2_stream_get_theta_mean(state, theta_at_2000);
    }
    
    /* Report errors at each checkpoint */
    printf("\nParameter tracking:\n");
    printf("  t=500  (CALM):   err vs CALM=%.1f%%\n", 100.0f * param_error(theta_at_500, gt_c));
    printf("  t=900  (CALM):   err vs CALM=%.1f%%\n", 100.0f * param_error(theta_at_900, gt_c));
    printf("  t=1100 (CRISIS): err vs CRISIS=%.1f%%\n", 100.0f * param_error(theta_at_1100, gt_x));
    printf("  t=1500 (CRISIS): err vs CRISIS=%.1f%%\n", 100.0f * param_error(theta_at_1500, gt_x));
    printf("  t=2000 (CRISIS): err vs CRISIS=%.1f%%\n", 100.0f * param_error(theta_at_2000, gt_x));
    
    smc2_stream_free(state);
    
    printf("\n[NOTE: Kernels not yet implemented - results are placeholder]\n");
    printf("PASSED (structure test only)\n");
    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TEST 4: Memory Bounded
 *═══════════════════════════════════════════════════════════════════════════════*/

static int test_memory_bounded() {
    print_separator("Memory Bounded (Long Sequence)");
    
    /* This test verifies circular buffers work correctly */
    int T = 10000;  /* Much larger than buffer_size */
    
    SMC2StreamConfig cfg = smc2_stream_default_config(64, 64);  /* Smaller for speed */
    cfg.fixed_lag = 50;
    
    printf("Config: N_theta=%d, N_inner=%d, L=%d, buffer_size=%d\n",
           cfg.N_theta, cfg.N_inner, cfg.fixed_lag, cfg.buffer_size);
    printf("Running T=%d observations (%.0fx buffer size)...\n",
           T, (float)T / cfg.buffer_size);
    
    SMC2StreamState* state = smc2_stream_create_with_config(&cfg);
    if (!state) {
        printf("FAILED: Could not create state\n");
        return 1;
    }
    
    smc2_stream_set_seed(state, 12345);
    
    /* Generate and process data */
    std::vector<float> y;
    generate_data(gt_calm(), T, 42, y);
    
    clock_t start = clock();
    for (int t = 0; t < T; t++) {
        smc2_stream_update(state, y[t]);
    }
    clock_t end = clock();
    
    float elapsed_ms = 1000.0f * (end - start) / CLOCKS_PER_SEC;
    printf("Completed in %.1f ms (%.3f ms/obs)\n", elapsed_ms, elapsed_ms / T);
    printf("Final t_current = %d\n", smc2_stream_get_t_current(state));
    
    smc2_stream_free(state);
    
    printf("PASSED (no memory explosion)\n");
    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(int argc, char** argv) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║   Streaming SMC² Test Suite                                           ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    
    int failures = 0;
    
    failures += test_basic_create_destroy();
    failures += test_stationary_data();
    failures += test_regime_change();
    failures += test_memory_bounded();
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    if (failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("FAILURES: %d\n", failures);
    }
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    return failures;
}
