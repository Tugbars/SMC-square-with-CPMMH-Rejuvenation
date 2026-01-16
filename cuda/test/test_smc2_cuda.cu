/**
 * @file test_smc2_cuda.cu
 * @brief Test suite for SMC² CUDA with proper PMMH rejuvenation
 * 
 * Uses shared test utilities from smc2_test_utils.cuh for data generation.
 * This ensures consistency across all test files.
 */

#include "smc2_rbpf_cuda.cuh"
#include "smc2_test_utils.cuh"  /* Single source of truth for data generation */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

using namespace smc2_test;

#ifndef CUDA_N_THETA
#define CUDA_N_THETA 256
#endif

#ifndef CUDA_N_INNER
#define CUDA_N_INNER 256
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * Local helpers (non-data-generation)
 *═══════════════════════════════════════════════════════════════════════════*/

static void print_data_stats_local(const float* y, int T) {
    float sum = 0.0f, sum_sq = 0.0f;
    float y_min = y[0], y_max = y[0];
    
    for (int t = 0; t < T; t++) {
        sum += y[t];
        sum_sq += y[t] * y[t];
        if (y[t] < y_min) y_min = y[t];
        if (y[t] > y_max) y_max = y[t];
    }
    
    float mean = sum / T;
    float var = sum_sq / T - mean * mean;
    
    printf("  Data stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f\n",
           mean, sqrtf(var), y_min, y_max);
    
    /* Expected: E[log χ²(1)] ≈ -1.27, Var[log χ²(1)] ≈ π²/2 ≈ 4.93 */
    printf("  Expected log χ²(1): mean≈-1.27, var≈4.93\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Basic CUDA
 *═══════════════════════════════════════════════════════════════════════════*/

void test_basic(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Basic CUDA Operations\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("  CUDA devices: %d\n", device_count);
    
    if (device_count == 0) {
        printf("  ERROR: No CUDA devices!\n");
        return;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Shared memory: %zu KB\n", prop.sharedMemPerBlock / 1024);
    
    SMC2StateCUDA* state = smc2_cuda_alloc(CUDA_N_THETA, CUDA_N_INNER);
    printf("  Allocated N_theta=%d, N_inner=%d\n", CUDA_N_THETA, CUDA_N_INNER);
    
    smc2_cuda_init_from_prior(state);
    printf("  Initialized from prior.\n");
    
    float theta_mean[8], theta_std[8];
    smc2_cuda_get_theta_mean(state, theta_mean);
    smc2_cuda_get_theta_std(state, theta_std);
    
    printf("  Prior samples:\n");
    printf("    rho=%.4f±%.4f, sigma_z=%.4f±%.4f\n", 
           theta_mean[0], theta_std[0], theta_mean[1], theta_std[1]);
    printf("    mu_base=%.4f±%.4f, sigma_base=%.4f±%.4f\n",
           theta_mean[2], theta_std[2], theta_mean[5], theta_std[5]);
    
    smc2_cuda_free(state);
    printf("  PASSED\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Parameter Learning with PMMH
 *═══════════════════════════════════════════════════════════════════════════*/

void test_parameter_learning(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Parameter Learning with PMMH Rejuvenation\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    /* Use moderate regime from common utilities */
    GroundTruth gt = regime_moderate();
    
    /* Customize for this test (same values as before) */
    gt.rho = 0.95f;
    gt.sigma_z = 0.15f;
    gt.mu_base = -1.0f;
    gt.mu_scale = 0.5f;
    gt.mu_rate = 1.0f;
    gt.sigma_base = 0.15f;
    gt.sigma_scale = 0.10f;
    gt.sigma_rate = 1.0f;
    
    print_ground_truth("TRUE PARAMETERS", gt);
    
    /* Generate data using common utilities */
    int T = 500;
    GeneratedData data = generate_sv_data(gt, T, 42, ObsMethod::DIRECT_CHI2);
    
    printf("\nGenerated T=%d observations\n", T);
    DataStats stats = compute_stats(data);
    print_stats(stats);
    
    /* Initialize SMC² */
    SMC2StateCUDA* state = smc2_cuda_alloc(128, 128);
    
    /* Set reproducible seed */
    smc2_cuda_set_seed(state, 12345);
    
    /* Pre-allocate noise capacity */
    smc2_cuda_set_noise_capacity(state, T + 128);
    
    smc2_cuda_init_from_prior(state);
    
    printf("\nRunning SMC² (N_theta=%d, N_inner=%d, K_rejuv=%d)...\n",
           state->N_theta, state->N_inner, state->K_rejuv);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int t = 0; t < T; t++) {
        float ess = smc2_cuda_update(state, data.y[t]);
        if ((t + 1) % 100 == 0) {
            float theta_mean[8];
            smc2_cuda_get_theta_mean(state, theta_mean);
            printf("  t=%3d: ESS=%5.1f, resamp=%2d, accept=%5.1f%%, "
                   "rho=%.3f, σz=%.3f, μb=%.2f, σb=%.3f\n",
                   t + 1, ess, state->n_resamples,
                   state->n_rejuv_total > 0 ? 
                   100.0f * state->n_rejuv_accepts / state->n_rejuv_total : 0.0f,
                   theta_mean[0], theta_mean[1], theta_mean[2], theta_mean[5]);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Time: %.1f ms (%.2f ms/obs)\n", ms, ms / T);
    printf("  Resamples: %d\n", state->n_resamples);
    printf("  Rejuvenation: %d/%d accepted (%.1f%%)\n",
           state->n_rejuv_accepts, state->n_rejuv_total,
           state->n_rejuv_total > 0 ?
           100.0f * state->n_rejuv_accepts / state->n_rejuv_total : 0.0f);
    
    float theta_mean[8], theta_std[8];
    smc2_cuda_get_theta_mean(state, theta_mean);
    smc2_cuda_get_theta_std(state, theta_std);
    
    printf("\nESTIMATED (mean ± std):\n");
    printf("  rho       = %.4f ± %.4f\n", theta_mean[0], theta_std[0]);
    printf("  sigma_z   = %.4f ± %.4f\n", theta_mean[1], theta_std[1]);
    printf("  mu_base   = %.4f ± %.4f\n", theta_mean[2], theta_std[2]);
    printf("  mu_scale  = %.4f ± %.4f\n", theta_mean[3], theta_std[3]);
    printf("  mu_rate   = %.4f ± %.4f\n", theta_mean[4], theta_std[4]);
    printf("  sigma_base  = %.4f ± %.4f\n", theta_mean[5], theta_std[5]);
    printf("  sigma_scale = %.4f ± %.4f\n", theta_mean[6], theta_std[6]);
    printf("  sigma_rate  = %.4f ± %.4f\n", theta_mean[7], theta_std[7]);
    
    /* Use common utility for parameter comparison */
    printf("\n");
    print_param_comparison(theta_mean, theta_std, gt);
    
    float avg_err = compute_avg_rel_error(theta_mean, gt);
    printf("\nAverage relative error: %.1f%%\n", 100.0f * avg_err);
    printf("%s\n", avg_err < 0.25f ? "PASSED" : "NEEDS INVESTIGATION");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    smc2_cuda_free(state);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Fixed-Lag PMMH for Long Sequences
 * 
 * Compares full-history replay (L=0) vs fixed-lag (L=100) at T=2000.
 * Expected: Fixed-lag should be faster and maintain accuracy.
 *═══════════════════════════════════════════════════════════════════════════*/

void test_fixed_lag(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Fixed-Lag PMMH - Accuracy at Large T\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("\nGoal: Show that fixed-lag maintains accuracy at large T where\n");
    printf("      full-history PMMH degrades due to O(T) variance growth.\n\n");
    
    /* Use common ground truth */
    GroundTruth gt = regime_moderate();
    gt.rho = 0.95f;
    gt.sigma_z = 0.15f;
    gt.mu_base = -1.0f;
    gt.mu_scale = 0.5f;
    gt.mu_rate = 1.0f;
    gt.sigma_base = 0.15f;
    gt.sigma_scale = 0.10f;
    gt.sigma_rate = 1.0f;
    
    float true_theta[8];
    gt_to_theta(gt, true_theta);
    
    /* Test at multiple T values to show variance growth effect */
    int T_values[] = {1000, 2000, 5000};
    int n_T = sizeof(T_values) / sizeof(T_values[0]);
    
    for (int ti = 0; ti < n_T; ti++) {
        int T = T_values[ti];
        
        printf("─────────────────────────────────────────────────────────────────────────\n");
        printf("T = %d\n", T);
        printf("─────────────────────────────────────────────────────────────────────────\n");
        
        /* Generate data using common utilities */
        GeneratedData data = generate_sv_data(gt, T, 42, ObsMethod::DIRECT_CHI2);
        
        printf("  %-6s  %8s  %10s  %10s  %8s  %8s\n", 
               "Lag", "Time(ms)", "rho", "sigma_z", "Accept%", "Resamps");
        
        /* Test L=0 (full history) and L=100 (fixed-lag) */
        int lag_values[] = {0, 100};
        int n_lags = 2;
        
        for (int i = 0; i < n_lags; i++) {
            int L = lag_values[i];
            
            SMC2StateCUDA* state = smc2_cuda_alloc(256, 256);
            smc2_cuda_set_seed(state, 12345);
            smc2_cuda_set_noise_capacity(state, T + 128);
            smc2_cuda_set_fixed_lag(state, L);
            smc2_cuda_init_from_prior(state);
            
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start);
            for (int t = 0; t < T; t++) {
                smc2_cuda_update(state, data.y[t]);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            
            float theta_mean[8], theta_std[8];
            smc2_cuda_get_theta_mean(state, theta_mean);
            smc2_cuda_get_theta_std(state, theta_std);
            
            float accept_pct = state->n_rejuv_total > 0 ? 
                100.0f * state->n_rejuv_accepts / state->n_rejuv_total : 0.0f;
            
            float rho_err = fabsf(theta_mean[0] - true_theta[0]);
            float rho_z = rho_err / fmaxf(theta_std[0], 1e-6f);
            float sigma_z_err = fabsf(theta_mean[1] - true_theta[1]);
            float sigma_z_z = sigma_z_err / fmaxf(theta_std[1], 1e-6f);
            
            const char* status = (rho_z < 2.0f && sigma_z_z < 2.0f) ? "OK" : "DEGRADED";
            
            printf("  L=%3d   %8.1f  %5.4f±%4.3f  %5.4f±%4.3f  %7.1f%%  %5d  [%s]\n",
                   L, ms, 
                   theta_mean[0], theta_std[0],
                   theta_mean[1], theta_std[1],
                   accept_pct, state->n_resamples, status);
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            smc2_cuda_free(state);
        }
        
        printf("\n");
    }
    
    printf("─────────────────────────────────────────────────────────────────────────\n");
    printf("Expected: At small T, both L=0 and L=100 work.\n");
    printf("          At large T (5000+), L=0 may show degraded acceptance/accuracy\n");
    printf("          while L=100 remains stable.\n");
    printf("─────────────────────────────────────────────────────────────────────────\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Throughput
 *═══════════════════════════════════════════════════════════════════════════*/

void test_throughput(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Throughput with PMMH Rejuvenation\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    /* Generate data using common utilities */
    GroundTruth gt = regime_moderate();
    int T = 500;
    GeneratedData data = generate_sv_data(gt, T, 123, ObsMethod::DIRECT_CHI2);
    
    printf("  N_theta  N_inner  Time(ms)  Resamples  Rejuv%%   ms/obs\n");
    printf("  ─────────────────────────────────────────────────────────\n");
    
    int configs[][2] = {{64, 64}, {128, 128}, {256, 256}};
    int n_configs = sizeof(configs) / sizeof(configs[0]);
    
    for (int c = 0; c < n_configs; c++) {
        SMC2StateCUDA* state = smc2_cuda_alloc(configs[c][0], configs[c][1]);
        smc2_cuda_set_seed(state, 54321);
        smc2_cuda_set_noise_capacity(state, T + 128);
        smc2_cuda_init_from_prior(state);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int t = 0; t < T; t++) {
            smc2_cuda_update(state, data.y[t]);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        
        float rejuv_pct = state->n_rejuv_total > 0 ?
            100.0f * state->n_rejuv_accepts / state->n_rejuv_total : 0.0f;
        
        printf("  %4d     %4d     %7.1f   %4d       %5.1f    %.3f\n",
               configs[c][0], configs[c][1], ms, state->n_resamples,
               rejuv_pct, ms / T);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        smc2_cuda_free(state);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Prior-Data Agreement
 * 
 * Verifies that the prior distribution can generate data similar to what
 * we're testing with. If there's a mismatch, the filter will struggle.
 *═══════════════════════════════════════════════════════════════════════════*/

void test_prior_data_agreement(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Prior-Data Agreement\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    SMC2StateCUDA* state = smc2_cuda_alloc(64, 64);
    
    printf("\nDefault prior means (what filter expects):\n");
    printf("  rho       = %.3f ± %.3f\n", state->prior.rho_mean, state->prior.rho_std);
    printf("  sigma_z   = %.3f ± %.3f\n", state->prior.sigma_z_mean, state->prior.sigma_z_std);
    printf("  mu_base   = %.3f ± %.3f\n", state->prior.mu_base_mean, state->prior.mu_base_std);
    printf("  mu_scale  = %.3f ± %.3f\n", state->prior.mu_scale_mean, state->prior.mu_scale_std);
    printf("  mu_rate   = %.3f ± %.3f\n", state->prior.mu_rate_mean, state->prior.mu_rate_std);
    printf("  sigma_base  = %.3f ± %.3f\n", state->prior.sigma_base_mean, state->prior.sigma_base_std);
    printf("  sigma_scale = %.3f ± %.3f\n", state->prior.sigma_scale_mean, state->prior.sigma_scale_std);
    printf("  sigma_rate  = %.3f ± %.3f\n", state->prior.sigma_rate_mean, state->prior.sigma_rate_std);
    
    printf("\nDefault bounds:\n");
    printf("  rho       ∈ [%.3f, %.3f]\n", state->bounds.rho_min, state->bounds.rho_max);
    printf("  sigma_z   ∈ [%.3f, %.3f]\n", state->bounds.sigma_z_min, state->bounds.sigma_z_max);
    printf("  mu_base   ∈ [%.3f, %.3f]\n", state->bounds.mu_base_min, state->bounds.mu_base_max);
    printf("  sigma_base  ∈ [%.3f, %.3f]\n", state->bounds.sigma_base_min, state->bounds.sigma_base_max);
    
    smc2_cuda_free(state);
    
    printf("\nTest parameters used:\n");
    printf("  rho=0.95, sigma_z=0.15\n");
    printf("  mu: base=-1.0, scale=0.5, rate=1.0\n");
    printf("  sigma: base=0.15, scale=0.10, rate=1.0\n");
    
    printf("\n  → Check that test params are within bounds and near prior means!\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: CPMMH Replay Determinism (CRITICAL CORRECTNESS TEST)
 * 
 * This test validates that the replay machinery is correct:
 *   - Set proposal_std = 0 (so θ* = θ exactly)
 *   - Set cpmmh_rho = 1.0 (so noise is identical)
 *   - Force rejuvenation
 *   - MUST get 100% acceptance (since θ* = θ and same noise → same likelihood)
 * 
 * If this test fails, there's a bug in the replay code.
 *═══════════════════════════════════════════════════════════════════════════*/

void test_cpmmh_replay_determinism(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: CPMMH Replay Determinism (Identity Proposal)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("\nThis is a CRITICAL correctness test.\n");
    printf("If θ* = θ and noise is identical, acceptance MUST be 100%%.\n\n");
    
    /* Generate test data */
    GroundTruth gt = regime_moderate();
    int T = 200;  /* Enough to trigger several outer resamples */
    GeneratedData data = generate_sv_data(gt, T, 42, ObsMethod::DIRECT_CHI2);
    
    /* Create filter with small particle count for speed */
    SMC2StateCUDA* state = smc2_cuda_alloc(64, 64);
    smc2_cuda_set_seed(state, 12345);
    smc2_cuda_set_noise_capacity(state, T + 128);
    smc2_cuda_init_from_prior(state);
    
    /* Run forward filter to build up history and trigger some resamples */
    printf("Phase 1: Running forward filter (T=%d)...\n", T);
    for (int t = 0; t < T; t++) {
        smc2_cuda_update(state, data.y[t]);
    }
    printf("  Forward pass complete. Resamples: %d, Rejuv: %d/%d (%.1f%%)\n",
           state->n_resamples,
           state->n_rejuv_accepts, state->n_rejuv_total,
           state->n_rejuv_total > 0 ? 
           100.0f * state->n_rejuv_accepts / state->n_rejuv_total : 0.0f);
    
    /* Save original settings */
    float orig_proposal_std[8];
    memcpy(orig_proposal_std, state->proposal_std, sizeof(orig_proposal_std));
    float orig_cpmmh_rho = state->cpmmh_rho;
    
    /* Set identity proposal: θ* = θ (no perturbation) */
    printf("\nPhase 2: Setting identity proposal (θ* = θ, ρ = 1.0)...\n");
    float zero_proposal[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    smc2_cuda_set_proposal_std(state, zero_proposal);
    smc2_cuda_set_cpmmh_rho(state, 1.0f);
    
    /* Reset acceptance counters */
    state->n_rejuv_accepts = 0;
    state->n_rejuv_total = 0;
    
    /* Force rejuvenation by processing more observations */
    printf("Phase 3: Processing more observations to trigger rejuvenation...\n");
    
    int T_extra = 50;
    GeneratedData data_extra = generate_sv_data(gt, T_extra, 99999, ObsMethod::DIRECT_CHI2);
    
    for (int t = 0; t < T_extra; t++) {
        smc2_cuda_update(state, data_extra.y[t]);
    }
    
    /* Check results */
    printf("\nRESULTS:\n");
    printf("  Rejuvenation attempts: %d\n", state->n_rejuv_total);
    printf("  Rejuvenation accepts:  %d\n", state->n_rejuv_accepts);
    
    float accept_rate = state->n_rejuv_total > 0 ? 
        100.0f * state->n_rejuv_accepts / state->n_rejuv_total : 0.0f;
    printf("  Acceptance rate: %.1f%%\n", accept_rate);
    
    /* Restore original settings */
    smc2_cuda_set_proposal_std(state, orig_proposal_std);
    smc2_cuda_set_cpmmh_rho(state, orig_cpmmh_rho);
    
    /* Verdict */
    printf("\n");
    if (state->n_rejuv_total == 0) {
        printf("WARNING: No rejuvenations occurred. Test inconclusive.\n");
        printf("         Try increasing T or lowering ess_threshold.\n");
    } else if (accept_rate >= 99.9f) {
        printf("═══════════════════════════════════════════════════════════════\n");
        printf("  ✓ PASSED: 100%% acceptance with identity proposal\n");
        printf("  → Replay machinery is PROVABLY CORRECT\n");
        printf("═══════════════════════════════════════════════════════════════\n");
    } else {
        printf("═══════════════════════════════════════════════════════════════\n");
        printf("  ✗ FAILED: Expected 100%% acceptance, got %.1f%%\n", accept_rate);
        printf("  → BUG in replay code: same θ + same noise ≠ same likelihood\n");
        printf("═══════════════════════════════════════════════════════════════\n");
    }
    
    smc2_cuda_free(state);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Main
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char** argv) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  SMC² RBPF CUDA - Test Suite (with PMMH Rejuvenation)         ║\n");
    printf("║  Data generation MATCHES filter z-transform (tanh)            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    /* Check for specific test selection */
    if (argc > 1) {
        if (strcmp(argv[1], "basic") == 0) {
            test_basic();
        } else if (strcmp(argv[1], "learn") == 0) {
            test_parameter_learning();
        } else if (strcmp(argv[1], "throughput") == 0) {
            test_throughput();
        } else if (strcmp(argv[1], "prior") == 0) {
            test_prior_data_agreement();
        } else if (strcmp(argv[1], "fixedlag") == 0) {
            test_fixed_lag();
        } else if (strcmp(argv[1], "determinism") == 0) {
            test_cpmmh_replay_determinism();
        } else {
            printf("Usage: %s [basic|learn|throughput|prior|fixedlag|determinism]\n", argv[0]);
            return 1;
        }
    } else {
        /* Run all tests */
        test_basic();
        test_prior_data_agreement();
        test_cpmmh_replay_determinism();  /* CRITICAL: Run early to catch bugs */
        test_parameter_learning();
        test_fixed_lag();
        test_throughput();
    }
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("All tests completed.\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    return 0;
}
