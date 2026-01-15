/**
 * @file test_smc2_cuda.cu
 * @brief Test suite for SMC² CUDA with proper PMMH rejuvenation
 * 
 * CRITICAL: Data generation MUST match the filter's model exactly:
 *   - z̃ follows unconstrained AR(1): z̃_t = ρ·z̃_{t-1} + σ_z·ε_t
 *   - z = 1.5·(1 + tanh(z̃)) ∈ (0, 3) for curve evaluation
 *   - y_t = h_t + log(χ²(1)) where χ²(1) = ε² for ε ~ N(0,1)
 */

#include "smc2_rbpf_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*═══════════════════════════════════════════════════════════════════════════
 * Host RNG (xorshift64* for quality)
 *═══════════════════════════════════════════════════════════════════════════*/

static unsigned long long h_rng = 12345678901234567ULL;

static void seed_host_rng(unsigned long long seed) {
    h_rng = seed ? seed : 12345678901234567ULL;
}

static float host_uniform(void) {
    h_rng ^= h_rng << 13;
    h_rng ^= h_rng >> 7;
    h_rng ^= h_rng << 17;
    return (h_rng >> 11) * (1.0f / 9007199254740992.0f);
}

static float host_normal(void) {
    /* Box-Muller transform */
    float u1 = host_uniform();
    float u2 = host_uniform();
    while (u1 < 1e-10f) u1 = host_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Generate SV Data - MATCHES FILTER MODEL EXACTLY
 * 
 * Model:
 *   z̃_t = ρ·z̃_{t-1} + σ_z·ε^z_t           (unconstrained AR(1))
 *   z_t = 1.5·(1 + tanh(z̃_t)) ∈ (0, 3)    (bounded transform)
 *   
 *   θ(z) = θ_base + θ_scale·(1 - exp(-θ_rate·z))
 *   μ(z) = μ_base + μ_scale·(1 - exp(-μ_rate·z))
 *   σ_h(z) = σ_base + σ_scale·(1 - exp(-σ_rate·z))
 *   
 *   h_t = (1-θ(z_t))·h_{t-1} + θ(z_t)·μ(z_t) + σ_h(z_t)·ε^h_t
 *   y_t = h_t + log(χ²(1))
 *═══════════════════════════════════════════════════════════════════════════*/

void generate_sv_data(
    float* y, float* h_true, float* z_true, int T,
    float rho, float sigma_z,
    float mu_base, float mu_scale, float mu_rate,
    float sigma_base, float sigma_scale, float sigma_rate,
    float theta_base, float theta_scale, float theta_rate
) {
    /* Initialize z̃ from stationary distribution: N(0, σ_z²/(1-ρ²)) */
    float one_minus_rho_sq = fmaxf(1.0f - rho * rho, 1e-6f);
    float z_tilde_stat_std = sigma_z / sqrtf(one_minus_rho_sq);
    float z_tilde = z_tilde_stat_std * host_normal();
    
    /* Transform to bounded z ∈ (0, 3) — MUST MATCH FILTER */
    float z = 1.5f * (1.0f + tanhf(z_tilde));
    
    /* Evaluate curves at initial z */
    float theta_z = theta_base + theta_scale * (1.0f - expf(-theta_rate * z));
    float mu_z = mu_base + mu_scale * (1.0f - expf(-mu_rate * z));
    float sigma_h = sigma_base + sigma_scale * (1.0f - expf(-sigma_rate * z));
    float phi = 1.0f - theta_z;
    
    /* Initialize h from approximate stationary distribution */
    float h_stat_var = (sigma_h * sigma_h) / fmaxf(1.0f - phi * phi, 1e-6f);
    float h = mu_z + sqrtf(h_stat_var) * host_normal();
    
    for (int t = 0; t < T; t++) {
        /* Store true states */
        if (h_true) h_true[t] = h;
        if (z_true) z_true[t] = z;
        
        /* Generate observation: y_t = h_t + log(χ²(1))
         * where χ²(1) = ε² for ε ~ N(0,1) */
        float eps = host_normal();
        float chi2_1 = eps * eps;
        /* Add small constant to avoid log(0) for very small eps */
        y[t] = h + logf(chi2_1 + 1e-10f);
        
        /* Transition z̃: unconstrained AR(1) — MATCHES FILTER */
        z_tilde = rho * z_tilde + sigma_z * host_normal();
        
        /* Transform to bounded z ∈ (0, 3) */
        z = 1.5f * (1.0f + tanhf(z_tilde));
        
        /* Evaluate curves at new z */
        theta_z = theta_base + theta_scale * (1.0f - expf(-theta_rate * z));
        mu_z = mu_base + mu_scale * (1.0f - expf(-mu_rate * z));
        sigma_h = sigma_base + sigma_scale * (1.0f - expf(-sigma_rate * z));
        phi = 1.0f - theta_z;
        
        /* Transition h: mean-reverting AR(1) with regime-dependent parameters */
        h = phi * h + theta_z * mu_z + sigma_h * host_normal();
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * Print Data Statistics
 *═══════════════════════════════════════════════════════════════════════════*/

void print_data_stats(const float* y, int T) {
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
    
    /* 
     * Realistic parameter values for daily log-volatility:
     * - mu_base = -3.0 implies baseline vol ≈ exp(-3/2) ≈ 22%
     * - mu_base + mu_scale = -2.0 implies high-regime vol ≈ exp(-2/2) ≈ 37%
     * 
     * These should match or be close to the prior means in the filter.
     */
    float true_rho = 0.95f;
    float true_sigma_z = 0.15f;
    float true_mu_base = -1.0f;
    float true_mu_scale = 0.5f;
    float true_mu_rate = 1.0f;
    float true_sigma_base = 0.15f;
    float true_sigma_scale = 0.10f;
    float true_sigma_rate = 1.0f;
    
    /* Theta curve (fixed in filter) */
    float theta_base = 0.02f;
    float theta_scale = 0.08f;
    float theta_rate = 1.5f;
    
    printf("TRUE PARAMETERS:\n");
    printf("  rho=%.3f, sigma_z=%.3f\n", true_rho, true_sigma_z);
    printf("  mu: base=%.3f, scale=%.3f, rate=%.3f\n", 
           true_mu_base, true_mu_scale, true_mu_rate);
    printf("  sigma: base=%.3f, scale=%.3f, rate=%.3f\n",
           true_sigma_base, true_sigma_scale, true_sigma_rate);
    printf("  theta (fixed): base=%.3f, scale=%.3f, rate=%.3f\n",
           theta_base, theta_scale, theta_rate);
    
    /* Generate data */
    seed_host_rng(42);  /* Reproducible */
    int T = 500;
    float* y = (float*)malloc(T * sizeof(float));
    float* h_true = (float*)malloc(T * sizeof(float));
    
    generate_sv_data(y, h_true, NULL, T,
                     true_rho, true_sigma_z,
                     true_mu_base, true_mu_scale, true_mu_rate,
                     true_sigma_base, true_sigma_scale, true_sigma_rate,
                     theta_base, theta_scale, theta_rate);
    
    printf("\nGenerated T=%d observations\n", T);
    print_data_stats(y, T);
    
    /* Print some h_true statistics */
    float h_sum = 0.0f, h_sum_sq = 0.0f;
    for (int t = 0; t < T; t++) {
        h_sum += h_true[t];
        h_sum_sq += h_true[t] * h_true[t];
    }
    float h_mean = h_sum / T;
    float h_std = sqrtf(h_sum_sq / T - h_mean * h_mean);
    printf("  True h: mean=%.3f, std=%.3f\n", h_mean, h_std);
    
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
        float ess = smc2_cuda_update(state, y[t]);
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
    
    /* Parameter recovery table */
    float true_params[8] = {true_rho, true_sigma_z, true_mu_base, true_mu_scale,
                           true_mu_rate, true_sigma_base, true_sigma_scale, true_sigma_rate};
    const char* names[8] = {"rho", "sigma_z", "mu_base", "mu_scale",
                            "mu_rate", "sigma_base", "sigma_scale", "sigma_rate"};
    
    printf("\n%-12s  %8s  %8s  %8s  %7s  %s\n", 
           "Parameter", "True", "Est", "Std", "z-score", "Status");
    printf("─────────────────────────────────────────────────────────────────\n");
    
    int n_ok = 0;
    for (int i = 0; i < 8; i++) {
        float err = theta_mean[i] - true_params[i];
        float z_score = fabsf(err) / fmaxf(theta_std[i], 1e-6f);
        const char* status = (z_score <= 2.0f) ? "OK" : (z_score <= 3.0f) ? "WARN" : "MISS";
        if (z_score <= 2.0f) n_ok++;
        
        printf("%-12s  %8.4f  %8.4f  %8.4f  %7.2f  [%s]\n", 
               names[i], true_params[i], theta_mean[i], theta_std[i], z_score, status);
    }
    
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("OVERALL: %d/8 within 2σ of true value\n", n_ok);
    printf("%s\n", n_ok >= 6 ? "PASSED" : "NEEDS INVESTIGATION");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(y);
    free(h_true);
    smc2_cuda_free(state);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Throughput
 *═══════════════════════════════════════════════════════════════════════════*/

void test_throughput(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Throughput with PMMH Rejuvenation\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    seed_host_rng(123);
    int T = 500;
    float* y = (float*)malloc(T * sizeof(float));
    generate_sv_data(y, NULL, NULL, T,
                     0.95f, 0.15f, -1.0f, 0.5f, 1.0f,
                     0.15f, 0.10f, 1.0f, 0.02f, 0.08f, 1.5f);
    
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
            smc2_cuda_update(state, y[t]);
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
    
    free(y);
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
        } else {
            printf("Usage: %s [basic|learn|throughput|prior]\n", argv[0]);
            return 1;
        }
    } else {
        /* Run all tests */
        test_basic();
        test_prior_data_agreement();
        test_parameter_learning();
        test_throughput();
    }
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("All tests completed.\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    return 0;
}
