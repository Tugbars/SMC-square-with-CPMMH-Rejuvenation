/**
 * @file test_smc2_cuda.cu
 * @brief Test suite for SMC² CUDA with proper PMMH rejuvenation
 */

#include "smc2_rbpf_cuda.cuh"
#include "sv_data_generator.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#ifndef CUDA_N_THETA
#define CUDA_N_THETA 256
#endif

#ifndef CUDA_N_INNER
#define CUDA_N_INNER 256
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * External Data File Reader
 * 
 * File format (binary):
 *   - Header: T (int32), n_params (int32), params[n_params] (float32)
 *   - Data: y[T], h_true[T], z_true[T] (all float32)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    int T;
    float rho, sigma_z;
    float mu_base, mu_scale, mu_rate;
    float sigma_base, sigma_scale, sigma_rate;
    float theta_base, theta_scale, theta_rate;
    float* y;
    float* h_true;
    float* z_true;
} SVDataFile;

static int load_sv_data(const char* filename, SVDataFile* data) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s\n", filename);
        return -1;
    }
    
    /* Read header */
    int32_t T, n_params;
    if (fread(&T, sizeof(int32_t), 1, f) != 1 ||
        fread(&n_params, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "Error: failed to read header\n");
        fclose(f);
        return -1;
    }
    
    if (n_params != 11) {
        fprintf(stderr, "Error: expected 11 params, got %d\n", n_params);
        fclose(f);
        return -1;
    }
    
    data->T = T;
    
    /* Read parameters */
    float params[11];
    if (fread(params, sizeof(float), 11, f) != 11) {
        fprintf(stderr, "Error: failed to read params\n");
        fclose(f);
        return -1;
    }
    
    data->rho = params[0];
    data->sigma_z = params[1];
    data->mu_base = params[2];
    data->mu_scale = params[3];
    data->mu_rate = params[4];
    data->sigma_base = params[5];
    data->sigma_scale = params[6];
    data->sigma_rate = params[7];
    data->theta_base = params[8];
    data->theta_scale = params[9];
    data->theta_rate = params[10];
    
    /* Allocate and read data */
    data->y = (float*)malloc(T * sizeof(float));
    data->h_true = (float*)malloc(T * sizeof(float));
    data->z_true = (float*)malloc(T * sizeof(float));
    
    if (fread(data->y, sizeof(float), T, f) != (size_t)T ||
        fread(data->h_true, sizeof(float), T, f) != (size_t)T ||
        fread(data->z_true, sizeof(float), T, f) != (size_t)T) {
        fprintf(stderr, "Error: failed to read data arrays\n");
        free(data->y); free(data->h_true); free(data->z_true);
        fclose(f);
        return -1;
    }
    
    fclose(f);
    return 0;
}

static void free_sv_data(SVDataFile* data) {
    free(data->y);
    free(data->h_true);
    free(data->z_true);
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
    
    /* Generate data using SVDataGenerator */
    SVDataGenerator gen;
    gen.seed(42);
    gen.T = 800;
    /* Use default params: rho=0.95, sigma_z=0.15, etc. */
    gen.generate();
    
    printf("TRUE PARAMETERS:\n");
    printf("  rho=%.3f, sigma_z=%.3f\n", gen.rho, gen.sigma_z);
    printf("  mu: base=%.3f, scale=%.3f, rate=%.3f\n", 
           gen.mu_base, gen.mu_scale, gen.mu_rate);
    printf("  sigma: base=%.3f, scale=%.3f, rate=%.3f\n",
           gen.sigma_base, gen.sigma_scale, gen.sigma_rate);
    printf("  theta (fixed): base=%.3f, scale=%.3f, rate=%.3f\n",
           gen.theta_base, gen.theta_scale, gen.theta_rate);
    
    printf("\nGenerated T=%d observations\n", gen.T);
    print_data_stats(gen.y, gen.T);
    
    /* Print some h_true statistics */
    float h_sum = 0.0f, h_sum_sq = 0.0f;
    for (int t = 0; t < gen.T; t++) {
        h_sum += gen.h_true[t];
        h_sum_sq += gen.h_true[t] * gen.h_true[t];
    }
    float h_mean = h_sum / gen.T;
    float h_std = sqrtf(h_sum_sq / gen.T - h_mean * h_mean);
    printf("  True h: mean=%.3f, std=%.3f\n", h_mean, h_std);
    
    /* Initialize SMC² */
    SMC2StateCUDA* state = smc2_cuda_alloc(512, 128);
    
    /* Set reproducible seed */
    smc2_cuda_set_seed(state, 12345);
    
    /* Pre-allocate noise capacity */
    smc2_cuda_set_noise_capacity(state, gen.T + 128);

    smc2_cuda_set_fixed_lag(state, 100); 

    state->ess_threshold_outer = 0.25f;

    smc2_cuda_init_from_prior(state);
    
    printf("\nRunning SMC² (N_theta=%d, N_inner=%d, K_rejuv=%d)...\n",
           state->N_theta, state->N_inner, state->K_rejuv);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int t = 0; t < gen.T; t++) {
        float ess = smc2_cuda_update(state, gen.y[t]);
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
    printf("  Time: %.1f ms (%.2f ms/obs)\n", ms, ms / gen.T);
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
    float true_params[8] = {gen.rho, gen.sigma_z, gen.mu_base, gen.mu_scale,
                           gen.mu_rate, gen.sigma_base, gen.sigma_scale, gen.sigma_rate};
    const char* names[8] = {"rho", "sigma_z", "mu_base", "mu_scale",
                            "mu_rate", "sigma_base", "sigma_scale", "sigma_rate"};
    
    printf("\n%-12s  %8s  %8s  %8s  %7s  %7s  %s\n", 
           "Parameter", "True", "Est", "Std", "Err%", "z-score", "Status");
    printf("─────────────────────────────────────────────────────────────────────────\n");
    
    int n_ok = 0;
    int n_within_15pct = 0;
    for (int i = 0; i < 8; i++) {
        float err = theta_mean[i] - true_params[i];
        float z_score = fabsf(err) / fmaxf(theta_std[i], 1e-6f);
        
        /* Percentage error: use absolute for params near zero */
        float pct_err;
        if (fabsf(true_params[i]) < 0.01f) {
            pct_err = err * 100.0f;  /* Absolute as percentage */
        } else {
            pct_err = 100.0f * err / true_params[i];
        }
        
        const char* status = (z_score <= 2.0f) ? "OK" : (z_score <= 3.0f) ? "WARN" : "MISS";
        if (z_score <= 2.0f) n_ok++;
        if (fabsf(pct_err) <= 15.0f) n_within_15pct++;
        
        printf("%-12s  %8.4f  %8.4f  %8.4f  %+6.1f%%  %7.2f  [%s]\n", 
               names[i], true_params[i], theta_mean[i], theta_std[i], pct_err, z_score, status);
    }
    
    printf("─────────────────────────────────────────────────────────────────────────\n");
    printf("OVERALL: %d/8 within 2σ, %d/8 within 15%% relative error\n", n_ok, n_within_15pct);
    printf("%s\n", n_ok >= 6 ? "PASSED" : "NEEDS INVESTIGATION");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    smc2_cuda_free(state);
    /* gen destructor frees y, h_true, z_true automatically */
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
    
    /* Test at multiple T values to show variance growth effect */
    int T_values[] = {1000, 2000, 5000};
    int n_T = sizeof(T_values) / sizeof(T_values[0]);
    
    for (int ti = 0; ti < n_T; ti++) {
        int T = T_values[ti];
        
        printf("─────────────────────────────────────────────────────────────────────────\n");
        printf("T = %d\n", T);
        printf("─────────────────────────────────────────────────────────────────────────\n");
        
        SVDataGenerator gen;
        gen.seed(42);
        gen.T = T;
        gen.generate();
        
        printf("  %-6s  %8s  %10s  %10s  %8s  %8s\n", 
               "Lag", "Time(ms)", "rho", "sigma_z", "Accept%", "Resamps");
        
        /* Test L=0 (full history) and L=100 (fixed-lag) */
        int lag_values[] = {0, 100};
        int n_lags = 2;
        
        for (int i = 0; i < n_lags; i++) {
            int L = lag_values[i];
            
            SMC2StateCUDA* state = smc2_cuda_alloc(256, 256);
            smc2_cuda_set_seed(state, 12345);
            smc2_cuda_set_noise_capacity(state, gen.T + 128);
            smc2_cuda_set_fixed_lag(state, L);
            smc2_cuda_init_from_prior(state);
            
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start);
            for (int t = 0; t < gen.T; t++) {
                smc2_cuda_update(state, gen.y[t]);
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
            
            float rho_err = fabsf(theta_mean[0] - gen.rho);
            float rho_z = rho_err / fmaxf(theta_std[0], 1e-6f);
            float sigma_z_err = fabsf(theta_mean[1] - gen.sigma_z);
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
        /* gen destructor frees arrays automatically */
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
    
    SVDataGenerator gen;
    gen.seed(123);
    gen.T = 500;
    gen.generate();
    
    printf("  N_theta  N_inner  Time(ms)  Resamples  Rejuv%%   ms/obs\n");
    printf("  ─────────────────────────────────────────────────────────\n");
    
    int configs[][2] = {{64, 64}, {128, 128}, {256, 256}};
    int n_configs = sizeof(configs) / sizeof(configs[0]);
    
    for (int c = 0; c < n_configs; c++) {
        SMC2StateCUDA* state = smc2_cuda_alloc(configs[c][0], configs[c][1]);
        smc2_cuda_set_seed(state, 54321);
        smc2_cuda_set_noise_capacity(state, gen.T + 128);
        smc2_cuda_init_from_prior(state);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int t = 0; t < gen.T; t++) {
            smc2_cuda_update(state, gen.y[t]);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        
        float rejuv_pct = state->n_rejuv_total > 0 ?
            100.0f * state->n_rejuv_accepts / state->n_rejuv_total : 0.0f;
        
        printf("  %4d     %4d     %7.1f   %4d       %5.1f    %.3f\n",
               configs[c][0], configs[c][1], ms, state->n_resamples,
               rejuv_pct, ms / gen.T);
        
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
 * Test: Run from External Data File
 * 
 * Usage: test_smc2_cuda file <path.bin> [L=100] [seed=12345]
 *═══════════════════════════════════════════════════════════════════════════*/

void test_from_file(const char* filename, int fixed_lag_L, uint64_t filter_seed) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Parameter Learning from External Data File\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("File: %s\n", filename);
    printf("Fixed-lag L: %d\n", fixed_lag_L);
    printf("Filter seed: %llu\n", (unsigned long long)filter_seed);
    
    /* Load data */
    SVDataFile data;
    if (load_sv_data(filename, &data) != 0) {
        printf("ERROR: Failed to load data file\n");
        return;
    }
    
    printf("\nLoaded T=%d observations\n", data.T);
    printf("\nTRUE PARAMETERS (from file):\n");
    printf("  rho         = %.4f\n", data.rho);
    printf("  sigma_z     = %.4f\n", data.sigma_z);
    printf("  mu_base     = %.4f\n", data.mu_base);
    printf("  mu_scale    = %.4f\n", data.mu_scale);
    printf("  mu_rate     = %.4f\n", data.mu_rate);
    printf("  sigma_base  = %.4f\n", data.sigma_base);
    printf("  sigma_scale = %.4f\n", data.sigma_scale);
    printf("  sigma_rate  = %.4f\n", data.sigma_rate);
    printf("  theta (fixed): base=%.3f, scale=%.3f, rate=%.3f\n",
           data.theta_base, data.theta_scale, data.theta_rate);
    
    /* Data statistics */
    print_data_stats(data.y, data.T);
    
    float h_sum = 0.0f, h_sum_sq = 0.0f;
    for (int t = 0; t < data.T; t++) {
        h_sum += data.h_true[t];
        h_sum_sq += data.h_true[t] * data.h_true[t];
    }
    float h_mean = h_sum / data.T;
    float h_std = sqrtf(h_sum_sq / data.T - h_mean * h_mean);
    printf("  True h: mean=%.3f, std=%.3f\n", h_mean, h_std);
    
    /* Initialize SMC² */
    SMC2StateCUDA* state = smc2_cuda_alloc(256, 256);
    smc2_cuda_set_seed(state, filter_seed);
    smc2_cuda_set_noise_capacity(state, data.T + 128);
    smc2_cuda_set_fixed_lag(state, fixed_lag_L);
    smc2_cuda_init_from_prior(state);
    
    printf("\nRunning SMC² (N_theta=%d, N_inner=%d, K_rejuv=%d, L=%d)...\n",
           state->N_theta, state->N_inner, state->K_rejuv, fixed_lag_L);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int t = 0; t < data.T; t++) {
        float ess = smc2_cuda_update(state, data.y[t]);
        if ((t + 1) % 100 == 0 || t == data.T - 1) {
            float theta_mean[8];
            smc2_cuda_get_theta_mean(state, theta_mean);
            printf("  t=%4d: ESS=%5.1f, resamp=%2d, accept=%5.1f%%, "
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
    printf("  Time: %.1f ms (%.2f ms/obs)\n", ms, ms / data.T);
    printf("  Resamples: %d\n", state->n_resamples);
    printf("  Rejuvenation: %d/%d accepted (%.1f%%)\n",
           state->n_rejuv_accepts, state->n_rejuv_total,
           state->n_rejuv_total > 0 ?
           100.0f * state->n_rejuv_accepts / state->n_rejuv_total : 0.0f);
    
    float theta_mean[8], theta_std[8];
    smc2_cuda_get_theta_mean(state, theta_mean);
    smc2_cuda_get_theta_std(state, theta_std);
    
    printf("\nESTIMATED (mean ± std):\n");
    printf("  rho         = %.4f ± %.4f\n", theta_mean[0], theta_std[0]);
    printf("  sigma_z     = %.4f ± %.4f\n", theta_mean[1], theta_std[1]);
    printf("  mu_base     = %.4f ± %.4f\n", theta_mean[2], theta_std[2]);
    printf("  mu_scale    = %.4f ± %.4f\n", theta_mean[3], theta_std[3]);
    printf("  mu_rate     = %.4f ± %.4f\n", theta_mean[4], theta_std[4]);
    printf("  sigma_base  = %.4f ± %.4f\n", theta_mean[5], theta_std[5]);
    printf("  sigma_scale = %.4f ± %.4f\n", theta_mean[6], theta_std[6]);
    printf("  sigma_rate  = %.4f ± %.4f\n", theta_mean[7], theta_std[7]);
    
    /* Parameter recovery table */
    float true_params[8] = {data.rho, data.sigma_z, data.mu_base, data.mu_scale,
                           data.mu_rate, data.sigma_base, data.sigma_scale, data.sigma_rate};
    const char* names[8] = {"rho", "sigma_z", "mu_base", "mu_scale",
                            "mu_rate", "sigma_base", "sigma_scale", "sigma_rate"};
    
    printf("\n%-12s  %8s  %8s  %8s  %7s  %7s  %s\n", 
           "Parameter", "True", "Est", "Std", "Err%", "z-score", "Status");
    printf("─────────────────────────────────────────────────────────────────────────\n");
    
    int n_ok = 0;
    int n_within_15pct = 0;
    for (int i = 0; i < 8; i++) {
        float err = theta_mean[i] - true_params[i];
        float z_score = fabsf(err) / fmaxf(theta_std[i], 1e-6f);
        
        float pct_err;
        if (fabsf(true_params[i]) < 0.01f) {
            pct_err = err * 100.0f;
        } else {
            pct_err = 100.0f * err / true_params[i];
        }
        
        const char* status = (z_score <= 2.0f) ? "OK" : (z_score <= 3.0f) ? "WARN" : "MISS";
        if (z_score <= 2.0f) n_ok++;
        if (fabsf(pct_err) <= 15.0f) n_within_15pct++;
        
        printf("%-12s  %8.4f  %8.4f  %8.4f  %+6.1f%%  %7.2f  [%s]\n", 
               names[i], true_params[i], theta_mean[i], theta_std[i], pct_err, z_score, status);
    }
    
    printf("─────────────────────────────────────────────────────────────────────────\n");
    printf("OVERALL: %d/8 within 2σ, %d/8 within 15%% relative error\n", n_ok, n_within_15pct);
    printf("%s\n", n_ok >= 6 ? "PASSED" : "NEEDS INVESTIGATION");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free_sv_data(&data);
    smc2_cuda_free(state);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Main
 *═══════════════════════════════════════════════════════════════════════════*/

void print_usage(const char* prog) {
    printf("Usage:\n");
    printf("  %s                       Run all tests\n", prog);
    printf("  %s basic                 Run basic smoke test\n", prog);
    printf("  %s learn                 Run parameter learning test\n", prog);
    printf("  %s throughput            Run throughput benchmark\n", prog);
    printf("  %s prior                 Check prior/data agreement\n", prog);
    printf("  %s fixedlag              Test fixed-lag performance\n", prog);
    printf("  %s file <data.bin> [L] [seed]   Run from external data file\n", prog);
}

int main(int argc, char** argv) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  SMC² RBPF CUDA - Test Suite                                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
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
        } else if (strcmp(argv[1], "file") == 0) {
            if (argc < 3) {
                print_usage(argv[0]);
                return 1;
            }
            int L = (argc > 3) ? atoi(argv[3]) : 100;
            uint64_t seed = (argc > 4) ? strtoull(argv[4], NULL, 10) : 12345;
            test_from_file(argv[2], L, seed);
        } else if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    } else {
        test_basic();
        test_prior_data_agreement();
        test_parameter_learning();
        test_fixed_lag();
        test_throughput();
    }
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("All tests completed.\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    return 0;
}
