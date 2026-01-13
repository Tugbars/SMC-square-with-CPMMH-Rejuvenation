/**
 * @file test_smc2_cuda.cu
 * @brief Test suite for SMC² CUDA implementation
 */

#include "smc2_rbpf_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*═══════════════════════════════════════════════════════════════════════════
 * Simple host-side RNG for data generation
 *═══════════════════════════════════════════════════════════════════════════*/

static unsigned long long h_rng_state = 12345678901234567ULL;

static float host_rand_uniform(void) {
    h_rng_state ^= h_rng_state << 13;
    h_rng_state ^= h_rng_state >> 7;
    h_rng_state ^= h_rng_state << 17;
    return (h_rng_state >> 11) * (1.0f / 9007199254740992.0f);
}

static float host_rand_normal(void) {
    float u1 = host_rand_uniform();
    float u2 = host_rand_uniform();
    while (u1 < 1e-10f) u1 = host_rand_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Generate Synthetic SV Data
 *═══════════════════════════════════════════════════════════════════════════*/

void generate_sv_data(
    float* y_out, float* h_out, float* z_out,
    int T,
    float rho, float sigma_z,
    float mu_base, float mu_scale, float mu_rate,
    float sigma_base, float sigma_scale, float sigma_rate,
    float theta_base, float theta_scale, float theta_rate
) {
    /* Initial values */
    float z = 0.2f;
    float h = mu_base;
    
    for (int t = 0; t < T; t++) {
        /* Store state */
        if (h_out) h_out[t] = h;
        if (z_out) z_out[t] = z;
        
        /* Generate observation */
        float eps = host_rand_normal();
        y_out[t] = h + logf(eps * eps + 1e-10f);
        
        /* Evaluate curves */
        float theta_z = theta_base + theta_scale * (1.0f - expf(-theta_rate * z));
        float mu_z = mu_base + mu_scale * (1.0f - expf(-mu_rate * z));
        float sigma_z_h = sigma_base + sigma_scale * (1.0f - expf(-sigma_rate * z));
        
        /* Transition */
        float phi = 1.0f - theta_z;
        z = rho * z + sigma_z * host_rand_normal();
        z = fmaxf(0.0f, fminf(3.0f, z));
        h = phi * h + theta_z * mu_z + sigma_z_h * host_rand_normal();
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Basic CUDA Operations
 *═══════════════════════════════════════════════════════════════════════════*/

void test_cuda_basic(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Test: Basic CUDA Operations\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    /* Check CUDA device */
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("  CUDA devices found: %d\n", device_count);
    
    if (device_count == 0) {
        printf("  ERROR: No CUDA devices found!\n");
        return;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  Device 0: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Warp size: %d\n", prop.warpSize);
    
    /* Allocate SMC² state */
    printf("\n  Allocating SMC² state (N_theta=%d, N_inner=%d)...\n", 
           CUDA_N_THETA, CUDA_N_INNER);
    
    SMC2StateCUDA* state = smc2_cuda_alloc(CUDA_N_THETA, CUDA_N_INNER);
    if (!state) {
        printf("  ERROR: Failed to allocate SMC² state!\n");
        return;
    }
    printf("  Allocation successful.\n");
    
    /* Initialize from prior */
    printf("  Initializing from prior...\n");
    smc2_cuda_init_from_prior(state);
    printf("  Initialization successful.\n");
    
    /* Get initial theta estimates */
    float theta_mean[8], theta_std[8];
    smc2_cuda_get_theta_mean(state, theta_mean);
    smc2_cuda_get_theta_std(state, theta_std);
    
    printf("\n  Initial θ estimates (from prior):\n");
    printf("    rho:         %.4f ± %.4f\n", theta_mean[0], theta_std[0]);
    printf("    sigma_z:     %.4f ± %.4f\n", theta_mean[1], theta_std[1]);
    printf("    mu_base:     %.4f ± %.4f\n", theta_mean[2], theta_std[2]);
    printf("    sigma_base:  %.4f ± %.4f\n", theta_mean[5], theta_std[5]);
    
    smc2_cuda_free(state);
    printf("\n  Basic CUDA test: PASSED\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: RBPF Step Performance
 *═══════════════════════════════════════════════════════════════════════════*/

void test_rbpf_step_performance(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Test: RBPF Step Performance\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    SMC2StateCUDA* state = smc2_cuda_alloc(CUDA_N_THETA, CUDA_N_INNER);
    smc2_cuda_init_from_prior(state);
    
    /* Generate test data */
    int T = 100;
    float* y = (float*)malloc(T * sizeof(float));
    generate_sv_data(y, NULL, NULL, T,
                     0.96f, 0.08f,
                     -0.8f, 0.4f, 1.2f,
                     0.12f, 0.08f, 1.0f,
                     0.02f, 0.08f, 1.5f);
    
    /* Warm-up */
    for (int t = 0; t < 10; t++) {
        smc2_cuda_update(state, y[t]);
    }
    cudaDeviceSynchronize();
    
    /* Time T steps */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int t = 10; t < T; t++) {
        smc2_cuda_update(state, y[t]);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    int steps = T - 10;
    printf("  %d RBPF steps in %.2f ms\n", steps, ms);
    printf("  %.3f ms per step\n", ms / steps);
    printf("  %.1f steps per second\n", steps * 1000.0f / ms);
    printf("  %.1f μs per θ-particle per step\n", (ms * 1000.0f) / (steps * CUDA_N_THETA));
    
    /* Check ESS */
    float ess = smc2_cuda_get_outer_ess(state);
    printf("\n  Final outer ESS: %.1f / %d (%.1f%%)\n", 
           ess, state->N_theta, 100.0f * ess / state->N_theta);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(y);
    smc2_cuda_free(state);
    
    printf("\n  Performance test: PASSED\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Parameter Learning (No Rejuvenation)
 *═══════════════════════════════════════════════════════════════════════════*/

void test_parameter_learning(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Test: Parameter Learning (No Rejuvenation)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    /* True parameters */
    float true_rho = 0.96f;
    float true_sigma_z = 0.08f;
    float true_mu_base = -0.8f;
    float true_mu_scale = 0.4f;
    float true_mu_rate = 1.2f;
    float true_sigma_base = 0.12f;
    float true_sigma_scale = 0.08f;
    float true_sigma_rate = 1.0f;
    
    printf("TRUE parameters:\n");
    printf("  rho=%.4f, sigma_z=%.4f\n", true_rho, true_sigma_z);
    printf("  mu: base=%.4f, scale=%.4f, rate=%.4f\n", 
           true_mu_base, true_mu_scale, true_mu_rate);
    printf("  sigma: base=%.4f, scale=%.4f, rate=%.4f\n",
           true_sigma_base, true_sigma_scale, true_sigma_rate);
    
    /* Generate data */
    int T = 200;
    float* y = (float*)malloc(T * sizeof(float));
    float* h = (float*)malloc(T * sizeof(float));
    float* z = (float*)malloc(T * sizeof(float));
    
    generate_sv_data(y, h, z, T,
                     true_rho, true_sigma_z,
                     true_mu_base, true_mu_scale, true_mu_rate,
                     true_sigma_base, true_sigma_scale, true_sigma_rate,
                     0.02f, 0.08f, 1.5f);
    
    printf("\nGenerated %d observations\n", T);
    printf("  y: mean=%.2f, std=%.2f\n", 
           h[T/2], sqrtf(true_sigma_base * true_sigma_base));
    
    /* Allocate and initialize */
    SMC2StateCUDA* state = smc2_cuda_alloc(256, 256);
    smc2_cuda_init_from_prior(state);
    
    /* Run SMC² */
    printf("\nRunning SMC² (N_theta=%d, N_inner=%d)...\n", 
           state->N_theta, state->N_inner);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int t = 0; t < T; t++) {
        float ess = smc2_cuda_update(state, y[t]);
        if ((t + 1) % 50 == 0) {
            printf("  t=%d/%d, ESS_θ=%.1f\n", t + 1, T, ess);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Results\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Elapsed: %.1f ms (%.3f ms/obs)\n", ms, ms / T);
    printf("  Resamples: %d\n", state->n_resamples);
    
    /* Get posterior estimates */
    float theta_mean[8], theta_std[8];
    smc2_cuda_get_theta_mean(state, theta_mean);
    smc2_cuda_get_theta_std(state, theta_std);
    
    printf("\nESTIMATED (mean ± std):\n");
    printf("  rho=%.4f ± %.4f\n", theta_mean[0], theta_std[0]);
    printf("  sigma_z=%.4f ± %.4f\n", theta_mean[1], theta_std[1]);
    printf("  mu: base=%.4f ± %.4f, scale=%.4f ± %.4f, rate=%.4f ± %.4f\n",
           theta_mean[2], theta_std[2], theta_mean[3], theta_std[3], 
           theta_mean[4], theta_std[4]);
    printf("  sigma: base=%.4f ± %.4f, scale=%.4f ± %.4f, rate=%.4f ± %.4f\n",
           theta_mean[5], theta_std[5], theta_mean[6], theta_std[6],
           theta_mean[7], theta_std[7]);
    
    /* Check parameter recovery */
    float true_params[8] = {true_rho, true_sigma_z, true_mu_base, true_mu_scale,
                           true_mu_rate, true_sigma_base, true_sigma_scale, true_sigma_rate};
    const char* param_names[8] = {"rho", "sigma_z", "mu_base", "mu_scale",
                                  "mu_rate", "sigma_base", "sigma_scale", "sigma_rate"};
    
    printf("\nParameter recovery (z-scores):\n");
    int n_ok = 0;
    for (int i = 0; i < 8; i++) {
        float z_score = fabsf(theta_mean[i] - true_params[i]) / fmaxf(theta_std[i], 1e-6f);
        const char* status = (z_score <= 2.0f) ? "OK" : "MISS";
        if (z_score <= 2.0f) n_ok++;
        printf("  %-12s: z=%.1f [%s]\n", param_names[i], z_score, status);
    }
    
    printf("\n  OVERALL: %d/8 parameters within 2σ\n", n_ok);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(y);
    free(h);
    free(z);
    smc2_cuda_free(state);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Throughput Scaling
 *═══════════════════════════════════════════════════════════════════════════*/

void test_throughput_scaling(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Test: Throughput Scaling\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    /* Generate test data */
    int T = 500;
    float* y = (float*)malloc(T * sizeof(float));
    generate_sv_data(y, NULL, NULL, T,
                     0.96f, 0.08f,
                     -0.8f, 0.4f, 1.2f,
                     0.12f, 0.08f, 1.0f,
                     0.02f, 0.08f, 1.5f);
    
    printf("  N_theta    N_inner    Time(ms)    Steps/sec    μs/θ/step\n");
    printf("  ─────────────────────────────────────────────────────────\n");
    
    int N_configs[][2] = {{64, 64}, {128, 128}, {256, 256}, {512, 256}};
    int n_configs = sizeof(N_configs) / sizeof(N_configs[0]);
    
    for (int c = 0; c < n_configs; c++) {
        int N_theta = N_configs[c][0];
        int N_inner = N_configs[c][1];
        
        SMC2StateCUDA* state = smc2_cuda_alloc(N_theta, N_inner);
        smc2_cuda_init_from_prior(state);
        
        /* Warm-up */
        for (int t = 0; t < 20; t++) {
            smc2_cuda_update(state, y[t]);
        }
        cudaDeviceSynchronize();
        
        /* Timed run */
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        int T_run = 100;
        cudaEventRecord(start);
        for (int t = 20; t < 20 + T_run; t++) {
            smc2_cuda_update(state, y[t]);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        
        float steps_per_sec = T_run * 1000.0f / ms;
        float us_per_theta = (ms * 1000.0f) / (T_run * N_theta);
        
        printf("  %4d       %4d       %7.1f     %8.0f     %7.2f\n",
               N_theta, N_inner, ms, steps_per_sec, us_per_theta);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        smc2_cuda_free(state);
    }
    
    free(y);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Main
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char** argv) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  SMC² with RBPF Inner Filter - CUDA Test Suite                ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    test_cuda_basic();
    test_rbpf_step_performance();
    test_parameter_learning();
    test_throughput_scaling();
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("All tests completed.\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    return 0;
}
