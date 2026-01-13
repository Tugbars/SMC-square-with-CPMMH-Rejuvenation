/**
 * @file test_smc2_cuda.cu
 * @brief Test suite for SMC² CUDA with proper PMMH rejuvenation
 */

#include "smc2_rbpf_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*═══════════════════════════════════════════════════════════════════════════
 * Host RNG
 *═══════════════════════════════════════════════════════════════════════════*/

static unsigned long long h_rng = 12345678901234567ULL;

static float host_uniform(void) {
    h_rng ^= h_rng << 13;
    h_rng ^= h_rng >> 7;
    h_rng ^= h_rng << 17;
    return (h_rng >> 11) * (1.0f / 9007199254740992.0f);
}

static float host_normal(void) {
    float u1 = host_uniform();
    float u2 = host_uniform();
    while (u1 < 1e-10f) u1 = host_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Generate SV Data
 *═══════════════════════════════════════════════════════════════════════════*/

void generate_sv_data(
    float* y, float* h_true, float* z_true, int T,
    float rho, float sigma_z,
    float mu_base, float mu_scale, float mu_rate,
    float sigma_base, float sigma_scale, float sigma_rate,
    float theta_base, float theta_scale, float theta_rate
) {
    float z = 0.2f;
    float h = mu_base;
    
    for (int t = 0; t < T; t++) {
        if (h_true) h_true[t] = h;
        if (z_true) z_true[t] = z;
        
        float eps = host_normal();
        y[t] = h + logf(eps * eps + 1e-10f);
        
        float theta_z = theta_base + theta_scale * (1.0f - expf(-theta_rate * z));
        float mu_z = mu_base + mu_scale * (1.0f - expf(-mu_rate * z));
        float sigma_h = sigma_base + sigma_scale * (1.0f - expf(-sigma_rate * z));
        
        float phi = 1.0f - theta_z;
        z = rho * z + sigma_z * host_normal();
        z = fmaxf(0.0f, fminf(3.0f, z));
        h = phi * h + theta_z * mu_z + sigma_h * host_normal();
    }
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
    
    /* True parameters */
    float true_rho = 0.96f;
    float true_sigma_z = 0.08f;
    float true_mu_base = -0.8f;
    float true_mu_scale = 0.4f;
    float true_mu_rate = 1.2f;
    float true_sigma_base = 0.12f;
    float true_sigma_scale = 0.08f;
    float true_sigma_rate = 1.0f;
    
    printf("TRUE:\n");
    printf("  rho=%.3f, sigma_z=%.3f\n", true_rho, true_sigma_z);
    printf("  mu: base=%.3f, scale=%.3f, rate=%.3f\n", 
           true_mu_base, true_mu_scale, true_mu_rate);
    printf("  sigma: base=%.3f, scale=%.3f, rate=%.3f\n",
           true_sigma_base, true_sigma_scale, true_sigma_rate);
    
    int T = 200;
    float* y = (float*)malloc(T * sizeof(float));
    generate_sv_data(y, NULL, NULL, T,
                     true_rho, true_sigma_z,
                     true_mu_base, true_mu_scale, true_mu_rate,
                     true_sigma_base, true_sigma_scale, true_sigma_rate,
                     0.02f, 0.08f, 1.5f);
    
    printf("\nGenerated T=%d observations\n", T);
    
    SMC2StateCUDA* state = smc2_cuda_alloc(256, 256);
    smc2_cuda_init_from_prior(state);
    
    printf("\nRunning SMC² (N_theta=%d, N_inner=%d, K_rejuv=%d)...\n",
           state->N_theta, state->N_inner, state->K_rejuv);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int t = 0; t < T; t++) {
        float ess = smc2_cuda_update(state, y[t]);
        if ((t + 1) % 50 == 0) {
            printf("  t=%d: ESS=%.1f, resamples=%d, rejuv_accept=%.1f%%\n",
                   t + 1, ess, state->n_resamples,
                   state->n_rejuv_total > 0 ? 
                   100.0f * state->n_rejuv_accepts / state->n_rejuv_total : 0.0f);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Results\n");
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
    printf("  rho=%.4f±%.4f, sigma_z=%.4f±%.4f\n",
           theta_mean[0], theta_std[0], theta_mean[1], theta_std[1]);
    printf("  mu: base=%.4f±%.4f, scale=%.4f±%.4f, rate=%.4f±%.4f\n",
           theta_mean[2], theta_std[2], theta_mean[3], theta_std[3],
           theta_mean[4], theta_std[4]);
    printf("  sigma: base=%.4f±%.4f, scale=%.4f±%.4f, rate=%.4f±%.4f\n",
           theta_mean[5], theta_std[5], theta_mean[6], theta_std[6],
           theta_mean[7], theta_std[7]);
    
    /* Z-scores */
    float true_params[8] = {true_rho, true_sigma_z, true_mu_base, true_mu_scale,
                           true_mu_rate, true_sigma_base, true_sigma_scale, true_sigma_rate};
    const char* names[8] = {"rho", "sigma_z", "mu_base", "mu_scale",
                            "mu_rate", "sigma_base", "sigma_scale", "sigma_rate"};
    
    printf("\nParameter recovery:\n");
    int n_ok = 0;
    for (int i = 0; i < 8; i++) {
        float z = fabsf(theta_mean[i] - true_params[i]) / fmaxf(theta_std[i], 1e-6f);
        const char* status = (z <= 2.0f) ? "OK" : "MISS";
        if (z <= 2.0f) n_ok++;
        printf("  %-12s: z=%.2f [%s]\n", names[i], z, status);
    }
    
    printf("\n  OVERALL: %d/8 within 2σ\n", n_ok);
    printf("  %s\n", n_ok >= 6 ? "PASSED" : "NEEDS TUNING");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(y);
    smc2_cuda_free(state);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Throughput with PMMH
 *═══════════════════════════════════════════════════════════════════════════*/

void test_throughput(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Throughput with PMMH Rejuvenation\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    int T = 300;
    float* y = (float*)malloc(T * sizeof(float));
    generate_sv_data(y, NULL, NULL, T,
                     0.96f, 0.08f, -0.8f, 0.4f, 1.2f,
                     0.12f, 0.08f, 1.0f, 0.02f, 0.08f, 1.5f);
    
    printf("  N_theta  N_inner  Time(ms)  Resamples  Rejuv%%  ms/obs\n");
    printf("  ────────────────────────────────────────────────────────\n");
    
    int configs[][2] = {{128, 128}, {256, 256}};
    int n_configs = sizeof(configs) / sizeof(configs[0]);
    
    for (int c = 0; c < n_configs; c++) {
        SMC2StateCUDA* state = smc2_cuda_alloc(configs[c][0], configs[c][1]);
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
        
        printf("  %4d     %4d     %7.1f   %4d       %5.1f   %.2f\n",
               configs[c][0], configs[c][1], ms, state->n_resamples,
               rejuv_pct, ms / T);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        smc2_cuda_free(state);
    }
    
    free(y);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Main
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  SMC² RBPF CUDA - Test Suite (with PMMH Rejuvenation)         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    test_basic();
    test_parameter_learning();
    test_throughput();
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("All tests completed.\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    return 0;
}
