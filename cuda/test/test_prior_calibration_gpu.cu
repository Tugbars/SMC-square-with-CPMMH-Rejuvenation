/*
 * test_prior_calibration_gpu.cu
 * 
 * Tests for GPU-native prior calibration.
 * 
 * Compile:
 *   nvcc -o test_prior_calibration_gpu test_prior_calibration_gpu.cu -std=c++17 -O3
 * 
 * Run:
 *   ./test_prior_calibration_gpu
 */

#include "smc2_prior_calibration_gpu.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

/* ============================================================================
 * Test Utilities
 * ============================================================================ */

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

static int g_tests_run = 0;
static int g_tests_passed = 0;

void test_pass(const char* name) {
    g_tests_run++;
    g_tests_passed++;
    printf("  ✓ %s\n", name);
}

void test_fail(const char* name, const char* msg) {
    g_tests_run++;
    printf("  ✗ %s\n    %s\n", name, msg);
}

/* Simple xorshift RNG for reproducibility */
uint64_t rng_state = 12345678901234567ULL;

float rand_normal() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    float u1 = (rng_state >> 11) * (1.0f / 9007199254740992.0f);
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    float u2 = (rng_state >> 11) * (1.0f / 9007199254740992.0f);
    while (u1 < 1e-10f) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        u1 = (rng_state >> 11) * (1.0f / 9007199254740992.0f);
    }
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

/* Generate synthetic SV data */
void generate_sv_data(float* returns, int n, float mu_base, float rho, float sigma_z) {
    float h = mu_base;
    float z_tilde = 0.0f;
    
    for (int t = 0; t < n; t++) {
        /* Observation */
        float eps = rand_normal();
        returns[t] = expf(h / 2.0f) * eps;
        
        /* Transition */
        z_tilde = rho * z_tilde + sigma_z * rand_normal();
        float theta = 0.02f + 0.08f * (1.0f - expf(-1.5f * 1.5f * (1.0f + tanhf(z_tilde))));
        float mu_z = mu_base + 0.5f * (1.0f - expf(-1.0f * 1.5f * (1.0f + tanhf(z_tilde))));
        float sigma_h = 0.15f + 0.10f * (1.0f - expf(-1.0f * 1.5f * (1.0f + tanhf(z_tilde))));
        h = (1.0f - theta) * h + theta * mu_z + sigma_h * rand_normal();
    }
}

/* ============================================================================
 * Tests
 * ============================================================================ */

void test_basic_calibration() {
    const int n = 100;
    float h_returns[n];
    
    /* Generate calm-regime data */
    rng_state = 42;
    generate_sv_data(h_returns, n, -10.0f, 0.95f, 0.10f);
    
    /* Copy to GPU */
    float* d_returns;
    SMC2PriorGPU* d_prior;
    CUDA_CHECK(cudaMalloc(&d_returns, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prior, sizeof(SMC2PriorGPU)));
    CUDA_CHECK(cudaMemcpy(d_returns, h_returns, n * sizeof(float), cudaMemcpyHostToDevice));
    
    /* Calibrate */
    CUDA_CHECK(smc2_calibrate_prior_gpu(d_returns, n, d_prior));
    
    /* Copy back for verification */
    SMC2PriorGPU h_prior;
    CUDA_CHECK(cudaMemcpy(&h_prior, d_prior, sizeof(SMC2PriorGPU), cudaMemcpyDeviceToHost));
    
    /* Check reasonable bounds */
    bool ok = true;
    char msg[256] = "";
    
    if (h_prior.rho_mean < 0.5f || h_prior.rho_mean > 1.0f) {
        snprintf(msg, sizeof(msg), "rho_mean=%.3f out of range [0.5, 1.0]", h_prior.rho_mean);
        ok = false;
    }
    if (h_prior.sigma_z_mean < 0.01f || h_prior.sigma_z_mean > 0.8f) {
        snprintf(msg, sizeof(msg), "sigma_z_mean=%.3f out of range [0.01, 0.8]", h_prior.sigma_z_mean);
        ok = false;
    }
    if (h_prior.mu_base_mean > 0.0f || h_prior.mu_base_mean < -15.0f) {
        snprintf(msg, sizeof(msg), "mu_base_mean=%.3f out of range [-15, 0]", h_prior.mu_base_mean);
        ok = false;
    }
    
    if (ok) {
        test_pass("Basic calibration");
    } else {
        test_fail("Basic calibration", msg);
    }
    
    cudaFree(d_returns);
    cudaFree(d_prior);
}

void test_calm_vs_crisis() {
    const int n = 100;
    float h_calm[n], h_crisis[n];
    
    /* Generate calm data (~10% vol) */
    rng_state = 123;
    generate_sv_data(h_calm, n, -10.5f, 0.98f, 0.08f);
    
    /* Generate crisis data (~50% vol) */
    rng_state = 456;
    generate_sv_data(h_crisis, n, -7.5f, 0.85f, 0.30f);
    
    /* Allocate */
    float* d_calm;
    float* d_crisis;
    SMC2PriorGPU* d_prior_calm;
    SMC2PriorGPU* d_prior_crisis;
    
    CUDA_CHECK(cudaMalloc(&d_calm, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_crisis, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prior_calm, sizeof(SMC2PriorGPU)));
    CUDA_CHECK(cudaMalloc(&d_prior_crisis, sizeof(SMC2PriorGPU)));
    
    CUDA_CHECK(cudaMemcpy(d_calm, h_calm, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_crisis, h_crisis, n * sizeof(float), cudaMemcpyHostToDevice));
    
    /* Calibrate both */
    CUDA_CHECK(smc2_calibrate_prior_gpu(d_calm, n, d_prior_calm));
    CUDA_CHECK(smc2_calibrate_prior_gpu(d_crisis, n, d_prior_crisis));
    
    /* Copy back */
    SMC2PriorGPU prior_calm, prior_crisis;
    CUDA_CHECK(cudaMemcpy(&prior_calm, d_prior_calm, sizeof(SMC2PriorGPU), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&prior_crisis, d_prior_crisis, sizeof(SMC2PriorGPU), cudaMemcpyDeviceToHost));
    
    /* Crisis should have higher mu_base (less negative = higher vol) */
    bool ok = prior_crisis.mu_base_mean > prior_calm.mu_base_mean;
    
    if (ok) {
        test_pass("Calm vs Crisis discrimination");
        printf("      Calm:   mu_base=%.2f, rho=%.3f, sigma_z=%.3f\n",
               prior_calm.mu_base_mean, prior_calm.rho_mean, prior_calm.sigma_z_mean);
        printf("      Crisis: mu_base=%.2f, rho=%.3f, sigma_z=%.3f\n",
               prior_crisis.mu_base_mean, prior_crisis.rho_mean, prior_crisis.sigma_z_mean);
    } else {
        char msg[256];
        snprintf(msg, sizeof(msg), "Expected crisis mu_base > calm, got calm=%.2f, crisis=%.2f",
                 prior_calm.mu_base_mean, prior_crisis.mu_base_mean);
        test_fail("Calm vs Crisis discrimination", msg);
    }
    
    cudaFree(d_calm);
    cudaFree(d_crisis);
    cudaFree(d_prior_calm);
    cudaFree(d_prior_crisis);
}

void test_warmup_buffer() {
    WarmupBufferGPU warmup;
    CUDA_CHECK(warmup_buffer_init(&warmup, 100));
    
    /* Add observations */
    rng_state = 789;
    for (int i = 0; i < 100; i++) {
        float y = rand_normal() * 0.01f;  /* ~1% daily returns */
        CUDA_CHECK(warmup_buffer_add(&warmup, y));
    }
    
    if (!warmup_buffer_ready(&warmup, 100)) {
        test_fail("Warmup buffer", "Buffer not ready after 100 adds");
        warmup_buffer_free(&warmup);
        return;
    }
    
    /* Calibrate from buffer */
    SMC2PriorGPU* d_prior;
    CUDA_CHECK(cudaMalloc(&d_prior, sizeof(SMC2PriorGPU)));
    CUDA_CHECK(smc2_calibrate_prior_gpu(warmup.d_buffer, warmup.count, d_prior));
    
    /* Verify */
    SMC2PriorGPU h_prior;
    CUDA_CHECK(cudaMemcpy(&h_prior, d_prior, sizeof(SMC2PriorGPU), cudaMemcpyDeviceToHost));
    
    if (h_prior.rho_mean > 0.5f && h_prior.rho_std > 0.0f) {
        test_pass("Warmup buffer workflow");
    } else {
        test_fail("Warmup buffer workflow", "Invalid prior from buffer");
    }
    
    warmup_buffer_free(&warmup);
    cudaFree(d_prior);
}

void test_verbose_output() {
    const int n = 100;
    float h_returns[n];
    
    rng_state = 999;
    generate_sv_data(h_returns, n, -9.0f, 0.94f, 0.15f);
    
    float* d_returns;
    SMC2PriorGPU* d_prior;
    SMC2PriorGPU h_prior;
    
    CUDA_CHECK(cudaMalloc(&d_returns, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prior, sizeof(SMC2PriorGPU)));
    CUDA_CHECK(cudaMemcpy(d_returns, h_returns, n * sizeof(float), cudaMemcpyHostToDevice));
    
    printf("\n  Verbose output test:\n");
    CUDA_CHECK(smc2_calibrate_prior_gpu_verbose(d_returns, n, d_prior, &h_prior));
    
    test_pass("Verbose output");
    
    cudaFree(d_returns);
    cudaFree(d_prior);
}

void test_minimum_observations() {
    float* d_returns;
    SMC2PriorGPU* d_prior;
    
    CUDA_CHECK(cudaMalloc(&d_returns, 20 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prior, sizeof(SMC2PriorGPU)));
    
    /* Should fail with < 30 observations */
    cudaError_t err = smc2_calibrate_prior_gpu(d_returns, 20, d_prior);
    
    if (err == cudaErrorInvalidValue) {
        test_pass("Reject too few observations");
    } else {
        test_fail("Reject too few observations", "Should return cudaErrorInvalidValue");
    }
    
    cudaFree(d_returns);
    cudaFree(d_prior);
}

void test_prior_width_from_bounds() {
    const int n = 100;
    float h_returns[n];
    
    rng_state = 1111;
    generate_sv_data(h_returns, n, -9.0f, 0.90f, 0.20f);
    
    float* d_returns;
    SMC2PriorGPU* d_prior;
    
    CUDA_CHECK(cudaMalloc(&d_returns, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prior, sizeof(SMC2PriorGPU)));
    CUDA_CHECK(cudaMemcpy(d_returns, h_returns, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(smc2_calibrate_prior_gpu(d_returns, n, d_prior));
    
    SMC2PriorGPU h_prior;
    CUDA_CHECK(cudaMemcpy(&h_prior, d_prior, sizeof(SMC2PriorGPU), cudaMemcpyDeviceToHost));
    
    /* Check widths are derived from SPY bounds */
    /* rho_std should be ~(0.99 - 0.665) / 4 = 0.081 */
    /* sigma_z_std should be ~(0.65 - 0.031) / 4 = 0.155 */
    
    bool ok = true;
    char msg[256] = "";
    
    float expected_rho_std = (0.990f - 0.665f) / 4.0f;
    float expected_sigma_z_std = (0.650f - 0.031f) / 4.0f;
    
    if (fabsf(h_prior.rho_std - expected_rho_std) > 0.01f) {
        snprintf(msg, sizeof(msg), "rho_std=%.4f, expected=%.4f", 
                 h_prior.rho_std, expected_rho_std);
        ok = false;
    }
    if (fabsf(h_prior.sigma_z_std - expected_sigma_z_std) > 0.01f) {
        snprintf(msg, sizeof(msg), "sigma_z_std=%.4f, expected=%.4f",
                 h_prior.sigma_z_std, expected_sigma_z_std);
        ok = false;
    }
    
    if (ok) {
        test_pass("Prior width from historical bounds");
    } else {
        test_fail("Prior width from historical bounds", msg);
    }
    
    cudaFree(d_returns);
    cudaFree(d_prior);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║  GPU Prior Calibration Tests                                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Basic Tests:\n");
    test_basic_calibration();
    test_minimum_observations();
    test_prior_width_from_bounds();
    
    printf("\nRegime Detection:\n");
    test_calm_vs_crisis();
    
    printf("\nWorkflow Tests:\n");
    test_warmup_buffer();
    test_verbose_output();
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Results: %d/%d tests passed\n", g_tests_passed, g_tests_run);
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
