/**
 * @file sv_data_generator.cuh
 * @brief Synthetic SV data generation - include and call directly from tests
 * 
 * Usage:
 *   #include "sv_data_generator.cuh"
 *   
 *   SVDataGenerator gen;
 *   gen.seed(42);
 *   gen.T = 500;
 *   gen.generate();
 *   
 *   // Use gen.y, gen.h_true, gen.z_true directly
 *   for (int t = 0; t < gen.T; t++) {
 *       smc2_cuda_update(state, gen.y[t]);
 *   }
 * 
 * Model (matches SMC² filter exactly):
 *   - z̃ follows AR(1): z̃_t = ρ·z̃_{t-1} + σ_z·ε_t
 *   - z = 1.5·(1 + tanh(z̃)) ∈ (0, 3)
 *   - y_t = h_t + log(χ²(1))
 */

#pragma once

#include <cstdlib>
#include <cstdint>
#include <cmath>

struct SVDataGenerator {
    /* Parameters (set before calling generate()) */
    int T = 500;
    float rho = 0.95f;
    float sigma_z = 0.15f;
    float mu_base = -1.0f;
    float mu_scale = 0.5f;
    float mu_rate = 1.0f;
    float sigma_base = 0.15f;
    float sigma_scale = 0.10f;
    float sigma_rate = 1.0f;
    float theta_base = 0.02f;
    float theta_scale = 0.08f;
    float theta_rate = 1.5f;
    
    /* Output arrays (allocated by generate()) */
    float* y = nullptr;
    float* h_true = nullptr;
    float* z_true = nullptr;
    
    /* RNG state */
    uint64_t rng_state = 12345678901234567ULL;
    
    void seed(uint64_t s) {
        rng_state = s ? s : 12345678901234567ULL;
    }
    
    void generate() {
        free_arrays();
        y = (float*)malloc(T * sizeof(float));
        h_true = (float*)malloc(T * sizeof(float));
        z_true = (float*)malloc(T * sizeof(float));
        
        /* Initialize z̃ from stationary distribution */
        float var_stat = (sigma_z * sigma_z) / fmaxf(1.0f - rho * rho, 1e-6f);
        float z_tilde = sqrtf(var_stat) * normal();
        float z = z_tilde_to_z(z_tilde);
        
        /* Initialize h */
        float theta_z = eval_curve(theta_base, theta_scale, theta_rate, z);
        float mu_z = eval_curve(mu_base, mu_scale, mu_rate, z);
        float sigma_h = eval_curve(sigma_base, sigma_scale, sigma_rate, z);
        float phi = 1.0f - theta_z;
        float h_var = (sigma_h * sigma_h) / fmaxf(1.0f - phi * phi, 1e-6f);
        float h = mu_z + sqrtf(h_var) * normal();
        
        for (int t = 0; t < T; t++) {
            h_true[t] = h;
            z_true[t] = z;
            
            /* Observation: y = h + log(χ²(1)) */
            float eps = normal();
            y[t] = h + logf(eps * eps + 1e-10f);
            
            /* Transition z̃ */
            z_tilde = rho * z_tilde + sigma_z * normal();
            z = z_tilde_to_z(z_tilde);
            
            /* Transition h */
            theta_z = eval_curve(theta_base, theta_scale, theta_rate, z);
            mu_z = eval_curve(mu_base, mu_scale, mu_rate, z);
            sigma_h = eval_curve(sigma_base, sigma_scale, sigma_rate, z);
            phi = 1.0f - theta_z;
            h = phi * h + theta_z * mu_z + sigma_h * normal();
        }
    }
    
    void free_arrays() {
        if (y) { free(y); y = nullptr; }
        if (h_true) { free(h_true); h_true = nullptr; }
        if (z_true) { free(z_true); z_true = nullptr; }
    }
    
    ~SVDataGenerator() { free_arrays(); }
    
private:
    float uniform() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return (rng_state >> 11) * (1.0f / 9007199254740992.0f);
    }
    
    float normal() {
        float u1 = uniform();
        float u2 = uniform();
        while (u1 < 1e-10f) u1 = uniform();
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
    }
    
    static float z_tilde_to_z(float z_tilde) {
        return 1.5f * (1.0f + tanhf(z_tilde));
    }
    
    static float eval_curve(float base, float scale, float rate, float z) {
        return base + scale * (1.0f - expf(-rate * z));
    }
};
