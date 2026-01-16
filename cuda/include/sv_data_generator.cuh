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
 *   
 *   SMC2StateCUDA* state = smc2_cuda_alloc(256, 256);
 *   
 *   // Validate BEFORE generating data
 *   if (!validate_generator_vs_filter(gen, state)) {
 *       fprintf(stderr, "Fix parameters before running test!\n");
 *       exit(1);
 *   }
 *   
 *   gen.generate();
 *   smc2_cuda_init_from_prior(state);
 *   
 *   for (int t = 0; t < gen.T; t++) {
 *       smc2_cuda_update(state, gen.y[t]);
 *   }
 * 
 * Model (matches SMC² filter exactly):
 *   - z̃ follows AR(1): z̃_t = ρ·z̃_{t-1} + σ_z·ε_t
 *   - z = 1.5·(1 + tanh(z̃)) ∈ (0, 3)
 *   - h follows regime-switching OU: h_t = φ(z)·h_{t-1} + θ(z)·μ(z) + σ_h(z)·ε_t
 *   - y_t = h_t + log(χ²(1))
 */

#pragma once

#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstdio>

/* Forward declaration - full definition in smc2_rbpf_cuda.cuh */
struct SMC2StateCUDA;

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
        
        /* Initialize h from stationary distribution conditional on z */
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
    
    /* Print true parameters for reference */
    void print_true_params() const {
        printf("TRUE PARAMETERS:\n");
        printf("  rho=%.3f, sigma_z=%.3f\n", rho, sigma_z);
        printf("  mu: base=%.3f, scale=%.3f, rate=%.3f\n", mu_base, mu_scale, mu_rate);
        printf("  sigma: base=%.3f, scale=%.3f, rate=%.3f\n", sigma_base, sigma_scale, sigma_rate);
        printf("  theta (fixed): base=%.3f, scale=%.3f, rate=%.3f\n", theta_base, theta_scale, theta_rate);
    }
    
    /* Print data statistics */
    void print_data_stats() const {
        if (!y || T <= 0) {
            printf("No data generated yet.\n");
            return;
        }
        
        float y_mean = 0.0f, y_min = y[0], y_max = y[0];
        float h_mean = 0.0f, h_min = h_true[0], h_max = h_true[0];
        
        for (int t = 0; t < T; t++) {
            y_mean += y[t];
            h_mean += h_true[t];
            y_min = fminf(y_min, y[t]);
            y_max = fmaxf(y_max, y[t]);
            h_min = fminf(h_min, h_true[t]);
            h_max = fmaxf(h_max, h_true[t]);
        }
        y_mean /= T;
        h_mean /= T;
        
        float y_var = 0.0f, h_var = 0.0f;
        for (int t = 0; t < T; t++) {
            y_var += (y[t] - y_mean) * (y[t] - y_mean);
            h_var += (h_true[t] - h_mean) * (h_true[t] - h_mean);
        }
        y_var /= T;
        h_var /= T;
        
        printf("Generated T=%d observations\n", T);
        printf("  y: mean=%.3f, std=%.3f, range=[%.3f, %.3f]\n", 
               y_mean, sqrtf(y_var), y_min, y_max);
        printf("  h_true: mean=%.3f, std=%.3f, range=[%.3f, %.3f]\n",
               h_mean, sqrtf(h_var), h_min, h_max);
        printf("  Expected log χ²(1): mean≈-1.27, var≈4.93\n");
    }
    
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

/*═══════════════════════════════════════════════════════════════════════════════
 * VALIDATION: Generator vs Filter Parameter Compatibility
 * 
 * Call this BEFORE generate() to catch configuration errors early.
 * 
 * Checks:
 *   1. ERROR: Parameters outside filter bounds (cannot learn)
 *   2. ERROR: theta_curve mismatch (fixed parameter, must match exactly)
 *   3. WARNING: Parameters >3σ from prior (slow convergence)
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Validate generator parameters against filter configuration
 * 
 * @param gen    Data generator with true parameters
 * @param state  SMC² filter state (must be allocated, bounds/prior set)
 * @return true if valid (can proceed), false if fatal errors detected
 * 
 * Prints detailed error/warning messages to stderr.
 */
inline bool validate_generator_vs_filter(const SVDataGenerator& gen, const SMC2StateCUDA* state) {
    if (!state) {
        fprintf(stderr, "ERROR: SMC² state is NULL\n");
        return false;
    }
    
    bool valid = true;
    int warnings = 0;
    
    fprintf(stderr, "\n═══ Validating Generator vs Filter Configuration ═══\n\n");
    
    /* ─────────────────────────────────────────────────────────────────────────
     * Check 1: Parameters within bounds (FATAL if violated)
     * ─────────────────────────────────────────────────────────────────────────*/
    
    #define CHECK_BOUND(param, min_field, max_field, name) \
        do { \
            float val = gen.param; \
            float lo = state->bounds.min_field; \
            float hi = state->bounds.max_field; \
            if (val < lo || val > hi) { \
                fprintf(stderr, "  ERROR: " name " = %.4f outside bounds [%.4f, %.4f]\n", \
                        val, lo, hi); \
                valid = false; \
            } \
        } while(0)
    
    CHECK_BOUND(rho, rho_min, rho_max, "rho");
    CHECK_BOUND(sigma_z, sigma_z_min, sigma_z_max, "sigma_z");
    CHECK_BOUND(mu_base, mu_base_min, mu_base_max, "mu_base");
    CHECK_BOUND(mu_scale, mu_scale_min, mu_scale_max, "mu_scale");
    CHECK_BOUND(mu_rate, mu_rate_min, mu_rate_max, "mu_rate");
    CHECK_BOUND(sigma_base, sigma_base_min, sigma_base_max, "sigma_base");
    CHECK_BOUND(sigma_scale, sigma_scale_min, sigma_scale_max, "sigma_scale");
    CHECK_BOUND(sigma_rate, sigma_rate_min, sigma_rate_max, "sigma_rate");
    
    #undef CHECK_BOUND
    
    /* ─────────────────────────────────────────────────────────────────────────
     * Check 2: theta_curve must match exactly (FATAL if mismatched)
     * ─────────────────────────────────────────────────────────────────────────*/
    
    const float eps = 1e-5f;
    bool theta_mismatch = false;
    
    if (fabsf(gen.theta_base - state->theta_curve.base) > eps) {
        fprintf(stderr, "  ERROR: theta_base mismatch: gen=%.4f vs filter=%.4f\n",
                gen.theta_base, state->theta_curve.base);
        theta_mismatch = true;
    }
    if (fabsf(gen.theta_scale - state->theta_curve.scale) > eps) {
        fprintf(stderr, "  ERROR: theta_scale mismatch: gen=%.4f vs filter=%.4f\n",
                gen.theta_scale, state->theta_curve.scale);
        theta_mismatch = true;
    }
    if (fabsf(gen.theta_rate - state->theta_curve.rate) > eps) {
        fprintf(stderr, "  ERROR: theta_rate mismatch: gen=%.4f vs filter=%.4f\n",
                gen.theta_rate, state->theta_curve.rate);
        theta_mismatch = true;
    }
    
    if (theta_mismatch) {
        fprintf(stderr, "         theta_curve is FIXED in the filter (not learned).\n");
        fprintf(stderr, "         Generator and filter MUST use identical values.\n");
        valid = false;
    }
    
    /* ─────────────────────────────────────────────────────────────────────────
     * Check 3: Parameters far from prior (WARNING only)
     * ─────────────────────────────────────────────────────────────────────────*/
    
    #define CHECK_PRIOR(param, mean_field, std_field, name) \
        do { \
            float val = gen.param; \
            float mu = state->prior.mean_field; \
            float sigma = state->prior.std_field; \
            float z_score = fabsf(val - mu) / fmaxf(sigma, 1e-6f); \
            if (z_score > 3.0f) { \
                fprintf(stderr, "  WARNING: " name " = %.4f is %.1fσ from prior " \
                        "(mean=%.4f, std=%.4f)\n", val, z_score, mu, sigma); \
                fprintf(stderr, "           Learning will be slow. Consider adjusting prior.\n"); \
                warnings++; \
            } \
        } while(0)
    
    CHECK_PRIOR(rho, rho_mean, rho_std, "rho");
    CHECK_PRIOR(sigma_z, sigma_z_mean, sigma_z_std, "sigma_z");
    CHECK_PRIOR(mu_base, mu_base_mean, mu_base_std, "mu_base");
    CHECK_PRIOR(mu_scale, mu_scale_mean, mu_scale_std, "mu_scale");
    CHECK_PRIOR(mu_rate, mu_rate_mean, mu_rate_std, "mu_rate");
    CHECK_PRIOR(sigma_base, sigma_base_mean, sigma_base_std, "sigma_base");
    CHECK_PRIOR(sigma_scale, sigma_scale_mean, sigma_scale_std, "sigma_scale");
    CHECK_PRIOR(sigma_rate, sigma_rate_mean, sigma_rate_std, "sigma_rate");
    
    #undef CHECK_PRIOR
    
    /* ─────────────────────────────────────────────────────────────────────────
     * Summary
     * ─────────────────────────────────────────────────────────────────────────*/
    
    if (valid && warnings == 0) {
        fprintf(stderr, "  ✓ All parameters valid and well-matched to prior.\n");
    } else if (valid) {
        fprintf(stderr, "\n  ⚠ %d warning(s) - filter will run but may converge slowly.\n", warnings);
    } else {
        fprintf(stderr, "\n  ✗ FATAL: Cannot proceed with current configuration.\n");
        fprintf(stderr, "    Fix errors above before running the filter.\n");
    }
    
    fprintf(stderr, "\n════════════════════════════════════════════════════════\n\n");
    
    return valid;
}

/**
 * @brief Copy generator's true parameters to filter's prior (for oracle testing)
 * 
 * Sets prior mean = true value with small std. Useful for:
 *   - Debugging filter correctness (should converge quickly)
 *   - Baseline comparisons
 * 
 * @param gen    Data generator with true parameters
 * @param state  SMC² filter state (modified in place)
 * @param prior_std_factor  Prior std = true_value * factor (default 0.1 = 10%)
 */
inline void set_prior_from_generator(const SVDataGenerator& gen, SMC2StateCUDA* state, 
                                      float prior_std_factor = 0.1f) {
    if (!state) return;
    
    state->prior.rho_mean = gen.rho;
    state->prior.rho_std = fmaxf(0.01f, fabsf(gen.rho) * prior_std_factor);
    
    state->prior.sigma_z_mean = gen.sigma_z;
    state->prior.sigma_z_std = fmaxf(0.01f, gen.sigma_z * prior_std_factor);
    
    state->prior.mu_base_mean = gen.mu_base;
    state->prior.mu_base_std = fmaxf(0.1f, fabsf(gen.mu_base) * prior_std_factor);
    
    state->prior.mu_scale_mean = gen.mu_scale;
    state->prior.mu_scale_std = fmaxf(0.1f, fabsf(gen.mu_scale) * prior_std_factor);
    
    state->prior.mu_rate_mean = gen.mu_rate;
    state->prior.mu_rate_std = fmaxf(0.1f, gen.mu_rate * prior_std_factor);
    
    state->prior.sigma_base_mean = gen.sigma_base;
    state->prior.sigma_base_std = fmaxf(0.01f, gen.sigma_base * prior_std_factor);
    
    state->prior.sigma_scale_mean = gen.sigma_scale;
    state->prior.sigma_scale_std = fmaxf(0.01f, fabsf(gen.sigma_scale) * prior_std_factor);
    
    state->prior.sigma_rate_mean = gen.sigma_rate;
    state->prior.sigma_rate_std = fmaxf(0.1f, gen.sigma_rate * prior_std_factor);
    
    /* theta_curve is fixed, just copy directly */
    state->theta_curve.base = gen.theta_base;
    state->theta_curve.scale = gen.theta_scale;
    state->theta_curve.rate = gen.theta_rate;
    
    fprintf(stderr, "Prior set from generator (std_factor=%.2f)\n", prior_std_factor);
}
