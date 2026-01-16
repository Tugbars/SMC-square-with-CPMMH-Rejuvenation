/**
 * @file smc2_test_utils.cuh
 * @brief Common test utilities for SMC² - Single source of truth for data generation
 * 
 * IMPORTANT: All synthetic data generation MUST use this file.
 * Do NOT create local data generation functions in individual test files.
 * 
 * This ensures:
 *   - Consistent OCSN mixture parameters
 *   - Consistent ground truth definitions
 *   - Consistent state-space model implementation
 *   - Tests that combine cleanly
 */

#ifndef SMC2_TEST_UTILS_CUH
#define SMC2_TEST_UTILS_CUH

#include <vector>
#include <random>
#include <cmath>
#include <cstdio>

/*═══════════════════════════════════════════════════════════════════════════════
 * OCSN MIXTURE PARAMETERS (Omori et al. 2007, Table 1)
 * 
 * 10-component Gaussian mixture approximation to log(χ²(1))
 * CANONICAL SOURCE - Do not duplicate elsewhere!
 *═══════════════════════════════════════════════════════════════════════════════*/

namespace smc2_test {

/* Use existing OCSN_K if defined by smc2_rbpf_cuda.cuh, otherwise define here */
#ifndef OCSN_K
#define OCSN_K 10
#endif

constexpr int OCSN_NUM_COMPONENTS = 10;

constexpr float OCSN_WEIGHTS[OCSN_NUM_COMPONENTS] = {
    0.00609f, 0.04775f, 0.13057f, 0.20674f, 0.22715f,
    0.18842f, 0.12047f, 0.05591f, 0.01575f, 0.00115f
};

constexpr float OCSN_MEANS[OCSN_NUM_COMPONENTS] = {
    1.92677f,  1.34744f,  0.73504f,  0.02266f, -0.85173f,
   -1.97278f, -3.46788f, -5.55246f, -8.68384f, -14.65000f
};

constexpr float OCSN_VARS[OCSN_NUM_COMPONENTS] = {
    0.11265f, 0.17788f, 0.26768f, 0.40611f, 0.62699f,
    0.98583f, 1.57469f, 2.54498f, 4.16591f, 7.33342f
};

/*═══════════════════════════════════════════════════════════════════════════════
 * GROUND TRUTH PARAMETER STRUCTURE
 * 
 * Defines all parameters for the regime-switching SV model.
 *═══════════════════════════════════════════════════════════════════════════════*/

struct GroundTruth {
    /* Regime (z) dynamics */
    float rho;           /**< AR(1) persistence for z̃ */
    float sigma_z;       /**< Innovation std for z̃ */
    float z_floor;       /**< Lower bound for z (after transform) */
    float z_ceil;        /**< Upper bound for z (after transform) */
    
    /* Volatility mean curve: μ(z) = mu_base + mu_scale * (1 - exp(-mu_rate * z)) */
    float mu_base;
    float mu_scale;
    float mu_rate;
    
    /* Vol-of-vol curve: σ_h(z) = sigma_base + sigma_scale * (1 - exp(-sigma_rate * z)) */
    float sigma_base;
    float sigma_scale;
    float sigma_rate;
    
    /* Mean-reversion curve: θ(z) = theta_base + theta_scale * (1 - exp(-theta_rate * z)) */
    float theta_base;
    float theta_scale;
    float theta_rate;
    
    /* Default constructor with sensible defaults */
    GroundTruth() :
        rho(0.95f), sigma_z(0.10f), z_floor(0.0f), z_ceil(3.0f),
        mu_base(-2.0f), mu_scale(1.0f), mu_rate(1.0f),
        sigma_base(0.10f), sigma_scale(0.10f), sigma_rate(1.0f),
        theta_base(0.02f), theta_scale(0.08f), theta_rate(1.5f) {}
};

/*═══════════════════════════════════════════════════════════════════════════════
 * PREDEFINED TEST REGIMES
 * 
 * Use these for consistent testing across all test files.
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Calm market regime
 * 
 * Low volatility, slow regime changes, tight vol-of-vol.
 * Vol ≈ exp(-3/2) ≈ 22%, slow mean-reversion.
 */
inline GroundTruth regime_calm() {
    GroundTruth gt;
    gt.rho = 0.98f;
    gt.sigma_z = 0.05f;
    gt.z_floor = 0.0f;
    gt.z_ceil = 3.0f;
    gt.mu_base = -3.0f;
    gt.mu_scale = 0.5f;
    gt.mu_rate = 1.0f;
    gt.sigma_base = 0.08f;
    gt.sigma_scale = 0.05f;
    gt.sigma_rate = 1.0f;
    gt.theta_base = 0.02f;
    gt.theta_scale = 0.08f;
    gt.theta_rate = 1.5f;
    return gt;
}

/**
 * @brief Crisis market regime
 * 
 * High volatility, fast regime switching, wide vol-of-vol.
 * Vol ≈ exp(-1.5/2) ≈ 47%, fast mean-reversion in stress.
 */
inline GroundTruth regime_crisis() {
    GroundTruth gt;
    gt.rho = 0.90f;
    gt.sigma_z = 0.15f;
    gt.z_floor = 0.0f;
    gt.z_ceil = 3.0f;
    gt.mu_base = -1.5f;
    gt.mu_scale = 1.5f;
    gt.mu_rate = 0.8f;
    gt.sigma_base = 0.15f;
    gt.sigma_scale = 0.20f;
    gt.sigma_rate = 0.8f;
    gt.theta_base = 0.02f;
    gt.theta_scale = 0.08f;
    gt.theta_rate = 1.5f;
    return gt;
}

/**
 * @brief Original CPMMH test regime (from old codebase)
 * 
 * High signal regime for easier parameter identification.
 * Use this for validating against legacy tests.
 */
inline GroundTruth regime_legacy_cpmmh() {
    GroundTruth gt;
    gt.rho = 0.985f;
    gt.sigma_z = 0.06f;
    gt.z_floor = 0.0f;
    gt.z_ceil = 3.0f;
    gt.mu_base = -4.2f;
    gt.mu_scale = 2.8f;
    gt.mu_rate = 0.35f;
    gt.sigma_base = 0.07f;
    gt.sigma_scale = 0.35f;
    gt.sigma_rate = 0.25f;
    gt.theta_base = 0.005f;
    gt.theta_scale = 0.12f;
    gt.theta_rate = 0.30f;
    return gt;
}

/**
 * @brief Moderate regime (balanced)
 * 
 * Middle-ground parameters for general testing.
 */
inline GroundTruth regime_moderate() {
    GroundTruth gt;
    gt.rho = 0.95f;
    gt.sigma_z = 0.10f;
    gt.z_floor = 0.0f;
    gt.z_ceil = 3.0f;
    gt.mu_base = -2.0f;
    gt.mu_scale = 1.0f;
    gt.mu_rate = 1.0f;
    gt.sigma_base = 0.10f;
    gt.sigma_scale = 0.10f;
    gt.sigma_rate = 1.0f;
    gt.theta_base = 0.02f;
    gt.theta_scale = 0.08f;
    gt.theta_rate = 1.5f;
    return gt;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * OCSN SAMPLING
 * 
 * Sample from y | h using different methods.
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Observation sampling method
 */
enum class ObsMethod {
    OCSN_MIXTURE,   /**< OCSN 10-component mixture (matches filter likelihood) */
    DIRECT_CHI2     /**< Direct log(χ²(1)) = log(ε²) (true model) */
};

/**
 * @brief Sample observation y given log-volatility h using OCSN mixture
 * 
 * Uses OCSN 10-component mixture approximation to log(χ²(1)).
 * This MATCHES the filter's likelihood model exactly.
 * 
 * @param h    Current log-volatility
 * @param rng  Random number generator (std::mt19937)
 * @return Observation y ~ p_OCSN(y | h)
 */
inline float sample_ocsn(float h, std::mt19937& rng) {
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    
    /* Sample mixture component */
    float u = uniform(rng);
    float cumsum = 0.0f;
    int k = 0;
    for (k = 0; k < OCSN_NUM_COMPONENTS - 1; k++) {
        cumsum += OCSN_WEIGHTS[k];
        if (u < cumsum) break;
    }
    
    /* Sample from selected component */
    return h + OCSN_MEANS[k] + std::sqrt(OCSN_VARS[k]) * normal(rng);
}

/**
 * @brief Sample observation y given log-volatility h using direct log(χ²(1))
 * 
 * Uses the TRUE model: y = h + log(ε²) where ε ~ N(0,1).
 * This is the actual SV observation equation, but the filter uses
 * OCSN approximation, so there's slight model mismatch.
 * 
 * @param h    Current log-volatility
 * @param rng  Random number generator (std::mt19937)
 * @return Observation y = h + log(ε²)
 */
inline float sample_direct_chi2(float h, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    float eps = normal(rng);
    float chi2_1 = eps * eps;
    /* Add small constant to avoid log(0) */
    return h + std::log(chi2_1 + 1e-10f);
}

/**
 * @brief Sample observation using specified method
 */
inline float sample_observation(float h, std::mt19937& rng, ObsMethod method) {
    switch (method) {
        case ObsMethod::OCSN_MIXTURE:
            return sample_ocsn(h, rng);
        case ObsMethod::DIRECT_CHI2:
            return sample_direct_chi2(h, rng);
        default:
            return sample_ocsn(h, rng);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * Z-SPACE TRANSFORM
 * 
 * Transform between unconstrained z̃ ∈ ℝ and bounded z ∈ (z_floor, z_ceil).
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Transform z̃ (unconstrained) to z (bounded)
 * 
 * z = center * (1 + tanh(z̃)) where center = (z_ceil - z_floor) / 2
 */
inline float z_tilde_to_z(float z_tilde, float z_floor, float z_ceil) {
    float center = (z_ceil + z_floor) / 2.0f;
    float half_range = (z_ceil - z_floor) / 2.0f;
    return center + half_range * std::tanh(z_tilde);
}

/**
 * @brief Transform z (bounded) to z̃ (unconstrained)
 * 
 * z̃ = atanh((z - center) / half_range)
 */
inline float z_to_z_tilde(float z, float z_floor, float z_ceil) {
    float center = (z_ceil + z_floor) / 2.0f;
    float half_range = (z_ceil - z_floor) / 2.0f;
    float normalized = (z - center) / half_range;
    /* Clamp to avoid atanh(±1) = ±∞ */
    normalized = std::fmax(-0.999f, std::fmin(0.999f, normalized));
    return std::atanh(normalized);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA GENERATION
 * 
 * THE canonical implementation. Do not create alternatives.
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Generated data output
 */
struct GeneratedData {
    std::vector<float> y;      /**< Observations */
    std::vector<float> z;      /**< True regime values (bounded) */
    std::vector<float> z_tilde;/**< True regime values (unconstrained) */
    std::vector<float> h;      /**< True log-volatility values */
    int T;                     /**< Number of observations */
};

/**
 * @brief Generate synthetic regime-switching SV data
 * 
 * This is THE function to use for all synthetic data generation.
 * 
 * @param gt      Ground truth parameters
 * @param T       Number of observations
 * @param seed    RNG seed for reproducibility
 * @param method  Observation sampling method (default: DIRECT_CHI2 for true model)
 * @return GeneratedData struct with all time series
 * 
 * NOTE: Use DIRECT_CHI2 for testing against true model.
 *       Use OCSN_MIXTURE if you want data that exactly matches filter likelihood.
 */
inline GeneratedData generate_sv_data(
    const GroundTruth& gt, 
    int T, 
    uint32_t seed,
    ObsMethod method = ObsMethod::DIRECT_CHI2
) {
    GeneratedData data;
    data.T = T;
    data.y.resize(T);
    data.z.resize(T);
    data.z_tilde.resize(T);
    data.h.resize(T);
    
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    
    /* Stationary variance for z̃: Var[z̃] = σ_z² / (1 - ρ²) */
    float one_minus_rho_sq = std::fmax(1.0f - gt.rho * gt.rho, 1e-6f);
    float z_tilde_stat_std = gt.sigma_z / std::sqrt(one_minus_rho_sq);
    
    /* Initialize z̃ from stationary distribution */
    float z_tilde = z_tilde_stat_std * normal(rng);
    float z = z_tilde_to_z(z_tilde, gt.z_floor, gt.z_ceil);
    
    /* Compute initial z-dependent parameters */
    float theta_z = gt.theta_base + gt.theta_scale * (1.0f - std::exp(-gt.theta_rate * z));
    float mu_z = gt.mu_base + gt.mu_scale * (1.0f - std::exp(-gt.mu_rate * z));
    float sigma_h_z = gt.sigma_base + gt.sigma_scale * (1.0f - std::exp(-gt.sigma_rate * z));
    
    /* Initialize h from conditional stationary distribution */
    float phi = 1.0f - theta_z;
    float h_stat_var = sigma_h_z * sigma_h_z / std::fmax(1.0f - phi * phi, 1e-6f);
    float h = mu_z + std::sqrt(h_stat_var) * normal(rng);
    
    for (int t = 0; t < T; t++) {
        /* Store current state */
        data.z_tilde[t] = z_tilde;
        data.z[t] = z;
        data.h[t] = h;
        
        /* Generate observation using specified method */
        data.y[t] = sample_observation(h, rng, method);
        
        /* Propagate z̃ (AR(1) in unconstrained space) */
        z_tilde = gt.rho * z_tilde + gt.sigma_z * normal(rng);
        z = z_tilde_to_z(z_tilde, gt.z_floor, gt.z_ceil);
        
        /* Update z-dependent parameters */
        theta_z = gt.theta_base + gt.theta_scale * (1.0f - std::exp(-gt.theta_rate * z));
        mu_z = gt.mu_base + gt.mu_scale * (1.0f - std::exp(-gt.mu_rate * z));
        sigma_h_z = gt.sigma_base + gt.sigma_scale * (1.0f - std::exp(-gt.sigma_rate * z));
        
        /* Propagate h (AR(1) with regime-dependent mean) */
        h = (1.0f - theta_z) * h + theta_z * mu_z + sigma_h_z * normal(rng);
    }
    
    return data;
}

/**
 * @brief Generate regime-switching data with mid-sequence change
 * 
 * @param gt1         Parameters for first regime
 * @param T1          Duration of first regime
 * @param gt2         Parameters for second regime
 * @param T2          Duration of second regime
 * @param seed        RNG seed
 * @param switch_idx  Output: index where regime switches
 * @param method      Observation sampling method
 * @return Combined data from both regimes
 */
inline GeneratedData generate_regime_switch_data(
    const GroundTruth& gt1, int T1,
    const GroundTruth& gt2, int T2,
    uint32_t seed,
    int* switch_idx = nullptr,
    ObsMethod method = ObsMethod::DIRECT_CHI2
) {
    GeneratedData data1 = generate_sv_data(gt1, T1, seed, method);
    GeneratedData data2 = generate_sv_data(gt2, T2, seed + 10000, method);
    
    GeneratedData combined;
    combined.T = T1 + T2;
    combined.y.reserve(combined.T);
    combined.z.reserve(combined.T);
    combined.z_tilde.reserve(combined.T);
    combined.h.reserve(combined.T);
    
    /* Concatenate */
    combined.y.insert(combined.y.end(), data1.y.begin(), data1.y.end());
    combined.y.insert(combined.y.end(), data2.y.begin(), data2.y.end());
    
    combined.z.insert(combined.z.end(), data1.z.begin(), data1.z.end());
    combined.z.insert(combined.z.end(), data2.z.begin(), data2.z.end());
    
    combined.z_tilde.insert(combined.z_tilde.end(), data1.z_tilde.begin(), data1.z_tilde.end());
    combined.z_tilde.insert(combined.z_tilde.end(), data2.z_tilde.begin(), data2.z_tilde.end());
    
    combined.h.insert(combined.h.end(), data1.h.begin(), data1.h.end());
    combined.h.insert(combined.h.end(), data2.h.begin(), data2.h.end());
    
    if (switch_idx) *switch_idx = T1;
    
    return combined;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DATA STATISTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Compute summary statistics for generated data
 */
struct DataStats {
    float y_mean, y_std, y_min, y_max;
    float z_mean, z_std;
    float h_mean, h_std;
};

inline DataStats compute_stats(const GeneratedData& data) {
    DataStats stats = {0};
    int T = data.T;
    
    /* y statistics */
    stats.y_min = data.y[0];
    stats.y_max = data.y[0];
    for (int t = 0; t < T; t++) {
        stats.y_mean += data.y[t];
        stats.y_min = std::fmin(stats.y_min, data.y[t]);
        stats.y_max = std::fmax(stats.y_max, data.y[t]);
    }
    stats.y_mean /= T;
    for (int t = 0; t < T; t++) {
        float d = data.y[t] - stats.y_mean;
        stats.y_std += d * d;
    }
    stats.y_std = std::sqrt(stats.y_std / T);
    
    /* z statistics */
    for (int t = 0; t < T; t++) stats.z_mean += data.z[t];
    stats.z_mean /= T;
    for (int t = 0; t < T; t++) {
        float d = data.z[t] - stats.z_mean;
        stats.z_std += d * d;
    }
    stats.z_std = std::sqrt(stats.z_std / T);
    
    /* h statistics */
    for (int t = 0; t < T; t++) stats.h_mean += data.h[t];
    stats.h_mean /= T;
    for (int t = 0; t < T; t++) {
        float d = data.h[t] - stats.h_mean;
        stats.h_std += d * d;
    }
    stats.h_std = std::sqrt(stats.h_std / T);
    
    return stats;
}

inline void print_stats(const DataStats& stats) {
    printf("  y: mean=%.3f, std=%.3f, min=%.3f, max=%.3f\n",
           stats.y_mean, stats.y_std, stats.y_min, stats.y_max);
    printf("  z: mean=%.3f, std=%.3f\n", stats.z_mean, stats.z_std);
    printf("  h: mean=%.3f, std=%.3f\n", stats.h_mean, stats.h_std);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TEST UTILITIES
 *═══════════════════════════════════════════════════════════════════════════════*/

inline void print_ground_truth(const char* label, const GroundTruth& gt) {
    printf("%s:\n", label);
    printf("  rho=%.3f, sigma_z=%.3f\n", gt.rho, gt.sigma_z);
    printf("  mu: base=%.2f, scale=%.2f, rate=%.2f\n", gt.mu_base, gt.mu_scale, gt.mu_rate);
    printf("  sigma: base=%.2f, scale=%.2f, rate=%.2f\n", gt.sigma_base, gt.sigma_scale, gt.sigma_rate);
    printf("  theta: base=%.3f, scale=%.2f, rate=%.2f\n", gt.theta_base, gt.theta_scale, gt.theta_rate);
}

/**
 * @brief Extract θ array from GroundTruth (for comparison with estimates)
 * 
 * Order: [rho, sigma_z, mu_base, mu_scale, mu_rate, sigma_base, sigma_scale, sigma_rate]
 */
inline void gt_to_theta(const GroundTruth& gt, float theta[8]) {
    theta[0] = gt.rho;
    theta[1] = gt.sigma_z;
    theta[2] = gt.mu_base;
    theta[3] = gt.mu_scale;
    theta[4] = gt.mu_rate;
    theta[5] = gt.sigma_base;
    theta[6] = gt.sigma_scale;
    theta[7] = gt.sigma_rate;
}

/**
 * @brief Compute average relative error between estimate and ground truth
 */
inline float compute_avg_rel_error(const float* estimate, const GroundTruth& gt) {
    float theta_true[8];
    gt_to_theta(gt, theta_true);
    
    float err = 0.0f;
    for (int p = 0; p < 8; p++) {
        err += std::fabs(estimate[p] - theta_true[p]) / std::fabs(theta_true[p]);
    }
    return err / 8.0f;
}

/**
 * @brief Print parameter comparison table
 */
inline void print_param_comparison(const float* estimate, const float* std, const GroundTruth& gt) {
    const char* names[8] = {
        "rho", "sigma_z", "mu_base", "mu_scale", "mu_rate",
        "sigma_base", "sigma_scale", "sigma_rate"
    };
    float theta_true[8];
    gt_to_theta(gt, theta_true);
    
    printf("Parameter         True       Est       Std     Err%%  z-score  Status\n");
    printf("─────────────────────────────────────────────────────────────────────────\n");
    
    int ok_count = 0;
    for (int p = 0; p < 8; p++) {
        float err_pct = (estimate[p] - theta_true[p]) / theta_true[p] * 100.0f;
        float z_score = std[p] > 1e-10f ? std::fabs(estimate[p] - theta_true[p]) / std[p] : 999.0f;
        const char* status = (z_score < 2.0f) ? "[OK]" : (z_score < 3.0f) ? "[WARN]" : "[MISS]";
        if (z_score < 2.0f) ok_count++;
        
        printf("%-12s  %10.4f  %8.4f  %8.4f  %+6.1f%%  %7.2f  %s\n",
               names[p], theta_true[p], estimate[p], std[p], err_pct, z_score, status);
    }
    
    printf("─────────────────────────────────────────────────────────────────────────\n");
    printf("OVERALL: %d/8 within 2σ\n", ok_count);
}

} /* namespace smc2_test */

#endif /* SMC2_TEST_UTILS_CUH */
