/*
 * smc2_prior_calibration.cuh
 * 
 * Prior calibration for SMC² cold start problem.
 * 
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  THE PROBLEM                                                            │
 * │                                                                         │
 * │  SMC² needs priors BEFORE seeing data, but bad priors cause failure.   │
 * │  We want to center near the truth (efficient) while covering regime    │
 * │  uncertainty (robust).                                                  │
 * └─────────────────────────────────────────────────────────────────────────┘
 * 
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  THE SOLUTION                                                           │
 * │                                                                         │
 * │                         PRIOR                                           │
 * │                                                                         │
 * │      Center (where)     ←───── Warmup data (current market)            │
 * │      Width (how wide)   ←───── Historical bounds (past regimes)        │
 * │                                                                         │
 * └─────────────────────────────────────────────────────────────────────────┘
 * 
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  THE MAPPING                                                            │
 * │                                                                         │
 * │  We observe returns y_t, not latent volatility h_t. But:               │
 * │                                                                         │
 * │      log(y_t²) = h_t + log(ε_t²)                                       │
 * │                        └─────── known noise (mean=-1.27, var=π²/2)     │
 * │                                                                         │
 * │  So statistics of log(y²) reveal latent parameters:                    │
 * │                                                                         │
 * │      ACF₁(log y²)   ──────►  ρ (persistence)                           │
 * │      Var(log y²)    ──────►  σ_z (vol-of-vol)                          │
 * │      Mean(log y²)   ──────►  μ_base (level)                            │
 * │                                                                         │
 * └─────────────────────────────────────────────────────────────────────────┘
 * 
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  WHY IT WORKS                                                           │
 * │                                                                         │
 * │  Scenario                      │ Result                                │
 * │  ─────────────────────────────────────────────────────────────────     │
 * │  Warmup=calm, stays calm       │ Center near truth, fast convergence   │
 * │  Warmup=calm, crisis hits      │ Wide prior covers crisis → adapts     │
 * │  Warmup=crisis, calms down     │ Wide prior covers calm → adapts       │
 * │                                                                         │
 * │  Width spans historical extremes, so filter always has particles       │
 * │  in the right region. May converge slower if center is off, but        │
 * │  won't fail catastrophically.                                          │
 * └─────────────────────────────────────────────────────────────────────────┘
 * 
 * Usage:
 *   // 1. Buffer warmup data
 *   float warmup[100];
 *   for (int i = 0; i < 100; i++) warmup[i] = get_return();
 *   
 *   // 2. Calibrate priors
 *   SMC2Prior prior;
 *   smc2_calibrate_priors(warmup, 100, &SPY_BOUNDS, &prior);
 *   
 *   // 3. Apply to state and start filtering
 *   smc2_cuda_set_prior(state, &prior);
 *   smc2_cuda_init_from_prior(state);
 */

#ifndef SMC2_PRIOR_CALIBRATION_CUH
#define SMC2_PRIOR_CALIBRATION_CUH

#include <cmath>
#include <cstdio>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Historical Regime Bounds
 * 
 * Computed from historical data covering calm and crisis periods.
 * Used to set prior widths that cover plausible parameter ranges.
 * ============================================================================ */

typedef struct {
    /* Persistence parameter rho */
    float rho_min;          /* Crisis: faster mean reversion */
    float rho_max;          /* Calm: high persistence */
    float rho_calm;         /* Typical calm value */
    float rho_crisis;       /* Typical crisis value */
    
    /* Vol-of-vol parameter sigma_z */
    float sigma_z_min;      /* Calm: low vol-of-vol */
    float sigma_z_max;      /* Crisis: high vol-of-vol */
    float sigma_z_calm;     /* Typical calm value */
    float sigma_z_crisis;   /* Typical crisis value */
    
    /* Log-variance level mu_base */
    float mu_base_min;
    float mu_base_max;
} HistoricalRegimeBounds;

/*
 * SPY bounds - generated from yfinance data (2007-2024)
 * Covers: 2008 crisis, 2011 eurozone, 2015 china, 2018 Q4, 
 *         2020 COVID, 2022 hiking, plus calm periods
 */
static const HistoricalRegimeBounds SPY_BOUNDS = {
    /* rho */
    .rho_min = 0.665f,
    .rho_max = 0.990f,
    .rho_calm = 0.990f,
    .rho_crisis = 0.842f,
    
    /* sigma_z */
    .sigma_z_min = 0.031f,
    .sigma_z_max = 0.650f,
    .sigma_z_calm = 0.091f,
    .sigma_z_crisis = 0.336f,
    
    /* mu_base */
    .mu_base_min = -11.88f,
    .mu_base_max = -5.80f,
};

/* Default wide bounds if no historical data available */
static const HistoricalRegimeBounds DEFAULT_BOUNDS = {
    .rho_min = 0.70f,
    .rho_max = 0.99f,
    .rho_calm = 0.95f,
    .rho_crisis = 0.85f,
    
    .sigma_z_min = 0.02f,
    .sigma_z_max = 0.50f,
    .sigma_z_calm = 0.08f,
    .sigma_z_crisis = 0.25f,
    
    .mu_base_min = -12.0f,
    .mu_base_max = -6.0f,
};

/* ============================================================================
 * Prior Structure
 * 
 * Matches the prior fields in SMC2StateCUDA.
 * ============================================================================ */

typedef struct {
    /* Core SV parameters */
    float rho_mean, rho_std;
    float sigma_z_mean, sigma_z_std;
    float mu_base_mean, mu_base_std;
    
    /* Extended model parameters (if used) */
    float mu_scale_mean, mu_scale_std;
    float mu_rate_mean, mu_rate_std;
    float sigma_base_mean, sigma_base_std;
    float sigma_scale_mean, sigma_scale_std;
    float sigma_rate_mean, sigma_rate_std;
} SMC2Prior;

/* ============================================================================
 * Warmup Statistics
 * 
 * Intermediate results from warmup analysis.
 * Useful for debugging/logging.
 * ============================================================================ */

typedef struct {
    int n_obs;                  /* Number of observations used */
    
    /* Raw statistics from log(y²) */
    float mean_log_y2;          /* Mean of log(y²) */
    float std_log_y2;           /* Std of log(y²) */
    float var_log_y2;           /* Variance of log(y²) */
    float acf1_log_y2;          /* ACF at lag 1 */
    
    /* Derived estimates */
    float var_h_est;            /* Estimated Var(h) after noise removal */
    float realized_vol;         /* Realized volatility (not annualized) */
    
    /* Parameter estimates (prior centers) */
    float est_rho;
    float est_sigma_z;
    float est_mu_base;
} WarmupStats;

/* ============================================================================
 * Constants
 * ============================================================================ */

/* Variance of log(χ²(1)) ≈ π²/2 */
#define LOG_CHI2_VARIANCE 4.9348f

/* Mean of log(χ²(1)) ≈ -1.27 */
#define LOG_CHI2_MEAN -1.2704f

/* Minimum warmup observations */
#define MIN_WARMUP_OBS 30

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static inline float smc2_clamp(float x, float lo, float hi) {
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

/* ============================================================================
 * Warmup Statistics Computation
 * ============================================================================ */

/*
 * Compute statistics from warmup returns.
 * 
 * For SV model: y_t = exp(h_t/2) × ε_t
 * Therefore:    log(y²) = h_t + log(ε²)
 * 
 * The noise term log(ε²) has known distribution (log-chi-squared).
 * We can partially back out properties of h_t.
 */
static inline int smc2_compute_warmup_stats(
    const float* returns,
    int n,
    WarmupStats* stats
) {
    if (n < MIN_WARMUP_OBS) {
        fprintf(stderr, "[Calibration] Error: need at least %d observations, got %d\n",
                MIN_WARMUP_OBS, n);
        return -1;
    }
    
    memset(stats, 0, sizeof(WarmupStats));
    stats->n_obs = n;
    
    /* Compute log(y²) for non-zero returns */
    float* log_y2 = (float*)alloca(n * sizeof(float));
    int valid = 0;
    
    for (int i = 0; i < n; i++) {
        if (returns[i] != 0.0f) {
            log_y2[valid++] = logf(returns[i] * returns[i]);
        }
    }
    
    if (valid < MIN_WARMUP_OBS) {
        fprintf(stderr, "[Calibration] Error: too many zero returns\n");
        return -1;
    }
    
    /* Mean of log(y²) */
    double sum = 0.0;
    for (int i = 0; i < valid; i++) {
        sum += log_y2[i];
    }
    stats->mean_log_y2 = (float)(sum / valid);
    
    /* Variance of log(y²) */
    double sum_sq = 0.0;
    for (int i = 0; i < valid; i++) {
        float diff = log_y2[i] - stats->mean_log_y2;
        sum_sq += diff * diff;
    }
    stats->var_log_y2 = (float)(sum_sq / (valid - 1));
    stats->std_log_y2 = sqrtf(stats->var_log_y2);
    
    /* ACF(1) of log(y²) */
    double acf_num = 0.0, acf_den = 0.0;
    for (int i = 0; i < valid - 1; i++) {
        float diff_i = log_y2[i] - stats->mean_log_y2;
        float diff_i1 = log_y2[i + 1] - stats->mean_log_y2;
        acf_num += diff_i * diff_i1;
        acf_den += diff_i * diff_i;
    }
    stats->acf1_log_y2 = (acf_den > 0) ? (float)(acf_num / acf_den) : 0.0f;
    
    /* Realized volatility (not annualized) */
    double ret_sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        ret_sum_sq += returns[i] * returns[i];
    }
    stats->realized_vol = sqrtf((float)(ret_sum_sq / n));
    
    /* ========================================================================
     * Map to SV parameters
     * 
     * Theory:
     *   Var(log y²) = Var(h) + π²/2
     *   Var(h) = σ_z² / (1 - ρ²)
     *   ACF_log_y2(1) ≈ ρ × Var(h) / Var(log y²)
     * ======================================================================== */
    
    /* Estimate Var(h) by subtracting noise variance */
    stats->var_h_est = fmaxf(stats->var_log_y2 - LOG_CHI2_VARIANCE, 0.1f);
    
    /* Estimate ρ from ACF - but ACF is attenuated by noise */
    /* ACF_observed ≈ ρ × Var(h) / (Var(h) + noise_var) */
    /* So: ρ ≈ ACF × (1 + noise_var / Var(h)) */
    float attenuation = (stats->var_h_est + LOG_CHI2_VARIANCE) / stats->var_h_est;
    float rho_raw = stats->acf1_log_y2 * attenuation;
    
    /* ACF from daily data is often low/negative due to noise */
    /* Use a baseline + adjustment approach */
    float rho_baseline = 0.95f;  /* Typical SV persistence */
    
    /* If ACF suggests lower persistence, adjust down */
    if (rho_raw < 0.5f) {
        stats->est_rho = rho_baseline - 0.05f * (0.5f - rho_raw);
    } else if (rho_raw > 1.0f) {
        stats->est_rho = rho_baseline + 0.02f;
    } else {
        stats->est_rho = rho_baseline + 0.5f * (rho_raw - 0.5f) * 0.1f;
    }
    
    /* Adjust based on realized vol (high vol → likely lower persistence) */
    float rv_ann = stats->realized_vol * sqrtf(252.0f);
    if (rv_ann > 0.40f) {
        stats->est_rho -= 0.10f;
    } else if (rv_ann > 0.25f) {
        stats->est_rho -= 0.05f;
    }
    
    stats->est_rho = smc2_clamp(stats->est_rho, 0.70f, 0.99f);
    
    /* Estimate σ_z from Var(h) = σ_z² / (1 - ρ²) */
    float one_minus_rho_sq = 1.0f - stats->est_rho * stats->est_rho;
    stats->est_sigma_z = sqrtf(stats->var_h_est * one_minus_rho_sq);
    stats->est_sigma_z = smc2_clamp(stats->est_sigma_z, 0.02f, 0.50f);
    
    /* Adjust σ_z for high realized vol */
    if (rv_ann > 0.40f) {
        stats->est_sigma_z = fmaxf(stats->est_sigma_z, 0.20f);
    } else if (rv_ann > 0.25f) {
        stats->est_sigma_z = fmaxf(stats->est_sigma_z, 0.12f);
    }
    
    /* Estimate μ_base: mean(log y²) - E[log ε²] */
    stats->est_mu_base = stats->mean_log_y2 - LOG_CHI2_MEAN;
    
    return 0;
}

/* ============================================================================
 * Main Calibration Function
 * ============================================================================ */

/*
 * Calibrate priors from warmup data and historical bounds.
 * 
 * Strategy:
 *   - Prior CENTER = warmup estimate (where we think we are)
 *   - Prior STD = (historical_max - historical_min) / 4 (covers regime range at 2σ)
 * 
 * Parameters:
 *   returns     - Warmup return data (percentage returns, not log returns)
 *   n           - Number of warmup observations
 *   bounds      - Historical regime bounds (use SPY_BOUNDS or DEFAULT_BOUNDS)
 *   out_prior   - Output prior structure
 * 
 * Returns:
 *   0 on success, -1 on error
 */
static inline int smc2_calibrate_priors(
    const float* returns,
    int n,
    const HistoricalRegimeBounds* bounds,
    SMC2Prior* out_prior
) {
    /* Use default bounds if not provided */
    if (bounds == NULL) {
        bounds = &DEFAULT_BOUNDS;
    }
    
    /* Compute warmup statistics */
    WarmupStats stats;
    if (smc2_compute_warmup_stats(returns, n, &stats) != 0) {
        return -1;
    }
    
    /* =======================================================================
     * Set prior centers from warmup estimates
     * ======================================================================= */
    
    out_prior->rho_mean = stats.est_rho;
    out_prior->sigma_z_mean = stats.est_sigma_z;
    out_prior->mu_base_mean = stats.est_mu_base;
    
    /* =======================================================================
     * Set prior widths from historical bounds
     * Width = (max - min) / 4 → covers range at ~2σ
     * ======================================================================= */
    
    out_prior->rho_std = (bounds->rho_max - bounds->rho_min) / 4.0f;
    out_prior->sigma_z_std = (bounds->sigma_z_max - bounds->sigma_z_min) / 4.0f;
    out_prior->mu_base_std = (bounds->mu_base_max - bounds->mu_base_min) / 4.0f;
    
    /* Ensure minimum width */
    out_prior->rho_std = fmaxf(out_prior->rho_std, 0.05f);
    out_prior->sigma_z_std = fmaxf(out_prior->sigma_z_std, 0.10f);
    out_prior->mu_base_std = fmaxf(out_prior->mu_base_std, 1.0f);
    
    /* =======================================================================
     * Extended parameters: use reasonable defaults
     * These are less critical - filter will adapt
     * ======================================================================= */
    
    out_prior->mu_scale_mean = 0.5f;
    out_prior->mu_scale_std = 0.5f;
    
    out_prior->mu_rate_mean = 1.0f;
    out_prior->mu_rate_std = 1.0f;
    
    out_prior->sigma_base_mean = 0.15f;
    out_prior->sigma_base_std = 0.10f;
    
    out_prior->sigma_scale_mean = 0.10f;
    out_prior->sigma_scale_std = 0.10f;
    
    out_prior->sigma_rate_mean = 1.0f;
    out_prior->sigma_rate_std = 0.5f;
    
    return 0;
}

/* ============================================================================
 * Logging / Debug
 * ============================================================================ */

static inline void smc2_print_warmup_stats(const WarmupStats* stats) {
    printf("\n┌─────────────────────────────────────────────────────┐\n");
    printf("│  Warmup Statistics (n=%d)                          \n", stats->n_obs);
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│  log(y²):  mean=%.2f  std=%.2f  ACF1=%.3f           \n",
           stats->mean_log_y2, stats->std_log_y2, stats->acf1_log_y2);
    printf("│  Var(h) estimate: %.3f                              \n", stats->var_h_est);
    printf("│  Realized vol (ann): %.1f%%                         \n", 
           stats->realized_vol * sqrtf(252.0f) * 100.0f);
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│  Parameter estimates:                               \n");
    printf("│    ρ      = %.3f                                    \n", stats->est_rho);
    printf("│    σ_z    = %.3f                                    \n", stats->est_sigma_z);
    printf("│    μ_base = %.2f                                    \n", stats->est_mu_base);
    printf("└─────────────────────────────────────────────────────┘\n\n");
}

static inline void smc2_print_prior(const SMC2Prior* prior) {
    printf("\n┌─────────────────────────────────────────────────────┐\n");
    printf("│  Calibrated Prior                                   \n");
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│  ρ      ~ N(%.3f, %.3f)                             \n", 
           prior->rho_mean, prior->rho_std);
    printf("│  σ_z    ~ N(%.3f, %.3f)                             \n",
           prior->sigma_z_mean, prior->sigma_z_std);
    printf("│  μ_base ~ N(%.2f, %.2f)                             \n",
           prior->mu_base_mean, prior->mu_base_std);
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│  Extended parameters:                               \n");
    printf("│    μ_scale  ~ N(%.2f, %.2f)                         \n",
           prior->mu_scale_mean, prior->mu_scale_std);
    printf("│    σ_base   ~ N(%.2f, %.2f)                         \n",
           prior->sigma_base_mean, prior->sigma_base_std);
    printf("└─────────────────────────────────────────────────────┘\n\n");
}

/* ============================================================================
 * Convenience: Calibrate and Print
 * ============================================================================ */

static inline int smc2_calibrate_priors_verbose(
    const float* returns,
    int n,
    const HistoricalRegimeBounds* bounds,
    SMC2Prior* out_prior
) {
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  SMC² Prior Calibration\n");
    printf("══════════════════════════════════════════════════════\n");
    
    /* Compute stats */
    WarmupStats stats;
    if (smc2_compute_warmup_stats(returns, n, &stats) != 0) {
        printf("  ERROR: Failed to compute warmup statistics\n");
        return -1;
    }
    
    smc2_print_warmup_stats(&stats);
    
    /* Calibrate */
    if (smc2_calibrate_priors(returns, n, bounds, out_prior) != 0) {
        printf("  ERROR: Failed to calibrate priors\n");
        return -1;
    }
    
    smc2_print_prior(out_prior);
    
    printf("══════════════════════════════════════════════════════\n\n");
    
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* SMC2_PRIOR_CALIBRATION_CUH */
