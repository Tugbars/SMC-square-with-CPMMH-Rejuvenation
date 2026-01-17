/*
 * smc2_convergence_v2.cuh
 * 
 * SMC² Convergence & Health Diagnostics (v2)
 * 
 * WHAT THIS MEASURES:
 *   ✓ Stability    - Has the posterior stopped moving?
 *   ✓ Degeneracy   - Is ESS healthy?
 *   ✓ Inner health - Are likelihood estimates reliable?
 *   ✗ Accuracy     - Is the posterior CORRECT? (Cannot measure without ground truth)
 *   ✗ Identifiability - Are parameters distinguishable? (Requires simulation study)
 * 
 * A stable filter can converge confidently to WRONG parameters.
 * Use simulation studies to validate accuracy.
 * 
 * CHANGES FROM V1:
 *   - Fixed ESS CV calculation (was double-dividing)
 *   - Fixed param drift normalization (added floor for late-stage stability)
 *   - Replaced log-lik CV with absolute std threshold (CV meaningless for SV)
 *   - Added inner ESS tracking (critical for likelihood estimator quality)
 *   - Renamed "converged" → "stable" for honesty
 *   - Added "healthy" (inner filter OK) and "ready" (stable && healthy)
 * 
 * COST: ~0 extra GPU work. All metrics piggyback on existing computations.
 * 
 * Usage:
 *   SMC2ConvergenceTracker tracker;
 *   smc2_conv_init(&tracker, N_theta, N_inner);
 *   smc2_conv_set_priors(&tracker, prior_mean, prior_std);
 *   
 *   // Main loop
 *   smc2_cuda_update(state, y_t);
 *   smc2_conv_update(&tracker, ess, ll_inc, theta_mean, theta_var, inner_ess_array);
 *   
 *   if (t % 100 == 0) {
 *       SMC2ConvergenceDiag diag = smc2_conv_check(&tracker);
 *       smc2_conv_print(&diag);
 *       if (diag.ready) { feed_params_to_rbpf(); }
 *   }
 */

#ifndef SMC2_CONVERGENCE_V2_CUH
#define SMC2_CONVERGENCE_V2_CUH

#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration
 * ============================================================================ */

#define SMC2_CONV_HISTORY_LEN     64    /* Circular buffer size */
#define SMC2_CONV_WINDOW          32    /* Fixed window for checks (don't grow) */
#define SMC2_CONV_NUM_PARAMS      8     /* Number of θ parameters */

/* Stability thresholds */
#define SMC2_CONV_ESS_CV_THRESH         0.30f   /* ESS coefficient of variation */
#define SMC2_CONV_DRIFT_THRESH          0.05f   /* Normalized parameter drift (RMS) */
#define SMC2_CONV_VAR_RATIO_LO          0.70f   /* Variance ratio bounds */
#define SMC2_CONV_VAR_RATIO_HI          1.30f
#define SMC2_CONV_STABLE_STREAK         3       /* Consecutive stable checks needed */

/* Log-likelihood thresholds (absolute, not CV) */
#define SMC2_CONV_LL_STD_THRESH         10.0f   /* If std > this, something is wrong */
#define SMC2_CONV_LL_EXPLODE_THRESH     50.0f   /* Catastrophic instability */

/* Inner filter health thresholds */
#define SMC2_CONV_INNER_ESS_BAD_FRAC    0.10f   /* Fraction with ESS < threshold */
#define SMC2_CONV_INNER_ESS_MIN_RATIO   0.10f   /* ESS < 0.1*N_inner is "bad" */

/* Prior divergence */
#define SMC2_CONV_PRIOR_ZSCORE_WARN     2.0f    /* |z| > 2 = railing */
#define SMC2_CONV_PRIOR_ZSCORE_CRIT     3.0f    /* |z| > 3 = critical */

/* ============================================================================
 * Data Structures
 * ============================================================================ */

/*
 * Diagnostic result
 */
typedef struct {
    /* === Outer filter stability === */
    
    /* ESS health */
    float ess_mean;                 /* Mean ESS over window */
    float ess_std;                  /* Std of ESS */
    float ess_cv;                   /* Coefficient of variation (std/mean) */
    int   ess_stable;               /* CV < threshold */
    
    /* Parameter drift */
    float param_drift;              /* RMS normalized drift across params */
    float param_drift_per_dim[SMC2_CONV_NUM_PARAMS];
    int   params_stable;            /* drift < threshold */
    
    /* Variance stability */
    float var_ratio[SMC2_CONV_NUM_PARAMS];  /* var_now / var_prev */
    int   var_stable;               /* All ratios in bounds */
    
    /* Log-likelihood health (absolute std, NOT CV) */
    float ll_mean;                  /* Mean incremental log-lik */
    float ll_std;                   /* Std of incremental log-lik */
    int   ll_healthy;               /* std < threshold */
    int   ll_exploding;             /* std > explosion threshold */
    
    /* === Inner filter health (CRITICAL for accuracy) === */
    
    float inner_ess_mean;           /* Mean inner ESS across θ-particles */
    float inner_ess_min;            /* Minimum inner ESS */
    float inner_ess_q10;            /* 10th percentile */
    float inner_ess_q50;            /* Median */
    float inner_ess_bad_frac;       /* Fraction below threshold */
    int   inner_healthy;            /* bad_frac < limit */
    
    /* === Prior divergence === */
    
    float prior_zscore[SMC2_CONV_NUM_PARAMS];
    int   num_railing;              /* |z| > 2 */
    int   num_critical;             /* |z| > 3 */
    
    /* === Overall status === */
    
    int   observations;             /* Total observations processed */
    int   stable_streak;            /* Consecutive stable checks */
    
    int   outer_stable;             /* ESS + params + var all stable */
    int   inner_healthy_flag;       /* Inner ESS OK */
    int   stable;                   /* Outer filter stabilized (was "converged") */
    int   healthy;                  /* Inner filters OK */
    int   ready;                    /* stable && healthy - OK to use params */
    
} SMC2ConvergenceDiag;

/*
 * Tracker state
 */
typedef struct {
    /* Configuration */
    int N_theta;                    /* Number of outer particles */
    int N_inner;                    /* Number of inner particles per θ */
    float inner_ess_threshold;      /* ESS below this is "bad" */
    
    /* Circular buffers for time series */
    float ess_history[SMC2_CONV_HISTORY_LEN];
    float ll_history[SMC2_CONV_HISTORY_LEN];
    float theta_mean_history[SMC2_CONV_HISTORY_LEN][SMC2_CONV_NUM_PARAMS];
    float theta_var_history[SMC2_CONV_HISTORY_LEN][SMC2_CONV_NUM_PARAMS];
    
    /* Inner ESS tracking (most recent only - no history needed) */
    float* inner_ess_snapshot;      /* [N_theta] - allocated dynamically */
    int    inner_ess_valid;         /* Has been updated */
    
    /* Buffer state */
    int head;                       /* Next write position */
    int count;                      /* Valid entries (up to HISTORY_LEN) */
    int total_updates;              /* Total observations seen */
    
    /* Convergence state */
    int stable_streak;
    int is_stable;                  /* Latched stable flag */
    int stable_at;                  /* When stability detected */
    
    /* Priors (for z-score) */
    float prior_mean[SMC2_CONV_NUM_PARAMS];
    float prior_std[SMC2_CONV_NUM_PARAMS];
    int   priors_set;
    
} SMC2ConvergenceTracker;

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/* Compute mean and std of array segment in circular buffer */
static inline void smc2_conv_stats(
    const float* buf,
    int head,
    int window,
    int buf_size,
    float* out_mean,
    float* out_std
) {
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < window; i++) {
        int idx = (head - 1 - i + buf_size) % buf_size;
        double v = buf[idx];
        sum += v;
        sum_sq += v * v;
    }
    double mean = sum / window;
    double var = (sum_sq / window) - (mean * mean);
    *out_mean = (float)mean;
    *out_std = sqrtf(fmaxf((float)var, 0.0f));
}

/* Quickselect for percentile (modifies array) */
static inline float smc2_conv_percentile(float* arr, int n, float p) {
    if (n <= 0) return 0.0f;
    int k = (int)(p * (n - 1));
    k = (k < 0) ? 0 : (k >= n ? n - 1 : k);
    std::nth_element(arr, arr + k, arr + n);
    return arr[k];
}

/* ============================================================================
 * API Functions
 * ============================================================================ */

/*
 * Initialize tracker
 */
static inline void smc2_conv_init(
    SMC2ConvergenceTracker* t,
    int N_theta,
    int N_inner
) {
    memset(t, 0, sizeof(SMC2ConvergenceTracker));
    t->N_theta = N_theta;
    t->N_inner = N_inner;
    t->inner_ess_threshold = SMC2_CONV_INNER_ESS_MIN_RATIO * N_inner;
    t->inner_ess_snapshot = (float*)malloc(N_theta * sizeof(float));
    t->inner_ess_valid = 0;
    t->stable_at = -1;
}

/*
 * Free tracker resources
 */
static inline void smc2_conv_free(SMC2ConvergenceTracker* t) {
    if (t->inner_ess_snapshot) {
        free(t->inner_ess_snapshot);
        t->inner_ess_snapshot = NULL;
    }
}

/*
 * Set prior reference (for z-score computation)
 */
static inline void smc2_conv_set_priors(
    SMC2ConvergenceTracker* t,
    const float* prior_mean,
    const float* prior_std
) {
    memcpy(t->prior_mean, prior_mean, SMC2_CONV_NUM_PARAMS * sizeof(float));
    memcpy(t->prior_std, prior_std, SMC2_CONV_NUM_PARAMS * sizeof(float));
    t->priors_set = 1;
}

/*
 * Update tracker with latest values
 * 
 * Call after each smc2_cuda_update().
 * 
 * Parameters:
 *   t            - Tracker
 *   ess          - Outer ESS (from resampling decision)
 *   ll_inc       - Log p(y_t | y_{1:t-1}) - incremental log-likelihood
 *   theta_mean   - θ posterior mean [8] (reuse from adaptive proposals)
 *   theta_var    - θ posterior variance [8] (diagonal of cov)
 *   inner_ess    - Inner ESS per θ-particle [N_theta], or NULL to skip
 */
static inline void smc2_conv_update(
    SMC2ConvergenceTracker* t,
    float ess,
    float ll_inc,
    const float* theta_mean,
    const float* theta_var,
    const float* inner_ess      /* Can be NULL */
) {
    int idx = t->head;
    
    /* Store scalar metrics */
    t->ess_history[idx] = ess;
    t->ll_history[idx] = ll_inc;
    
    /* Store θ moments */
    if (theta_mean && theta_var) {
        memcpy(t->theta_mean_history[idx], theta_mean, SMC2_CONV_NUM_PARAMS * sizeof(float));
        memcpy(t->theta_var_history[idx], theta_var, SMC2_CONV_NUM_PARAMS * sizeof(float));
    }
    
    /* Store inner ESS snapshot (most recent only) */
    if (inner_ess && t->inner_ess_snapshot) {
        memcpy(t->inner_ess_snapshot, inner_ess, t->N_theta * sizeof(float));
        t->inner_ess_valid = 1;
    }
    
    /* Advance buffer */
    t->head = (t->head + 1) % SMC2_CONV_HISTORY_LEN;
    if (t->count < SMC2_CONV_HISTORY_LEN) t->count++;
    t->total_updates++;
}

/*
 * Check convergence/health
 * 
 * Call periodically (e.g., every 50-100 updates)
 */
static inline SMC2ConvergenceDiag smc2_conv_check(SMC2ConvergenceTracker* t) {
    SMC2ConvergenceDiag d;
    memset(&d, 0, sizeof(d));
    
    d.observations = t->total_updates;
    
    /* Not enough history */
    if (t->count < SMC2_CONV_WINDOW) {
        return d;
    }
    
    int window = SMC2_CONV_WINDOW;  /* Fixed window, don't grow */
    
    /* ================================================================
     * 1. ESS Stability
     * ================================================================ */
    smc2_conv_stats(t->ess_history, t->head, window, SMC2_CONV_HISTORY_LEN,
                    &d.ess_mean, &d.ess_std);
    
    /* CV = std / mean (FIXED: was computing wrong before) */
    d.ess_cv = d.ess_std / fmaxf(d.ess_mean, 1e-6f);
    d.ess_stable = (d.ess_cv < SMC2_CONV_ESS_CV_THRESH) && (d.ess_mean > 1.0f);
    
    /* ================================================================
     * 2. Log-Likelihood Health (absolute std, NOT CV)
     * ================================================================ */
    smc2_conv_stats(t->ll_history, t->head, window, SMC2_CONV_HISTORY_LEN,
                    &d.ll_mean, &d.ll_std);
    
    d.ll_healthy = (d.ll_std < SMC2_CONV_LL_STD_THRESH);
    d.ll_exploding = (d.ll_std > SMC2_CONV_LL_EXPLODE_THRESH);
    
    /* ================================================================
     * 3. Parameter Drift (FIXED normalization)
     * ================================================================ */
    int oldest_idx = (t->head - window + SMC2_CONV_HISTORY_LEN) % SMC2_CONV_HISTORY_LEN;
    int newest_idx = (t->head - 1 + SMC2_CONV_HISTORY_LEN) % SMC2_CONV_HISTORY_LEN;
    
    double drift_sq_sum = 0.0;
    d.params_stable = 1;
    d.var_stable = 1;
    
    for (int p = 0; p < SMC2_CONV_NUM_PARAMS; p++) {
        float mean_old = t->theta_mean_history[oldest_idx][p];
        float mean_new = t->theta_mean_history[newest_idx][p];
        float var_old = t->theta_var_history[oldest_idx][p];
        float var_new = t->theta_var_history[newest_idx][p];
        float std_new = sqrtf(fmaxf(var_new, 1e-10f));
        
        /* 
         * FIXED: Drift normalization with floor
         * 
         * Problem: As t→∞, std_new→0, making delta→∞ from noise
         * Solution: Floor at 1% of |mean| or prior_std
         */
        float floor_val = fmaxf(fabsf(mean_new) * 0.01f, 1e-6f);
        if (t->priors_set) {
            floor_val = fmaxf(floor_val, t->prior_std[p] * 0.01f);
        }
        float denom = fmaxf(std_new, floor_val);
        
        float delta = (mean_new - mean_old) / denom;
        d.param_drift_per_dim[p] = fabsf(delta);
        drift_sq_sum += delta * delta;
        
        /* Variance ratio */
        d.var_ratio[p] = var_new / fmaxf(var_old, 1e-10f);
        if (d.var_ratio[p] < SMC2_CONV_VAR_RATIO_LO || 
            d.var_ratio[p] > SMC2_CONV_VAR_RATIO_HI) {
            d.var_stable = 0;
        }
        
        /* Prior z-score */
        if (t->priors_set) {
            d.prior_zscore[p] = (mean_new - t->prior_mean[p]) / 
                                fmaxf(t->prior_std[p], 1e-6f);
            if (fabsf(d.prior_zscore[p]) > SMC2_CONV_PRIOR_ZSCORE_WARN) {
                d.num_railing++;
            }
            if (fabsf(d.prior_zscore[p]) > SMC2_CONV_PRIOR_ZSCORE_CRIT) {
                d.num_critical++;
            }
        }
    }
    
    d.param_drift = sqrtf((float)(drift_sq_sum / SMC2_CONV_NUM_PARAMS));
    d.params_stable = (d.param_drift < SMC2_CONV_DRIFT_THRESH) && d.var_stable;
    
    /* ================================================================
     * 4. Inner Filter Health (CRITICAL for likelihood quality)
     * ================================================================ */
    if (t->inner_ess_valid && t->inner_ess_snapshot) {
        /* Make a copy for percentile computation (modifies array) */
        float* ess_copy = (float*)alloca(t->N_theta * sizeof(float));
        memcpy(ess_copy, t->inner_ess_snapshot, t->N_theta * sizeof(float));
        
        /* Compute statistics */
        double sum = 0.0;
        float min_val = ess_copy[0];
        int bad_count = 0;
        
        for (int i = 0; i < t->N_theta; i++) {
            sum += ess_copy[i];
            if (ess_copy[i] < min_val) min_val = ess_copy[i];
            if (ess_copy[i] < t->inner_ess_threshold) bad_count++;
        }
        
        d.inner_ess_mean = (float)(sum / t->N_theta);
        d.inner_ess_min = min_val;
        d.inner_ess_q10 = smc2_conv_percentile(ess_copy, t->N_theta, 0.10f);
        d.inner_ess_q50 = smc2_conv_percentile(ess_copy, t->N_theta, 0.50f);
        d.inner_ess_bad_frac = (float)bad_count / t->N_theta;
        
        d.inner_healthy = (d.inner_ess_bad_frac < SMC2_CONV_INNER_ESS_BAD_FRAC);
    } else {
        /* No inner ESS data - assume healthy (can't check) */
        d.inner_healthy = 1;
        d.inner_ess_mean = (float)t->N_inner;  /* Assume full */
    }
    
    /* ================================================================
     * 5. Overall Status
     * ================================================================ */
    d.outer_stable = d.ess_stable && d.params_stable && !d.ll_exploding;
    d.inner_healthy_flag = d.inner_healthy;
    
    /* Update stable streak */
    if (d.outer_stable) {
        t->stable_streak++;
    } else {
        t->stable_streak = 0;
    }
    d.stable_streak = t->stable_streak;
    
    /* Stable = outer filter stabilized (but may not be accurate!) */
    d.stable = (t->stable_streak >= SMC2_CONV_STABLE_STREAK);
    if (d.stable && !t->is_stable) {
        t->is_stable = 1;
        t->stable_at = t->total_updates;
    }
    
    /* Healthy = inner filters OK */
    d.healthy = d.inner_healthy;
    
    /* Ready = stable AND healthy - OK to use parameters */
    d.ready = d.stable && d.healthy;
    
    return d;
}

/*
 * Print diagnostic summary
 */
static inline void smc2_conv_print(const SMC2ConvergenceDiag* d) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  SMC² Diagnostic (t=%d)                                     \n", d->observations);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    
    /* Outer filter */
    printf("║  OUTER FILTER                                                 \n");
    printf("║    ESS:    mean=%.1f  std=%.1f  CV=%.3f  %s\n",
           d->ess_mean, d->ess_std, d->ess_cv,
           d->ess_stable ? "[OK]" : "[UNSTABLE]");
    printf("║    Drift:  %.4f  %s\n",
           d->param_drift,
           d->params_stable ? "[OK]" : "[MOVING]");
    printf("║    LogLik: mean=%.2f  std=%.2f  %s%s\n",
           d->ll_mean, d->ll_std,
           d->ll_healthy ? "[OK]" : "[HIGH VAR]",
           d->ll_exploding ? " [EXPLODING!]" : "");
    
    /* Inner filter */
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  INNER FILTER (Likelihood Quality)                           \n");
    printf("║    ESS:    mean=%.1f  q10=%.1f  q50=%.1f  min=%.1f\n",
           d->inner_ess_mean, d->inner_ess_q10, d->inner_ess_q50, d->inner_ess_min);
    printf("║    Bad:    %.1f%%  %s\n",
           d->inner_ess_bad_frac * 100.0f,
           d->inner_healthy ? "[OK]" : "[DEGRADED]");
    
    /* Prior divergence */
    if (d->num_railing > 0 || d->num_critical > 0) {
        printf("╠══════════════════════════════════════════════════════════════╣\n");
        printf("║  WARNING: %d params railing (|z|>2), %d critical (|z|>3)     \n",
               d->num_railing, d->num_critical);
    }
    
    /* Status */
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  STATUS                                                       \n");
    printf("║    Stable streak: %d/%d\n", d->stable_streak, SMC2_CONV_STABLE_STREAK);
    printf("║    Outer stable:  %s\n", d->stable ? "YES" : "no");
    printf("║    Inner healthy: %s\n", d->healthy ? "YES" : "no");
    printf("║    ────────────────────────────────\n");
    printf("║    READY TO USE:  %s\n", 
           d->ready ? "*** YES ***" : "NOT YET");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
}

/*
 * Print compact one-liner status
 */
static inline void smc2_conv_print_compact(const SMC2ConvergenceDiag* d) {
    printf("[t=%5d] ESS=%.0f(cv=%.2f) drift=%.4f inner=%.0f(%.0f%% bad) | %s%s%s\n",
           d->observations,
           d->ess_mean, d->ess_cv,
           d->param_drift,
           d->inner_ess_mean, d->inner_ess_bad_frac * 100.0f,
           d->stable ? "STABLE " : "",
           d->healthy ? "HEALTHY " : "",
           d->ready ? "-> READY" : "");
}

/*
 * Quick status checks
 */
static inline int smc2_conv_is_stable(const SMC2ConvergenceTracker* t) {
    return t->is_stable;
}

static inline int smc2_conv_is_ready(const SMC2ConvergenceTracker* t, const SMC2ConvergenceDiag* d) {
    return d->ready;
}

static inline int smc2_conv_stable_at(const SMC2ConvergenceTracker* t) {
    return t->stable_at;
}

/*
 * Reset (e.g., after regime change detected)
 */
static inline void smc2_conv_reset(SMC2ConvergenceTracker* t) {
    t->head = 0;
    t->count = 0;
    t->stable_streak = 0;
    t->is_stable = 0;
    t->stable_at = -1;
    t->inner_ess_valid = 0;
    /* Keep total_updates and priors */
}

#ifdef __cplusplus
}
#endif

#endif /* SMC2_CONVERGENCE_V2_CUH */
