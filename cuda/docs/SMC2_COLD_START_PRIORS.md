# SMC² Cold Start: Prior Calibration for Unknown Market Data

**Status**: Design Complete  
**Date**: January 2026  
**Context**: Deploying Adaptive SMC² to real market data where true parameters are unknown

---

## The Problem

We tuned SMC² priors knowing the true synthetic parameters (e.g., σ_z = 0.15). In production, you don't know the true values. If priors are misspecified:

| Prior Issue | Consequence |
|-------------|-------------|
| Too tight, wrong center | Posterior stuck at prior mean, ignores data |
| Too tight, right center | Works, but fragile to regime changes |
| Too wide | Slow convergence, high variance, poor early estimates |
| Completely wrong region | Likelihood ≈ 0 for all particles → filter dies at t=1 |

**Example Failure**: If real market σ_z ≈ 0.05 but prior is N(0.15, 0.05), initial particles will have σ_z ∈ [0.10, 0.20]. All particles produce bad likelihoods → immediate degeneracy.

---

## Why "Adaptive Widening" Fails

A tempting solution: monitor posterior vs prior divergence, widen priors if posterior hits boundaries.

**This breaks SMC mathematics.**

```
Particle weights at time t:
    w_t ∝ p(y₁:t | θ) × π(θ) / q(θ)

If you change π(θ) → π'(θ) mid-filter:
    - Old particles were sampled assuming π(θ)
    - To make them valid for π'(θ), must reweight by π'(θ)/π(θ)
    - If π' is wider, tail particles get massive weight boost
    - → Instant degeneracy → Resample → Back to square one
```

**The Rule**: Priors are set ONCE, before filtering starts. They are FROZEN during the filter run.

---

## The Correct Solution: Empirical Bayes Warmup

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: WARMUP (Host-side, ~50-100 observations)          │
│                                                             │
│  • Buffer initial market data (do NOT run filter yet)       │
│  • Compute simple statistics from data                      │
│  • Map statistics → prior centers                           │
│  • Set WIDE priors around estimated centers                 │
│  • Upload priors to GPU, initialize particles               │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  PHASE 2: FILTERING (GPU, priors FROZEN)                    │
│                                                             │
│  • Run SMC² normally                                        │
│  • Monitor posterior-prior divergence (diagnostic only)     │
│  • If divergence persists: ALERT, do not auto-adjust        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Mapping Data Statistics to Priors

For a stochastic volatility model, use the **log-variance proxy**:

```cpp
// Given returns y_t, the latent log-volatility h_t relates to:
//     log(y_t²) ≈ h_t + log(ε_t²)
// 
// Statistics of log(y²) give hints about latent process parameters.

std::vector<float> log_sq(warmup_len);
for (int i = 0; i < warmup_len; i++) {
    log_sq[i] = logf(fmaxf(y[i] * y[i], 1e-10f));
}

// Mean of log(y²) → relates to μ_base (volatility level)
float h_mean = mean(log_sq);

// Std of log(y²) → relates to σ_z (vol-of-vol)
float h_std = std(log_sq);

// ACF(1) of log(y²) → relates to ρ (persistence)
float h_acf1 = autocorr_lag1(log_sq);
```

### Heuristic Mapping

| Statistic | Parameter | Mapping |
|-----------|-----------|---------|
| ACF₁(log y²) | ρ | `est_rho = clamp(acf1, 0.7, 0.98)` |
| Std(log y²) | σ_z | `est_sigma_z = clamp(h_std * 0.3, 0.05, 0.4)` |
| Mean(log y²) | μ_base | `est_mu_base = h_mean` |

These are rough estimates—the key is to get priors in the right **ballpark**, then let the filter refine.

### Setting Prior Widths

**Critical**: Even with good center estimates, keep priors **wide**:

```cpp
// Centered on empirical estimates, but WIDE variance
state->prior.rho_mean = est_rho;
state->prior.rho_std = 0.08f;       // Not 0.02 - allow flexibility

state->prior.sigma_z_mean = est_sigma_z;
state->prior.sigma_z_std = 0.15f;   // Not 0.05 - our problem child

state->prior.mu_base_mean = est_mu_base;
state->prior.mu_base_std = 1.5f;    // Very wide for level
```

**Rule of thumb**: Prior std should cover ±50% of plausible range at 2σ.

---

## Implementation

### `smc2_calibrate_priors()`

```cpp
void smc2_calibrate_priors(
    SMC2StateCUDA* state,
    const float* warmup_data,
    int warmup_len
) {
    // 1. Compute log-variance proxy
    std::vector<float> log_sq(warmup_len);
    for (int i = 0; i < warmup_len; i++) {
        log_sq[i] = logf(fmaxf(warmup_data[i] * warmup_data[i], 1e-10f));
    }
    
    // 2. Statistics
    float h_mean = 0;
    for (int i = 0; i < warmup_len; i++) h_mean += log_sq[i];
    h_mean /= warmup_len;
    
    float h_var = 0;
    for (int i = 0; i < warmup_len; i++) {
        h_var += (log_sq[i] - h_mean) * (log_sq[i] - h_mean);
    }
    h_var /= (warmup_len - 1);
    float h_std = sqrtf(h_var);
    
    // ACF(1)
    float acf_num = 0, acf_den = 0;
    for (int i = 0; i < warmup_len - 1; i++) {
        acf_num += (log_sq[i] - h_mean) * (log_sq[i+1] - h_mean);
        acf_den += (log_sq[i] - h_mean) * (log_sq[i] - h_mean);
    }
    float acf1 = acf_num / fmaxf(acf_den, 1e-10f);
    
    // 3. Map to parameters (conservative)
    float est_rho = fminf(fmaxf(acf1, 0.7f), 0.98f);
    float est_sigma_z = fminf(fmaxf(h_std * 0.3f, 0.05f), 0.4f);
    float est_mu_base = h_mean;
    
    // 4. Set priors: centered but WIDE
    state->prior.rho_mean = est_rho;
    state->prior.rho_std = 0.08f;
    
    state->prior.sigma_z_mean = est_sigma_z;
    state->prior.sigma_z_std = 0.15f;
    
    state->prior.mu_base_mean = est_mu_base;
    state->prior.mu_base_std = 1.5f;
    
    // Other parameters: reasonable defaults
    state->prior.mu_scale_mean = 0.5f;
    state->prior.mu_scale_std = 0.5f;
    state->prior.mu_rate_mean = 1.0f;
    state->prior.mu_rate_std = 1.0f;
    state->prior.sigma_base_mean = 0.15f;
    state->prior.sigma_base_std = 0.1f;
    state->prior.sigma_scale_mean = 0.1f;
    state->prior.sigma_scale_std = 0.1f;
    state->prior.sigma_rate_mean = 1.0f;
    state->prior.sigma_rate_std = 0.5f;
    
    printf("[Calibration] %d obs: log_var_mean=%.2f, log_var_std=%.2f, ACF1=%.3f\n",
           warmup_len, h_mean, h_std, acf1);
    printf("[Calibration] Priors: rho=%.3f±%.3f, sigma_z=%.3f±%.3f, mu_base=%.2f±%.2f\n",
           est_rho, 0.08f, est_sigma_z, 0.15f, est_mu_base, 1.5f);
}
```

### Deployment Workflow

```cpp
int main() {
    // 1. Allocate state
    SMC2StateCUDA* state = smc2_cuda_alloc(512, 256);
    state->ess_threshold_outer = 0.25f;
    smc2_cuda_set_fixed_lag(state, 100);
    
    // 2. Collect warmup data (DO NOT filter yet)
    std::vector<float> warmup_buffer;
    for (int i = 0; i < 100; i++) {
        float y = get_next_market_tick();
        warmup_buffer.push_back(y);
    }
    
    // 3. Calibrate priors from warmup
    smc2_calibrate_priors(state, warmup_buffer.data(), warmup_buffer.size());
    
    // 4. Initialize particles from calibrated priors
    smc2_cuda_init_from_prior(state);
    
    // 5. Optionally process warmup as first observations
    for (float y : warmup_buffer) {
        smc2_cuda_update(state, y);
    }
    
    // 6. Continue with live data (priors now FROZEN)
    while (market_open()) {
        float y = get_next_market_tick();
        smc2_cuda_update(state, y);
        
        // Monitor but don't adjust
        SMC2Diagnostic diag = smc2_get_diagnostic(state);
        if (diag.any_railing) {
            log_warning("Parameter hitting prior boundary");
        }
    }
}
```

---

## Diagnostics: Stability ≠ Accuracy

**Critical distinction**: A stable filter can converge confidently to WRONG parameters.

| What diagnostics measure | What they DON'T measure |
|--------------------------|------------------------|
| ✓ Stability - posterior stopped moving | ✗ Accuracy - posterior is correct |
| ✓ Degeneracy - ESS health | ✗ Model specification |
| ✓ Inner filter quality | ✗ Identifiability |

**To verify accuracy**: Run simulation studies with known ground truth.

### Convergence Diagnostics (v2)

Even with good calibration, monitor filter health:

```cpp
typedef struct {
    float z_scores[8];    // (posterior_mean - prior_mean) / prior_std
    bool railing[8];      // |z_score| > 2.0
    bool any_railing;     // Any parameter railing
} SMC2Diagnostic;

SMC2Diagnostic smc2_get_diagnostic(SMC2StateCUDA* state) {
    SMC2Diagnostic diag = {0};
    
    float post_mean[8], post_std[8];
    smc2_cuda_get_theta_mean(state, post_mean);
    smc2_cuda_get_theta_std(state, post_std);
    
    float prior_means[8] = { /* ... from state->prior ... */ };
    float prior_stds[8] = { /* ... from state->prior ... */ };
    
    for (int i = 0; i < 8; i++) {
        diag.z_scores[i] = (post_mean[i] - prior_means[i]) / prior_stds[i];
        diag.railing[i] = fabsf(diag.z_scores[i]) > 2.0f;
        diag.any_railing |= diag.railing[i];
    }
    
    return diag;
}
```

### What To Do When Railing

| Duration | Action |
|----------|--------|
| Transient (< 10 updates) | Normal - posterior is learning |
| Persistent (> 50 updates) | Alert - model may need manual restart |
| Extreme (z > 4) | Critical - priors badly misspecified |

**Do NOT auto-widen priors.** If railing persists:

1. Log the incident
2. Alert human operator
3. Consider: restart with fresh calibration from recent data window

---

## Summary

| Strategy | Verdict | Notes |
|----------|---------|-------|
| Empirical Bayes Warmup | ✅ **Use this** | Calibrate priors ONCE before filtering |
| Wide priors after calibration | ✅ **Essential** | Don't trust point estimates |
| Z-score monitoring | ✅ **Do this** | Diagnose problems early |
| Adaptive widening mid-filter | ❌ **Never** | Breaks importance sampling math |
| Manual restart on persistent railing | ✅ **Last resort** | Better than corrupted estimates |

### Key Equations

**Why adaptive widening fails:**
```
w_t ∝ p(y|θ) × π_old(θ) / q(θ)

After changing prior to π_new:
w_t^corrected = w_t × π_new(θ) / π_old(θ)

If π_new wider → tail particles get huge weights → degeneracy
```

**Log-variance proxy:**
```
y_t = exp(h_t/2) × ε_t,  ε_t ~ N(0,1)
log(y_t²) = h_t + log(ε_t²)
E[log(ε²)] ≈ -1.27  (constant offset)

Therefore:
mean(log y²) ≈ E[h_t] → relates to μ_base
std(log y²)  ≈ std(h_t) → relates to σ_z  
ACF(log y²)  ≈ ACF(h_t) → relates to ρ
```

---

## Files

- `smc2_rbpf_cuda.cu` - Add `smc2_calibrate_priors()` function
- `smc2_rbpf_cuda.cuh` - Add `SMC2Diagnostic` struct

## References

- Chopin & Papaspiliopoulos (2020) - *An Introduction to Sequential Monte Carlo*, Ch. 17
- Doucet & Johansen (2009) - *A Tutorial on Particle Filtering*, Section on initialization