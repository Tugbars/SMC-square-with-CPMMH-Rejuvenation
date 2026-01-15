# CPMMH + RBPF: Analysis and Path Forward

## Executive Summary

Our GPU-accelerated CPMMH implementation with RBPF inner filter achieves 400ms runtime but fails to converge on variance parameters (σ_base, σ_scale, σ_rate). After analysis, we identified that **CPMMH's coupling mechanism is partially broken** when combined with RBPF due to the deterministic nature of Kalman updates. This document outlines the problem, failed approaches, and viable solutions.

---

## 1. The Problem

### Symptoms
- R-hat stuck at 2-4 for variance parameters after 2000+ iterations
- Acceptance rate ~25% (healthy range)
- Fast parameters (ρ, μ curves) converge normally
- Slow parameters (σ curves) show no mixing

### Working Reference
Bootstrap PF + CPMMH (CPU, 6 seconds):
```
Parameter Recovery:
  rho         0.9850 → 0.9604  (2.5% error)
  sigma_base  0.0700 → 0.0837  (19.5% error, but converged)
  sigma_scale 0.3500 → 0.3604  (3.0% error)
  sigma_rate  0.2500 → 0.2705  (8.2% error)
```

---

## 2. Root Cause Analysis

### CPMMH Coupling Requirements

CPMMH works by correlating random number streams between current and proposed parameter evaluations:

```
u_prop = ρ · u_curr + √(1-ρ²) · ξ_fresh
```

This correlation ensures that `log p̂(y|θ_prop, u_prop) - log p̂(y|θ_curr, u_curr)` varies smoothly with θ, enabling efficient MH exploration.

### What Bootstrap PF Correlates

In bootstrap PF, there are **two noise streams** that affect the likelihood:

| Stream | Purpose | Effect on Likelihood |
|--------|---------|---------------------|
| `z_noise[t,n]` | Propagate latent z | Direct (state dynamics) |
| `h_noise[t,n]` | Propagate latent h | **Direct** (observation model) |
| `u0[t]` | Systematic resampling | Indirect (ancestor selection) |

The h-noise is critical because variance parameters (σ_base, σ_scale, σ_rate) directly scale it:
```cpp
h_new = (1 - θ_z) * h_old + θ_z * μ_z + σ_h * h_noise  // ← σ_h depends on variance params
```

### What RBPF Removes

RBPF marginalizes h analytically via Kalman filtering:

```cpp
// Kalman predict (deterministic given z)
μ_pred = φ * μ_h + θ_z * μ_z
σ²_pred = φ² * σ²_h + σ_h²

// Kalman update (deterministic given y)
K = σ²_pred / (σ²_pred + σ²_obs)
μ_post = μ_pred + K * (y - μ_pred)
σ²_post = (1 - K) * σ²_pred
```

**There is no h-noise to correlate.** The variance parameters affect likelihood through the deterministic Kalman gain, not through sampled innovations.

### Why This Breaks Variance Parameter Learning

1. Variance parameters (σ_base, σ_scale, σ_rate) change the Kalman prediction variance
2. This changes the Kalman gain K, which changes μ_post
3. Small parameter changes can cause **discontinuous jumps** in likelihood when:
   - Resampling decisions change (ancestor selection)
   - The "best" mixture component changes in OCSN update
4. Without correlatable h-noise, these discontinuities cannot be smoothed

---

## 3. Failed/Incomplete Approaches

### 3.1 Separate u0 Stream (Implemented, +3% improvement)

**Change:** Separated resampling uniform from z-propagation noise
```cpp
// Before: u0 derived from z_noise
float u0 = fmodf(fabsf(z_noise[t, 0]), 1.0f);

// After: independent correlated stream  
float u0_prop = ρ * u0_curr + √(1-ρ²) * ξ;
u0_prop = u0_prop - floorf(u0_prop);  // wrap to [0,1)
```

**Result:** Marginal improvement. Confirms coupling issue exists but doesn't fix the core problem.

### 3.2 FP16 → FP32 (Not tested, unlikely to help)

Precision is not the bottleneck. The issue is structural (missing noise stream), not numerical.

### 3.3 More Particles (Tested, no improvement)

Increasing N_inner from 128 to 256 or 512 reduces likelihood variance but doesn't restore coupling.

---

## 4. Viable Solutions (Ordered by Complexity)

### 4.1 Bucket Sort by h After Resampling (Experiment 1) ✅ Implemented

**Hypothesis:** Resampling permutes particle slots randomly, breaking the noise→state mapping that CPMMH relies on. Sorting by h (or μ_h in RBPF) restores "slot identity."

**Implementation:**
```cpp
// After resampling, before propagation:
if ((t % SORT_EVERY_K) == 0) {
    // O(N) bucket sort by μ_h
    int bin = (μ_h - H_MIN) * BINS / (H_MAX - H_MIN);
    // Scatter to sorted positions
    // Particle i now holds the i-th quantile of h-distribution
}
```

**Expected outcome:** If this significantly improves variance parameter convergence, the problem is slot identity, not missing h-noise.

**Status:** Code complete, awaiting test.

---

### 4.2 Mixture-Augmented RBPF (Path A) — Recommended

**Concept:** The OCSN observation model is a 10-component mixture. Currently we approximate it with a single Gaussian. Instead, sample the mixture indicator explicitly:

```
s_t ∈ {1, ..., 10}  // Which mixture component generated y_t
y_t | h_t, s_t ~ N(h_t + m_s, v_s)  // Exact Gaussian given s
```

**Why this works:**
1. Sampling s_t introduces **new correlatable randomness**
2. Conditional on s_{1:T}, the model is exactly linear-Gaussian in h
3. Kalman update becomes exact (no approximation)
4. Mixture allocation is a major source of likelihood discontinuity — correlating it restores CPMMH's smoothing

**Implementation sketch:**
```cpp
// Store uniform stream for mixture selection
half* d_s_noise[2];  // [N_theta * N_inner * (T+1)]

// In PF step:
float s_noise_prop = ρ * s_noise_curr + scale * s_noise_fresh;
s_noise_prop = s_noise_prop - floorf(s_noise_prop);  // wrap to [0,1)

// Map to mixture component via posterior CDF
float posterior_weights[OCSN_K];
compute_ocsn_posterior(y, μ_pred, σ²_pred, posterior_weights);
int s = sample_categorical(posterior_weights, s_noise_prop);

// Exact Kalman update with selected component
float obs_mean = OCSN_MEANS[s];
float obs_var = OCSN_VARS[s];
kalman_update_exact(μ_pred, σ²_pred, y - obs_mean, obs_var, &μ_post, &σ²_post);
```

**Complexity:** Medium (2-4 hours). Requires:
- New noise buffer for s
- Modified OCSN update to sample then condition
- Same correlation/replay logic as z and u0

---

### 4.3 Jittered RBPF (Path B) — Simpler but Approximate

**Concept:** Add artificial correlated noise to h, then correct via importance weighting.

```cpp
// Instead of deterministic Kalman:
float h_jitter = correlated_noise * jitter_scale;
μ_post_jittered = μ_post + h_jitter;

// Importance weight correction
log_w += log_p(h_jitter | 0, jitter_scale);  // proposal
log_w -= log_p(h_jitter | 0, σ_h);           // target (approximate)
```

**Tradeoff:** Easier to implement than Path A, but introduces bias if not carefully tuned.

---

### 4.4 Abandon CPMMH, Use PMMH (Path C) — Fallback

If RBPF's variance reduction is sufficient, standard PMMH may work despite ~20% acceptance:

```cpp
// No correlation, just fresh noise each proposal
u_prop ~ N(0, I)
```

**Pros:** Simpler code, no coupling to debug
**Cons:** Lower acceptance, slower mixing, may need more iterations

---

## 5. Decision Matrix

| Approach | Effort | Expected Improvement | Risk |
|----------|--------|---------------------|------|
| Bucket sort (4.1) | 1 hour | Unknown (diagnostic) | Low |
| Mixture-augmented (4.2) | 4 hours | High (proper fix) | Medium |
| Jittered RBPF (4.3) | 2 hours | Medium | Medium (bias) |
| PMMH fallback (4.4) | 30 min | Low-Medium | Low |

**Recommended path:**
1. Test bucket sort first (quick diagnostic)
2. If insufficient, implement mixture-augmented RBPF (proper fix)
3. If timeline pressure, fall back to PMMH

---

## 6. Key Insights

### The Deeper Rule

> **CPMMH needs correlated randomness in the parts of the estimator that dominate variance and discontinuity.**

Bootstrap PF guarantees this because everything is sampled. RBPF removes sampling from h, so correlation must be moved to what remains random:
- Mixture indicators (Path A)
- Resampling uniforms (already done)
- Ancestor ordering via sorting (Experiment 1)

### Why "RBPF is Incompatible with CPMMH" Was Too Strong

RBPF doesn't have to be fully deterministic. The Kalman filter is deterministic **conditional on the observation model**. Our observation model (OCSN mixture) is not Gaussian — the moment we approximate it, we lose exactness.

By augmenting with mixture indicators (Path A), we restore:
1. Exactness (no Gaussian approximation)
2. Randomness (s_t is sampled)
3. Correlation (s_t stream can be coupled)

---

## 7. References

- Dahlin & Lindsten (2015): Particle Metropolis-Hastings with Correlated Noise
- Doucet et al. (2000): Rao-Blackwellised Particle Filtering
- Kim, Shephard & Chib (1998): OCSN mixture approximation for log-χ²

---

## Appendix: Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Forward kernel with sorting | `smc2_rbpf_cuda_optimized.cu` | 520-730 |
| Joint CPMMH with sorting | `smc2_rbpf_cuda_optimized.cu` | 939-1250 |
| Blocked CPMMH with sorting | `smc2_rbpf_cuda_optimized.cu` | 1351-1736 |
| Sort configuration | `smc2_rbpf_cuda_optimized.cu` | 91-107 |
| Working bootstrap reference | `cpmmh_gpu_learn_v3.cu` | (full file) |
