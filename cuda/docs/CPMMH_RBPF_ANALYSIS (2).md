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

### 4.1 Bucket Sort by h After Resampling (Experiment 1) ❌ No improvement

**Hypothesis:** Resampling permutes particle slots randomly, breaking the noise→state mapping that CPMMH relies on. Sorting by h (or μ_h in RBPF) restores "slot identity."

**Result:** No improvement. Confirms the problem is **missing correlatable randomness**, not slot identity.

---

### 4.2 Mixture-Augmented RBPF (Path A) ❌ Reverted

**What we tried:** Sample mixture component explicitly and correlate that noise.

**Why it failed:** 
- Sampling introduces discrete "shot noise" into trajectories
- Makes likelihood function rougher, not smoother
- Actually increases variance instead of reducing it

**Supervisor's insight:** Marginalizing out a random variable is mathematically equivalent to perfectly coupling it. A deterministic function of correlated inputs (z_noise, u0_noise) is inherently correlated.

**Resolution:** Reverted to deterministic moment-matching OCSN update. Removed `d_s_noise` buffer entirely.

---

### 4.3 Current Status: Deterministic RBPF with CPMMH

The marginalized OCSN update is the CORRECT approach:
```cpp
/* MARGINALIZED OCSN update - deterministic moment-matching
 * - Optimal MMSE estimator for Gaussian approximation
 * - Deterministic function of correlated inputs is correlated
 * - Avoids discrete "shot noise" from mixture sampling
 */
ocsn_kalman_update(y_obs, mu_pred, var_pred, &mu_post, &var_post, &log_lik);
```

**Remaining issues to investigate:**
1. Data generation - extreme observations when ε² is very small
2. Prior/bounds tuning for variance parameters
3. CPMMH correlation strength (ρ = 0.99 may need adjustment)

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
