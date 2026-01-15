# Prior Selection for Production Systems

When running SMC² in production, you don't know the true parameters. The priors must be wide enough to cover any plausible market regime, yet informative enough to enable convergence in reasonable time.

---

## The Prior Truncation Problem

If your prior bounds exclude the true parameter values, the filter **cannot** find them:

```
Prior: mu_base ∈ [-2, 0]
True:  mu_base = -4.2

Posterior estimate: mu_base ≈ -2.0  (stuck at boundary!)

     Prior density
         ▲
         │    ┌────────────┐
         │    │            │
         │    │  ALLOWED   │
         │    │   REGION   │
         │    │            │
         └────┴────────────┴──────────────────▶ mu_base
             -2            0
              ▲
              │
         Posterior piles up here
         (can't go lower!)
                                    TRUE VALUE
                                        ↓
         ─────────────────────────────(−4.2)───
```

**Symptoms of prior truncation:**
- Posterior mean very close to prior boundary
- Posterior std suspiciously small (artificially constrained)
- Z-scores explode when you know ground truth
- NLL remains elevated even after convergence

---

## Two Philosophies

### Philosophy 1: Wide Priors (Robust)

Use very wide, weakly informative priors that cover any plausible market regime:

```cpp
// "I don't know much, but I know physics"
// Vol can't be negative, rho must be < 1, etc.

PriorBounds wide_priors = {
    .rho_min = 0.80f,      .rho_max = 0.999f,
    .sigma_z_min = 0.01f,  .sigma_z_max = 0.30f,
    
    .mu_base_min = -10.0f, .mu_base_max = 0.0f,   // 0.7% to 100% vol
    .mu_scale_min = 0.0f,  .mu_scale_max = 6.0f,  // up to ~400× vol increase
    .mu_rate_min = 0.05f,  .mu_rate_max = 5.0f,
    
    .sigma_base_min = 0.01f,  .sigma_base_max = 0.50f,
    .sigma_scale_min = 0.0f,  .sigma_scale_max = 1.0f,
    .sigma_rate_min = 0.05f,  .sigma_rate_max = 5.0f,
};
```

| Pros | Cons |
|------|------|
| Robust to unknown regimes | Slower convergence |
| Won't miss true values | Needs more data (T > 3000) |
| Simple — no adaptation logic | Needs more particles |
| Safe default choice | May explore implausible regions |

**When to use:** Production systems, unknown market regimes, when robustness matters more than speed.

### Philosophy 2: Adaptive Priors

Start with reasonable priors based on domain knowledge, but detect and respond to boundary hitting:

```
┌─────────────────────────────────────────────────────────────────┐
│  ADAPTIVE PRIOR STRATEGY                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. START with "typical market" priors                          │
│     mu_base ∈ [-4, -1]    // 10-60% annual vol                 │
│     mu_scale ∈ [0, 2]     // up to 3× vol in stress            │
│                                                                 │
│  2. RUN SMC² and monitor posterior                              │
│                                                                 │
│  3. DETECT boundary hitting:                                    │
│     posterior_mean - 2σ < prior_min  →  hitting lower bound    │
│     posterior_mean + 2σ > prior_max  →  hitting upper bound    │
│                                                                 │
│  4. RESPOND:                                                    │
│     Option A: Expand bounds and re-run                          │
│     Option B: Flag result as unreliable                         │
│     Option C: Use drift to gradually migrate                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| Pros | Cons |
|------|------|
| Faster convergence initially | Complex logic |
| Domain knowledge helps | May need re-runs |
| Tighter posteriors | Risk of missing regimes |

**When to use:** Research, known stable regimes, when you have good domain priors.

---

## Recommended Prior Bounds

Based on empirical analysis of equity markets across multiple regimes (2008 crisis, 2020 COVID, normal periods):

### Log-Volatility Scale

| μ_base | exp(μ/2) | Interpretation |
|--------|----------|----------------|
| -6.0 | 5% | Very calm (bonds, low-vol stocks) |
| -4.0 | 14% | Normal equity vol |
| -2.0 | 37% | Elevated (earnings, mild stress) |
| -1.0 | 61% | High stress |
| 0.0 | 100% | Crisis levels |

### Production Priors

```cpp
// Recommended bounds for equity markets
PriorBounds production_priors = {
    // Regime persistence
    .rho_min = 0.85f,       // min ~7 tick half-life
    .rho_max = 0.995f,      // max ~140 tick half-life
    
    // Regime innovation
    .sigma_z_min = 0.02f,   // very slow regime changes
    .sigma_z_max = 0.25f,   // fast regime switching
    
    // Volatility mean curve: μ(z) = μ_base + μ_scale * (1 - exp(-μ_rate * z))
    .mu_base_min = -8.0f,   // ~2% vol floor (calm)
    .mu_base_max = -1.0f,   // ~60% vol floor (already stressed)
    .mu_scale_min = 0.0f,   // no stress effect
    .mu_scale_max = 5.0f,   // up to ~12× vol increase
    .mu_rate_min = 0.1f,    // slow saturation
    .mu_rate_max = 3.0f,    // fast saturation
    
    // Vol-of-vol curve: σ_h(z) = σ_base + σ_scale * (1 - exp(-σ_rate * z))
    .sigma_base_min = 0.02f,
    .sigma_base_max = 0.30f,
    .sigma_scale_min = 0.0f,
    .sigma_scale_max = 0.60f,
    .sigma_rate_min = 0.1f,
    .sigma_rate_max = 3.0f,
};
```

---

## Boundary Detection

Implement runtime checks to detect when posteriors are truncated:

```cpp
typedef struct {
    bool is_truncated;
    int param_idx;
    bool at_lower;
    float posterior_mean;
    float posterior_std;
    float boundary;
} TruncationInfo;

TruncationInfo check_prior_truncation(
    const float* theta_mean,
    const float* theta_std,
    const PriorBounds* bounds,
    float margin  // e.g., 0.05 for 5%
) {
    TruncationInfo info = {0};
    
    const float* mins = (const float*)bounds;  // Assume interleaved min/max
    const float* maxs = mins + 1;
    
    for (int p = 0; p < N_PARAMS; p++) {
        float mean = theta_mean[p];
        float std = theta_std[p];
        float lo = mins[p * 2];
        float hi = maxs[p * 2];
        float range = hi - lo;
        
        // Check lower bound
        if (mean - 2*std < lo + margin * range) {
            info.is_truncated = true;
            info.param_idx = p;
            info.at_lower = true;
            info.posterior_mean = mean;
            info.posterior_std = std;
            info.boundary = lo;
            return info;  // Return first truncation found
        }
        
        // Check upper bound
        if (mean + 2*std > hi - margin * range) {
            info.is_truncated = true;
            info.param_idx = p;
            info.at_lower = false;
            info.posterior_mean = mean;
            info.posterior_std = std;
            info.boundary = hi;
            return info;
        }
    }
    
    return info;  // is_truncated = false
}
```

### Response Strategies

When truncation is detected:

**Strategy A: Expand and Re-run**
```cpp
if (info.is_truncated) {
    if (info.at_lower) {
        bounds->min[info.param_idx] *= 1.5f;  // Expand by 50%
    } else {
        bounds->max[info.param_idx] *= 1.5f;
    }
    smc2_reset(state);
    // Re-run from beginning with new bounds
}
```

**Strategy B: Flag and Continue**
```cpp
if (info.is_truncated) {
    result.confidence = LOW;
    result.warning = "Posterior truncated at prior boundary";
    // Continue but downstream systems should reduce position sizes
}
```

**Strategy C: Rely on Drift (Streaming Mode)**
```cpp
// In streaming mode with drift Q, particles will gradually
// migrate even if starting near boundaries.
// No explicit re-run needed, but convergence is slower.

if (info.is_truncated) {
    // Increase drift for truncated parameter
    Q[info.param_idx] *= 2.0f;
}
```

---

## Prior Initialization

How to set initial θ-particles:

### Option 1: Sample from Prior (Uninformative)

```cpp
for (int j = 0; j < N_theta; j++) {
    for (int p = 0; p < N_PARAMS; p++) {
        theta[j][p] = uniform(prior_min[p], prior_max[p]);
    }
}
```

**Pros:** Unbiased, explores full space
**Cons:** Many particles start in implausible regions, slow initial convergence

### Option 2: Concentrated Initialization

```cpp
// Start near prior midpoint with small jitter
for (int j = 0; j < N_theta; j++) {
    for (int p = 0; p < N_PARAMS; p++) {
        float mid = (prior_min[p] + prior_max[p]) / 2;
        float range = prior_max[p] - prior_min[p];
        theta[j][p] = mid + 0.1f * range * randn();
    }
}
```

**Pros:** Fast initial convergence if prior midpoint is reasonable
**Cons:** May miss true value if prior is misspecified

### Option 3: Warm Start from Previous Estimate

```cpp
// Use HCRBPF's current operating parameters as starting point
float* prev_theta = hcrbpf_get_params();
for (int j = 0; j < N_theta; j++) {
    for (int p = 0; p < N_PARAMS; p++) {
        theta[j][p] = prev_theta[p] + jitter_std[p] * randn();
    }
}
```

**Pros:** Best for recalibration — starts near known-good region
**Cons:** May be trapped if previous params were wrong

### Recommendation

For a recalibration system:

```cpp
if (first_run || nll_excess > crisis_threshold) {
    // Full exploration mode — sample from wide prior
    init_from_prior(state, wide_priors);
} else {
    // Refinement mode — warm start from current params
    init_from_current(state, hcrbpf_params, small_jitter);
}
```

---

## Domain-Specific Prior Elicitation

Questions to ask when setting priors:

### Volatility Level (μ_base, μ_scale)

1. What's the lowest realistic annualized vol for this asset? (→ μ_base_min)
2. What's the highest vol observed historically? (→ μ_base + μ_scale max)
3. How much can vol increase during stress? (→ μ_scale_max)

**Example for SPY:**
- Lowest: ~8% (2017 calm) → μ_base_min ≈ -5.0
- Highest: ~80% (March 2020) → μ_base + μ_scale ≈ -0.4
- Stress multiplier: ~10× → μ_scale_max ≈ 4.0

### Regime Dynamics (ρ, σ_z)

1. How long do vol regimes typically last? (→ ρ)
2. How quickly can regimes switch? (→ σ_z)

**Example:**
- VIX mean-reversion half-life: ~20-40 days → ρ ≈ 0.95-0.98
- Flash crash regime switch: <1 day → σ_z up to 0.2

### Vol-of-Vol (σ_base, σ_scale)

1. How uncertain is vol estimation in calm markets? (→ σ_base)
2. How much more uncertain during stress? (→ σ_scale)

---

## Summary: Decision Tree

```
START
  │
  ▼
Is this a known, stable regime?
  │
  ├─ YES → Use informative priors + warm start
  │        Prior width: ±2σ around expected values
  │        Init: from previous estimate
  │
  └─ NO (unknown regime or recalibration after drift)
       │
       ▼
     Use wide priors + full exploration
       │
       ├─ Prior width: cover all plausible values
       │
       ├─ Init: sample from prior (not concentrated)
       │
       └─ Enable boundary detection
            │
            ▼
          Truncation detected?
            │
            ├─ NO → Continue normally
            │
            └─ YES → Expand bounds or increase drift Q
```

---

## References

- Gelman, A. (2006). "Prior distributions for variance parameters in hierarchical models"
- Stan Development Team. "Prior Choice Recommendations"
- Murphy, K. (2012). "Machine Learning: A Probabilistic Perspective" — Ch. 5 on Bayesian inference
