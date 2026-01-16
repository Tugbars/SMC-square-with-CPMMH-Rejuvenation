# SMC² Weight Accumulation Problem and Block SMC² Fix

## Update: Root Cause May Be Simpler

**IMPORTANT DISCOVERY**: The default `K_rejuv = 1` (only ONE MH step per rejuvenation) was likely the primary cause of high-T failure, not fundamental weight accumulation.

With K=1:
- After resample: 256 clones of ~10 distinct θ
- Rejuvenation: 1 MH step, ~30% accept → ~70% still exact clones
- Continue filtering: same lineages dominate
- Repeat death spiral

**First fix to try**: Increase `K_rejuv` to 5-10:
```cpp
smc2_cuda_set_rejuv_steps(state, 10);
```

If this fixes high-T, the "weight accumulation problem" was actually "insufficient diversification" all along.

If high-T still fails after K=10-20, then implement Block SMC² (weight reset) as described below.

---

## The Problem

SMC² maintains a population of θ-particles, each with a weight proportional to its cumulative likelihood:

```
log w_θ(T) = Σₜ₌₁ᵀ log p(yₜ | θ)
```

This multiplicative accumulation is the core problem.

### Why Weights Diverge

Even with well-specified models and good parameters, different θ values produce slightly different likelihoods at each timestep:

| T | Best θ log-likelihood | Worst θ log-likelihood | Spread |
|---|----------------------|------------------------|--------|
| 100 | -150 | -180 | 30 |
| 500 | -750 | -900 | 150 |
| 1000 | -1500 | -1800 | 300 |
| 2000 | -3000 | -3600 | 600 |

A spread of 600 log-points means:
```
w_best / w_worst = exp(600) ≈ 10²⁶⁰
```

**Result**: One particle has weight ≈ 1.0, the other 255 have weight ≈ 0.

### The Death Spiral

1. **T=500**: Weights spread ~150 log-points → ESS drops → Resample
2. **After resample**: 256 copies of ~10 distinct θ lineages  
3. **Rejuvenation**: 5 MH steps → slight jitter, still ~10 lineages
4. **T=1000**: Same lineages, weights diverge again → Resample
5. **After resample**: 256 copies of ~3 distinct θ lineages
6. **T=2000**: Effectively 1 lineage with 256 clones

Rejuvenation cannot fix this because:
- It starts from a **degenerate population** (clones)
- K=5 MH steps is not enough to recreate diversity
- Even K=100 steps can't turn clones into independent samples

### Contrast with Standalone CPMMH

Standalone CPMMH (single chain, 3000 iterations) doesn't have this problem:

| SMC² | CPMMH |
|------|-------|
| Population with weights | Single chain, no weights |
| Weights multiply over T | No accumulation |
| Bad θ gets resampled away | Bad θ gets rejected, chain stays |
| Population can collapse | Chain can't "collapse" - just slow mixing |
| Need to maintain diversity | Just need ergodicity |

CPMMH at T=2000 works fine. SMC² at T=2000 collapses. Same data, same model.

## The Fix: Block SMC² (Weight Reset)

### Core Insight

After rejuvenation:
- **θ-particle VALUES** encode the posterior (they were resampled + diversified)
- **θ-particle WEIGHTS** encode stale cumulative likelihood (harmful)

Solution: **Reset weights to uniform after rejuvenation.**

```cpp
// After resample + rejuvenate completes:
for (int i = 0; i < N_theta; i++) {
    d_outer_log_weights[i] = 0.0f;  // Uniform weights
}
```

### Why This Works

1. **Resample** selected θ values proportional to their likelihood (information preserved in particle values)
2. **Rejuvenation** diversified the clones via MCMC (restored population diversity)  
3. **Weight reset** prevents O(T) accumulation (fresh start for next block)

Each "block" between rejuvenations is an independent SMC run. Information flows through **particle values**, not **weight accumulation**.

### Theoretical Justification

This is equivalent to:
- **IBIS** (Iterated Batch Importance Sampling) with adaptive batch boundaries
- **SMC² with refresh** as described in Chopin & Papaspiliopoulos (2020)
- **Resample-move** SMC with full weight reset

The key insight: In particle MCMC methods, the **particle values after MCMC moves** are approximate posterior samples. Their relative weights before the moves are already incorporated via resampling. Continuing to accumulate likelihood on top of this double-counts evidence.

### Implementation

```cpp
void smc2_cuda_rejuvenate(SMC2StateCUDA* state) {
    // 1. Resample θ-particles by weight
    resample_outer_particles(state);
    
    // 2. Run K CPMMH steps per particle
    for (int k = 0; k < K_rejuv; k++) {
        cpmmh_step(state);
    }
    
    // 3. RESET WEIGHTS (the fix)
    reset_outer_weights_to_uniform<<<...>>>(state->d_outer_log_weights, N_theta);
    
    // 4. Reset cumulative log-likelihood accumulators
    cudaMemset(state->d_theta_log_likelihoods, 0, N_theta * sizeof(float));
}
```

### Expected Behavior

| Metric | Before Fix (T=2000) | After Fix (T=2000) |
|--------|--------------------|--------------------|
| Effective θ lineages | 1-3 | ~50-100 |
| Parameter recovery | Biased | Unbiased |
| Posterior variance | Underestimated | Correct |

## Alternative Approaches (Less Recommended)

### Continuous Tempering

Use `p(yₜ|θ)^γ` with γ < 1 to slow divergence:
```cpp
log_weight += gamma * log_likelihood;  // gamma = 0.1
```

**Problems**: 
- Introduces bias (not targeting true posterior)
- Requires tuning γ
- Still accumulates, just slower

### Aggressive Rejuvenation

Trigger at ESS < 0.9N, use K=100 MH steps.

**Problems**:
- Computational cost
- Still fighting the fundamental issue
- Clones don't become independent samples easily

### Surprise-Based Scaling

Scale proposal variance when likelihood drops.

**Problems**:
- Addresses adaptation speed, not weight accumulation
- Proposing from clones with bigger steps still gives clones

## Two Orthogonal Fixes

Adaptive proposals and blocking operate on **orthogonal axes**:

| | Adaptive Proposals | Blocking (Weight Reset) |
|---|---|---|
| **Fixes** | MCMC efficiency | Weight dynamics |
| **Axis** | Within population (local) | Across time (global) |
| **Problem** | "Rejuvenation doesn't mix well" | "Weights explode over T" |
| **Symptom** | Low acceptance, slow exploration | Population collapse to clones |
| **When it matters** | During rejuvenation | Between rejuvenations |

### What Adaptive Proposals Fix (Local MCMC Pathology)

- Vanishing CPMMH acceptance rates
- Proposal–posterior scale mismatch  
- Ineffective rejuvenation late in time
- Over-reliance on resampling

This is a **within-population** fix. It does not change the fundamental weight dynamics.

### What Blocking Fixes (Global Weight Pathology)

- Multiplicative likelihood explosion
- Irreversible early selection
- Winner-take-all dynamics
- Loss of global posterior mass

This is a **between-time** fix.

### Why You Need Both

**Without blocking**: Population is clones before rejuvenation even starts → adaptive proposals proposing from clones still gives clones

**Without adaptive proposals**: Population is diverse but rejuvenation is inefficient → 5 steps with bad proposals don't diversify enough

**Blocking is the prerequisite. Adaptive proposals are the optimization.**

Implementation order:
1. Implement blocking first
2. Verify it fixes high-T collapse
3. Add adaptive proposals if acceptance rates are still poor

## Summary

| Problem | Weight accumulation: log w ∝ T |
|---------|-------------------------------|
| Symptom | Population collapses to clones at high T |
| Root cause | Multiplicative likelihood over T steps |
| Fix | Reset weights to uniform after rejuvenation |
| Complexity | One line of code |
| Theory | Standard in IBIS / SMC² with refresh |

The fix transforms SMC² from "one long run with accumulating weights" to "sequence of fresh SMC blocks connected by particle values."