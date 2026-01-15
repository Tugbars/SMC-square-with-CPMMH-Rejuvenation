# SMC² with CPMMH Rejuvenation: Design Document

## 1. Introduction

### 1.1 The Problem

We want to perform **online Bayesian parameter estimation** for a regime-switching stochastic volatility model. As financial data streams in tick-by-tick, we need to:

1. **Track the hidden volatility state** h_t (filtering)
2. **Learn the model parameters** θ (inference)
3. **Do it fast enough for HFT applications** (<1ms per observation)

This is hard because:
- The parameter space is 8-dimensional
- The likelihood p(y₁:T | θ) is intractable (no closed form)
- Standard MCMC is too slow for online use
- Standard particle filters don't learn parameters

### 1.2 Our Solution

We present a **hybrid algorithm** that combines:

| Component | Source | What It Provides |
|-----------|--------|------------------|
| SMC² | Chopin et al. (2013) | Massive parallelism (N_θ × N_inner particles) |
| PMMH | Andrieu et al. (2010) | Correct Bayesian parameter updates |
| CPMMH | Deligiannidis et al. (2018) | Low-variance likelihood estimates |

The key insight: **Use SMC² for parallel forward filtering, but inject CPMMH-style rejuvenation when particle diversity drops.** This gives us the parallelism of SMC² with the mixing quality of CPMMH.

---

## 2. The Stochastic Volatility Model

### 2.1 Model Structure

We model log-volatility h_t with a **regime-switching** structure. A latent regime variable z_t ∈ (0, 3) controls the volatility dynamics:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      GENERATIVE PROCESS                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Regime Layer (slow-moving):                                       │
│   ┌─────────┐      ρ·z̃ + σ_z·ε      ┌─────────┐                     │
│   │  z̃_t-1  │ ───────────────────▶ │   z̃_t   │    z̃ ∈ ℝ           │
│   └─────────┘                       └────┬────┘                     │
│                                          │                          │
│                                          ▼ z = 1.5·(1 + tanh(z̃))   │
│                                     ┌─────────┐                     │
│                                     │   z_t   │    z ∈ (0, 3)       │
│                                     └────┬────┘                     │
│                                          │                          │
│              ┌───────────────────────────┼───────────────────────┐  │
│              ▼                           ▼                       ▼  │
│         ┌─────────┐                 ┌─────────┐             ┌──────┐│
│         │  θ(z)   │                 │  μ(z)   │             │σ_h(z)││
│         └────┬────┘                 └────┬────┘             └──┬───┘│
│              │ mean-reversion            │ long-run mean       │    │
│              │ speed                     │                     │    │
│              └───────────────┬───────────┴─────────────────────┘    │
│                              ▼                                      │
│   Volatility Layer:     h_t = (1-θ)·h_{t-1} + θ·μ + σ_h·ε          │
│   ┌─────────┐                                   ┌─────────┐         │
│   │  h_t-1  │ ─────────────────────────────────▶│   h_t   │         │
│   └─────────┘                                   └────┬────┘         │
│                                                      │              │
│   Observation Layer:                                 ▼              │
│                                                 ┌─────────┐         │
│                    y_t = exp(h_t/2) · ε_t       │   y_t   │         │
│                                                 └─────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Parameter Table

We estimate **8 parameters** that control the regime-dependent dynamics:

| Parameter | Symbol | Range | Physical Meaning |
|-----------|--------|-------|------------------|
| **Regime persistence** | ρ | (0.8, 0.999) | How slowly the regime changes. ρ=0.95 means half-life ≈ 14 steps |
| **Regime volatility** | σ_z | (0.01, 1.0) | How much the regime fluctuates |
| **Base mean** | μ_base | (-10, 5) | Log-volatility in low regime. μ=-2 → vol ≈ 37% |
| **Mean scale** | μ_scale | (0, 5) | Additional log-vol in high regime |
| **Mean rate** | μ_rate | (0.1, 5) | How fast μ(z) saturates |
| **Base noise** | σ_base | (0.01, 1) | Volatility-of-volatility in low regime |
| **Noise scale** | σ_scale | (0, 1) | Additional vol-of-vol in high regime |
| **Noise rate** | σ_rate | (0.1, 5) | How fast σ_h(z) saturates |

### 2.3 Curve Functions

The regime variable z controls three curves:

```
θ(z) = θ_base + θ_scale · (1 - exp(-θ_rate · z))     [mean-reversion speed]
μ(z) = μ_base + μ_scale · (1 - exp(-μ_rate · z))     [long-run mean]
σ_h(z) = σ_base + σ_scale · (1 - exp(-σ_rate · z))   [innovation noise]
```

```
         Curve Shape: base + scale·(1 - exp(-rate·z))
         
    value
      ▲
      │                          ●────────────── base + scale
      │                     ●
      │                ●
      │           ●
      │       ●
      │    ●
      │  ●
      │ ●
      │●─────────────────────────────────────── base
      └──────────────────────────────────────▶ z
      0              1              2         3
      
      Low regime              High regime
      (calm market)           (stressed market)
```

---

## 3. Background: SMC², PMMH, and CPMMH

Before describing our hybrid, we briefly review the three algorithms we combine.

### 3.1 Sequential Monte Carlo Squared (SMC²)

SMC² maintains a population of **θ-particles**, each running its own **particle filter** on the latent states:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SMC² STRUCTURE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  θ-particle 1:  θ₁ ──┬── [h₁,₁, h₁,₂, ..., h₁,N]  (inner filter)   │
│                      │                                              │
│  θ-particle 2:  θ₂ ──┼── [h₂,₁, h₂,₂, ..., h₂,N]  (inner filter)   │
│                      │                                              │
│  θ-particle 3:  θ₃ ──┼── [h₃,₁, h₃,₂, ..., h₃,N]  (inner filter)   │
│                      │                                              │
│       ⋮              │           ⋮                                  │
│                      │                                              │
│  θ-particle M:  θ_M ─┴── [h_M,1, h_M,2, ..., h_M,N] (inner filter)  │
│                                                                     │
│  Total particles: M × N  (e.g., 256 × 256 = 65,536)                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Strengths**: Massively parallel (each θ-particle independent)
**Weakness**: Outer particles degenerate; standard rejuvenation uses random-walk MH

### 3.2 Particle Marginal Metropolis-Hastings (PMMH)

PMMH treats the particle filter likelihood estimate as an unbiased estimator in an MH accept/reject step:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PMMH ITERATION                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Current state: θ with likelihood estimate p̂(y|θ)               │
│                                                                     │
│  2. Propose: θ* ~ q(θ*|θ)                                          │
│                                                                     │
│  3. Run particle filter with θ* to get p̂(y|θ*)                     │
│     ┌─────────────────────────────────────────┐                     │
│     │  PF(θ*): t=1 ──▶ t=2 ──▶ ... ──▶ t=T   │ ◀── SEQUENTIAL!    │
│     └─────────────────────────────────────────┘                     │
│                                                                     │
│  4. Accept with probability:                                        │
│                                                                     │
│              p̂(y|θ*) · p(θ*) · q(θ|θ*)                             │
│     α = min(1, ─────────────────────────────)                       │
│              p̂(y|θ)  · p(θ)  · q(θ*|θ)                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Strength**: Correct Bayesian updates (the noise in p̂ cancels!)
**Weakness**: Step 3 is sequential over T — cannot parallelize

### 3.3 Correlated PMMH (CPMMH)

CPMMH reduces the variance of the acceptance ratio by **correlating** the random numbers between current and proposed:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CPMMH CORRELATION                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Standard PMMH:                                                     │
│    p̂(y|θ)  uses random numbers U = [u₁, u₂, ..., u_T]              │
│    p̂(y|θ*) uses random numbers U* = [u₁*, u₂*, ..., u_T*]  (fresh) │
│                                                                     │
│    ──▶ High variance in ratio p̂(y|θ*)/p̂(y|θ)                       │
│                                                                     │
│  CPMMH:                                                             │
│    p̂(y|θ*) uses U* where:                                          │
│                                                                     │
│        u_i* = ρ · u_i + √(1-ρ²) · ε_i    (ρ ≈ 0.99)                │
│                 ▲           ▲                                       │
│                 │           │                                       │
│           from current    fresh                                     │
│                                                                     │
│    ──▶ U* ≈ U, so p̂(y|θ*) ≈ p̂(y|θ) when θ* ≈ θ                    │
│    ──▶ Much lower variance in acceptance ratio!                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Strength**: 10-100x variance reduction → better mixing
**Weakness**: Still sequential (must replay from t=1 to T)

---

## 4. The Hybrid Algorithm

### 4.1 The Key Insight

We observe that:

1. **SMC² forward filtering is embarrassingly parallel** — each (θ, inner particle) is independent
2. **SMC² rejuvenation is the bottleneck** — random-walk MH has poor mixing
3. **CPMMH has excellent mixing but is sequential** — can't parallelize the replay

**Our insight**: Run SMC² normally until ESS drops, then inject **CPMMH-style rejuvenation** for each θ-particle **in parallel**. Each rejuvenation is sequential over T, but all M rejuvenations run simultaneously on the GPU.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE HYBRID ALGORITHM                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  for t = 1, 2, ..., T:                                              │
│                                                                     │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │ STEP 1: Forward Filter (PARALLEL over all M×N particles)   │  │
│    │                                                             │  │
│    │   for each θ-particle j = 1..M:        ─┐                   │  │
│    │     for each inner particle i = 1..N:   │ GPU: M×N threads  │  │
│    │       propagate h[j,i]                  │                   │  │
│    │       weight by p(y_t | h[j,i])        ─┘                   │  │
│    │                                                             │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                               │                                     │
│                               ▼                                     │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │ STEP 2: Check Outer ESS                                     │  │
│    │                                                             │  │
│    │   ESS = 1 / Σ w_j²                                          │  │
│    │                                                             │  │
│    │   if ESS > threshold:  continue to next t                   │  │
│    │   if ESS < threshold:  trigger rejuvenation ───────────┐    │  │
│    └─────────────────────────────────────────────────────────│────┘  │
│                                                              │      │
│                                                              ▼      │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │ STEP 3: Resample θ-particles                                │  │
│    │                                                             │  │
│    │   Multinomial resampling based on weights w_j               │  │
│    │   Copy inner particles, noise arrays, states                │  │
│    │                                                             │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                               │                                     │
│                               ▼                                     │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │ STEP 4: CPMMH Rejuvenation (PARALLEL over M θ-particles)   │  │
│    │                                                             │  │
│    │   for each θ-particle j = 1..M:        ─┐                   │  │
│    │                                          │                  │  │
│    │     θ* ~ proposal(θ_j)                   │                  │  │
│    │                                          │ GPU: M blocks    │  │
│    │     ┌─────────────────────────────────┐  │ N threads each   │  │
│    │     │ Replay filter t=1..T with       │  │                  │  │
│    │     │ CORRELATED noise (ρ=0.99)       │  │ Sequential       │  │
│    │     │ to get p̂(y|θ*)                  │  │ within block     │  │
│    │     └─────────────────────────────────┘  │                  │  │
│    │                                          │                  │  │
│    │     Accept/reject via MH ratio          ─┘                  │  │
│    │                                                             │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Parallelism Analysis

| Operation | Parallelism | GPU Mapping |
|-----------|-------------|-------------|
| Forward filter | M × N | M×N threads |
| Inner resample | M × N | M×N threads |
| Outer ESS | M | M threads |
| Outer resample | M | M threads |
| CPMMH replay | M × N (spatial) × T (temporal) | M blocks × N threads, T iterations |

The CPMMH replay is **sequential in T** but **parallel in M**. With M=256 rejuvenations running simultaneously, we achieve high GPU utilization despite the sequential inner loop.

### 4.3 Why Not Standard SMC² Rejuvenation?

Standard SMC² uses a random-walk MH kernel:

```
θ* = θ + ε,    ε ~ N(0, Σ)
```

This requires tuning Σ and has poor mixing in high dimensions. Each proposal is essentially "blind" — it doesn't account for the complex likelihood surface.

Our CPMMH rejuvenation **re-runs the particle filter** for each proposal, giving us:
1. **Exact likelihood ratio** (up to Monte Carlo error)
2. **Correlated noise** to reduce that error
3. **Proper Bayesian updates** without tuning step sizes

---

## 5. Making Coupling Work: The Sorting Trick

### 5.1 The Problem with Resampling

CPMMH correlates random numbers to reduce variance:

```
u*[i] = ρ · u[i] + √(1-ρ²) · fresh[i]
```

This only helps if `u[i]` affects the "same" particle in both current and proposed runs. But **resampling scrambles particle identities**:

```
┌─────────────────────────────────────────────────────────────────────┐
│              COUPLING FAILURE WITHOUT SORTING                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Before Resampling (t=49):                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Particle   │  μ_h   │  Uses noise  │  State region          │    │
│  ├────────────┼────────┼──────────────┼────────────────────────┤    │
│  │     0      │  -2.1  │    u[0]      │  Low volatility        │    │
│  │     1      │  -1.5  │    u[1]      │  Low volatility        │    │
│  │     2      │  +0.3  │    u[2]      │  Medium volatility     │    │
│  │     3      │  +1.8  │    u[3]      │  High volatility       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Resampling with ancestors = [3, 3, 0, 3]:                          │
│                                                                     │
│  After Resampling (t=50):                                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Particle   │  μ_h   │  Uses noise  │  State region          │    │
│  ├────────────┼────────┼──────────────┼────────────────────────┤    │
│  │     0      │  +1.8  │    u[0]      │  High volatility  ✗    │    │
│  │     1      │  +1.8  │    u[1]      │  High volatility  ✗    │    │
│  │     2      │  -2.1  │    u[2]      │  Low volatility   ✗    │    │
│  │     3      │  +1.8  │    u[3]      │  High volatility  ✓    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Particle 0 now has HIGH volatility but uses u[0], which was       │
│  calibrated for LOW volatility. The coupling is BROKEN!            │
│                                                                     │
│  RESULT: u*[0] ≈ u[0] is useless — states are completely different │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 The Solution: Sort After Resampling

We **sort particles by μ_h** after every resampling step:

```
┌─────────────────────────────────────────────────────────────────────┐
│              COUPLING RESTORED WITH SORTING                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  After Resampling (t=50), BEFORE sort:                              │
│  ┌────────────────────────────────────────────────────┐             │
│  │ Particle   │  μ_h   │  State region                │             │
│  ├────────────┼────────┼──────────────────────────────┤             │
│  │     0      │  +1.8  │  High volatility             │             │
│  │     1      │  +1.8  │  High volatility             │             │
│  │     2      │  -2.1  │  Low volatility              │             │
│  │     3      │  +1.8  │  High volatility             │             │
│  └────────────────────────────────────────────────────┘             │
│                          │                                          │
│                          ▼ SORT BY μ_h                              │
│                                                                     │
│  After Sorting:                                                     │
│  ┌────────────────────────────────────────────────────┐             │
│  │ Particle   │  μ_h   │  State region                │             │
│  ├────────────┼────────┼──────────────────────────────┤             │
│  │     0      │  -2.1  │  Low volatility         ✓    │             │
│  │     1      │  +1.8  │  High volatility        ✓    │             │
│  │     2      │  +1.8  │  High volatility        ✓    │             │
│  │     3      │  +1.8  │  High volatility        ✓    │             │
│  └────────────────────────────────────────────────────┘             │
│                                                                     │
│  Now particle i ALWAYS represents the i-th quantile of the         │
│  h-distribution. The noise u[i] consistently affects the same      │
│  "region" of state space.                                           │
│                                                                     │
│  RESULT: Coupling ρ=0.99 actually reduces variance!                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 Why Bitonic Sort?

We need a **deterministic** sort — if the sort order depends on thread scheduling, the coupling breaks across runs. We use **bitonic sort**:

- O(N log²N) comparisons
- Fully deterministic (no thread-order dependence)
- In-place (no extra memory)
- Fast on GPU for N ≤ 512

---

## 6. Tractable Likelihoods: The OCSN Mixture

### 6.1 The Problem

The observation model is:

```
y_t = exp(h_t / 2) · ε_t,    ε_t ~ N(0, 1)
```

Taking logs to linearize:

```
log(y_t²) = h_t + log(ε_t²)
          = h_t + log(χ²(1))
```

The `log(χ²(1))` term is **non-Gaussian** — it breaks Kalman filtering.

```
        Distribution of log(χ²(1))
        
    density
      ▲
      │
      │    ●
      │   ● ●
      │  ●   ●
      │ ●     ●●
      │●        ●●●
      │           ●●●●●●●●●●●●●●───
    ──┼───────────────────────────────▶ x
     -10    -5     0     5
           ▲
           │
     Mean ≈ -1.27
     Highly skewed!
     NOT Gaussian!
```

### 6.2 The OCSN Approximation

Omori, Chib, Shephard & Nakajima (2007) approximate `log(χ²(1))` as a **10-component Gaussian mixture**:

```
log(χ²(1)) ≈ Σᵢ wᵢ · N(mᵢ, vᵢ²)
```

| Component | Weight wᵢ | Mean mᵢ | Variance vᵢ² |
|-----------|-----------|---------|--------------|
| 1 | 0.00609 | 1.927 | 0.113² |
| 2 | 0.04775 | 1.347 | 0.178² |
| 3 | 0.13057 | 0.735 | 0.268² |
| 4 | 0.20674 | 0.023 | 0.406² |
| 5 | 0.22715 | -0.852 | 0.627² |
| 6 | 0.18842 | -1.973 | 0.986² |
| 7 | 0.12047 | -3.468 | 1.575² |
| 8 | 0.05591 | -5.552 | 2.545² |
| 9 | 0.01575 | -8.684 | 4.166² |
| 10 | 0.00115 | -14.654 | 7.333² |

```
        OCSN Mixture Approximation
        
    density
      ▲
      │
      │    True log(χ²(1))     ─────
      │    10-component mix    ●●●●●
      │
      │    ●
      │   ●─●
      │  ● ─ ●
      │ ●  ─  ●●
      │●   ─    ●●●●●●●●●●●●●●───
    ──┼───────────────────────────────▶ x
     -10    -5     0     5
     
     Approximation is nearly exact!
```

### 6.3 Moment Matching vs Component Sampling

Two approaches to use the mixture:

**Approach 1: Sample component**
- Draw component index k ~ Categorical(w)
- Condition on that component
- Requires tracking discrete state

**Approach 2: Moment matching** (our choice)
- Compute marginal mean and variance
- Use a single Gaussian approximation
- Smoother likelihood surface for MH

We use moment matching because it makes the likelihood surface smoother, which benefits CPMMH exploration.

---

## 7. Proper Dynamics: The Z-Space Transform

### 7.1 The Problem with Clamping

The regime variable z must be in (0, 3) for curve evaluation. Naive clamping:

```
z_raw = ρ · z_{t-1} + σ_z · ε
z = clamp(z_raw, 0.001, 2.999)
```

This **distorts the likelihood**:

```
┌─────────────────────────────────────────────────────────────────────┐
│              PROBABILITY PILEUP AT BOUNDARIES                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  True AR(1) density for z:           Clamped density:               │
│                                                                     │
│      ▲                                   ▲                          │
│      │     ●●●                           █     ●●●                  │
│      │   ●     ●                         █   ●     ●                │
│      │  ●       ●                        █  ●       ●               │
│      │ ●         ●                       █ ●         ●              │
│      │●           ●                      █●           ●    █        │
│      └──────────────▶                    └──────────────▶           │
│      0      z       3                    0      z       3           │
│                                          ▲                 ▲        │
│                                          │                 │        │
│                                     Probability       Probability   │
│                                     pileup!           pileup!       │
│                                                                     │
│  Clamping creates point masses at boundaries, making the            │
│  transition density incorrect and biasing parameter estimates.      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 The Tanh Reparameterization

We introduce an **unconstrained** variable z̃ ∈ ℝ and transform:

```
z̃_t = ρ · z̃_{t-1} + σ_z · ε_t    (exact Gaussian AR(1))
z_t = 1.5 · (1 + tanh(z̃_t))       (smooth map to (0, 3))
```

```
┌─────────────────────────────────────────────────────────────────────┐
│              SMOOTH TRANSFORM: NO BOUNDARY ISSUES                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  z̃ space (unconstrained):                z space (bounded):         │
│                                                                     │
│      ▲                                   ▲                          │
│      │     ●●●                           │       ●●●●               │
│      │   ●     ●                         │     ●      ●             │
│      │  ●       ●                        │    ●        ●            │
│      │ ●         ●                       │  ●            ●          │
│      │●           ●                      │●               ●         │
│      └──────────────▶                    └───────────────────▶      │
│     -∞      z̃      +∞                    0        z          3      │
│                                                                     │
│        Exact Gaussian          ──▶      Smooth, no pileup           │
│        transition density               Proper likelihood           │
│                                                                     │
│  The tanh transform smoothly squashes ℝ into (0, 3).               │
│  No probability mass accumulates at boundaries.                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- AR(1) on z̃ has exact Gaussian transition density
- No Jacobian needed (we track z̃, not z)
- Smooth likelihood surface for CPMMH

---

## 8. Scaling to Large T: Fixed-Lag PMMH

### 8.1 The Variance Growth Problem

The CPMMH likelihood estimate is a product of T terms:

```
p̂(y₁:T | θ) = ∏ₜ p̂(yₜ | y₁:ₜ₋₁, θ)
```

In log space, this is a **sum of T noisy terms**:

```
log p̂(y₁:T | θ) = Σₜ log p̂(yₜ | y₁:ₜ₋₁, θ)
```

By CLT, the variance grows linearly:

```
Var[log p̂(y₁:T | θ)] ∝ T
```

```
┌─────────────────────────────────────────────────────────────────────┐
│              VARIANCE GROWTH WITH T                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Variance                                                           │
│      ▲                                                              │
│      │                                              ●               │
│      │                                           ●                  │
│      │                                        ●   Full history      │
│  5000│                                     ●      (L = 0)           │
│      │                                  ●                           │
│      │                               ●                              │
│      │                            ●                                 │
│      │                         ●                                    │
│      │                      ●                                       │
│      │                   ●                                          │
│  1000│                ●                                             │
│      │             ●                                                │
│      │          ●                                                   │
│      │       ●                                                      │
│   100│─ ─ ●─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  Fixed-lag (L=100)  │
│      │    ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●         │
│      │                                                              │
│      └──────────────────────────────────────────────────────────▶ T │
│          500    1000   2000       5000      10000                   │
│                                                                     │
│  At T=5000, full-history variance is 50× the fixed-lag variance!   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**High variance → erratic MH acceptance → poor mixing → degraded estimates**

### 8.2 The Fixed-Lag Solution

Instead of replaying the full history [0, T], we only replay a window [T-L, T]:

```
┌─────────────────────────────────────────────────────────────────────┐
│              FIXED-LAG PMMH                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Full History (L = 0):                                              │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ t=0  t=1  t=2  ...  t=T-L  ...  t=T-1  t=T         │            │
│  │  ●────●────●────●────●────●────●────●────●          │            │
│  │  └──────────────────────────────────────┘          │            │
│  │            Replay entire history                   │            │
│  │            Cost: O(T)  Variance: O(T)              │            │
│  └─────────────────────────────────────────────────────┘            │
│                                                                     │
│  Fixed-Lag (L = 100):                                               │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ t=0  t=1  t=2  ...  t=T-L  ...  t=T-1  t=T         │            │
│  │  ○────○────○────○────●════●════●════●════●          │            │
│  │                      ▲    └────────────────┘        │            │
│  │                      │    Replay only window        │            │
│  │                Checkpoint                           │            │
│  │                      │                              │            │
│  │            Cost: O(L)  Variance: O(L)              │            │
│  └─────────────────────────────────────────────────────┘            │
│                                                                     │
│  ○ = Frozen (use checkpoint state)                                  │
│  ● = Replayed (run filter with proposed θ)                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.3 The Bias-Variance Tradeoff

Fixed-lag introduces a small **bias** because we ignore observations before T-L:

```
Bias ≈ ρ^L
```

For ρ = 0.95 and L = 100:

```
Bias ≈ 0.95^100 ≈ 0.006 (0.6%)
```

This is negligible for practical purposes.

| L | Bias | Variance Bound | Recommendation |
|---|------|----------------|----------------|
| 0 | 0 | O(T) | Only for short sequences |
| 50 | 8% | O(50) | Aggressive |
| 100 | 0.6% | O(100) | **Recommended** |
| 200 | 0.003% | O(200) | Conservative |

### 8.4 Checkpoint System

We save particle state every L steps:

```
┌─────────────────────────────────────────────────────────────────────┐
│              CHECKPOINT TIMELINE                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  L = 100                                                            │
│                                                                     │
│  t:  0    50   100   150   200   250   300   350   400             │
│      │     │     │     │     │     │     │     │     │              │
│      ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼              │
│      ●─────●─────●─────●─────●─────●─────●─────●─────●              │
│                  ▲           ▲           ▲           ▲              │
│                  │           │           │           │              │
│                SAVE       SAVE        SAVE        SAVE              │
│             checkpoint  checkpoint  checkpoint  checkpoint          │
│                                                                     │
│  When rejuvenation triggered at t=350:                              │
│    - Load checkpoint from t=300                                     │
│    - Replay t=301..350 with proposed θ                              │
│    - Compare window likelihoods                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Checkpoint contains:**
- Inner particle states: z̃, μ_h, var_h, log_w
- Accumulated likelihood up to checkpoint: log p̂(y₁:T_checkpoint | θ)

---

## 9. CUDA Implementation

### 9.1 Memory Layout

We use **Structure of Arrays (SoA)** for coalesced memory access:

```
┌─────────────────────────────────────────────────────────────────────┐
│              MEMORY LAYOUT                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Array of Structures (AoS) — BAD for GPU:                          │
│  ┌────────────────────────────────────────────────────────┐         │
│  │ [θ₀, h₀, w₀] [θ₁, h₁, w₁] [θ₂, h₂, w₂] [θ₃, h₃, w₃]  │         │
│  └────────────────────────────────────────────────────────┘         │
│       ▲              Thread 1 reads θ₁                              │
│       Thread 0 reads θ₀              ▲                              │
│                                      Memory access scattered!       │
│                                                                     │
│  Structure of Arrays (SoA) — GOOD for GPU:                         │
│  ┌────────────────────────────────────────────────────────┐         │
│  │ [θ₀, θ₁, θ₂, θ₃]  [h₀, h₁, h₂, h₃]  [w₀, w₁, w₂, w₃] │         │
│  └────────────────────────────────────────────────────────┘         │
│    ▲   ▲   ▲   ▲                                                    │
│    │   │   │   │     Memory access coalesced!                       │
│    T0  T1  T2  T3    Single cache line serves all threads           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 FP16 Noise Arrays

Random numbers are stored in **half precision (FP16)** to reduce memory bandwidth:

```
┌─────────────────────────────────────────────────────────────────────┐
│              NOISE STORAGE                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Noise array shape: [N_theta] × [N_inner] × [T+1]                  │
│                                                                     │
│  With N_theta=256, N_inner=256, T=5000:                            │
│                                                                     │
│    FP32: 256 × 256 × 5001 × 4 bytes = 1.22 GB                      │
│    FP16: 256 × 256 × 5001 × 2 bytes = 0.61 GB  ◀── 2× smaller      │
│                                                                     │
│  FP16 precision (3-4 decimal digits) is sufficient for noise.      │
│  We convert to FP32 only during computation.                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.3 Kernel Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│              KERNEL ARCHITECTURE                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  One Update Step:                                                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ kernel_rbpf_step                                        │        │
│  │ Grid: N_theta blocks, N_inner threads each              │        │
│  │                                                         │        │
│  │   1. Load observation y_t                               │        │
│  │   2. Propagate z̃ (AR(1) transition)                     │        │
│  │   3. Transform z = 1.5·(1 + tanh(z̃))                    │        │
│  │   4. Evaluate curves θ(z), μ(z), σ_h(z)                 │        │
│  │   5. Kalman predict h                                   │        │
│  │   6. OCSN mixture likelihood                            │        │
│  │   7. Kalman update h                                    │        │
│  │   8. Update weights                                     │        │
│  │   9. Compute inner ESS                                  │        │
│  │  10. Inner resample (if ESS low)                        │        │
│  │  11. Sort by μ_h (for CPMMH coupling)                   │        │
│  │  12. Accumulate outer likelihood                        │        │
│  └─────────────────────────────────────────────────────────┘        │
│                          │                                          │
│                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ kernel_compute_outer_ess                                │        │
│  │ Grid: 1 block, N_theta threads                          │        │
│  └─────────────────────────────────────────────────────────┘        │
│                          │                                          │
│                          ▼ (if ESS < threshold)                     │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ kernel_outer_resample                                   │        │
│  │ kernel_copy_theta_particles                             │        │
│  └─────────────────────────────────────────────────────────┘        │
│                          │                                          │
│                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ kernel_cpmmh_rejuvenate_fused                           │        │
│  │ Grid: N_theta blocks, N_inner threads each              │        │
│  │                                                         │        │
│  │   For each θ-particle (in parallel):                    │        │
│  │     1. Propose θ* from current θ                        │        │
│  │     2. Load checkpoint (if fixed-lag)                   │        │
│  │     3. Replay filter from t_start to T                  │        │
│  │        - Correlate noise: u* = ρ·u + √(1-ρ²)·fresh     │        │
│  │        - Inner resample + sort                          │        │
│  │        - Accumulate likelihood                          │        │
│  │     4. MH accept/reject                                 │        │
│  │     5. Swap buffers if accepted                         │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.4 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│              DATA FLOW DIAGRAM                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                      ┌─────────────┐                                │
│                      │  Host CPU   │                                │
│                      └──────┬──────┘                                │
│                             │ y_t (observation)                     │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                        GPU MEMORY                            │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  │   │
│  │  │  θ-particles    │  │  Inner states   │  │  Noise arrays│  │   │
│  │  │  ─────────────  │  │  ─────────────  │  │  ──────────  │  │   │
│  │  │  rho[N_theta]   │  │  z[N_θ × N_i]   │  │  z_noise     │  │   │
│  │  │  sigma_z[...]   │  │  mu_h[...]      │  │  [N_θ×N_i×T] │  │   │
│  │  │  mu_base[...]   │  │  var_h[...]     │  │  (FP16)      │  │   │
│  │  │  ...            │  │  log_w[...]     │  │              │  │   │
│  │  │  weight[...]    │  │                 │  │  u0_noise    │  │   │
│  │  │  log_lik[...]   │  │                 │  │  [N_θ × T]   │  │   │
│  │  └────────┬────────┘  └────────┬────────┘  └──────┬───────┘  │   │
│  │           │                    │                  │          │   │
│  │           └──────────┬─────────┴──────────────────┘          │   │
│  │                      │                                       │   │
│  │                      ▼                                       │   │
│  │           ┌─────────────────────┐                            │   │
│  │           │   CUDA Kernels      │                            │   │
│  │           │   (fused ops)       │                            │   │
│  │           └──────────┬──────────┘                            │   │
│  │                      │                                       │   │
│  │                      ▼                                       │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                    │   │
│  │  │  Checkpoint     │  │  Scratch buffer │                    │   │
│  │  │  (for fixed-lag)│  │  (for resample) │                    │   │
│  │  └─────────────────┘  └─────────────────┘                    │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│                      ┌─────────────┐                                │
│                      │  Host CPU   │                                │
│                      │  ─────────  │                                │
│                      │  θ_mean     │                                │
│                      │  θ_std      │                                │
│                      │  ESS        │                                │
│                      └─────────────┘                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. Tuning Guide

### 10.1 Particle Counts

| Parameter | Low | Medium | High | Notes |
|-----------|-----|--------|------|-------|
| N_theta | 64 | 256 | 512 | More = better posterior approximation |
| N_inner | 64 | 256 | 512 | More = lower likelihood variance |

**Rule of thumb**: N_theta × N_inner ≈ 65,536 (256 × 256) is a good balance.

### 10.2 Fixed-Lag Window

| ρ | Recommended L | Bias | Notes |
|---|---------------|------|-------|
| 0.90 | 50 | 0.5% | Fast-moving regimes |
| 0.95 | 100 | 0.6% | **Typical** |
| 0.99 | 200 | 1.3% | Slow-moving regimes |

**Formula**: L ≈ 7 × half-life, where half-life = log(0.5) / log(ρ)

### 10.3 CPMMH Correlation

| cpmmh_rho | Acceptance Rate | Variance Reduction |
|-----------|-----------------|-------------------|
| 0.90 | ~25% | Low |
| 0.95 | ~35% | Medium |
| 0.99 | ~50% | High |
| 0.999 | ~60% | Very high (may reduce exploration) |

**Recommendation**: Start with 0.99, increase if acceptance < 30%.

### 10.4 ESS Thresholds

| Threshold | Rejuvenation Frequency | Notes |
|-----------|------------------------|-------|
| 0.3 | Rare | May lose diversity |
| 0.5 | Moderate | **Recommended** |
| 0.7 | Frequent | More computation |

---

## 11. API Reference

### 11.1 Allocation and Setup

```cpp
// Allocate state
SMC2StateCUDA* state = smc2_cuda_alloc(N_theta, N_inner);

// Configure
smc2_cuda_set_seed(state, 12345);           // Reproducibility
smc2_cuda_set_noise_capacity(state, T+128); // Pre-allocate for T observations
smc2_cuda_set_fixed_lag(state, 100);        // Enable fixed-lag with L=100

// Initialize from prior
smc2_cuda_init_from_prior(state);
```

### 11.2 Online Update

```cpp
// Process each observation
for (int t = 0; t < T; t++) {
    float ess = smc2_cuda_update(state, y[t]);
    
    // Optionally monitor
    if (ess < threshold) {
        printf("Warning: low ESS at t=%d\n", t);
    }
}
```

### 11.3 Extract Results

```cpp
// Get posterior mean and std
float theta_mean[8], theta_std[8];
smc2_cuda_get_theta_mean(state, theta_mean);
smc2_cuda_get_theta_std(state, theta_std);

// Parameters: [rho, sigma_z, mu_base, mu_scale, mu_rate, 
//              sigma_base, sigma_scale, sigma_rate]
```

### 11.4 Cleanup

```cpp
smc2_cuda_free(state);
```

---

## 12. References

1. **SMC²**: Chopin, Jacob, Papaspiliopoulos (2013). "SMC²: An efficient algorithm for sequential analysis of state space models." *JRSS-B*.

2. **PMMH**: Andrieu, Doucet, Holenstein (2010). "Particle Markov chain Monte Carlo methods." *JRSS-B*.

3. **CPMMH**: Deligiannidis, Doucet, Pitt (2018). "The Correlated Pseudo-Marginal Method." *JRSS-B*.

4. **OCSN Mixture**: Omori, Chib, Shephard, Nakajima (2007). "Stochastic Volatility with Leverage: Fast and Efficient Likelihood Inference." *J. Econometrics*.

5. **Bitonic Sort**: Batcher (1968). "Sorting networks and their applications." *AFIPS*.

---

## Appendix A: Mathematical Details

### A.1 State Space Model

**Regime dynamics** (unconstrained):
```
z̃_t = ρ · z̃_{t-1} + σ_z · ε^z_t,    ε^z_t ~ N(0,1)
```

**Regime transform** (bounded):
```
z_t = 1.5 · (1 + tanh(z̃_t)) ∈ (0, 3)
```

**Volatility dynamics**:
```
h_t = (1 - θ(z_t)) · h_{t-1} + θ(z_t) · μ(z_t) + σ_h(z_t) · ε^h_t
```

**Observation**:
```
y_t = exp(h_t / 2) · ε^y_t,    ε^y_t ~ N(0,1)
```

### A.2 OCSN Likelihood

The observation in log-squared form:
```
log(y_t²) = h_t + log(χ²(1))
```

With OCSN approximation:
```
p(log(y_t²) | h_t) ≈ Σᵢ wᵢ · N(log(y_t²); h_t + mᵢ, vᵢ²)
```

Marginal likelihood (moment-matched):
```
E[log(y_t²) | h_t] = h_t + Σᵢ wᵢ · mᵢ
Var[log(y_t²) | h_t] = Σᵢ wᵢ · (vᵢ² + mᵢ²) - (Σᵢ wᵢ · mᵢ)²
```

### A.3 Fixed-Lag MH Ratio

Full likelihood ratio:
```
α = [p̂(y₁:T | θ*) · p(θ*)] / [p̂(y₁:T | θ) · p(θ)]
```

Fixed-lag approximation (from checkpoint at T-L):
```
α ≈ [p̂(y_{T-L+1}:T | θ*, x_{T-L}) · p(θ*)] / [p̂(y_{T-L+1}:T | θ, x_{T-L}) · p(θ)]
```

The prefix likelihood p̂(y₁:{T-L} | θ) cancels because both use the same checkpoint state x_{T-L}.

---

*Document version: 1.0*
*Last updated: January 2026*
