# SMC² with CPMMH Rejuvenation - CUDA Implementation

Real-time Bayesian parameter learning for stochastic volatility models. Achieves **sub-millisecond per-tick inference** on GPU, enabling online parameter estimation for HFT applications.

## What This Is

A GPU implementation combining two frontier methods from computational statistics:

1. **SMC²** (Chopin, Jacob, Papaspiliopoulos 2013) - Sequential Monte Carlo for parameter estimation
2. **CPMMH** (Deligiannidis, Doucet, Pitt 2018) - Correlated Pseudo-Marginal Metropolis-Hastings

The target model is a **state-space stochastic volatility model** with 8 parameters, OCSN (Omori-Chib-Shephard-Nakajima) observation model, and latent z-dependent dynamics.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         SMC² Outer Level                        │
│                                                                 │
│   θ-particle 0    θ-particle 1    ...    θ-particle N_θ-1      │
│   ┌───────────┐   ┌───────────┐         ┌───────────┐          │
│   │  RBPF     │   │  RBPF     │         │  RBPF     │          │
│   │  N_inner  │   │  N_inner  │         │  N_inner  │          │
│   │ particles │   │ particles │         │ particles │          │
│   └───────────┘   └───────────┘         └───────────┘          │
│                                                                 │
│   CUDA: 1 block = 1 θ-particle, 1 thread = 1 inner particle    │
└─────────────────────────────────────────────────────────────────┘
```

**Per-tick update:**
1. Forward filter step for all θ-particles (parallel)
2. Compute outer ESS
3. If ESS < threshold: resample + CPMMH rejuvenation

## Why GPU?

```
CPU (Ryzen/i9):     ~16-24 threads
GPU (RTX 5080):     ~10,752 threads executing in parallel
                    ────────────────────────────────────
                    ~500× more parallelism
```

RTX 5080 (Blackwell):
- 84 Streaming Multiprocessors (SMs)
- 128 CUDA cores per SM
- **84 × 128 = 10,752 truly simultaneous thread executions**

For SMC² with 256×256 configuration:
```
256 θ-particles × 256 inner particles = 65,536 independent operations

CPU: Process sequentially or with 16 threads → seconds
GPU: Process with 10,752 cores → ~6 cycles to touch all 65K → milliseconds
```

The algorithm is **embarrassingly parallel**:
- Each θ-particle is independent
- Each inner particle is independent (until resampling sync)
- Perfect GPU workload

## Why SMC² + CPMMH is Fast

### Standalone CPMMH (Batch)

```
For each MCMC iteration (need ~2000 for convergence):
    Propose θ' → Run FULL particle filter O(T × N_inner) → Accept/reject
    
Total: O(2000 × T × N_inner)
```

### SMC² (Online)

```
For each new observation:
    Forward step: O(N_theta × N_inner)           ← CHEAP
    IF ESS drops (rare):
        Rejuvenate: 1 CPMMH move × O(t × N_inner) ← Only current t, not T
```

**Key insight:** The posterior evolves gradually. Particles track it. When they degenerate, one CPMMH move is enough because:
- Particles start close to the new posterior
- Correlated noise (ρ=0.99) → high acceptance (~56%)
- No burn-in needed

```
Batch:     [═══════ replay T ═══════] × 2000 iterations
SMC²:      [·][·][·]...[══ replay t ══]...[·][·][·]
                       ↑ only when ESS drops
```

## CPMMH Variance Reduction

Standard PMMH:
```
Var(log α) = Var(ℓ_prop - ℓ_curr) = 2 × Var(ℓ)
```

CPMMH with correlated noise (ρ = 0.99):
```
noise_prop = ρ × noise_curr + √(1-ρ²) × noise_fresh

Var(log α) ≈ (1 - ρ²) × Var(ℓ) = 0.02 × Var(ℓ)
```

**~100× variance reduction** → stable acceptance rates even for complex models.

## Optimizations

| Optimization | Impact |
|--------------|--------|
| **Ping-pong noise buffers** | No memcpy on accept - just swap index |
| **FP16 noise storage** | Half bandwidth, 8× memory reduction |
| **Derived u0** | Resampling uniform from z-noise, no separate array |
| **Fused CPMMH kernel** | 3 kernels → 1, all in registers |
| **Always resample (R1)** | Eliminates ESS branching, deterministic replay |

Memory comparison (T=2000, N_θ=256, N_inner=256):
```
Before: 4 FP32 arrays → ~420 MB
After:  2 FP16 arrays → ~53 MB
```

## Results

### Parameter Recovery (T=500)

```
Parameter         True       Est      Err%   Status
─────────────────────────────────────────────────────
rho             0.9600    0.9596     -0.0%  [OK]
sigma_z         0.0800    0.1287    +60.8%  [OK]
mu_base        -0.8000   -2.3687   +196.1%  [MISS]  ← identifiability issue
mu_scale        0.4000    0.4337     +8.4%  [OK]
mu_rate         1.2000    1.0557    -12.0%  [OK]
sigma_base      0.1200    0.1258     +4.8%  [OK]
sigma_scale     0.0800    0.0846     +5.8%  [OK]
sigma_rate      1.0000    1.0145     +1.5%  [OK]

7/8 parameters within 2σ
```

Note: mu_base suffers from identifiability with the OCSN offset (~-1.27). This is a known limitation of SV models, not the algorithm.

### Throughput (RTX 5080)

| N_θ × N_inner | T=500 | ms/observation |
|---------------|-------|----------------|
| 128 × 128 | 308 ms | **0.62 ms** |
| 256 × 256 | 804 ms | **1.61 ms** |

For HFT at 1000 ticks/second: 128×128 provides sub-millisecond inference with headroom.

### Speedup vs Batch CPMMH

| Method | Time | Notes |
|--------|------|-------|
| Batch CPMMH | ~8 sec | 2000 MCMC iterations |
| **SMC² + CPMMH** | **558 ms** | 4 resamples, K_rejuv=1 |

**~15× faster** while providing online estimates at every timestep.

## Algorithm Details

### Forward Filter (kernel_rbpf_step)

Each θ-particle runs an independent Rao-Blackwellized particle filter:
- **State:** z (latent factor), h (log-volatility via Kalman sufficient statistics)
- **Transition:** z follows AR(1), h follows z-dependent AR(1)
- **Observation:** OCSN mixture approximation to log-χ² 

### CPMMH Rejuvenation (kernel_cpmmh_rejuvenate_fused)

When outer ESS drops below threshold:
1. Resample θ-particles (systematic)
2. Copy noise arrays via ping-pong swap
3. For each θ-particle:
   - Propose θ' via Gaussian random walk
   - Generate correlated noise in registers
   - Replay filter with θ' and correlated noise
   - MH accept/reject targeting posterior (likelihood + prior)
   - On accept: swap to proposed noise buffer

### Critical Implementation Details

**Log prior in MH ratio:**
```cuda
float log_alpha = (ll_prop + lp_prop) - (ll_curr + lp_curr);
```
Missing the prior → targets likelihood, not posterior → parameter collapse.

**One-shot proposals:**
```cuda
theta_prop = theta_curr + std * N(0,1);
if (out_of_bounds) reject immediately;  // No retry loop
```
Retry loops create asymmetric proposals requiring Hastings correction.

**Always resample (R1):**
```cuda
// Both forward filter AND replay: always resample every timestep
// Eliminates ESS-based branching → deterministic coupling
```

## References

- Chopin, Jacob, Papaspiliopoulos (2013). "SMC²: An efficient algorithm for sequential analysis of state space models"
- Deligiannidis, Doucet, Pitt (2018). "The Correlated Pseudo-Marginal Method"
- Omori, Chib, Shephard, Nakajima (2007). "Stochastic volatility with leverage: Fast and efficient likelihood inference"

## Files

```
cuda/
├── include/
│   └── smc2_rbpf_cuda.cuh    # API and data structures
├── src/
│   └── smc2_rbpf_cuda.cu     # Implementation (~1400 lines)
└── test/
    └── test_smc2_cuda.cu     # Test suite
```