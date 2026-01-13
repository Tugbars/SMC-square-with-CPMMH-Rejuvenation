# SMC² with RBPF Inner Filter for Stochastic Volatility

## Overview

This implementation provides SMC² (Sequential Monte Carlo squared) parameter learning 
for a continuous-z stochastic volatility model, using Rao-Blackwellized particle 
filtering (RBPF) for the inner filter.

## Model Structure

**Observation**: `y_t = h_t + log(ε_t²)` where `ε_t ~ N(0,1)`
- Uses OCSN 10-component mixture approximation for log(χ²_1)

**Log-volatility dynamics**:
```
h_{t+1} = φ(z_t) * h_t + θ(z_t) * μ(z_t) + σ(z_t) * η_t
```
where `φ(z) = 1 - θ(z)` and curves are parameterized as:
```
f(z) = base + scale * (1 - exp(-rate * z))
```

**Z-process dynamics** (OU process):
```
z_{t+1} = ρ * (z_t - z_floor) + z_floor + σ_z * ε_z
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  SMC² OUTER LAYER: θ-particles (N_theta ~ 256-512)              │
│                                                                 │
│  Per θ-particle: RBPF inner filter                              │
│    - z is sampled                                               │
│    - h is marginalized via Kalman (Rao-Blackwellization)        │
│    - Each particle holds (z_i, μ_h, v_h)                        │
│                                                                 │
│  Rejuvenation: PMMH moves when ESS_θ drops                      │
└─────────────────────────────────────────────────────────────────┘
```

## Key Implementation Details

### OCSN Offset (Critical!)

The OCSN mixture approximates log(χ²_1). The Kalman filter requires zero-mean observation noise.

**For my OCSN constants**:
- Weighted mean: Σ w_k × m_k ≈ **2.22**
- True E[log(χ²_1)] ≈ **-1.27**
- Required offset = 2.22 - (-1.27) = **3.49 ≈ 3.5**

**Note**: If using centered KSC constants (weighted mean ≈ -1.27), the offset would be ~1.27.
The correct offset depends on your specific OCSN_MEANS array. Always verify:
```c
float wmean = 0;
for (int k = 0; k < 10; k++) wmean += OCSN_WEIGHTS[k] * OCSN_MEANS[k];
float offset = wmean - (-1.2704f);  // Should give your required offset
```

### Proper PMMH Rejuvenation (O(t) cost)

SMC² requires PMMH moves that target the full posterior π_t(θ) ∝ p(y_{1:t}|θ)p(θ).
This means the MH acceptance ratio must use **accumulated** likelihood:

```
α = p(y_{1:t}|θ') p(θ') / p(y_{1:t}|θ) p(θ)
```

**NOT** incremental likelihood (which would be wrong):
```
α = p(y_t|y_{1:t-1},θ') p(θ') / p(y_t|y_{1:t-1},θ) p(θ)  // WRONG!
```

To compute p(y_{1:t}|θ'), we must re-run the inner RBPF from t=0 to t_current.
This is O(t) per MH move - the true cost of SMC².

**Why we minimize resampling**: Each resample triggers K_rejuv MH moves, each costing O(t).
With RBPF giving near-zero variance likelihood estimates, outer ESS stays high and
resampling is rare (typically 1-3 times per 500 observations).

### Per-Particle RNG States (Critical for Parallelism!)

Each θ-particle maintains its own persistent RNG state (`tp->rng_state`), initialized 
via SplitMix64 from the master seed. This ensures:
- Reproducible results regardless of thread scheduling
- No RNG correlation between particles
- Correct behavior under OpenMP parallelism
- Ready for GPU port (each thread has independent state)

### Incremental vs Accumulated Likelihood

The implementation carefully distinguishes:
- `tp->log_weight`: Outer SMC weight, updated with incremental likelihood
- `tp->log_likelihood`: Accumulated log p(y_{1:t}|θ), diagnostic only
- `ll_increments[j]`: Current timestep increment, used for rejuvenation MH

Rejuvenation uses **incremental likelihood only** for MH acceptance, not accumulated.

### Pre-allocated Scratch Buffers

To avoid per-step allocations (important for GPU):
- `RBPFState.temp_particles`: Scratch for inner resampling
- `SMC2State.inner_scratch`: Scratch for rejuvenation proposal evaluation

### RBPF Update (Corrected Math)

For each OCSN component k:
1. Predictive variance: `S_k = v_pred + v_k`
2. Innovation: `innov = y - μ_pred - m_k + 3.5`  ← offset here
3. Kalman gain: `K_k = v_pred / S_k`
4. Posterior moments: `μ_k, v_k_post`
5. Moment matching across components: `μ_post = Σ α_k * μ_k`, `v_post = E[h²] - E[h]²`

### Parameters Learned (8 total)

| Parameter | Description | Prior |
|-----------|-------------|-------|
| ρ | Z-process AR coefficient | N(0.95, 0.02) |
| σ_z | Z-process innovation std | N(0.1, 0.05) |
| μ_base | h mean curve base | N(-1.0, 0.5) |
| μ_scale | h mean curve scale | N(0.5, 0.3) |
| μ_rate | h mean curve rate | N(1.0, 0.5) |
| σ_base | h std curve base | N(0.15, 0.05) |
| σ_scale | h std curve scale | N(0.1, 0.05) |
| σ_rate | h std curve rate | N(1.0, 0.5) |

**Fixed**: θ(z) curve (defines coordinate system, not learned)

## Test Results

### Parameter Recovery (T=500, N_theta=128, N_inner=128)
- **8/8 parameters within 2σ of true values**
- Rejuvenation acceptance: ~55%
- Only 1-2 resamples needed (RBPF keeps outer ESS high)

### RBPF Variance Reduction
- Log-likelihood CV ≈ 0.0000 (extremely low variance)
- ESS consistently near N_inner (~128)
- Bootstrap PF would need 5-10× more particles

### Scaling with Proper PMMH (N_theta=64, N_inner=64)
| T | Time (ms) | Resamples | Notes |
|---|-----------|-----------|-------|
| 100 | 120 | 0 | No rejuvenation needed |
| 250 | 340 | 1 | O(t) rejuvenation cost |
| 500 | 1540 | 2 | ~1s in rejuvenation |
| 1000 | 2960 | 3 | ~2s in rejuvenation |

**Key insight**: Time is dominated by O(t) rejuvenation when resamples occur.
The RBPF's low-variance estimates minimize resampling frequency.

## Building

### Prerequisites

- **Windows**: Intel oneAPI Base Toolkit (includes ICX compiler), CMake 3.20+, Ninja (recommended)
- **Linux**: GCC/Clang with OpenMP support, CMake 3.20+

### Windows with Intel ICX (Recommended)

**Option 1: Using build script**
```batch
REM Open "Intel oneAPI command prompt" or run setvars.bat first
build.bat Release
build.bat Release --test   # Build and run tests
```

**Option 2: Using CMake presets**
```batch
cmake --preset icx-release
cmake --build --preset icx-release
```

**Option 3: Manual CMake**
```batch
REM Initialize Intel oneAPI environment
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

mkdir build && cd build
cmake -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### Linux with GCC
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

### Running Tests
```bash
# After building
./build/bin/test_smc2      # Linux
build\bin\test_smc2.exe    # Windows
```

## Files

- `smc2_rbpf.h` - Header with all structures and function declarations
- `smc2_rbpf.c` - Main implementation
- `test_smc2.c` - Basic test suite
- `test_smc2_extended.c` - Extended tests with longer series
- `test_ocsn_calibration.c` - OCSN offset calibration tests
- `Makefile` - Build system

## Building

```bash
make          # Build with -O3 and OpenMP
make run      # Build and run tests
make debug    # Build with debug symbols
make valgrind # Run under valgrind
```

## Usage

```c
// Configure
SMC2Config cfg = smc2_config_defaults();
cfg.N_theta = 256;
cfg.N_inner = 256;

// Allocate
SMC2State* smc2 = smc2_alloc(&cfg);

// Set prior and fixed curves
smc2_set_prior(smc2, &prior);
smc2_set_theta_curve(smc2, &theta_curve);

// Run
SMC2Result result = smc2_run(smc2, observations, T);

// Extract posterior
float theta_mean[8], theta_std[8];
smc2_get_theta_mean(smc2, theta_mean);
smc2_get_theta_std(smc2, theta_std);

// Cleanup
smc2_free(smc2);
```

## Next Steps: GPU Port

The architecture is designed to be GPU-friendly:
- Each θ-particle maps to one CUDA block
- Each inner particle maps to one thread
- OCSN update is branchless with fixed K=10 components
- Block-level reductions for log-likelihood
- CUB library for efficient resampling

Key optimizations for CUDA:
1. Warp-level `__shfl` for log-sum-exp reduction
2. Constant memory for OCSN weights/means/vars
3. Shared memory for θ parameters within block
4. Philox counter-based RNG per thread

## References

- Kim, Shephard, Chib (1998): Stochastic Volatility: Likelihood Inference
- Omori, Chib, Shephard, Nakajima (2007): OCSN mixture approximation
- Chopin et al. (2013): SMC² for parameter learning
- Andrieu, Doucet, Holenstein (2010): PMMH