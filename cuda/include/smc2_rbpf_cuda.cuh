/**
 * @file smc2_rbpf_cuda.cuh
 * @brief SMC² with RBPF Inner Filter and CPMMH Rejuvenation - CUDA Implementation
 * 
 * @author TUGBARS
 * @date 2025
 * 
 * ═══════════════════════════════════════════════════════════════════════════════
 * ALGORITHM OVERVIEW
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * This implements SMC² (Chopin et al. 2013) for online Bayesian parameter
 * learning in a regime-switching stochastic volatility model.
 * 
 * Three-Level Structure:
 * ----------------------
 * 
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │ OUTER: SMC² over θ-particles (N_theta = 256)                │
 *   │   - Parameters: ρ, σ_z, μ_base, μ_scale, μ_rate,            │
 *   │                 σ_base, σ_scale, σ_rate                     │
 *   │   - Weights: accumulated likelihood p̂(y_{1:t} | θ)          │
 *   │   - Resample when ESS < threshold                           │
 *   │   - Rejuvenate via CPMMH moves                              │
 *   └─────────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │ INNER: RBPF over (z, h) state (N_inner = 256 per θ)         │
 *   │   - Regime z̃: particle approximation (N_inner samples)      │
 *   │   - Log-vol h: Rao-Blackwellized (analytic Kalman moments)  │
 *   │   - OCSN 10-component mixture for observation likelihood    │
 *   └─────────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │ CPMMH: Correlated Pseudo-Marginal MH for rejuvenation       │
 *   │   - Correlates noise: z' = ρ·z + √(1-ρ²)·ε  (ρ ≈ 0.99)      │
 *   │   - Bucket sort after resampling preserves coupling         │
 *   │   - Full-history replay for correct MH ratio                │
 *   └─────────────────────────────────────────────────────────────────┘
 * 
 * 
 * CPMMH Coupling: Why We Sort After Resampling
 * =============================================
 * 
 * Standard PMMH has high variance because p̂(y|θ) is noisy. CPMMH reduces
 * variance by correlating the random numbers between current and proposed:
 * 
 *     z_prop[i] = ρ · z_curr[i] + √(1-ρ²) · z_fresh[i]
 * 
 * With ρ = 0.99, proposed and current filters see nearly identical noise.
 * BUT this only helps if particle[i] represents the "same" state in both runs.
 * 
 * Problem: Resampling scrambles particle identities.
 * 
 *     Before: particle[3] has μ_h = -2.1, uses z_noise[3]
 *     After:  particle[3] copied from ancestor[7], now μ_h = +1.5
 *     
 * The coupling z_prop[3] ≈ z_curr[3] is useless — states differ wildly.
 * 
 * Solution: Sort particles by μ_h after resampling.
 * 
 *     particle[i] always holds the i-th quantile of h-distribution
 *     z_noise[i] consistently affects the same "region" of state space
 *     Correlation ρ = 0.99 actually reduces variance
 * 
 * This is THE key implementation detail that makes CPMMH work.
 * 
 * 
 * OCSN Mixture Approximation
 * ==========================
 * 
 * The SV observation equation is:
 * 
 *     y_t = exp(h_t/2) · ε_t,  ε_t ~ N(0,1)
 * 
 * Taking logs: log(y_t²) = h_t + log(ε_t²)
 *                        = h_t + log(χ²(1))
 * 
 * The log(χ²(1)) term is non-Gaussian, breaking the Kalman filter.
 * Omori, Chib, Shephard & Nakajima (2007) approximate it as a 10-component
 * Gaussian mixture, restoring (approximate) Kalman tractability.
 * 
 * We marginalize over mixture components (moment matching) rather than
 * sampling — this makes the likelihood surface smoother for CPMMH.
 * 
 * 
 * Z-Space Transform
 * =================
 * 
 * The regime variable z must be in (0, 3) for curve evaluation.
 * Instead of clamping (which distorts the likelihood), we reparameterize:
 * 
 *     z̃ ∈ ℝ              (unconstrained, exact Gaussian AR(1))
 *     z = 1.5·(1 + tanh(z̃)) ∈ (0, 3)   (bounded, for curves)
 * 
 * Benefits:
 *   - AR(1) on z̃ has exact Gaussian transition density
 *   - No probability mass pileup at boundaries
 *   - Smooth transform → no gradient discontinuities
 * 
 * 
 * Noise Precision Configuration
 * ==============================
 * 
 * By default, noise arrays use FP32 for full precision.
 * For bandwidth-critical applications with T < 500, define SMC2_NOISE_FP16:
 * 
 *     nvcc -DSMC2_NOISE_FP16 -o smc2_fp16 smc2_rbpf_cuda.cu
 * 
 * FP32: Full precision (DEFAULT), recommended for accuracy
 * FP16: ~50% bandwidth, ~0.1% relative error per step
 * 
 * 
 * Performance Notes
 * =================
 * 
 * This code is optimized for HFT latency. Key choices:
 * 
 *   - Fused kernels: One kernel does resample + sort + propagate + observe
 *   - FP16 noise: Half precision cuts bandwidth for stored noise arrays
 *   - Parallel scans: Hillis-Steele for CDF computation
 *   - Bucket sort: O(N) with 64 bins, avoids CUB overhead for N=256
 *   - No u0 arrays: Resampling uniform derived from z_noise via Φ(z)
 * 
 * Typical performance: ~2 seconds for T=500, N_theta=256, N_inner=256
 * (vs ~7 seconds for equivalent CPU CPMMH implementation)
 * 
 * 
 * References
 * ==========
 * 
 * [1] Chopin, Jacob, Papaspiliopoulos (2013). "SMC²: An efficient algorithm
 *     for sequential analysis of state space models." JRSS-B.
 * 
 * [2] Andrieu, Doucet, Holenstein (2010). "Particle Markov chain Monte Carlo
 *     methods." JRSS-B.
 * 
 * [3] Deligiannidis, Doucet, Pitt (2018). "The Correlated Pseudo-Marginal
 *     Method." JRSS-B.
 * 
 * [4] Omori, Chib, Shephard, Nakajima (2007). "Stochastic Volatility with
 *     Leverage: Fast and Efficient Likelihood Inference." J. Econometrics.
 * 
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#ifndef SMC2_RBPF_CUDA_CUH
#define SMC2_RBPF_CUDA_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdint.h>

/* Include noise precision abstraction - defines noise_t type */
#include "smc2_noise_precision.cuh"

/*═══════════════════════════════════════════════════════════════════════════════
 * SECTION 1: COMPILE-TIME CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/** Number of OCSN mixture components (Omori et al. 2007, Table 1) */
#define OCSN_K 10

#ifndef SORT_EVERY_K
#define SORT_EVERY_K  4   /**< Sort every K steps (4 = good balance for ρ≈0.99) */
#endif

/** Z-space transform constants: z = Z_CENTER * (1 + tanh(z̃)) */
#define Z_CENTER 1.5f
#define Z_SCALE  1.5f

/*═══════════════════════════════════════════════════════════════════════════════
 * SECTION 2: DATA STRUCTURES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Gaussian prior specification for θ parameters
 * 
 * Each parameter has independent N(mean, std²) prior.
 * Used in MH acceptance ratio: π(θ*)/π(θ).
 */
struct SVPrior {
    float rho_mean, rho_std;           /**< AR(1) persistence for z̃ */
    float sigma_z_mean, sigma_z_std;   /**< Innovation std for z̃ */
    float mu_base_mean, mu_base_std;   /**< Long-run mean curve: base */
    float mu_scale_mean, mu_scale_std; /**< Long-run mean curve: scale */
    float mu_rate_mean, mu_rate_std;   /**< Long-run mean curve: rate */
    float sigma_base_mean, sigma_base_std;   /**< Vol-of-vol curve: base */
    float sigma_scale_mean, sigma_scale_std; /**< Vol-of-vol curve: scale */
    float sigma_rate_mean, sigma_rate_std;   /**< Vol-of-vol curve: rate */
};

/**
 * @brief Hard bounds for parameter support
 * 
 * Parameters outside bounds → log_prior = -∞ (instant rejection).
 * Should be wider than prior ±3σ to avoid truncation artifacts.
 */
struct SVBounds {
    float rho_min, rho_max;
    float sigma_z_min, sigma_z_max;
    float mu_base_min, mu_base_max;
    float mu_scale_min, mu_scale_max;
    float mu_rate_min, mu_rate_max;
    float sigma_base_min, sigma_base_max;
    float sigma_scale_min, sigma_scale_max;
    float sigma_rate_min, sigma_rate_max;
};

/**
 * @brief Regime-dependent curve: f(z) = base + scale * (1 - exp(-rate * z))
 * 
 * Saturates at (base + scale) as z → ∞.
 * Used for θ(z), μ(z), σ_h(z).
 */
struct SVCurve {
    float base;
    float scale;
    float rate;
};

/**
 * @brief θ-particle population with embedded RBPF state (SoA layout)
 * 
 * Memory layout uses Structure-of-Arrays for coalesced GPU access.
 * 
 * Outer level indexing: array[theta_idx], size N_theta
 * Inner level indexing: array[theta_idx * N_inner + inner_idx], size N_theta * N_inner
 * 
 * The inner_z array stores z̃ (unconstrained), not z (bounded).
 */
struct ThetaParticlesSoA {
    /* ═══ θ-level arrays (N_theta elements) ═══ */
    float* rho;              /**< AR(1) coefficient for z̃ dynamics */
    float* sigma_z;          /**< Innovation std for z̃ */
    float* mu_base;          /**< μ(z) curve: base parameter */
    float* mu_scale;         /**< μ(z) curve: scale parameter */
    float* mu_rate;          /**< μ(z) curve: rate parameter */
    float* sigma_base;       /**< σ_h(z) curve: base parameter */
    float* sigma_scale;      /**< σ_h(z) curve: scale parameter */
    float* sigma_rate;       /**< σ_h(z) curve: rate parameter */
    
    float* log_weight;       /**< Unnormalized log w (reset after outer resample) */
    float* weight;           /**< Normalized weight (sums to 1) */
    float* log_likelihood;   /**< Accumulated log p̂(y_{1:t} | θ) for PMMH */
    float* ess_inner;        /**< Inner filter ESS (diagnostic only) */
    
    /* ═══ Inner RBPF arrays (N_theta × N_inner elements) ═══ */
    float* inner_z;          /**< Regime z̃ in unconstrained space */
    float* inner_mu_h;       /**< Kalman posterior mean E[h | y_{1:t}] */
    float* inner_var_h;      /**< Kalman posterior variance */
    float* inner_log_w;      /**< Inner particle log-weight */
    curandState* rng_states; /**< Per-particle RNG state */
};

/**
 * @brief Complete SMC² state container
 * 
 * Owns all GPU memory. Use smc2_cuda_alloc/free for lifecycle.
 */
struct SMC2StateCUDA {
    /* ═══ Dimensions ═══ */
    int N_theta;             /**< Number of outer (θ) particles */
    int N_inner;             /**< Number of inner (RBPF) particles per θ */
    
    /* ═══ Particle storage (double-buffered for resampling) ═══ */
    ThetaParticlesSoA d_particles;      /**< Current particles */
    ThetaParticlesSoA d_particles_temp; /**< Scratch for resampling/CPMMH */
    
    /* ═══ Observation history (for PMMH replay) ═══ */
    float* d_y_history;
    int y_history_len;
    int y_history_capacity;
    int t_current;           /**< Current timestep (0-indexed) */
    
    /* ═══ CPMMH noise buffers (FP16 or FP32 based on SMC2_NOISE_FP32, ping-pong) ═══ */
    noise_t* d_z_noise[2];   /**< Propagation noise [N_theta × N_inner × (T+1)] */
    noise_t* d_u0_noise[2];  /**< Resampling noise [N_theta × (T+1)] */
    int noise_buf;           /**< Active buffer index (0 or 1) */
    int noise_capacity;      /**< Max T for noise arrays */
    float cpmmh_rho;         /**< Noise correlation (default 0.99) */
    
    /* ═══ Scratch arrays ═══ */
    int* d_ancestors;        /**< Outer resampling ancestors */
    float* d_uniform;        /**< Single uniform for systematic resampling */
    float* d_ess;            /**< Output: outer ESS */
    int* d_accepts;          /**< CPMMH acceptance counter */
    int* d_swap_flags;       /**< Per-particle accept flags */
    
    /* ═══ Model specification ═══ */
    SVPrior prior;
    SVBounds bounds;
    SVCurve theta_curve;     /**< θ(z) = base + scale*(1 - exp(-rate*z)) */
    float proposal_std[8];   /**< Random walk proposal std per parameter */
    
    /* ═══ Algorithm settings ═══ */
    float ess_threshold_outer;  /**< Resample if ESS < threshold * N_theta */
    float ess_threshold_inner;  /**< (unused currently, always resample) */
    int K_rejuv;                /**< CPMMH moves per outer resample */
    
    /* ═══ Fixed-Lag PMMH ═══ */
    int fixed_lag_L;            /**< Window size (0 = full history, default) */
    int t_checkpoint;           /**< Timestamp of last checkpoint */
    float* d_checkpoint_z;      /**< Checkpoint: inner_z [N_theta × N_inner] */
    float* d_checkpoint_mu_h;   /**< Checkpoint: inner_mu_h */
    float* d_checkpoint_var_h;  /**< Checkpoint: inner_var_h */
    float* d_checkpoint_log_w;  /**< Checkpoint: inner_log_w */
    float* d_checkpoint_ll;     /**< Checkpoint: log_likelihood [N_theta] */
    
    /* ═══ Diagnostics ═══ */
    int n_resamples;
    int n_rejuv_accepts;
    int n_rejuv_total;
    
    /* ═══ Adaptive Proposals (Haario et al. 2001) ═══ */
    float* d_temp_mean;         /**< Scratch: particle mean [8] */
    float* d_temp_cov;          /**< Scratch: particle covariance [64] */
    bool use_adaptive_proposals; /**< Enable adaptive covariance (default: true) */
    
    /* ═══ RNG ═══ */
    uint64_t user_seed;         /**< User-provided seed (0 = time-based) */
    uint64_t host_rng_state;    /**< Host-side xorshift64* state */
};

/*═══════════════════════════════════════════════════════════════════════════════
 * SECTION 3: SHARED MEMORY HELPERS
 *═══════════════════════════════════════════════════════════════════════════════*/

/* Note: Actual shared memory calculations are in Section 4.4 after CUB include:
 *   - rbpf_shared_mem_size<BLOCK_SIZE>() for forward step
 *   - cpmmh_shared_mem_size<BLOCK_SIZE>() for rejuvenation
 * 
 * These are templates because CUB temp storage size depends on BLOCK_SIZE.
 */

/*═══════════════════════════════════════════════════════════════════════════════
 * SECTION 4: DEVICE HELPER FUNCTIONS
 * 
 * All __device__ __forceinline__ — zero overhead, just organization.
 *═══════════════════════════════════════════════════════════════════════════════*/

/*─────────────────────────────────────────────────────────────────────────────
 * 4.1 Warp and Block Reductions
 *─────────────────────────────────────────────────────────────────────────────*/

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/**
 * @brief Block-wide sum reduction
 * @param val    Thread's input value
 * @param shared Scratch space (need 32 floats)
 * @return Sum across all threads (broadcast to all)
 */
__device__ __forceinline__ 
float block_reduce_sum(float val, volatile float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int numWarps = (blockDim.x + 31) >> 5;
    
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

/**
 * @brief Block-wide max reduction
 */
__device__ __forceinline__
float block_reduce_max(float val, volatile float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int numWarps = (blockDim.x + 31) >> 5;
    
    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : -1e30f;
    if (wid == 0) val = warp_reduce_max(val);
    
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

/*─────────────────────────────────────────────────────────────────────────────
 * 4.2 Parallel Prefix Sum (Hillis-Steele)
 *─────────────────────────────────────────────────────────────────────────────*/

/**
 * @brief In-place inclusive scan in shared memory
 * 
 * O(n log n) work, O(log n) depth. Better latency than serial O(n).
 * Requires n <= blockDim.x.
 */
__device__ __forceinline__
void block_inclusive_scan(volatile float* data, int n) {
    int tid = threadIdx.x;
    
    for (int offset = 1; offset < n; offset *= 2) {
        float temp = 0.0f;
        if (tid >= offset && tid < n) {
            temp = data[tid - offset];
        }
        __syncthreads();
        
        if (tid >= offset && tid < n) {
            data[tid] += temp;
        }
        __syncthreads();
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * 4.3 Model Functions
 *─────────────────────────────────────────────────────────────────────────────*/

/**
 * @brief Evaluate regime-dependent curve
 * @param base, scale, rate  Curve parameters
 * @param z                  Regime value in (0, 3)
 * @return base + scale * (1 - exp(-rate * z))
 */
__device__ __forceinline__
float eval_curve(float base, float scale, float rate, float z) {
    return base + scale * (1.0f - __expf(-rate * z));
}

/**
 * @brief Transform unconstrained z̃ to bounded z ∈ (0, 3)
 * 
 * z = 1.5 * (1 + tanh(z̃))
 * Maps ℝ → (0, 3) smoothly.
 */
__device__ __forceinline__ float z_tilde_to_z(float z_tilde) {
    return Z_CENTER * (1.0f + tanhf(z_tilde));
}

/**
 * @brief Inverse transform: bounded z → unconstrained z̃
 * 
 * Used only for initialization from bounded prior samples.
 */
__device__ __forceinline__ float z_to_z_tilde(float z) {
    float normalized = (z - Z_CENTER) / Z_SCALE;
    normalized = fmaxf(-0.999f, fminf(0.999f, normalized));
    return atanhf(normalized);
}

/**
 * @brief Derive resampling uniform from Gaussian noise
 * 
 * Uses probability integral transform: Φ(z) ~ Uniform(0,1) if z ~ N(0,1).
 * This eliminates the need for separate u0 storage arrays.
 */
__device__ __forceinline__ float u0_from_noise(float z_noise) {
    float u = normcdff(z_noise);
    return fmaxf(1e-7f, fminf(1.0f - 1e-7f, u));
}

/*─────────────────────────────────────────────────────────────────────────────
 * 4.4 CPMMH Sorting
 * 
 * Deterministic sort of particles by μ_h to preserve noise coupling.
 * Implementation in smc2_sorting.cuh with build-time backend selection.
 *─────────────────────────────────────────────────────────────────────────────*/

#include "smc2_sorting.cuh"

/*═══════════════════════════════════════════════════════════════════════════════
 * SECTION 5: OCSN CONSTANTS (declared extern, defined in .cu)
 * 
 * 10-component Gaussian mixture approximation to log χ²(1).
 * Source: Omori, Chib, Shephard & Nakajima (2007), Table 1
 * 
 * These approximate the RAW log χ²(1) distribution:
 *   E[log χ²(1)] ≈ -1.2704
 *   Var[log χ²(1)] ≈ π²/2 ≈ 4.93
 * 
 * Arrays defined in smc2_rbpf_cuda.cu to avoid multiple definition errors.
 *═══════════════════════════════════════════════════════════════════════════════*/

extern __device__ __constant__ float d_OCSN_WEIGHTS[OCSN_K];
extern __device__ __constant__ float d_OCSN_MEANS[OCSN_K];
extern __device__ __constant__ float d_OCSN_VARS[OCSN_K];
extern __device__ __constant__ float d_OCSN_LOG_WEIGHTS[OCSN_K];
extern __device__ __constant__ float d_OCSN_INV_VARS[OCSN_K];
extern __device__ __constant__ float d_OCSN_LOG_VARS[OCSN_K];

/*─────────────────────────────────────────────────────────────────────────────
 * OCSN Kalman Update (Marginalized)
 * 
 * Computes posterior E[h|y] and Var[h|y] by moment-matching over the
 * 10-component mixture. This is deterministic — no sampling.
 * 
 * Why marginalize instead of sample:
 *   - Smoother likelihood surface for CPMMH
 *   - Deterministic = perfectly correlated between θ and θ'
 *   - Optimal MMSE estimator for the Gaussian approximation
 *─────────────────────────────────────────────────────────────────────────────*/

__device__ __forceinline__
void ocsn_kalman_update(
    float y,           /**< Observation: log(price_return²) */
    float mu_pred,     /**< Prior mean E[h] */
    float var_pred,    /**< Prior variance Var[h] */
    float* mu_post,    /**< [out] Posterior mean */
    float* var_post,   /**< [out] Posterior variance */
    float* log_lik     /**< [out] Log marginal likelihood */
) {
    /* Numerical guard: prevent division by zero and log instabilities */
    float safe_var = fmaxf(var_pred, 1e-6f);
    
    float log_alpha_tilde[OCSN_K];
    float log_max = -1e30f;
    
    /* Pass 1: Unnormalized log mixture weights */
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        float v_k = d_OCSN_VARS[k];
        float inv_v_k = d_OCSN_INV_VARS[k];
        float log_v_k = d_OCSN_LOG_VARS[k];
        
        float S = safe_var + v_k;
        float inv_S = 1.0f / S;
        float innov = y - mu_pred - d_OCSN_MEANS[k];
        float log_S = log_v_k + log1pf(safe_var * inv_v_k);
        
        float val = d_OCSN_LOG_WEIGHTS[k] - 0.5f * (log_S + innov * innov * inv_S);
        log_alpha_tilde[k] = val;
        log_max = fmaxf(log_max, val);
    }
    
    /* Normalize */
    float sum_exp = 0.0f;
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        sum_exp += __expf(log_alpha_tilde[k] - log_max);
    }
    float log_norm = log_max + __logf(sum_exp);
    
    /* Pass 2: Moment matching */
    float mu_out = 0.0f;
    float E_h_sq = 0.0f;
    
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        float w = __expf(log_alpha_tilde[k] - log_norm);
        
        float S = safe_var + d_OCSN_VARS[k];
        float inv_S = 1.0f / S;
        float innov = y - mu_pred - d_OCSN_MEANS[k];
        float K = safe_var * inv_S;
        
        float mu_k = mu_pred + K * innov;
        float var_k = (1.0f - K) * safe_var;
        
        mu_out += w * mu_k;
        E_h_sq += w * (var_k + mu_k * mu_k);
    }
    
    *mu_post = mu_out;
    *var_post = fmaxf(E_h_sq - mu_out * mu_out, 1e-6f);
    *log_lik = log_norm;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * SECTION 6: KERNEL DECLARATIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Initialize curandState for all particles
 */
__global__ void kernel_init_rng(
    curandState* states,
    unsigned long long seed,
    int N
);

/**
 * @brief Sample θ from prior, initialize inner RBPF at stationary distribution
 */
__global__ void kernel_init_from_prior(
    ThetaParticlesSoA particles,
    int N_theta, int N_inner,
    noise_t* d_z_noise,
    noise_t* d_u0_noise,
    int noise_capacity
);

/**
 * @brief RBPF forward step: resample → propagate → observe
 * 
 * This is the main filtering kernel. Each block handles one θ-particle.
 * blockDim.x = N_inner, gridDim.x = N_theta.
 * 
 * Steps:
 *   1. Normalize weights, compute CDF
 *   2. Systematic resampling with correlated uniform u0
 *   3. Bucket sort by μ_h (CPMMH coupling)
 *   4. Propagate z̃: z̃' = ρ·z̃ + σ_z·ε
 *   5. Kalman predict: μ_pred, σ²_pred from z̃'
 *   6. OCSN Kalman update: μ_post, σ²_post, log_lik
 *   7. Accumulate log-likelihood to outer particle
 */
__global__ void kernel_rbpf_step(
    ThetaParticlesSoA particles,
    float y_obs,
    int N_theta, int N_inner,
    noise_t* d_z_noise,
    noise_t* d_u0_noise,
    int t_current,
    int noise_capacity
);

/**
 * @brief Compute outer particle ESS and normalize weights
 */
__global__ void kernel_compute_outer_ess(
    ThetaParticlesSoA particles,
    float* d_ess_out,
    int N_theta
);

/**
 * @brief Systematic resampling of outer θ-particles
 */
__global__ void kernel_outer_resample(
    ThetaParticlesSoA particles,
    int* d_ancestors,
    float* d_uniform,
    int N_theta
);

/**
 * @brief Copy θ-particles according to ancestor indices
 */
__global__ void kernel_copy_theta_particles(
    ThetaParticlesSoA src,
    ThetaParticlesSoA dst,
    int* d_ancestors,
    int N_theta, int N_inner,
    unsigned long long resample_seed
);

/**
 * @brief Copy noise arrays after outer resampling (ping-pong)
 */
__global__ void kernel_copy_noise_arrays(
    const noise_t* src_z_noise,
    noise_t* dst_z_noise,
    const noise_t* src_u0_noise,
    noise_t* dst_u0_noise,
    const int* d_ancestors,
    int N_theta, int N_inner,
    int t_current, int noise_capacity
);

/**
 * @brief CPMMH rejuvenation with fused noise correlation and replay
 * 
 * This is the most complex kernel. It:
 *   1. Proposes θ* from random walk
 *   2. Generates correlated noise: z' = ρ·z + √(1-ρ²)·ε
 *   3. Replays full filter history with θ* and z'
 *   4. Accepts/rejects via MH ratio
 * 
 * All done in ONE kernel launch to minimize latency.
 */
__global__ void kernel_cpmmh_rejuvenate_fused(
    ThetaParticlesSoA particles,
    ThetaParticlesSoA particles_scratch,
    const float* y_history,
    noise_t* d_z_noise_curr,
    noise_t* d_z_noise_other,
    noise_t* d_u0_noise_curr,
    noise_t* d_u0_noise_other,
    int t_current,
    int N_theta, int N_inner,
    int noise_capacity,
    float cpmmh_rho,
    int* d_accepts,
    int* d_swap_flags,
    unsigned long long seed,
    int move_id,
    int block_id
);

/**
 * @brief Commit accepted CPMMH proposals (copy noise buffers)
 */
__global__ void kernel_commit_accepted_noise(
    noise_t* d_z_noise_0,
    noise_t* d_z_noise_1,
    noise_t* d_u0_noise_0,
    noise_t* d_u0_noise_1,
    const int* d_swap_flags,
    int N_theta, int N_inner,
    int t_current, int noise_capacity,
    int t_start
);

/*═══════════════════════════════════════════════════════════════════════════════
 * SECTION 7: HOST API
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocate SMC² state
 * @param N_theta  Number of outer particles (typically 256)
 * @param N_inner  Number of inner RBPF particles (typically 256)
 * @return Allocated state, or NULL on failure
 */
SMC2StateCUDA* smc2_cuda_alloc(int N_theta, int N_inner);

/**
 * @brief Free SMC² state
 */
void smc2_cuda_free(SMC2StateCUDA* state);

/**
 * @brief Set RNG seed for reproducibility
 * 
 * Call before smc2_cuda_init_from_prior(). Seed 0 means time-based (non-reproducible).
 */
void smc2_cuda_set_seed(SMC2StateCUDA* state, uint64_t seed);

/**
 * @brief Grow noise buffer capacity
 * 
 * Call if you know T will exceed current capacity.
 */
void smc2_cuda_set_noise_capacity(SMC2StateCUDA* state, int capacity);

/**
 * @brief Enable fixed-lag PMMH for long sequences
 * 
 * @param state  SMC² state
 * @param L      Window size (0 = full history replay, default)
 * 
 * For ρ ≈ 0.95, L = 100-200 is recommended (about 7 half-lives).
 * 
 * Benefits:
 *   - Bounds PMMH variance to O(L) instead of O(T)
 *   - Enables scaling to T > 2000 without accuracy degradation
 *   - Rejuvenation runs in O(L) instead of O(T) time
 * 
 * Trade-off:
 *   - Small bias: ignores info beyond lag L (typically ρ^L < 1%)
 *   - Parameters may "track" slowly-drifting data (feature for finance)
 */
void smc2_cuda_set_fixed_lag(SMC2StateCUDA* state, int L);

/**
 * @brief Set CPMMH noise correlation parameter
 * 
 * @param state  SMC² state
 * @param rho    Correlation coefficient (default 0.99)
 * 
 * Higher values reduce PMMH variance but increase autocorrelation.
 * Typical range: 0.95-0.999
 */
void smc2_cuda_set_cpmmh_rho(SMC2StateCUDA* state, float rho);

/**
 * @brief Set proposal standard deviations
 * 
 * @param state  SMC² state
 * @param std    Array of 8 floats, or NULL to reset to defaults
 * 
 * Order: [rho, sigma_z, mu_base, mu_scale, mu_rate, sigma_base, sigma_scale, sigma_rate]
 */
void smc2_cuda_set_proposal_std(SMC2StateCUDA* state, const float* std);

/**
 * @brief Initialize particles from prior
 * 
 * Samples θ ~ prior, initializes inner filters at stationary distribution.
 */
void smc2_cuda_init_from_prior(SMC2StateCUDA* state);

/**
 * @brief Process one observation
 * 
 * @param state   SMC² state
 * @param y_obs   Observation (typically log(return²))
 * @return Current outer ESS
 * 
 * This is the main entry point. Call once per timestep.
 * Automatically handles resampling and CPMMH rejuvenation.
 */
float smc2_cuda_update(SMC2StateCUDA* state, float y_obs);

/**
 * @brief Get posterior mean of θ parameters
 * @param theta_mean  Output array of size 8
 * 
 * Order: [rho, sigma_z, mu_base, mu_scale, mu_rate, sigma_base, sigma_scale, sigma_rate]
 */
void smc2_cuda_get_theta_mean(SMC2StateCUDA* state, float* theta_mean);

/**
 * @brief Get posterior std of θ parameters
 */
void smc2_cuda_get_theta_std(SMC2StateCUDA* state, float* theta_std);

/**
 * @brief Get current outer ESS
 */
float smc2_cuda_get_outer_ess(SMC2StateCUDA* state);

#ifdef __cplusplus
}
#endif

#endif /* SMC2_RBPF_CUDA_CUH */
