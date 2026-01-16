/**
 * @file smc2_streaming.cuh
 * @brief Streaming SMC² with Parameter Drift - CUDA Implementation
 * 
 * @author TUGBARS
 * @date 2026
 * 
 * ═══════════════════════════════════════════════════════════════════════════════
 * OVERVIEW
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * This implements a streaming variant of SMC² designed for infinite-horizon
 * operation with regime change adaptation. Key differences from batch SMC²:
 * 
 *   ┌─────────────────────────────────────────────────────────────────────────┐
 *   │ BATCH SMC² (smc2_rbpf_cuda.cu)                                          │
 *   │   - Noise arrays grow O(T) with time                                    │
 *   │   - Full history replay for CPMMH                                       │
 *   │   - Parameters treated as static                                        │
 *   │   - Suitable for T < 5000                                               │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                              vs
 *   ┌─────────────────────────────────────────────────────────────────────────┐
 *   │ STREAMING SMC² (this file)                                              │
 *   │   - Circular noise buffers O(L) fixed memory                            │
 *   │   - Fixed-lag CPMMH replay (last L steps only)                          │
 *   │   - Parameter DRIFT: θ_t = θ_{t-1} + w_t, w_t ~ N(0, Q)                │
 *   │   - Liu-West regularization prevents particle collapse                  │
 *   │   - Suitable for T → ∞                                                  │
 *   └─────────────────────────────────────────────────────────────────────────┘
 * 
 * 
 * THE THREE INFINITY PROBLEMS (AND SOLUTIONS)
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * 1. MEMORY EXPLOSION
 *    Problem:  Noise arrays grow O(T). At T=100k: 26 GB.
 *    Solution: Circular buffers. Only store last L+1 values. Memory = O(L).
 * 
 * 2. WEIGHT COLLAPSE
 *    Problem:  Old likelihood accumulates → new data can't change weights.
 *    Solution: Fixed-lag likelihood. Only last L observations matter.
 * 
 * 3. REGIME LOCK-IN
 *    Problem:  θ-particles calibrated on old regime can't reach new regime.
 *    Solution: Parameter drift. Inflate posterior by Q → particles explore.
 * 
 * 
 * PARAMETER DRIFT MODEL
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * Instead of treating θ as static:
 * 
 *     θ = constant
 * 
 * We model θ as slowly time-varying:
 * 
 *     θ_t = θ_{t-1} + w_t,  w_t ~ N(0, Q)
 * 
 * At each rejuvenation:
 *   1. Compute posterior moments: μ_θ, Σ_θ from weighted particles
 *   2. Inflate: Σ_inflated = Σ_θ + Q (add "drift covariance")
 *   3. CPMMH proposal: θ' = θ + Chol(Σ_inflated) · ε
 * 
 * This allows particles to explore OUTSIDE current posterior if regime changes.
 * 
 * 
 * LIU-WEST REGULARIZATION
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * After resampling, particles collapse to duplicates. Liu-West fixes this:
 * 
 *     θ'[j] = a · θ[j] + (1-a) · μ_θ + h · Chol(Σ_θ) · ε
 * 
 * Where h² = 1 - a² preserves total variance.
 * 
 *   - Shrinkage (a ≈ 0.98): Pull toward mean, prevents over-dispersion
 *   - Jitter (h ≈ 0.2): Add noise, maintains diversity
 * 
 * 
 * CIRCULAR BUFFER INDEXING
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * Buffer size must be power of 2 for fast masking:
 * 
 *     buffer_idx = t_absolute & buffer_mask
 * 
 * Example with L=100, buffer_size=128:
 * 
 *     t=0:   idx=0
 *     t=100: idx=100
 *     t=127: idx=127
 *     t=128: idx=0   (wraps!)
 *     t=200: idx=72
 * 
 * CPMMH replay from t_start to t_current maps each t to circular index.
 * 
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#ifndef SMC2_STREAMING_CUH
#define SMC2_STREAMING_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * COMPILE-TIME CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/** Number of θ parameters (ρ, σ_z, μ_base, μ_scale, μ_rate, σ_base, σ_scale, σ_rate) */
#define SMC2_N_PARAMS 8

/** Default fixed-lag window size */
#ifndef SMC2_DEFAULT_LAG
#define SMC2_DEFAULT_LAG 100
#endif

/** Default Liu-West shrinkage factor */
#ifndef SMC2_DEFAULT_LIU_WEST_A
#define SMC2_DEFAULT_LIU_WEST_A 0.98f
#endif

/** OCSN mixture components */
#define OCSN_K 10

/*═══════════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Drift matrix Q (diagonal, per-parameter)
 * 
 * Q[i] = variance of drift noise for parameter i per rejuvenation.
 * Larger Q → faster adaptation, more exploration.
 * Smaller Q → slower adaptation, more stability.
 */
typedef struct {
    float rho;           /**< Drift variance for persistence */
    float sigma_z;       /**< Drift variance for z innovation */
    float mu_base;       /**< Drift variance for vol mean base */
    float mu_scale;      /**< Drift variance for vol mean scale */
    float mu_rate;       /**< Drift variance for vol mean rate */
    float sigma_base;    /**< Drift variance for vol-of-vol base */
    float sigma_scale;   /**< Drift variance for vol-of-vol scale */
    float sigma_rate;    /**< Drift variance for vol-of-vol rate */
} SMC2DriftQ;

/**
 * @brief Streaming SMC² configuration
 */
typedef struct {
    int N_theta;              /**< Number of θ-particles */
    int N_inner;              /**< Number of inner particles per θ */
    int fixed_lag;            /**< Fixed-lag window L for CPMMH */
    int buffer_size;          /**< Circular buffer size (power of 2, >= L+1) */
    
    float ess_threshold;      /**< Outer ESS threshold for resampling (default 0.5) */
    float cpmmh_rho;          /**< CPMMH correlation (default 0.99) */
    int cpmmh_moves;          /**< CPMMH moves per rejuvenation (default 1) */
    
    float liu_west_a;         /**< Liu-West shrinkage factor (default 0.98) */
    SMC2DriftQ Q;             /**< Drift covariance (diagonal) */
    
    int enable_adaptive_Q;    /**< Scale Q based on NLL excess */
    float adaptive_Q_scale;   /**< Max Q multiplier under stress (default 3.0) */
    float nll_baseline;       /**< Expected NLL per tick (set after warmup) */
} SMC2StreamConfig;

/**
 * @brief Opaque state handle for streaming SMC²
 */
typedef struct SMC2StreamState SMC2StreamState;

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE API
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Create streaming SMC² state with default configuration
 * 
 * @param N_theta  Number of θ-particles (typically 256)
 * @param N_inner  Number of inner particles per θ (typically 256)
 * @return Allocated state, or NULL on failure
 */
SMC2StreamState* smc2_stream_create(int N_theta, int N_inner);

/**
 * @brief Create streaming SMC² state with custom configuration
 */
SMC2StreamState* smc2_stream_create_with_config(const SMC2StreamConfig* config);

/**
 * @brief Free streaming SMC² state
 */
void smc2_stream_free(SMC2StreamState* state);

/**
 * @brief Get default configuration (for modification before create)
 */
SMC2StreamConfig smc2_stream_default_config(int N_theta, int N_inner);

/*═══════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION API
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Set RNG seed for reproducibility
 * @param seed  Seed value (0 = use time-based seed)
 */
void smc2_stream_set_seed(SMC2StreamState* state, uint64_t seed);

/**
 * @brief Initialize particles from prior distribution
 * 
 * Samples θ uniformly from prior bounds, initializes inner RBPF filters.
 */
void smc2_stream_init_from_prior(SMC2StreamState* state);

/**
 * @brief Initialize from existing θ estimate (warm start)
 * 
 * @param theta_init  Initial θ values [N_PARAMS]
 * @param jitter_std  Jitter std for particle diversity [N_PARAMS]
 * 
 * Use for recalibration when you have a previous estimate.
 */
void smc2_stream_init_from_theta(
    SMC2StreamState* state,
    const float* theta_init,
    const float* jitter_std
);

/*═══════════════════════════════════════════════════════════════════════════════
 * STREAMING UPDATE API
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Process one observation
 * 
 * @param state  Streaming SMC² state
 * @param y_obs  Observation (typically log(return²))
 * @return Current outer ESS
 * 
 * Main entry point. Call once per tick. Handles:
 *   - Inner RBPF forward step
 *   - Outer weight update
 *   - Resampling if ESS < threshold
 *   - CPMMH rejuvenation with drift
 *   - Liu-West regularization
 *   - Circular buffer management
 */
float smc2_stream_update(SMC2StreamState* state, float y_obs);

/**
 * @brief Process batch of observations
 * 
 * @param state  Streaming SMC² state
 * @param y_obs  Observations array [n_obs]
 * @param n_obs  Number of observations
 * @return Final outer ESS
 * 
 * Convenience wrapper that calls smc2_stream_update() in a loop.
 */
float smc2_stream_update_batch(SMC2StreamState* state, const float* y_obs, int n_obs);

/*═══════════════════════════════════════════════════════════════════════════════
 * POSTERIOR QUERY API
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get posterior mean of θ parameters
 * @param theta_mean  Output array [SMC2_N_PARAMS]
 * 
 * Order: [rho, sigma_z, mu_base, mu_scale, mu_rate, sigma_base, sigma_scale, sigma_rate]
 */
void smc2_stream_get_theta_mean(SMC2StreamState* state, float* theta_mean);

/**
 * @brief Get posterior std of θ parameters
 * @param theta_std  Output array [SMC2_N_PARAMS]
 */
void smc2_stream_get_theta_std(SMC2StreamState* state, float* theta_std);

/**
 * @brief Get full posterior covariance of θ
 * @param theta_cov  Output array [SMC2_N_PARAMS × SMC2_N_PARAMS] (row-major)
 */
void smc2_stream_get_theta_cov(SMC2StreamState* state, float* theta_cov);

/**
 * @brief Get current outer ESS
 */
float smc2_stream_get_outer_ess(SMC2StreamState* state);

/**
 * @brief Get current absolute time step
 */
int smc2_stream_get_t_current(SMC2StreamState* state);

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS API
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Diagnostic information from last update
 */
typedef struct {
    int t_current;            /**< Absolute time step */
    float outer_ess;          /**< Outer ESS (before resample if triggered) */
    float outer_ess_ratio;    /**< ESS / N_theta */
    int did_resample;         /**< Whether outer resampling occurred */
    int did_rejuvenate;       /**< Whether CPMMH rejuvenation occurred */
    int cpmmh_accepts;        /**< Number of accepted CPMMH moves */
    int cpmmh_total;          /**< Total CPMMH proposals */
    float accept_rate;        /**< CPMMH acceptance rate */
    float mean_inner_ess;     /**< Average inner ESS across θ-particles */
    float nll_current;        /**< Current tick NLL */
    float nll_excess_accum;   /**< Accumulated NLL excess (for adaptive Q) */
    float Q_scale_factor;     /**< Current Q scaling (1.0 = baseline) */
} SMC2StreamDiag;

/**
 * @brief Get diagnostics from last update
 */
SMC2StreamDiag smc2_stream_get_diag(SMC2StreamState* state);

/**
 * @brief Reset NLL baseline (call after warmup period)
 * 
 * Sets nll_baseline to current average NLL per tick.
 * Required for adaptive Q scaling.
 */
void smc2_stream_reset_nll_baseline(SMC2StreamState* state);

/*═══════════════════════════════════════════════════════════════════════════════
 * RUNTIME CONFIGURATION API
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Set drift covariance Q
 * 
 * Can be called at any time to adjust drift rate.
 */
void smc2_stream_set_drift_Q(SMC2StreamState* state, const SMC2DriftQ* Q);

/**
 * @brief Get current drift covariance Q
 */
void smc2_stream_get_drift_Q(SMC2StreamState* state, SMC2DriftQ* Q);

/**
 * @brief Set Liu-West shrinkage factor
 * @param a  Shrinkage factor in (0, 1). Higher = less jitter. Default 0.98.
 */
void smc2_stream_set_liu_west_a(SMC2StreamState* state, float a);

/**
 * @brief Enable/disable adaptive Q scaling
 */
void smc2_stream_set_adaptive_Q(SMC2StreamState* state, int enable, float max_scale);

/**
 * @brief Manually trigger rejuvenation
 * 
 * Useful for testing or when external signal indicates regime change.
 */
void smc2_stream_force_rejuvenate(SMC2StreamState* state);

/*═══════════════════════════════════════════════════════════════════════════════
 * CHECKPOINT/RESTORE API
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get checkpoint size in bytes
 */
size_t smc2_stream_checkpoint_size(SMC2StreamState* state);

/**
 * @brief Save state to checkpoint buffer
 * @param buffer  Output buffer (must be at least checkpoint_size bytes)
 * @return Bytes written
 */
size_t smc2_stream_save_checkpoint(SMC2StreamState* state, void* buffer);

/**
 * @brief Restore state from checkpoint buffer
 * @param buffer  Input buffer from save_checkpoint
 * @return 0 on success, -1 on failure
 */
int smc2_stream_load_checkpoint(SMC2StreamState* state, const void* buffer);

#ifdef __cplusplus
}
#endif

#endif /* SMC2_STREAMING_CUH */
