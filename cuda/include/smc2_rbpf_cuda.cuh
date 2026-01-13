/**
 * @file smc2_rbpf_cuda.cuh
 * @brief SMC² with RBPF Inner Filter - CUDA Implementation
 *
 * Architecture:
 *   - 1 CUDA block = 1 θ-particle
 *   - 1 thread = 1 inner RBPF particle
 *   - Block-local resampling via shared memory
 *   - Warp-level reductions for log-sum-exp
 *
 * Memory Layout:
 *   - θ-particles in global memory (SoA)
 *   - Inner particles in shared memory during step
 *   - Observation history in global memory (for PMMH)
 */

#ifndef SMC2_RBPF_CUDA_CUH
#define SMC2_RBPF_CUDA_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

/*═══════════════════════════════════════════════════════════════════════════
 * Configuration Constants
 *═══════════════════════════════════════════════════════════════════════════*/

#define CUDA_N_THETA      256    /* θ-particles (CUDA blocks) */
#define CUDA_N_INNER      256    /* Inner particles per θ (threads per block) */
#define CUDA_WARP_SIZE    32
#define CUDA_N_WARPS      (CUDA_N_INNER / CUDA_WARP_SIZE)

/* OCSN mixture constants (K=10 components) */
#define OCSN_K 10
#define OCSN_OFFSET 3.5f

/*═══════════════════════════════════════════════════════════════════════════
 * OCSN Constants (in constant memory for fast broadcast)
 *═══════════════════════════════════════════════════════════════════════════*/

/* Declared in .cu file */
extern __constant__ float d_OCSN_WEIGHTS[OCSN_K];
extern __constant__ float d_OCSN_MEANS[OCSN_K];
extern __constant__ float d_OCSN_VARS[OCSN_K];
extern __constant__ float d_OCSN_LOG_WEIGHTS[OCSN_K];

/*═══════════════════════════════════════════════════════════════════════════
 * Parameter Structures
 *═══════════════════════════════════════════════════════════════════════════*/

/* Curve parameterization: f(z) = base + scale * (1 - exp(-rate * z)) */
struct SVCurve {
    float base;
    float scale;
    float rate;
};

/* Full SV model parameters (per θ-particle) */
struct SVParams {
    float rho;              /* z-process AR coefficient */
    float sigma_z;          /* z-process innovation std */
    SVCurve mu_curve;       /* h mean-reversion target */
    SVCurve sigma_curve;    /* h innovation std */
    SVCurve theta_curve;    /* h mean-reversion speed (fixed) */
    float z_floor;          /* z lower bound */
    float z_ceil;           /* z upper bound */
};

/* Prior specification */
struct SVPrior {
    float rho_mean, rho_std;
    float sigma_z_mean, sigma_z_std;
    float mu_base_mean, mu_base_std;
    float mu_scale_mean, mu_scale_std;
    float mu_rate_mean, mu_rate_std;
    float sigma_base_mean, sigma_base_std;
    float sigma_scale_mean, sigma_scale_std;
    float sigma_rate_mean, sigma_rate_std;
};

/* Parameter bounds */
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

/*═══════════════════════════════════════════════════════════════════════════
 * Inner RBPF Particle (one per thread)
 *═══════════════════════════════════════════════════════════════════════════*/

struct RBPFParticle {
    float z;        /* Latent regime variable (sampled) */
    float mu_h;     /* Kalman posterior mean for h */
    float var_h;    /* Kalman posterior variance for h */
};

/*═══════════════════════════════════════════════════════════════════════════
 * θ-particle State (SoA layout in global memory)
 *═══════════════════════════════════════════════════════════════════════════*/

struct ThetaParticlesSoA {
    /* Parameters [N_theta] */
    float* rho;
    float* sigma_z;
    float* mu_base;
    float* mu_scale;
    float* mu_rate;
    float* sigma_base;
    float* sigma_scale;
    float* sigma_rate;
    
    /* Inner RBPF particles [N_theta * N_inner] */
    float* inner_z;         /* z values */
    float* inner_mu_h;      /* h posterior means */
    float* inner_var_h;     /* h posterior variances */
    float* inner_weights;   /* Normalized weights */
    
    /* Per θ-particle state [N_theta] */
    float* log_weight;      /* Outer SMC weight (unnormalized) */
    float* weight;          /* Normalized weight */
    float* log_likelihood;  /* Accumulated log p(y_{1:t}|θ) */
    float* ess_inner;       /* Inner ESS */
    
    /* RNG states [N_theta * N_inner] - one per thread */
    curandState* rng_states;
};

/*═══════════════════════════════════════════════════════════════════════════
 * SMC² State (host-side management)
 *═══════════════════════════════════════════════════════════════════════════*/

struct SMC2StateCUDA {
    /* Device arrays */
    ThetaParticlesSoA d_particles;
    
    /* Observation history (device) */
    float* d_y_history;
    int y_history_len;
    int y_history_capacity;
    
    /* Scratch space for resampling */
    float* d_cumsum;        /* [N_theta] */
    int* d_ancestors;       /* [N_theta] */
    float* d_uniform;       /* [1] for systematic resampling */
    
    /* Configuration */
    int N_theta;
    int N_inner;
    float ess_threshold_outer;
    float ess_threshold_inner;
    int K_rejuv;
    
    /* Prior/bounds (host copies, transferred to constant memory) */
    SVPrior prior;
    SVBounds bounds;
    SVCurve theta_curve;    /* Fixed curve */
    float proposal_std[8];
    
    /* Diagnostics (host) */
    int n_resamples;
    int n_rejuv_accepts;
    int n_rejuv_total;
};

/*═══════════════════════════════════════════════════════════════════════════
 * API Functions
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef __cplusplus
extern "C" {
#endif

/* Initialization */
SMC2StateCUDA* smc2_cuda_alloc(int N_theta, int N_inner);
void smc2_cuda_free(SMC2StateCUDA* state);
void smc2_cuda_init_from_prior(SMC2StateCUDA* state);
void smc2_cuda_set_prior(SMC2StateCUDA* state, const SVPrior* prior);
void smc2_cuda_set_bounds(SMC2StateCUDA* state, const SVBounds* bounds);
void smc2_cuda_set_theta_curve(SMC2StateCUDA* state, const SVCurve* curve);

/* Main algorithm */
float smc2_cuda_update(SMC2StateCUDA* state, float y_obs);
void smc2_cuda_run(SMC2StateCUDA* state, const float* observations, int T);

/* Posterior extraction */
void smc2_cuda_get_theta_mean(SMC2StateCUDA* state, float* theta_mean);
void smc2_cuda_get_theta_std(SMC2StateCUDA* state, float* theta_std);
float smc2_cuda_get_outer_ess(SMC2StateCUDA* state);

#ifdef __cplusplus
}
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * Device Helper Functions (inline for performance)
 *═══════════════════════════════════════════════════════════════════════════*/

/* Curve evaluation */
__device__ __forceinline__ float eval_curve(float base, float scale, float rate, float z) {
    return base + scale * (1.0f - __expf(-rate * z));
}

/* Clamp value to range */
__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

/* Warp-level reduction for max */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = CUDA_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/* Warp-level reduction for sum */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = CUDA_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/* Block-level reduction for max (assumes N_INNER threads) */
__device__ float block_reduce_max(float val, float* shared_data);

/* Block-level reduction for sum */
__device__ float block_reduce_sum(float val, float* shared_data);

#endif /* SMC2_RBPF_CUDA_CUH */
