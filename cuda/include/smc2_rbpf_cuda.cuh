/**
 * @file smc2_rbpf_cuda.cuh
 * @brief SMC² with RBPF Inner Filter - CUDA Implementation
 */

#ifndef SMC2_RBPF_CUDA_CUH
#define SMC2_RBPF_CUDA_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <stdint.h>      /* For uint64_t */

/*═══════════════════════════════════════════════════════════════════════════
 * Configuration
 *═══════════════════════════════════════════════════════════════════════════*/

#define CUDA_N_THETA      256
#define CUDA_N_INNER      256
#define CUDA_WARP_SIZE    32
#define OCSN_K            10

/* PMMH Proposal Strategy (compile-time)
 * 
 * SMC2_BLOCKED_PMMH=0: Joint 8-parameter proposal (default)
 *   - 1 replay per rejuvenation
 *   - Lower acceptance (~10-20% for 8D)
 * 
 * SMC2_BLOCKED_PMMH=1: Blocked proposals
 *   - Block 1: (ρ, σ_z)                    — dynamics
 *   - Block 2: (μ_base, μ_scale, μ_rate)   — mean curve
 *   - Block 3: (σ_base, σ_scale, σ_rate)   — vol curve
 *   - 3 replays per rejuvenation
 *   - Higher acceptance per block (~60-70%)
 *   - Better mixing for strongly correlated posteriors
 * 
 * Build: nvcc -DSMC2_BLOCKED_PMMH=1 ... 
 */
#ifndef SMC2_BLOCKED_PMMH
#define SMC2_BLOCKED_PMMH 0
#endif

/* OCSN 10-component Gaussian mixture approximation to log χ²(1)
 * 
 * Source: Omori, Chib, Shephard & Nakajima (2007), Table 1
 * 
 * Weights, means, and variances are matched pairs approximating
 * RAW log χ²(1) (E ≈ -1.2704, Var ≈ π²/2 ≈ 4.93).
 * 
 * With OCSN_OFFSET = 0: innov = y - h_pred - m_k
 */
#define OCSN_OFFSET       0.0f

/*═══════════════════════════════════════════════════════════════════════════
 * Parameter Structures
 *═══════════════════════════════════════════════════════════════════════════*/

struct SVCurve {
    float base;
    float scale;
    float rate;
};

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
 * θ-particle SoA Layout
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
    
    /* Inner particles [N_theta * N_inner] */
    float* inner_z;      /* z̃ (unconstrained) - AR(1) state, transform via z = 1.5*(1+tanh(z̃)) */
    float* inner_mu_h;
    float* inner_var_h;
    float* inner_log_w;  /* Log weights for numerical stability */
    
    /* Per θ state [N_theta] */
    float* log_weight;
    float* weight;
    float* log_likelihood;
    float* ess_inner;
    
    /* RNG [N_theta * N_inner] */
    curandState* rng_states;
};

/*═══════════════════════════════════════════════════════════════════════════
 * SMC² State
 *═══════════════════════════════════════════════════════════════════════════*/

struct SMC2StateCUDA {
    ThetaParticlesSoA d_particles;
    ThetaParticlesSoA d_particles_temp;  /* For resampling swap */
    
    /* Observation history */
    float* d_y_history;
    int y_history_len;
    int y_history_capacity;
    int t_current;
    
    /* CPMMH: Ping-pong noise buffers for zero-copy swaps
     * FP16 storage for bandwidth reduction
     * u0 derived from z_noise via Φ(z) (no separate storage) */
    half* d_z_noise[2];     /* Ping-pong: [N_theta * N_inner * (T+1)] each */
    int noise_buf;          /* Current buffer index: 0 or 1 */
    int noise_capacity;     /* Max T for noise arrays */
    float cpmmh_rho;        /* Correlation: 0.99 typical */
    
    /* Host-side fast RNG for outer resampling (avoids curandGenerator overhead) */
    uint64_t host_rng_state;
    
    /* Scratch */
    int* d_ancestors;
    float* d_uniform;
    float* d_ess;
    int* d_accepts;
    int* d_swap_flags;  /* Per-particle accept flags for CPMMH */
    
    /* Config */
    int N_theta;
    int N_inner;
    float ess_threshold_outer;
    float ess_threshold_inner;
    int K_rejuv;
    
    /* Prior/bounds */
    SVPrior prior;
    SVBounds bounds;
    SVCurve theta_curve;
    float proposal_std[8];
    
    /* Diagnostics */
    int n_resamples;
    int n_rejuv_accepts;
    int n_rejuv_total;
    
    /* Reproducibility: user-provided seed (0 = use time-based seed) */
    uint64_t user_seed;
};

/*═══════════════════════════════════════════════════════════════════════════
 * API
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef __cplusplus
extern "C" {
#endif

SMC2StateCUDA* smc2_cuda_alloc(int N_theta, int N_inner);
void smc2_cuda_free(SMC2StateCUDA* state);
void smc2_cuda_set_seed(SMC2StateCUDA* state, uint64_t seed);  /* For reproducibility */
void smc2_cuda_set_noise_capacity(SMC2StateCUDA* state, int capacity);
void smc2_cuda_init_from_prior(SMC2StateCUDA* state);
float smc2_cuda_update(SMC2StateCUDA* state, float y_obs);
void smc2_cuda_get_theta_mean(SMC2StateCUDA* state, float* theta_mean);
void smc2_cuda_get_theta_std(SMC2StateCUDA* state, float* theta_std);
float smc2_cuda_get_outer_ess(SMC2StateCUDA* state);

#ifdef __cplusplus
}
#endif

#endif /* SMC2_RBPF_CUDA_CUH */
