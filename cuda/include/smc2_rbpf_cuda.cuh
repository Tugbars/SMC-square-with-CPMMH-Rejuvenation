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

/* OCSN mixture approximation for log χ²(1)
 * 
 * The stored d_OCSN_MEANS are from Kim-Shephard-Chib (1998) Table 4,
 * which directly approximates the RAW log χ²(1) distribution (mean ≈ -1.27).
 * 
 * Observation model: y = h + log(ε²) where ε ~ N(0,1), so log(ε²) ~ log χ²(1)
 * 
 * Innovation for component k: innov = y - h_pred - m_k
 * No offset needed since means are already for raw distribution.
 * 
 * NOTE: Omori et al. (2007) gives CENTERED means (mean = 0), which would
 * require OCSN_OFFSET = 1.27036. But we use KSC 1998 raw means, so offset = 0.
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
    float* inner_z;
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
};

/*═══════════════════════════════════════════════════════════════════════════
 * API
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef __cplusplus
extern "C" {
#endif

SMC2StateCUDA* smc2_cuda_alloc(int N_theta, int N_inner);
void smc2_cuda_free(SMC2StateCUDA* state);
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
