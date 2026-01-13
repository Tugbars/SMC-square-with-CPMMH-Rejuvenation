/**
 * @file smc2_rbpf.h
 * @brief SMC² Parameter Learning with RBPF Inner Filter
 *
 * Architecture:
 *   Outer SMC: θ-particles (8 parameters)
 *   Inner RBPF: z sampled, h marginalized via Kalman with OCSN mixture
 *
 * This is designed to be equivalent to cPMMH but more parallelizable.
 */

#ifndef SMC2_RBPF_H
#define SMC2_RBPF_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * OCSN CONSTANTS (Kim, Shephard, Chib 1998)
 *═══════════════════════════════════════════════════════════════════════════*/

#define OCSN_K 10

/* Exposed for potential CUDA constant memory */
extern const float OCSN_WEIGHTS[OCSN_K];
extern const float OCSN_MEANS[OCSN_K];
extern const float OCSN_VARS[OCSN_K];      /* v_k (variances, not inverse) */
extern const float OCSN_LOG_WEIGHTS[OCSN_K];

/*═══════════════════════════════════════════════════════════════════════════
 * MODEL STRUCTURES
 *═══════════════════════════════════════════════════════════════════════════*/

/* Curve: f(z) = base + scale * (1 - exp(-rate * z)) */
typedef struct {
    float base;
    float scale;
    float rate;
} SVCurve;

/* Full parameter set (8 learned + 1 fixed curve) */
typedef struct {
    /* z-dynamics (2 params) */
    float rho;          /* AR coefficient for z */
    float sigma_z;      /* Innovation std for z */
    
    /* h-dynamics curves (6 params) */
    SVCurve mu_curve;   /* μ(z): long-run mean of h */
    SVCurve sigma_curve;/* σ(z): innovation std of h */
    
    /* Fixed (not learned) */
    SVCurve theta_curve;/* θ(z): mean-reversion speed */
    
    /* Bounds for z */
    float z_floor;
    float z_ceil;
} SVParams;

/* Prior specification */
typedef struct {
    /* Means */
    float rho_mean, sigma_z_mean;
    float mu_base_mean, mu_scale_mean, mu_rate_mean;
    float sigma_base_mean, sigma_scale_mean, sigma_rate_mean;
    
    /* Standard deviations */
    float rho_std, sigma_z_std;
    float mu_base_std, mu_scale_std, mu_rate_std;
    float sigma_base_std, sigma_scale_std, sigma_rate_std;
} SVPrior;

/* Bounds for parameter validity */
typedef struct {
    float rho_min, rho_max;
    float sigma_z_min, sigma_z_max;
    float mu_base_min, mu_base_max;
    float mu_scale_min, mu_scale_max;
    float mu_rate_min, mu_rate_max;
    float sigma_base_min, sigma_base_max;
    float sigma_scale_min, sigma_scale_max;
    float sigma_rate_min, sigma_rate_max;
} SVBounds;

/*═══════════════════════════════════════════════════════════════════════════
 * RBPF INNER FILTER
 *═══════════════════════════════════════════════════════════════════════════*/

/* Single RBPF particle: z sampled, h marginalized */
typedef struct {
    float z;        /* Sampled z value */
    float mu_h;     /* Kalman mean of h */
    float var_h;    /* Kalman variance of h */
} RBPFParticle;

/* Inner RBPF state for one θ-particle */
typedef struct {
    RBPFParticle* particles;    /* [N_inner] */
    RBPFParticle* temp_particles; /* [N_inner] scratch for resampling */
    float* weights;             /* [N_inner] normalized */
    float* log_weights;         /* [N_inner] unnormalized */
    int* ancestors;             /* [N_inner] for resampling */
    float* cumsum;              /* [N_inner] CDF for resampling */
    
    int N;                      /* Number of inner particles */
    float ess_threshold;        /* Resample when ESS < this * N (0.3-0.7) */
    float log_likelihood;       /* Accumulated log p(y_{1:t}|θ) */
    float ess;                  /* Current ESS */
} RBPFState;

/* Allocate/free inner RBPF */
RBPFState* rbpf_alloc(int N_inner);
void rbpf_free(RBPFState* state);

/* Initialize from stationary distribution */
void rbpf_init_stationary(RBPFState* state, const SVParams* params, uint64_t* rng);

/* Run one timestep, return log-likelihood increment */
float rbpf_step(RBPFState* state, float y_obs, const SVParams* params, uint64_t* rng);

/* Deep copy (for rejuvenation) */
void rbpf_copy(RBPFState* dst, const RBPFState* src);

/*═══════════════════════════════════════════════════════════════════════════
 * SMC² OUTER LAYER
 *═══════════════════════════════════════════════════════════════════════════*/

/* θ-particle: parameters + inner RBPF state */
typedef struct {
    SVParams params;            /* Current parameter values */
    RBPFState* inner;           /* Inner RBPF filter state */
    float log_weight;           /* Outer SMC weight (unnormalized) */
    float weight;               /* Normalized weight */
    float log_likelihood;       /* Accumulated log p(y|θ) - diagnostic only */
    uint64_t rng_state;         /* Persistent per-particle RNG state */
} ThetaParticle;

/* SMC² state */
typedef struct {
    ThetaParticle* theta_particles; /* [N_theta] */
    int N_theta;                    /* Number of θ-particles */
    int N_inner;                    /* Number of inner particles per θ */
    
    /* Scratch for rejuvenation */
    RBPFState* inner_scratch;       /* Temporary RBPF state */
    
    /* Observation history (for proper PMMH rejuvenation) */
    float* y_history;               /* [T_max] observations seen so far */
    int y_history_len;              /* Current length of history */
    int y_history_capacity;         /* Allocated capacity */
    
    /* Configuration */
    float ess_threshold_outer;      /* Resample θ when ESS < this * N_theta */
    float ess_threshold_inner;      /* Resample inner when ESS < this * N_inner */
    int K_rejuv;                    /* PMMH moves per rejuvenation */
    
    /* Prior and bounds */
    SVPrior prior;
    SVBounds bounds;
    SVCurve theta_curve;            /* Fixed curve */
    
    /* Proposal standard deviations */
    float proposal_std[8];
    
    /* Diagnostics */
    int n_resamples;
    int n_rejuv_accepts;
    int n_rejuv_total;
    float* ess_history;             /* [T] if allocated */
    int ess_history_len;
    
    /* RNG */
    uint64_t rng_state;
} SMC2State;

/* Configuration */
typedef struct {
    int N_theta;                /* Number of θ-particles (256-512) */
    int N_inner;                /* Number of inner particles (256-512) */
    float ess_threshold_outer;  /* Outer resample threshold (0.3-0.7) */
    float ess_threshold_inner;  /* Inner resample threshold (0.3-0.7) */
    int K_rejuv;                /* Rejuvenation moves (2-5) */
    uint64_t seed;              /* RNG seed */
} SMC2Config;

/* Default configuration */
SMC2Config smc2_config_defaults(void);

/* Allocate/free SMC² state */
SMC2State* smc2_alloc(const SMC2Config* cfg);
void smc2_free(SMC2State* state);

/* Set prior and bounds */
void smc2_set_prior(SMC2State* state, const SVPrior* prior);
void smc2_set_bounds(SMC2State* state, const SVBounds* bounds);
void smc2_set_theta_curve(SMC2State* state, const SVCurve* theta_curve);
void smc2_set_proposal_std(SMC2State* state, const float* std);

/* Initialize θ-particles from prior */
void smc2_init_from_prior(SMC2State* state);

/* Process one observation, return marginal log-likelihood estimate */
float smc2_update(SMC2State* state, float y_obs);

/* Run full SMC² over observation sequence */
typedef struct {
    float log_marginal_likelihood;  /* log p(y_{1:T}) */
    float* theta_mean;              /* [8] posterior mean */
    float* theta_std;               /* [8] posterior std */
    int n_resamples;
    float acceptance_rate;
    double elapsed_ms;
} SMC2Result;

SMC2Result smc2_run(SMC2State* state, const float* observations, int T);

/* Extract posterior statistics */
void smc2_get_theta_mean(const SMC2State* state, float* mean);
void smc2_get_theta_std(const SMC2State* state, float* std);

/*═══════════════════════════════════════════════════════════════════════════
 * UTILITY FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

/* Fast xorshift RNG */
uint64_t xorshift64(uint64_t* state);
float rand_uniform(uint64_t* state);
float rand_normal(uint64_t* state);

/* Curve evaluation */
float eval_curve(const SVCurve* curve, float z);

/* Log-prior evaluation */
float smc2_log_prior(const SVParams* params, const SVPrior* prior, const SVBounds* bounds);

/* Default prior/bounds */
SVPrior smc2_default_prior(void);
SVBounds smc2_default_bounds(void);

#ifdef __cplusplus
}
#endif

#endif /* SMC2_RBPF_H */
