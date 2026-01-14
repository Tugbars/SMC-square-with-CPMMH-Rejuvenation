/**
 * @file smc2_rbpf.c
 * @brief SMC² Parameter Learning with RBPF Inner Filter
 *
 * Implementation of:
 *   - OCSN mixture likelihood and Kalman update with moment matching
 *   - RBPF inner filter (z sampled, h marginalized)
 *   - SMC² outer layer with rejuvenation
 *
 * Follows the corrected math from supervisor feedback.
 */

#include "smc2_rbpf.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * OCSN CONSTANTS (Kim, Shephard, Chib 1998)
 *═══════════════════════════════════════════════════════════════════════════*/

const float OCSN_WEIGHTS[OCSN_K] = {
    0.00609f, 0.04775f, 0.13057f, 0.20674f, 0.22715f,
    0.18842f, 0.12047f, 0.05591f, 0.01575f, 0.00115f
};

/* KSC Table 4 means - used directly, offset applied in likelihood calc */
const float OCSN_MEANS[OCSN_K] = {
    -10.12999f, -3.97281f, -0.57354f, 1.22474f, 2.58590f,
    3.72372f, 4.73732f, 5.69446f, 6.63386f, 8.06767f
};

/* Offset: OCSN weighted mean ≈ 2.22, E[log(χ²_1)] ≈ -1.27
 * The OCSN constants approximate some shifted version of log(χ²_1)
 * Required offset = OCSN_mean - E[log(χ²_1)] ≈ 2.22 - (-1.27) ≈ 3.49
 * 
 * Empirically validated: offset=3.5 gives correct h tracking and 
 * parameter recovery. offset=1.2704 causes systematic bias. */
#define OCSN_OFFSET 3.5f

/* VARIANCES (not inverse!) - computed from KSC Table 4 */
const float OCSN_VARS[OCSN_K] = {
    5.79596f, 2.61369f, 1.59719f, 1.19063f, 1.01300f,
    0.89981f, 0.81661f, 0.74999f, 0.69431f, 0.64015f
};

const float OCSN_LOG_WEIGHTS[OCSN_K] = {
    -5.10114f, -3.04139f, -2.03591f, -1.57665f, -1.48176f,
    -1.66932f, -2.11689f, -2.88456f, -4.15059f, -6.76773f
};

/*═══════════════════════════════════════════════════════════════════════════
 * UTILITY FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* SplitMix64 for generating independent RNG seeds */
static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

float rand_uniform(uint64_t* state) {
    return (float)(xorshift64(state) >> 11) * (1.0f / 9007199254740992.0f);
}

float rand_normal(uint64_t* state) {
    /* Box-Muller transform */
    float u1 = rand_uniform(state);
    float u2 = rand_uniform(state);
    /* Avoid log(0) */
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

float eval_curve(const SVCurve* curve, float z) {
    return curve->base + curve->scale * (1.0f - expf(-curve->rate * z));
}

static inline float clampf(float x, float lo, float hi) {
    return fmaxf(lo, fminf(hi, x));
}

/*═══════════════════════════════════════════════════════════════════════════
 * OCSN RBPF UPDATE (Corrected Math)
 *
 * Given: y_t observation, (μ_pred, v_pred) Kalman predictive moments
 * Returns: (μ_post, v_post) via mixture moment-matching, log p(y_t|μ_pred, v_pred)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    float mu_post;
    float var_post;
    float log_likelihood;
} OCSNUpdateResult;

static OCSNUpdateResult ocsn_kalman_update(float y, float mu_pred, float var_pred) {
    OCSNUpdateResult result;
    
    float log_alpha_tilde[OCSN_K];
    float S[OCSN_K];          /* Predictive variance per component */
    float K[OCSN_K];          /* Kalman gain per component */
    float mu_k[OCSN_K];       /* Posterior mean per component */
    float var_k_post[OCSN_K]; /* Posterior variance per component */
    
    /*═══════════════════════════════════════════════════════════════════
     * Step 1: Compute unnormalized log responsibilities
     * log α̃_k = log w_k - 0.5*log(S_k) - 0.5*(y - μ_pred - m_k + offset)²/S_k
     * 
     * The offset accounts for E[log(χ²_1)] = -1.2704
     * OCSN mixture has weighted mean ~2.22, so we adjust by +1.2704
     *═══════════════════════════════════════════════════════════════════*/
    float log_max = -1e30f;
    
    for (int k = 0; k < OCSN_K; k++) {
        /* Predictive variance for component k: v_pred + v_k */
        S[k] = var_pred + OCSN_VARS[k];
        
        /* Innovation: y - (μ_pred + m_k - offset) = y - μ_pred - m_k + offset */
        float innov = y - mu_pred - OCSN_MEANS[k] + OCSN_OFFSET;
        
        /* Log unnormalized responsibility */
        log_alpha_tilde[k] = OCSN_LOG_WEIGHTS[k]
                           - 0.5f * logf(S[k])
                           - 0.5f * innov * innov / S[k];
        
        if (log_alpha_tilde[k] > log_max) {
            log_max = log_alpha_tilde[k];
        }
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * Step 2: Normalize responsibilities via log-sum-exp
     *═══════════════════════════════════════════════════════════════════*/
    float sum_exp = 0.0f;
    for (int k = 0; k < OCSN_K; k++) {
        sum_exp += expf(log_alpha_tilde[k] - log_max);
    }
    float log_norm = log_max + logf(sum_exp);
    
    float alpha[OCSN_K];
    for (int k = 0; k < OCSN_K; k++) {
        alpha[k] = expf(log_alpha_tilde[k] - log_norm);
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * Step 3: Per-component Kalman updates
     *   K_k = v_pred / S_k
     *   μ_k = μ_pred + K_k * (y - μ_pred - m_k + offset)
     *   v_k_post = (1 - K_k) * v_pred
     *═══════════════════════════════════════════════════════════════════*/
    for (int k = 0; k < OCSN_K; k++) {
        /* Kalman gain */
        K[k] = var_pred / S[k];
        
        /* Innovation (same as step 1, with offset) */
        float innov = y - mu_pred - OCSN_MEANS[k] + OCSN_OFFSET;
        
        /* Posterior mean for component k */
        mu_k[k] = mu_pred + K[k] * innov;
        
        /* Posterior variance for component k */
        var_k_post[k] = (1.0f - K[k]) * var_pred;
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * Step 4: Moment matching across components
     *   μ_post = Σ α_k μ_k
     *   v_post = Σ α_k (v_k_post + μ_k²) - μ_post²
     *═══════════════════════════════════════════════════════════════════*/
    float mu_post = 0.0f;
    float E_h_sq = 0.0f;
    
    for (int k = 0; k < OCSN_K; k++) {
        mu_post += alpha[k] * mu_k[k];
        E_h_sq += alpha[k] * (var_k_post[k] + mu_k[k] * mu_k[k]);
    }
    
    float var_post = E_h_sq - mu_post * mu_post;
    
    /* Floor variance to prevent collapse */
    var_post = fmaxf(var_post, 1e-6f);
    
    /*═══════════════════════════════════════════════════════════════════
     * Step 5: Return results
     * Log likelihood WITHOUT Gaussian constant (cancels in ESS/MH)
     *═══════════════════════════════════════════════════════════════════*/
    result.mu_post = mu_post;
    result.var_post = var_post;
    result.log_likelihood = log_norm;  /* No -0.5*log(2π), saves FLOPs */
    
    return result;
}

/*═══════════════════════════════════════════════════════════════════════════
 * RBPF INNER FILTER
 *═══════════════════════════════════════════════════════════════════════════*/

RBPFState* rbpf_alloc(int N_inner) {
    RBPFState* state = (RBPFState*)calloc(1, sizeof(RBPFState));
    if (!state) return NULL;
    
    state->N = N_inner;
    state->ess_threshold = 0.5f;  /* Default, can be overridden */
    state->particles = (RBPFParticle*)calloc(N_inner, sizeof(RBPFParticle));
    state->temp_particles = (RBPFParticle*)calloc(N_inner, sizeof(RBPFParticle));
    state->weights = (float*)calloc(N_inner, sizeof(float));
    state->log_weights = (float*)calloc(N_inner, sizeof(float));
    state->ancestors = (int*)calloc(N_inner, sizeof(int));
    state->cumsum = (float*)calloc(N_inner, sizeof(float));
    
    if (!state->particles || !state->temp_particles || !state->weights || 
        !state->log_weights || !state->ancestors || !state->cumsum) {
        rbpf_free(state);
        return NULL;
    }
    
    return state;
}

void rbpf_free(RBPFState* state) {
    if (!state) return;
    free(state->particles);
    free(state->temp_particles);
    free(state->weights);
    free(state->log_weights);
    free(state->ancestors);
    free(state->cumsum);
    free(state);
}

void rbpf_copy(RBPFState* dst, const RBPFState* src) {
    if (!dst || !src || dst->N != src->N) return;
    
    memcpy(dst->particles, src->particles, src->N * sizeof(RBPFParticle));
    memcpy(dst->weights, src->weights, src->N * sizeof(float));
    memcpy(dst->log_weights, src->log_weights, src->N * sizeof(float));
    dst->log_likelihood = src->log_likelihood;
    dst->ess = src->ess;
}

void rbpf_init_stationary(RBPFState* state, const SVParams* params, uint64_t* rng) {
    const int N = state->N;
    
    /* z stationary distribution: N(z_floor, σ_z²/(1-ρ²)) truncated to [z_floor, z_ceil] */
    float one_minus_rho_sq = 1.0f - params->rho * params->rho;
    if (one_minus_rho_sq < 1e-6f) one_minus_rho_sq = 1e-6f;
    float z_stat_std = params->sigma_z / sqrtf(one_minus_rho_sq);
    
    for (int n = 0; n < N; n++) {
        /* Sample z from stationary */
        float z = params->z_floor + z_stat_std * rand_normal(rng);
        z = clampf(z, params->z_floor, params->z_ceil);
        
        /* Get h stationary parameters at this z */
        float theta_z = eval_curve(&params->theta_curve, z);
        float mu_z = eval_curve(&params->mu_curve, z);
        float sigma_z = eval_curve(&params->sigma_curve, z);
        
        float phi = 1.0f - theta_z;
        float one_minus_phi_sq = 1.0f - phi * phi;
        if (one_minus_phi_sq < 1e-6f) one_minus_phi_sq = 1e-6f;
        float h_stat_var = (sigma_z * sigma_z) / one_minus_phi_sq;
        
        /* Initialize particle */
        state->particles[n].z = z;
        state->particles[n].mu_h = mu_z;  /* Start at stationary mean */
        state->particles[n].var_h = h_stat_var;
        
        /* Uniform weights initially */
        state->weights[n] = 1.0f / N;
    }
    
    state->log_likelihood = 0.0f;
    state->ess = (float)N;
}

/* Systematic resampling */
static void systematic_resample(RBPFState* state, uint64_t* rng) {
    const int N = state->N;
    
    /* Build CDF */
    state->cumsum[0] = state->weights[0];
    for (int i = 1; i < N; i++) {
        state->cumsum[i] = state->cumsum[i-1] + state->weights[i];
    }
    
    /* Normalize CDF */
    float total = state->cumsum[N-1];
    if (total > 0.0f) {
        float inv_total = 1.0f / total;
        for (int i = 0; i < N; i++) {
            state->cumsum[i] *= inv_total;
        }
    }
    state->cumsum[N-1] = 1.0f;
    
    /* Systematic sampling */
    float u0 = rand_uniform(rng);
    float inv_N = 1.0f / (float)N;
    int idx = 0;
    
    for (int i = 0; i < N; i++) {
        float u = (u0 + (float)i) * inv_N;
        while (idx < N - 1 && state->cumsum[idx] < u) {
            idx++;
        }
        state->ancestors[i] = idx;
    }
}

float rbpf_step(RBPFState* state, float y_obs, const SVParams* params, uint64_t* rng) {
    const int N = state->N;
    
    /* Check if resampling needed (configurable ESS threshold) */
    if (state->ess < state->ess_threshold * N) {
        systematic_resample(state, rng);
        
        /* Apply ancestor indices using pre-allocated temp buffer */
        for (int n = 0; n < N; n++) {
            state->temp_particles[n] = state->particles[state->ancestors[n]];
        }
        memcpy(state->particles, state->temp_particles, N * sizeof(RBPFParticle));
        
        /* Reset weights */
        for (int n = 0; n < N; n++) {
            state->weights[n] = 1.0f / N;
        }
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * Pass 1: Propagate particles and find max log-weight (branch-free)
     *═══════════════════════════════════════════════════════════════════════*/
    float log_max = -1e30f;
    
    for (int n = 0; n < N; n++) {
        RBPFParticle* p = &state->particles[n];
        
        /* 1. Propagate z (sampled, OU dynamics) */
        float z_old = p->z;
        float z_mean = params->rho * (z_old - params->z_floor) + params->z_floor;
        float z_new = z_mean + params->sigma_z * rand_normal(rng);
        z_new = clampf(z_new, params->z_floor, params->z_ceil);
        
        /* 2. Evaluate curves at z_new */
        float theta_z = eval_curve(&params->theta_curve, z_new);
        float mu_z = eval_curve(&params->mu_curve, z_new);
        float sigma_z = eval_curve(&params->sigma_curve, z_new);
        
        float phi = 1.0f - theta_z;
        
        /* 3. Kalman predict for h */
        float mu_pred = phi * p->mu_h + theta_z * mu_z;
        float var_pred = phi * phi * p->var_h + sigma_z * sigma_z;
        
        /* Floor var_pred to prevent collapse when theta_z → 1 */
        var_pred = fmaxf(var_pred, 1e-8f);
        
        /* 4. OCSN update (marginal likelihood + moment-matched posterior) */
        OCSNUpdateResult update = ocsn_kalman_update(y_obs, mu_pred, var_pred);
        
        /* 5. Store updated state */
        p->z = z_new;
        p->mu_h = update.mu_post;
        p->var_h = update.var_post;
        
        state->log_weights[n] = update.log_likelihood;
        
        /* Branch-free max: use fmaxf instead of if */
        log_max = fmaxf(log_max, update.log_likelihood);
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * Pass 2: Normalize weights (branch-free via masked contributions)
     *═══════════════════════════════════════════════════════════════════════*/
    
    /* Handle degenerate case: if log_max is very negative, all particles dead */
    /* Use a floor to avoid NaN/Inf in exp, then mask below */
    const float LOG_MIN = -1e20f;
    float safe_log_max = fmaxf(log_max, LOG_MIN);
    
    float sum_w = 0.0f;
    for (int n = 0; n < N; n++) {
        /* Compute weight, will be ~0 if log_weight << log_max */
        float log_w = state->log_weights[n];
        float w = expf(log_w - safe_log_max);
        state->weights[n] = w;
        sum_w += w;
    }
    
    /* Safe normalize: floor sum_w to avoid division by zero */
    float safe_sum = fmaxf(sum_w, 1e-30f);
    float inv_sum = 1.0f / safe_sum;
    float sum_w_sq = 0.0f;
    
    for (int n = 0; n < N; n++) {
        state->weights[n] *= inv_sum;
        sum_w_sq += state->weights[n] * state->weights[n];
    }
    
    /* Compute ESS: floor denominator */
    state->ess = 1.0f / fmaxf(sum_w_sq, 1e-30f);
    
    /* Log-likelihood increment: log(mean weight) = log(sum_w/N) + max 
     * If degenerate (sum_w ≈ 0), return very negative value */
    float alive = (sum_w > 1e-30f) ? 1.0f : 0.0f;
    float ll_increment = alive * (safe_log_max + logf(safe_sum) - logf((float)N)) 
                       + (1.0f - alive) * (-1e30f);
    
    state->log_likelihood += ll_increment;
    
    return ll_increment;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SMC² OUTER LAYER
 *═══════════════════════════════════════════════════════════════════════════*/

SMC2Config smc2_config_defaults(void) {
    SMC2Config cfg;
    cfg.N_theta = 256;
    cfg.N_inner = 256;
    cfg.ess_threshold_outer = 0.5f;  /* Resample θ when ESS < 0.5 * N_theta */
    cfg.ess_threshold_inner = 0.5f;  /* Resample inner when ESS < 0.5 * N_inner */
    cfg.K_rejuv = 5;
    cfg.seed = 12345678901234567ULL;
    return cfg;
}

SVPrior smc2_default_prior(void) {
    SVPrior prior;
    
    /* z-dynamics */
    prior.rho_mean = 0.95f;       prior.rho_std = 0.02f;
    prior.sigma_z_mean = 0.1f;    prior.sigma_z_std = 0.05f;
    
    /* μ(z) curve */
    prior.mu_base_mean = -1.0f;   prior.mu_base_std = 0.5f;
    prior.mu_scale_mean = 0.5f;   prior.mu_scale_std = 0.3f;
    prior.mu_rate_mean = 1.0f;    prior.mu_rate_std = 0.5f;
    
    /* σ(z) curve */
    prior.sigma_base_mean = 0.15f;  prior.sigma_base_std = 0.05f;
    prior.sigma_scale_mean = 0.1f;  prior.sigma_scale_std = 0.05f;
    prior.sigma_rate_mean = 1.0f;   prior.sigma_rate_std = 0.5f;
    
    return prior;
}

SVBounds smc2_default_bounds(void) {
    SVBounds bounds;
    
    bounds.rho_min = 0.8f;        bounds.rho_max = 0.999f;
    bounds.sigma_z_min = 0.01f;   bounds.sigma_z_max = 0.5f;
    
    bounds.mu_base_min = -3.0f;   bounds.mu_base_max = 1.0f;
    bounds.mu_scale_min = 0.0f;   bounds.mu_scale_max = 2.0f;
    bounds.mu_rate_min = 0.1f;    bounds.mu_rate_max = 5.0f;
    
    bounds.sigma_base_min = 0.01f;  bounds.sigma_base_max = 0.5f;
    bounds.sigma_scale_min = 0.0f;  bounds.sigma_scale_max = 0.5f;
    bounds.sigma_rate_min = 0.1f;   bounds.sigma_rate_max = 5.0f;
    
    return bounds;
}

SMC2State* smc2_alloc(const SMC2Config* cfg) {
    SMC2State* state = (SMC2State*)calloc(1, sizeof(SMC2State));
    if (!state) return NULL;
    
    state->N_theta = cfg->N_theta;
    state->N_inner = cfg->N_inner;
    state->ess_threshold_outer = cfg->ess_threshold_outer;
    state->ess_threshold_inner = cfg->ess_threshold_inner;
    state->K_rejuv = cfg->K_rejuv;
    state->rng_state = cfg->seed;
    
    /* Allocate θ-particles */
    state->theta_particles = (ThetaParticle*)calloc(cfg->N_theta, sizeof(ThetaParticle));
    if (!state->theta_particles) {
        smc2_free(state);
        return NULL;
    }
    
    /* Allocate inner RBPF for each θ-particle */
    for (int j = 0; j < cfg->N_theta; j++) {
        state->theta_particles[j].inner = rbpf_alloc(cfg->N_inner);
        if (!state->theta_particles[j].inner) {
            smc2_free(state);
            return NULL;
        }
        /* Set inner ESS threshold from config */
        state->theta_particles[j].inner->ess_threshold = cfg->ess_threshold_inner;
    }
    
    /* Scratch for rejuvenation */
    state->inner_scratch = rbpf_alloc(cfg->N_inner);
    if (!state->inner_scratch) {
        smc2_free(state);
        return NULL;
    }
    state->inner_scratch->ess_threshold = cfg->ess_threshold_inner;
    
    /* Observation history for proper PMMH (initial capacity 1000, grows as needed) */
    state->y_history_capacity = 1000;
    state->y_history = (float*)malloc(state->y_history_capacity * sizeof(float));
    if (!state->y_history) {
        smc2_free(state);
        return NULL;
    }
    state->y_history_len = 0;
    
    /* Set defaults */
    state->prior = smc2_default_prior();
    state->bounds = smc2_default_bounds();
    
    /* Default proposal std */
    state->proposal_std[0] = 0.01f;   /* rho */
    state->proposal_std[1] = 0.02f;   /* sigma_z */
    state->proposal_std[2] = 0.1f;    /* mu_base */
    state->proposal_std[3] = 0.1f;    /* mu_scale */
    state->proposal_std[4] = 0.2f;    /* mu_rate */
    state->proposal_std[5] = 0.02f;   /* sigma_base */
    state->proposal_std[6] = 0.02f;   /* sigma_scale */
    state->proposal_std[7] = 0.2f;    /* sigma_rate */
    
    /* Default theta curve (fixed, not learned) */
    state->theta_curve.base = 0.02f;
    state->theta_curve.scale = 0.08f;
    state->theta_curve.rate = 1.5f;
    
    return state;
}

void smc2_free(SMC2State* state) {
    if (!state) return;
    
    if (state->theta_particles) {
        for (int j = 0; j < state->N_theta; j++) {
            rbpf_free(state->theta_particles[j].inner);
        }
        free(state->theta_particles);
    }
    
    rbpf_free(state->inner_scratch);
    free(state->y_history);
    free(state->ess_history);
    free(state);
}

void smc2_set_prior(SMC2State* state, const SVPrior* prior) {
    state->prior = *prior;
}

void smc2_set_bounds(SMC2State* state, const SVBounds* bounds) {
    state->bounds = *bounds;
}

void smc2_set_theta_curve(SMC2State* state, const SVCurve* theta_curve) {
    state->theta_curve = *theta_curve;
}

void smc2_set_proposal_std(SMC2State* state, const float* std) {
    memcpy(state->proposal_std, std, 8 * sizeof(float));
}

float smc2_log_prior(const SVParams* params, const SVPrior* prior, const SVBounds* bounds) {
    /* Check bounds first */
    if (params->rho < bounds->rho_min || params->rho > bounds->rho_max) return -1e30f;
    if (params->sigma_z < bounds->sigma_z_min || params->sigma_z > bounds->sigma_z_max) return -1e30f;
    if (params->mu_curve.base < bounds->mu_base_min || params->mu_curve.base > bounds->mu_base_max) return -1e30f;
    if (params->mu_curve.scale < bounds->mu_scale_min || params->mu_curve.scale > bounds->mu_scale_max) return -1e30f;
    if (params->mu_curve.rate < bounds->mu_rate_min || params->mu_curve.rate > bounds->mu_rate_max) return -1e30f;
    if (params->sigma_curve.base < bounds->sigma_base_min || params->sigma_curve.base > bounds->sigma_base_max) return -1e30f;
    if (params->sigma_curve.scale < bounds->sigma_scale_min || params->sigma_curve.scale > bounds->sigma_scale_max) return -1e30f;
    if (params->sigma_curve.rate < bounds->sigma_rate_min || params->sigma_curve.rate > bounds->sigma_rate_max) return -1e30f;
    
    /* Gaussian log-prior (ignoring constants) */
    float lp = 0.0f;
    
    float d;
    d = (params->rho - prior->rho_mean) / prior->rho_std;
    lp -= 0.5f * d * d;
    
    d = (params->sigma_z - prior->sigma_z_mean) / prior->sigma_z_std;
    lp -= 0.5f * d * d;
    
    d = (params->mu_curve.base - prior->mu_base_mean) / prior->mu_base_std;
    lp -= 0.5f * d * d;
    
    d = (params->mu_curve.scale - prior->mu_scale_mean) / prior->mu_scale_std;
    lp -= 0.5f * d * d;
    
    d = (params->mu_curve.rate - prior->mu_rate_mean) / prior->mu_rate_std;
    lp -= 0.5f * d * d;
    
    d = (params->sigma_curve.base - prior->sigma_base_mean) / prior->sigma_base_std;
    lp -= 0.5f * d * d;
    
    d = (params->sigma_curve.scale - prior->sigma_scale_mean) / prior->sigma_scale_std;
    lp -= 0.5f * d * d;
    
    d = (params->sigma_curve.rate - prior->sigma_rate_mean) / prior->sigma_rate_std;
    lp -= 0.5f * d * d;
    
    return lp;
}

/* Helper: params to array */
static void params_to_array(const SVParams* params, float* arr) {
    arr[0] = params->rho;
    arr[1] = params->sigma_z;
    arr[2] = params->mu_curve.base;
    arr[3] = params->mu_curve.scale;
    arr[4] = params->mu_curve.rate;
    arr[5] = params->sigma_curve.base;
    arr[6] = params->sigma_curve.scale;
    arr[7] = params->sigma_curve.rate;
}

/* Helper: array to params */
static void array_to_params(const float* arr, SVParams* params) {
    params->rho = arr[0];
    params->sigma_z = arr[1];
    params->mu_curve.base = arr[2];
    params->mu_curve.scale = arr[3];
    params->mu_curve.rate = arr[4];
    params->sigma_curve.base = arr[5];
    params->sigma_curve.scale = arr[6];
    params->sigma_curve.rate = arr[7];
}

void smc2_init_from_prior(SMC2State* state) {
    const int N_theta = state->N_theta;
    
    for (int j = 0; j < N_theta; j++) {
        ThetaParticle* tp = &state->theta_particles[j];
        SVParams* p = &tp->params;
        
        /* Initialize per-particle RNG state using splitmix64 */
        tp->rng_state = splitmix64(&state->rng_state);
        
        /* Sample from prior (truncated by bounds) */
        int valid = 0;
        while (!valid) {
            p->rho = state->prior.rho_mean + state->prior.rho_std * rand_normal(&tp->rng_state);
            p->sigma_z = state->prior.sigma_z_mean + state->prior.sigma_z_std * rand_normal(&tp->rng_state);
            p->mu_curve.base = state->prior.mu_base_mean + state->prior.mu_base_std * rand_normal(&tp->rng_state);
            p->mu_curve.scale = state->prior.mu_scale_mean + state->prior.mu_scale_std * rand_normal(&tp->rng_state);
            p->mu_curve.rate = state->prior.mu_rate_mean + state->prior.mu_rate_std * rand_normal(&tp->rng_state);
            p->sigma_curve.base = state->prior.sigma_base_mean + state->prior.sigma_base_std * rand_normal(&tp->rng_state);
            p->sigma_curve.scale = state->prior.sigma_scale_mean + state->prior.sigma_scale_std * rand_normal(&tp->rng_state);
            p->sigma_curve.rate = state->prior.sigma_rate_mean + state->prior.sigma_rate_std * rand_normal(&tp->rng_state);
            
            float lp = smc2_log_prior(p, &state->prior, &state->bounds);
            valid = (lp > -1e20f);
        }
        
        /* Copy fixed curve and bounds */
        p->theta_curve = state->theta_curve;
        p->z_floor = 0.0f;
        p->z_ceil = 3.0f;
        
        /* Initialize inner RBPF using per-particle RNG */
        rbpf_init_stationary(tp->inner, p, &tp->rng_state);
        
        /* Initialize weights */
        tp->log_weight = 0.0f;
        tp->weight = 1.0f / N_theta;
        tp->log_likelihood = 0.0f;
    }
    
    state->n_resamples = 0;
    state->n_rejuv_accepts = 0;
    state->n_rejuv_total = 0;
}

/* Rejuvenate single θ-particle using PROPER PMMH
 * 
 * CRITICAL: Must re-run inner RBPF from t=0 to t_current to compute
 * the FULL accumulated likelihood p(y_{1:t}|θ'), not just incremental.
 * 
 * This is O(t) per MH move, which is the true cost of SMC².
 */
static void rejuvenate_theta(SMC2State* state, int j, 
                             const float* y_history, int t_current) {
    ThetaParticle* tp = &state->theta_particles[j];
    SVParams params_curr = tp->params;
    
    /* Use FULL accumulated likelihood, not incremental */
    float ll_accum_curr = tp->log_likelihood;
    float lp_curr = smc2_log_prior(&params_curr, &state->prior, &state->bounds);
    
    for (int k = 0; k < state->K_rejuv; k++) {
        state->n_rejuv_total++;
        
        /* Split RNG streams: one for proposal, one for likelihood */
        uint64_t rng_prop = splitmix64(&tp->rng_state);
        uint64_t rng_like = splitmix64(&tp->rng_state);
        
        /* 1. Propose θ' using proposal RNG stream */
        SVParams params_prop = params_curr;
        float theta_arr[8];
        params_to_array(&params_curr, theta_arr);
        
        for (int i = 0; i < 8; i++) {
            theta_arr[i] += state->proposal_std[i] * rand_normal(&rng_prop);
        }
        array_to_params(theta_arr, &params_prop);
        params_prop.theta_curve = state->theta_curve;
        params_prop.z_floor = 0.0f;
        params_prop.z_ceil = 3.0f;
        
        /* 2. Check prior */
        float lp_prop = smc2_log_prior(&params_prop, &state->prior, &state->bounds);
        if (lp_prop < -1e20f) continue;
        
        /* 3. Re-run inner RBPF from t=0 to t_current with θ' */
        rbpf_init_stationary(state->inner_scratch, &params_prop, &rng_like);
        float ll_accum_prop = 0.0f;
        
        for (int t = 0; t <= t_current; t++) {
            ll_accum_prop += rbpf_step(state->inner_scratch, y_history[t], 
                                       &params_prop, &rng_like);
        }
        
        /* 4. MH acceptance using ACCUMULATED likelihood (proper PMMH) */
        float log_alpha = (ll_accum_prop + lp_prop) - (ll_accum_curr + lp_curr);
        
        float u = rand_uniform(&rng_prop);
        if (logf(u) < log_alpha) {
            /* Accept: update particle state */
            params_curr = params_prop;
            rbpf_copy(tp->inner, state->inner_scratch);
            ll_accum_curr = ll_accum_prop;
            lp_curr = lp_prop;
            state->n_rejuv_accepts++;
        }
        /* RNG state already advanced by splitmix64 calls above */
    }
    
    tp->params = params_curr;
    tp->log_likelihood = ll_accum_curr;
}

float smc2_update(SMC2State* state, float y_obs) {
    const int N_theta = state->N_theta;
    
    /*═══════════════════════════════════════════════════════════════════
     * 0. Append observation to history (needed for proper PMMH)
     *═══════════════════════════════════════════════════════════════════*/
    if (state->y_history_len >= state->y_history_capacity) {
        /* Grow capacity */
        state->y_history_capacity *= 2;
        state->y_history = (float*)realloc(state->y_history, 
                                           state->y_history_capacity * sizeof(float));
    }
    state->y_history[state->y_history_len] = y_obs;
    int t_current = state->y_history_len;
    state->y_history_len++;
    
    /*═══════════════════════════════════════════════════════════════════
     * 1. Run inner RBPF step for each θ-particle
     *    Uses per-particle RNG states for reproducibility and parallelism
     *═══════════════════════════════════════════════════════════════════*/
    #pragma omp parallel for if(N_theta >= 64)
    for (int j = 0; j < N_theta; j++) {
        ThetaParticle* tp = &state->theta_particles[j];
        
        /* Use persistent per-particle RNG state */
        float ll_incr = rbpf_step(tp->inner, y_obs, &tp->params, &tp->rng_state);
        
        /* Update outer SMC weight with incremental likelihood */
        tp->log_weight += ll_incr;
        
        /* Accumulated likelihood (used in PMMH rejuvenation) */
        tp->log_likelihood += ll_incr;
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * 2. Normalize θ-weights, compute ESS
     *═══════════════════════════════════════════════════════════════════*/
    float log_max = -1e30f;
    for (int j = 0; j < N_theta; j++) {
        log_max = fmaxf(log_max, state->theta_particles[j].log_weight);
    }
    
    float sum_w = 0.0f;
    for (int j = 0; j < N_theta; j++) {
        float w = expf(state->theta_particles[j].log_weight - log_max);
        state->theta_particles[j].weight = w;
        sum_w += w;
    }
    
    float sum_w_sq = 0.0f;
    for (int j = 0; j < N_theta; j++) {
        state->theta_particles[j].weight /= sum_w;
        sum_w_sq += state->theta_particles[j].weight * state->theta_particles[j].weight;
    }
    
    float ess_theta = 1.0f / sum_w_sq;
    
    /*═══════════════════════════════════════════════════════════════════
     * 3. Resample + Rejuvenate if ESS low
     *═══════════════════════════════════════════════════════════════════*/
    if (ess_theta < state->ess_threshold_outer * N_theta) {
        state->n_resamples++;
        
        /* Build CDF */
        float* cumsum = (float*)malloc(N_theta * sizeof(float));
        cumsum[0] = state->theta_particles[0].weight;
        for (int j = 1; j < N_theta; j++) {
            cumsum[j] = cumsum[j-1] + state->theta_particles[j].weight;
        }
        cumsum[N_theta-1] = 1.0f;
        
        /* Systematic resampling */
        int* ancestors = (int*)malloc(N_theta * sizeof(int));
        float u0 = rand_uniform(&state->rng_state);
        float inv_N = 1.0f / (float)N_theta;
        int idx = 0;
        for (int j = 0; j < N_theta; j++) {
            float u = (u0 + (float)j) * inv_N;
            while (idx < N_theta - 1 && cumsum[idx] < u) {
                idx++;
            }
            ancestors[j] = idx;
        }
        
        /* Copy resampled particles to temp
         * CRITICAL: Copy RNG state with particle for genealogical consistency */
        ThetaParticle* temp = (ThetaParticle*)malloc(N_theta * sizeof(ThetaParticle));
        for (int j = 0; j < N_theta; j++) {
            int a = ancestors[j];
            temp[j].params = state->theta_particles[a].params;
            temp[j].inner = rbpf_alloc(state->N_inner);
            temp[j].inner->ess_threshold = state->ess_threshold_inner;
            rbpf_copy(temp[j].inner, state->theta_particles[a].inner);
            temp[j].log_likelihood = state->theta_particles[a].log_likelihood;
            temp[j].rng_state = state->theta_particles[a].rng_state;
        }
        
        /* Free old inner states, copy back */
        for (int j = 0; j < N_theta; j++) {
            rbpf_free(state->theta_particles[j].inner);
            state->theta_particles[j].params = temp[j].params;
            state->theta_particles[j].inner = temp[j].inner;
            state->theta_particles[j].log_likelihood = temp[j].log_likelihood;
            state->theta_particles[j].rng_state = temp[j].rng_state;
            state->theta_particles[j].log_weight = 0.0f;  /* Reset */
            state->theta_particles[j].weight = 1.0f / N_theta;
        }
        free(temp);
        free(ancestors);
        free(cumsum);
        
        /* Rejuvenation: PROPER PMMH moves per θ-particle
         * Each move re-runs the inner filter from t=0 to t_current: O(t) per move */
        for (int j = 0; j < N_theta; j++) {
            rejuvenate_theta(state, j, state->y_history, t_current);
        }
    }
    
    /* Return marginal likelihood estimate */
    return log_max + logf(sum_w);
}

void smc2_get_theta_mean(const SMC2State* state, float* mean) {
    memset(mean, 0, 8 * sizeof(float));
    
    for (int j = 0; j < state->N_theta; j++) {
        float w = state->theta_particles[j].weight;
        float arr[8];
        params_to_array(&state->theta_particles[j].params, arr);
        
        for (int i = 0; i < 8; i++) {
            mean[i] += w * arr[i];
        }
    }
}

void smc2_get_theta_std(const SMC2State* state, float* std) {
    float mean[8];
    smc2_get_theta_mean(state, mean);
    
    float var[8] = {0};
    for (int j = 0; j < state->N_theta; j++) {
        float w = state->theta_particles[j].weight;
        float arr[8];
        params_to_array(&state->theta_particles[j].params, arr);
        
        for (int i = 0; i < 8; i++) {
            float d = arr[i] - mean[i];
            var[i] += w * d * d;
        }
    }
    
    for (int i = 0; i < 8; i++) {
        std[i] = sqrtf(var[i]);
    }
}

SMC2Result smc2_run(SMC2State* state, const float* observations, int T) {
    SMC2Result result;
    memset(&result, 0, sizeof(result));
    
    result.theta_mean = (float*)calloc(8, sizeof(float));
    result.theta_std = (float*)calloc(8, sizeof(float));
    
    /* Allocate ESS history */
    state->ess_history = (float*)malloc(T * sizeof(float));
    state->ess_history_len = T;
    
    clock_t start = clock();
    
    /* Initialize */
    smc2_init_from_prior(state);
    
    /* Process observations */
    float log_ml = 0.0f;
    for (int t = 0; t < T; t++) {
        float ll = smc2_update(state, observations[t]);
        log_ml += ll;
        
        /* Store ESS */
        float sum_w_sq = 0.0f;
        for (int j = 0; j < state->N_theta; j++) {
            sum_w_sq += state->theta_particles[j].weight * state->theta_particles[j].weight;
        }
        state->ess_history[t] = 1.0f / sum_w_sq;
        
        /* Progress */
        if ((t + 1) % 100 == 0 || t == T - 1) {
            printf("  t=%d/%d, ESS_θ=%.1f, resamples=%d\n", 
                   t + 1, T, state->ess_history[t], state->n_resamples);
        }
    }
    
    clock_t end = clock();
    
    /* Extract results */
    result.log_marginal_likelihood = log_ml;
    smc2_get_theta_mean(state, result.theta_mean);
    smc2_get_theta_std(state, result.theta_std);
    result.n_resamples = state->n_resamples;
    result.acceptance_rate = (state->n_rejuv_total > 0) 
                           ? (float)state->n_rejuv_accepts / state->n_rejuv_total 
                           : 0.0f;
    result.elapsed_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    
    return result;
}
