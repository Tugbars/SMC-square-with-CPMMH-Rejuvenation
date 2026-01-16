/**
 * @file smc2_rbpf_cuda.cu
 * @brief SMC² with RBPF Inner Filter - Kernel Implementations
 * 
 * See smc2_rbpf_cuda.cuh for algorithm documentation.
 * This file contains kernel bodies and host API implementation.
 */

#include "smc2_rbpf_cuda.cuh"
#include "smc2_noise_precision.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * CUDA ERROR CHECKING
 *═══════════════════════════════════════════════════════════════════════════════*/

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/*═══════════════════════════════════════════════════════════════════════════════
 * OCSN CONSTANT MEMORY DEFINITIONS
 * 
 * 10-component Gaussian mixture approximation to log χ²(1).
 * Source: Omori, Chib, Shephard & Nakajima (2007), Table 1
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ __constant__ float d_OCSN_WEIGHTS[OCSN_K] = {
    0.00609f, 0.04775f, 0.13057f, 0.20674f, 0.22715f,
    0.18842f, 0.12047f, 0.05591f, 0.01575f, 0.00115f
};

__device__ __constant__ float d_OCSN_MEANS[OCSN_K] = {
    1.92677f,  1.34744f,  0.73504f,  0.02266f, -0.85173f,
   -1.97278f, -3.46788f, -5.55246f, -8.68384f, -14.65000f
};

__device__ __constant__ float d_OCSN_VARS[OCSN_K] = {
    0.11265f, 0.17788f, 0.26768f, 0.40611f, 0.62699f,
    0.98583f, 1.57469f, 2.54498f, 4.16591f, 7.33342f
};

__device__ __constant__ float d_OCSN_LOG_WEIGHTS[OCSN_K] = {
    -5.1011072f, -3.0417762f, -2.0358458f, -1.5762933f, -1.4821447f,
    -1.6690818f, -2.1163545f, -2.8840120f, -4.1509149f, -6.7679933f
};

__device__ __constant__ float d_OCSN_INV_VARS[OCSN_K] = {
    8.87705282f, 5.62176748f, 3.73580395f, 2.46238704f, 1.59492177f,
    1.01437367f, 0.63504563f, 0.39293040f, 0.24004359f, 0.13636202f
};

__device__ __constant__ float d_OCSN_LOG_VARS[OCSN_K] = {
    -2.18346961f, -1.72664611f, -1.31796304f, -0.90113122f, -0.46682469f,
    -0.01427135f,  0.45405843f,  0.93412279f,  1.42693474f,  1.99244198f
};

/*═══════════════════════════════════════════════════════════════════════════════
 * CONSTANT MEMORY
 * 
 * Model parameters copied once at init, accessed by all kernels.
 *═══════════════════════════════════════════════════════════════════════════════*/

__constant__ SVPrior  d_prior;
__constant__ SVBounds d_bounds;
__constant__ SVCurve  d_theta_curve;
__constant__ float    d_proposal_std[8];

/*═══════════════════════════════════════════════════════════════════════════════
 * LOG PRIOR EVALUATION
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ float log_prior_theta(
    float rho, float sigma_z,
    float mu_base, float mu_scale, float mu_rate,
    float sigma_base, float sigma_scale, float sigma_rate
) {
    /* Bounds check */
    if (rho < d_bounds.rho_min || rho > d_bounds.rho_max) return -INFINITY;
    if (sigma_z < d_bounds.sigma_z_min || sigma_z > d_bounds.sigma_z_max) return -INFINITY;
    if (mu_base < d_bounds.mu_base_min || mu_base > d_bounds.mu_base_max) return -INFINITY;
    if (mu_scale < d_bounds.mu_scale_min || mu_scale > d_bounds.mu_scale_max) return -INFINITY;
    if (mu_rate < d_bounds.mu_rate_min || mu_rate > d_bounds.mu_rate_max) return -INFINITY;
    if (sigma_base < d_bounds.sigma_base_min || sigma_base > d_bounds.sigma_base_max) return -INFINITY;
    if (sigma_scale < d_bounds.sigma_scale_min || sigma_scale > d_bounds.sigma_scale_max) return -INFINITY;
    if (sigma_rate < d_bounds.sigma_rate_min || sigma_rate > d_bounds.sigma_rate_max) return -INFINITY;
    
    /* Gaussian log-prior (normalization constants cancel in MH ratio) */
    float lp = 0.0f;
    
    float d_rho = (rho - d_prior.rho_mean) / d_prior.rho_std;
    float d_sigma_z = (sigma_z - d_prior.sigma_z_mean) / d_prior.sigma_z_std;
    float d_mu_base = (mu_base - d_prior.mu_base_mean) / d_prior.mu_base_std;
    float d_mu_scale = (mu_scale - d_prior.mu_scale_mean) / d_prior.mu_scale_std;
    float d_mu_rate = (mu_rate - d_prior.mu_rate_mean) / d_prior.mu_rate_std;
    float d_sigma_base = (sigma_base - d_prior.sigma_base_mean) / d_prior.sigma_base_std;
    float d_sigma_scale = (sigma_scale - d_prior.sigma_scale_mean) / d_prior.sigma_scale_std;
    float d_sigma_rate = (sigma_rate - d_prior.sigma_rate_mean) / d_prior.sigma_rate_std;
    
    lp += -0.5f * (d_rho * d_rho + d_sigma_z * d_sigma_z +
                   d_mu_base * d_mu_base + d_mu_scale * d_mu_scale + d_mu_rate * d_mu_rate +
                   d_sigma_base * d_sigma_base + d_sigma_scale * d_sigma_scale + d_sigma_rate * d_sigma_rate);
    
    return lp;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: RNG Initialization
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_init_rng(curandState* states, unsigned long long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Initialize from Prior
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_init_from_prior(
    ThetaParticlesSoA particles,
    int N_theta, int N_inner,
    noise_t* d_z_noise,
    noise_t* d_u0_noise,
    int noise_capacity
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_inner + inner_idx;
    
    if (theta_idx >= N_theta) return;
    
    curandState* rng = &particles.rng_states[global_idx];
    
    /* Shared memory for θ parameters (sampled by thread 0) */
    __shared__ float s_rho, s_sigma_z;
    __shared__ float s_mu_base, s_mu_scale, s_mu_rate;
    __shared__ float s_sigma_base, s_sigma_scale, s_sigma_rate;
    
    /* Thread 0: sample θ from prior with rejection */
    if (inner_idx == 0) {
        int attempts = 0;
        int valid = 0;
        while (!valid && attempts < 1000) {
            s_rho = d_prior.rho_mean + d_prior.rho_std * curand_normal(rng);
            s_sigma_z = d_prior.sigma_z_mean + d_prior.sigma_z_std * curand_normal(rng);
            s_mu_base = d_prior.mu_base_mean + d_prior.mu_base_std * curand_normal(rng);
            s_mu_scale = d_prior.mu_scale_mean + d_prior.mu_scale_std * curand_normal(rng);
            s_mu_rate = d_prior.mu_rate_mean + d_prior.mu_rate_std * curand_normal(rng);
            s_sigma_base = d_prior.sigma_base_mean + d_prior.sigma_base_std * curand_normal(rng);
            s_sigma_scale = d_prior.sigma_scale_mean + d_prior.sigma_scale_std * curand_normal(rng);
            s_sigma_rate = d_prior.sigma_rate_mean + d_prior.sigma_rate_std * curand_normal(rng);
            
            valid = (s_rho >= d_bounds.rho_min && s_rho <= d_bounds.rho_max &&
                     s_sigma_z >= d_bounds.sigma_z_min && s_sigma_z <= d_bounds.sigma_z_max &&
                     s_mu_base >= d_bounds.mu_base_min && s_mu_base <= d_bounds.mu_base_max &&
                     s_mu_scale >= d_bounds.mu_scale_min && s_mu_scale <= d_bounds.mu_scale_max &&
                     s_mu_rate >= d_bounds.mu_rate_min && s_mu_rate <= d_bounds.mu_rate_max &&
                     s_sigma_base >= d_bounds.sigma_base_min && s_sigma_base <= d_bounds.sigma_base_max &&
                     s_sigma_scale >= d_bounds.sigma_scale_min && s_sigma_scale <= d_bounds.sigma_scale_max &&
                     s_sigma_rate >= d_bounds.sigma_rate_min && s_sigma_rate <= d_bounds.sigma_rate_max);
            attempts++;
        }
        
        /* Store θ parameters */
        particles.rho[theta_idx] = s_rho;
        particles.sigma_z[theta_idx] = s_sigma_z;
        particles.mu_base[theta_idx] = s_mu_base;
        particles.mu_scale[theta_idx] = s_mu_scale;
        particles.mu_rate[theta_idx] = s_mu_rate;
        particles.sigma_base[theta_idx] = s_sigma_base;
        particles.sigma_scale[theta_idx] = s_sigma_scale;
        particles.sigma_rate[theta_idx] = s_sigma_rate;
        
        particles.log_weight[theta_idx] = 0.0f;
        particles.weight[theta_idx] = 1.0f / N_theta;
        particles.log_likelihood[theta_idx] = 0.0f;
        particles.ess_inner[theta_idx] = (float)N_inner;
    }
    
    __syncthreads();
    
    float rho = s_rho;
    float sigma_z = s_sigma_z;
    
    /* Stationary distribution for z̃: N(0, σ_z²/(1-ρ²)) */
    float one_minus_rho_sq = fmaxf(1.0f - rho * rho, 1e-6f);
    float z_tilde_stat_std = sigma_z / sqrtf(one_minus_rho_sq);
    
    /* Generate t=0 noise with precision-appropriate round-trip */
    float z_noise_raw = curand_normal(rng);
    int64_t z_noise_idx = (int64_t)theta_idx * N_inner * (noise_capacity + 1) + inner_idx;
    float z_noise_init = noise_store_roundtrip(d_z_noise, z_noise_idx, z_noise_raw);
    
    /* Thread 0: generate u0 noise */
    if (inner_idx == 0) {
        float u0_noise_raw = curand_normal(rng);
        int64_t u0_noise_idx = (int64_t)theta_idx * (noise_capacity + 1);
        noise_store(d_u0_noise, u0_noise_idx, u0_noise_raw);
    }
    
    /* Initialize z̃ and compute derived quantities */
    float z_tilde = z_tilde_stat_std * z_noise_init;
    float z = z_tilde_to_z(z_tilde);
    
    float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z);
    float mu_z = eval_curve(s_mu_base, s_mu_scale, s_mu_rate, z);
    float sigma_h = eval_curve(s_sigma_base, s_sigma_scale, s_sigma_rate, z);
    
    float phi = 1.0f - theta_z;
    float one_minus_phi_sq = fmaxf(1.0f - phi * phi, 1e-6f);
    float h_stat_var = (sigma_h * sigma_h) / one_minus_phi_sq;
    
    /* Store inner particle state */
    particles.inner_z[global_idx] = z_tilde;
    particles.inner_mu_h[global_idx] = mu_z;
    particles.inner_var_h[global_idx] = h_stat_var;
    particles.inner_log_w[global_idx] = -__logf((float)N_inner);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: RBPF Forward Step
 * 
 * Main filtering kernel. Fused: resample → sort → propagate → observe
 * 
 * Template parameter N_INNER must match blockDim.x for CUB sort.
 *═══════════════════════════════════════════════════════════════════════════════*/

template<int N_INNER>
__global__ 
__launch_bounds__(N_INNER)
void kernel_rbpf_step_impl(
    ThetaParticlesSoA particles,
    float y_obs,
    int N_theta,
    noise_t* d_z_noise,
    noise_t* d_u0_noise,
    int t_current,
    int noise_capacity
) {
    static_assert(N_INNER <= 1024, "N_INNER must be <= 1024");
    
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_INNER + inner_idx;
    
    if (theta_idx >= N_theta || inner_idx >= N_INNER) return;
    
    /* Shared memory layout - NO ALIASING to avoid race conditions */
    extern __shared__ char shared_raw[];
    float* s_reduction = reinterpret_cast<float*>(shared_raw);
    float* s_z = &s_reduction[32];           /* Particle z state for resampling */
    float* s_mu = &s_z[N_INNER];             /* Particle mu state for resampling */
    float* s_var = &s_mu[N_INNER];           /* Particle var state for resampling */
    float* s_cumsum = &s_var[N_INNER];       /* CDF for resampling (dedicated, not aliased) */
    int* s_idx = reinterpret_cast<int*>(&s_cumsum[N_INNER]);
    void* s_cub_temp = reinterpret_cast<void*>(&s_idx[N_INNER]);
    
    /* Load θ parameters to shared memory */
    __shared__ float s_rho, s_sigma_z;
    __shared__ float s_mu_base, s_mu_scale, s_mu_rate;
    __shared__ float s_sigma_base, s_sigma_scale, s_sigma_rate;
    __shared__ float s_log_max, s_sum_w, s_u0;
    
    if (inner_idx == 0) {
        s_rho = particles.rho[theta_idx];
        s_sigma_z = particles.sigma_z[theta_idx];
        s_mu_base = particles.mu_base[theta_idx];
        s_mu_scale = particles.mu_scale[theta_idx];
        s_mu_rate = particles.mu_rate[theta_idx];
        s_sigma_base = particles.sigma_base[theta_idx];
        s_sigma_scale = particles.sigma_scale[theta_idx];
        s_sigma_rate = particles.sigma_rate[theta_idx];
    }
    __syncthreads();
    
    curandState local_rng = particles.rng_states[global_idx];
    
    /* Load particle state */
    float z_tilde = particles.inner_z[global_idx];
    float mu_h = particles.inner_mu_h[global_idx];
    float var_h = particles.inner_var_h[global_idx];
    float log_w = particles.inner_log_w[global_idx];
    
    /* Noise indices */
    int64_t z_noise_base = (int64_t)theta_idx * N_INNER * (noise_capacity + 1);
    int64_t z_noise_idx = z_noise_base + (int64_t)(t_current + 1) * N_INNER + inner_idx;
    int64_t u0_noise_idx = (int64_t)theta_idx * (noise_capacity + 1) + (t_current + 1);
    
    /* Generate and store propagation noise (precision-appropriate round-trip) */
    float z_noise_raw = curand_normal(&local_rng);
    float z_noise = noise_store_roundtrip(d_z_noise, z_noise_idx, z_noise_raw);
    
    /* Thread 0: generate resampling noise */
    if (inner_idx == 0) {
        float u0_noise_raw = curand_normal(&local_rng);
        float u0_stored = noise_store_roundtrip(d_u0_noise, u0_noise_idx, u0_noise_raw);
        s_u0 = u0_from_noise(u0_stored);
    }
    __syncthreads();
    
    /*─────────────────────────────────────────────────────────────────────────
     * RESAMPLE (always, for CPMMH determinism)
     * 
     * CRITICAL: Store particle state to shared memory BEFORE resampling.
     * This avoids race conditions when reading ancestor state.
     *─────────────────────────────────────────────────────────────────────────*/
    {
        /* Store ALL particle state to shared memory FIRST */
        s_z[inner_idx] = z_tilde;
        s_mu[inner_idx] = mu_h;
        s_var[inner_idx] = var_h;
        __syncthreads();
        
        float log_max = block_reduce_max(log_w, s_reduction);
        if (inner_idx == 0) s_log_max = log_max;
        __syncthreads();
        
        float w_unnorm = __expf(log_w - s_log_max);
        float sum_w = block_reduce_sum(w_unnorm, s_reduction);
        if (inner_idx == 0) s_sum_w = sum_w;
        __syncthreads();
        
        /* Build CDF directly (no need for separate weights array) */
        s_cumsum[inner_idx] = w_unnorm / s_sum_w;
        __syncthreads();
        block_inclusive_scan(s_cumsum, N_INNER);
        if (inner_idx == N_INNER - 1) s_cumsum[N_INNER - 1] = 1.0f;
        __syncthreads();
        
        /* Systematic resampling */
        float u = (s_u0 + (float)inner_idx) / (float)N_INNER;
        int lo = 0, hi = N_INNER - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (s_cumsum[mid] < u) lo = mid + 1;
            else hi = mid;
        }
        int ancestor = lo;
        
        /* Load ancestor state from SHARED memory (race-free!) */
        z_tilde = s_z[ancestor];
        mu_h = s_mu[ancestor];
        var_h = s_var[ancestor];
        log_w = -__logf((float)N_INNER);
        
        __syncthreads();
        
        /*─────────────────────────────────────────────────────────────────────
         * CPMMH Sort by μ_h (deterministic)
         *─────────────────────────────────────────────────────────────────────*/
        if ((t_current % SORT_EVERY_K) == 0) {
            s_z[inner_idx] = z_tilde;
            s_mu[inner_idx] = mu_h;
            s_var[inner_idx] = var_h;
            __syncthreads();
            
            cpmmh_sort<N_INNER>(s_z, s_mu, s_var, s_idx, s_cub_temp);
            
            z_tilde = s_z[inner_idx];
            mu_h = s_mu[inner_idx];
            var_h = s_var[inner_idx];
            __syncthreads();
        }
    }
    
    /*─────────────────────────────────────────────────────────────────────────
     * PROPAGATE z̃
     *─────────────────────────────────────────────────────────────────────────*/
    float z_tilde_new = s_rho * z_tilde + s_sigma_z * z_noise;
    float z = z_tilde_to_z(z_tilde_new);
    
    /*─────────────────────────────────────────────────────────────────────────
     * KALMAN PREDICT
     *─────────────────────────────────────────────────────────────────────────*/
    float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z);
    float mu_z = eval_curve(s_mu_base, s_mu_scale, s_mu_rate, z);
    float sigma_h = eval_curve(s_sigma_base, s_sigma_scale, s_sigma_rate, z);
    
    float phi = 1.0f - theta_z;
    float mu_pred = phi * mu_h + theta_z * mu_z;
    float var_pred = phi * phi * var_h + sigma_h * sigma_h;
    var_pred = fmaxf(var_pred, 1e-8f);
    
    /*─────────────────────────────────────────────────────────────────────────
     * OCSN KALMAN UPDATE (marginalized)
     *─────────────────────────────────────────────────────────────────────────*/
    float mu_post, var_post, log_lik;
    ocsn_kalman_update(y_obs, mu_pred, var_pred, &mu_post, &var_post, &log_lik);
    
    log_w += log_lik;
    
    /*─────────────────────────────────────────────────────────────────────────
     * NORMALIZE AND COMPUTE ESS
     *─────────────────────────────────────────────────────────────────────────*/
    float log_max = block_reduce_max(log_w, s_reduction);
    if (inner_idx == 0) s_log_max = log_max;
    __syncthreads();
    
    float w_unnorm = __expf(log_w - s_log_max);
    float sum_w = block_reduce_sum(w_unnorm, s_reduction);
    if (inner_idx == 0) s_sum_w = sum_w;
    __syncthreads();
    
    float w_norm = w_unnorm / fmaxf(s_sum_w, 1e-30f);
    float w_sq = w_norm * w_norm;
    float sum_w_sq = block_reduce_sum(w_sq, s_reduction);
    float ess = 1.0f / fmaxf(sum_w_sq, 1e-30f);
    
    float ll_incr = s_log_max + __logf(fmaxf(s_sum_w, 1e-30f)) - __logf((float)N_INNER);
    
    /*─────────────────────────────────────────────────────────────────────────
     * STORE RESULTS
     *─────────────────────────────────────────────────────────────────────────*/
    particles.inner_z[global_idx] = z_tilde_new;
    particles.inner_mu_h[global_idx] = mu_post;
    particles.inner_var_h[global_idx] = var_post;
    particles.inner_log_w[global_idx] = log_w;
    particles.rng_states[global_idx] = local_rng;
    
    if (inner_idx == 0) {
        particles.ess_inner[theta_idx] = ess;
        particles.log_weight[theta_idx] += ll_incr;
        particles.log_likelihood[theta_idx] += ll_incr;
    }
}

/* Wrapper to dispatch based on N_inner at runtime */
__global__ void kernel_rbpf_step(
    ThetaParticlesSoA particles,
    float y_obs,
    int N_theta, int N_inner,
    noise_t* d_z_noise,
    noise_t* d_u0_noise,
    int t_current,
    int noise_capacity
) {
    /* This wrapper exists for API compatibility but shouldn't be used directly.
     * The host code should call kernel_rbpf_step_impl<N_INNER> directly. */
    (void)particles; (void)y_obs; (void)N_theta; (void)N_inner;
    (void)d_z_noise; (void)d_u0_noise; (void)t_current; (void)noise_capacity;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Reset Outer Weights (SMC² with Refresh / IBIS-style)
 * 
 * After resample + MCMC rejuvenation, particles are approximate posterior samples.
 * Their weights should be uniform - the evidence is encoded in which particles
 * survived resampling and where MCMC moved them, not in cumulative likelihoods.
 * 
 * This prevents O(T) weight explosion and is standard in IBIS / SMC² with refresh.
 * Reference: Chopin & Papaspiliopoulos (2020), Section 17.3.3
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_reset_outer_weights(
    ThetaParticlesSoA particles,
    int N_theta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_theta) return;
    
    /* Reset to uniform weights */
    particles.log_weight[idx] = 0.0f;                    /* log(1) = 0 */
    particles.weight[idx] = 1.0f / (float)N_theta;       /* Normalized uniform */
    particles.log_likelihood[idx] = 0.0f;                /* Fresh accumulator */
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Compute Outer ESS
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_compute_outer_ess(
    ThetaParticlesSoA particles,
    float* d_ess_out,
    int N_theta
) {
    extern __shared__ float s_data[];
    int idx = threadIdx.x;
    
    float log_w = (idx < N_theta) ? particles.log_weight[idx] : -1e30f;
    
    /* Normalize weights */
    float log_max = block_reduce_max(log_w, s_data);
    __shared__ float s_log_max;
    if (idx == 0) s_log_max = log_max;
    __syncthreads();
    
    float w = (idx < N_theta) ? __expf(log_w - s_log_max) : 0.0f;
    float sum_w = block_reduce_sum(w, s_data);
    __shared__ float s_sum_w;
    if (idx == 0) s_sum_w = sum_w;
    __syncthreads();
    
    if (idx < N_theta) {
        w /= s_sum_w;
        particles.weight[idx] = w;
    }
    
    /* Compute ESS */
    float w_sq = (idx < N_theta) ? w * w : 0.0f;
    float sum_w_sq = block_reduce_sum(w_sq, s_data);
    
    if (idx == 0) {
        *d_ess_out = 1.0f / fmaxf(sum_w_sq, 1e-30f);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Outer Resampling
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_outer_resample(
    ThetaParticlesSoA particles,
    int* d_ancestors,
    float* d_uniform,
    int N_theta
) {
    extern __shared__ float s_cumsum[];
    int idx = threadIdx.x;
    
    if (idx < N_theta) s_cumsum[idx] = particles.weight[idx];
    __syncthreads();
    
    /* Serial prefix sum (N_theta = 256 is small) */
    if (idx == 0) {
        for (int i = 1; i < N_theta; i++) s_cumsum[i] += s_cumsum[i-1];
        s_cumsum[N_theta - 1] = 1.0f;
    }
    __syncthreads();
    
    /* Systematic resampling */
    if (idx < N_theta) {
        float u = (*d_uniform + (float)idx) / (float)N_theta;
        int lo = 0, hi = N_theta - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (s_cumsum[mid] < u) lo = mid + 1;
            else hi = mid;
        }
        d_ancestors[idx] = lo;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Copy θ-Particles After Resampling
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_copy_theta_particles(
    ThetaParticlesSoA src,
    ThetaParticlesSoA dst,
    int* d_ancestors,
    int N_theta, int N_inner,
    unsigned long long resample_seed
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (theta_idx >= N_theta) return;
    
    int ancestor = d_ancestors[theta_idx];
    
    /* Thread 0: copy θ parameters */
    if (inner_idx == 0) {
        dst.rho[theta_idx] = src.rho[ancestor];
        dst.sigma_z[theta_idx] = src.sigma_z[ancestor];
        dst.mu_base[theta_idx] = src.mu_base[ancestor];
        dst.mu_scale[theta_idx] = src.mu_scale[ancestor];
        dst.mu_rate[theta_idx] = src.mu_rate[ancestor];
        dst.sigma_base[theta_idx] = src.sigma_base[ancestor];
        dst.sigma_scale[theta_idx] = src.sigma_scale[ancestor];
        dst.sigma_rate[theta_idx] = src.sigma_rate[ancestor];
        
        dst.log_weight[theta_idx] = 0.0f;
        dst.weight[theta_idx] = 1.0f / N_theta;
        dst.log_likelihood[theta_idx] = src.log_likelihood[ancestor];
        dst.ess_inner[theta_idx] = src.ess_inner[ancestor];
    }
    
    /* All threads: copy inner particle state */
    if (inner_idx < N_inner) {
        int src_idx = ancestor * N_inner + inner_idx;
        int dst_idx = theta_idx * N_inner + inner_idx;
        
        dst.inner_z[dst_idx] = src.inner_z[src_idx];
        dst.inner_mu_h[dst_idx] = src.inner_mu_h[src_idx];
        dst.inner_var_h[dst_idx] = src.inner_var_h[src_idx];
        dst.inner_log_w[dst_idx] = src.inner_log_w[src_idx];
        
        /* Re-init RNG (critical: copied particles must have distinct RNG!) */
        curand_init(resample_seed, dst_idx, 0, &dst.rng_states[dst_idx]);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Copy Noise Arrays (Ping-Pong)
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_copy_noise_arrays(
    const noise_t* src_z_noise,
    noise_t* dst_z_noise,
    const noise_t* src_u0_noise,
    noise_t* dst_u0_noise,
    const int* d_ancestors,
    int N_theta, int N_inner,
    int t_current, int noise_capacity
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    
    int ancestor = d_ancestors[theta_idx];
    
    /* Copy z_noise: layout [theta][t][inner] */
    int64_t dst_z_base = (int64_t)theta_idx * N_inner * (noise_capacity + 1);
    int64_t src_z_base = (int64_t)ancestor * N_inner * (noise_capacity + 1);
    
    for (int t = 0; t <= t_current + 1; t++) {
        int64_t src_idx = src_z_base + t * N_inner + inner_idx;
        int64_t dst_idx = dst_z_base + t * N_inner + inner_idx;
        dst_z_noise[dst_idx] = src_z_noise[src_idx];
    }
    
    /* Copy u0_noise: layout [theta][t] - only thread 0 per block */
    if (inner_idx == 0) {
        int64_t dst_u0_base = (int64_t)theta_idx * (noise_capacity + 1);
        int64_t src_u0_base = (int64_t)ancestor * (noise_capacity + 1);
        
        for (int t = 0; t <= t_current + 1; t++) {
            dst_u0_noise[dst_u0_base + t] = src_u0_noise[src_u0_base + t];
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Copy Checkpoint Arrays (after outer resampling)
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_copy_checkpoint(
    const float* src_z,
    const float* src_mu_h,
    const float* src_var_h,
    const float* src_log_w,
    const float* src_ll,
    float* dst_z,
    float* dst_mu_h,
    float* dst_var_h,
    float* dst_log_w,
    float* dst_ll,
    const int* d_ancestors,
    int N_theta, int N_inner
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    
    int ancestor = d_ancestors[theta_idx];
    
    int src_global = ancestor * N_inner + inner_idx;
    int dst_global = theta_idx * N_inner + inner_idx;
    
    dst_z[dst_global] = src_z[src_global];
    dst_mu_h[dst_global] = src_mu_h[src_global];
    dst_var_h[dst_global] = src_var_h[src_global];
    dst_log_w[dst_global] = src_log_w[src_global];
    
    if (inner_idx == 0) {
        dst_ll[theta_idx] = src_ll[ancestor];
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: CPMMH Fused Rejuvenation
 * 
 * The big one. See header for algorithm documentation.
 * 
 * Template parameter N_INNER must match blockDim.x for CUB sort.
 *═══════════════════════════════════════════════════════════════════════════════*/

template<int N_INNER>
__global__
__launch_bounds__(N_INNER)
void kernel_cpmmh_rejuvenate_fused_impl(
    ThetaParticlesSoA particles,
    ThetaParticlesSoA particles_scratch,
    const float* y_history,
    noise_t* d_z_noise_curr,
    noise_t* d_z_noise_other,
    noise_t* d_u0_noise_curr,
    noise_t* d_u0_noise_other,
    int t_current,
    int N_theta,
    int noise_capacity,
    float cpmmh_rho,
    int* d_accepts,
    int* d_swap_flags,
    unsigned long long seed,
    int move_id,
    int block_id,
    /* Fixed-lag parameters */
    int t_checkpoint,
    const float* d_checkpoint_z,
    const float* d_checkpoint_mu_h,
    const float* d_checkpoint_var_h,
    const float* d_checkpoint_log_w,
    const float* d_checkpoint_ll
) {
    static_assert(N_INNER <= 1024, "N_INNER must be <= 1024");
    
    (void)seed; (void)move_id; (void)block_id;  /* Unused in joint mode */
    
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_INNER + inner_idx;
    
    if (theta_idx >= N_theta || inner_idx >= N_INNER) return;
    
    /* Shared memory layout for CUB sort */
    extern __shared__ char shared_raw[];
    float* s_reduction = reinterpret_cast<float*>(shared_raw);
    float* s_z = &s_reduction[32];
    float* s_mu = &s_z[N_INNER];
    float* s_var = &s_mu[N_INNER];
    float* s_cdf = &s_var[N_INNER];
    int* s_idx = reinterpret_cast<int*>(&s_cdf[N_INNER]);
    void* s_cub_temp = reinterpret_cast<void*>(&s_idx[N_INNER]);
    
    __shared__ float s_log_max, s_sum_w, s_ess_prop;
    __shared__ float s_rho_curr, s_sigma_z_curr;
    __shared__ float s_mu_base_curr, s_mu_scale_curr, s_mu_rate_curr;
    __shared__ float s_sigma_base_curr, s_sigma_scale_curr, s_sigma_rate_curr;
    __shared__ float s_rho_prop, s_sigma_z_prop;
    __shared__ float s_mu_base_prop, s_mu_scale_prop, s_mu_rate_prop;
    __shared__ float s_sigma_base_prop, s_sigma_scale_prop, s_sigma_rate_prop;
    __shared__ float s_ll_curr, s_ll_prop, s_lp_curr, s_lp_prop;
    __shared__ int s_accept, s_valid;
    __shared__ float s_u0_shared;
    
    curandState local_rng = particles.rng_states[global_idx];
    
    /*─────────────────────────────────────────────────────────────────────────
     * PROPOSE θ* (thread 0)
     *─────────────────────────────────────────────────────────────────────────*/
    if (inner_idx == 0) {
        s_rho_curr = particles.rho[theta_idx];
        s_sigma_z_curr = particles.sigma_z[theta_idx];
        s_mu_base_curr = particles.mu_base[theta_idx];
        s_mu_scale_curr = particles.mu_scale[theta_idx];
        s_mu_rate_curr = particles.mu_rate[theta_idx];
        s_sigma_base_curr = particles.sigma_base[theta_idx];
        s_sigma_scale_curr = particles.sigma_scale[theta_idx];
        s_sigma_rate_curr = particles.sigma_rate[theta_idx];
        
        s_ll_curr = particles.log_likelihood[theta_idx];
        s_lp_curr = log_prior_theta(s_rho_curr, s_sigma_z_curr,
                                     s_mu_base_curr, s_mu_scale_curr, s_mu_rate_curr,
                                     s_sigma_base_curr, s_sigma_scale_curr, s_sigma_rate_curr);
        
        /* Random walk proposal */
        s_rho_prop = s_rho_curr + d_proposal_std[0] * curand_normal(&local_rng);
        s_sigma_z_prop = s_sigma_z_curr + d_proposal_std[1] * curand_normal(&local_rng);
        s_mu_base_prop = s_mu_base_curr + d_proposal_std[2] * curand_normal(&local_rng);
        s_mu_scale_prop = s_mu_scale_curr + d_proposal_std[3] * curand_normal(&local_rng);
        s_mu_rate_prop = s_mu_rate_curr + d_proposal_std[4] * curand_normal(&local_rng);
        s_sigma_base_prop = s_sigma_base_curr + d_proposal_std[5] * curand_normal(&local_rng);
        s_sigma_scale_prop = s_sigma_scale_curr + d_proposal_std[6] * curand_normal(&local_rng);
        s_sigma_rate_prop = s_sigma_rate_curr + d_proposal_std[7] * curand_normal(&local_rng);
        
        s_lp_prop = log_prior_theta(s_rho_prop, s_sigma_z_prop,
                                     s_mu_base_prop, s_mu_scale_prop, s_mu_rate_prop,
                                     s_sigma_base_prop, s_sigma_scale_prop, s_sigma_rate_prop);
        
        s_valid = isfinite(s_lp_prop) ? 1 : 0;
        s_accept = 0;
    }
    __syncthreads();
    
    /* Early exit for invalid proposals */
    if (s_valid == 0) {
        if (inner_idx == 0) d_swap_flags[theta_idx] = 0;
        particles.rng_states[global_idx] = local_rng;
        return;
    }
    
    int64_t z_noise_base = (int64_t)theta_idx * N_INNER * (noise_capacity + 1);
    float scale = sqrtf(1.0f - cpmmh_rho * cpmmh_rho);
    
    /*─────────────────────────────────────────────────────────────────────────
     * INITIALIZATION
     * 
     * Full history (t_checkpoint < 0): Start from t=0, replay everything
     * Fixed-lag (t_checkpoint >= 0): Load checkpoint state, replay L steps
     * 
     * For fixed-lag, we load θ_curr's checkpoint state and run with θ_prop.
     * This creates small bias that decays as exp(-L/τ), but enables O(L) replay.
     *─────────────────────────────────────────────────────────────────────────*/
    
    float rho = s_rho_prop;
    float sigma_z = s_sigma_z_prop;
    float mu_base = s_mu_base_prop;
    float mu_scale = s_mu_scale_prop;
    float mu_rate = s_mu_rate_prop;
    float sigma_base = s_sigma_base_prop;
    float sigma_scale = s_sigma_scale_prop;
    float sigma_rate = s_sigma_rate_prop;
    
    float z_tilde, mu_h, var_h, log_w;
    float ll_accum = 0.0f;
    int t_start;
    
    if (t_checkpoint >= 0 && d_checkpoint_z != nullptr) {
        /* Fixed-lag: Load checkpoint state (saved at t_checkpoint)
         * This state was computed under θ_curr, but we run with θ_prop.
         * The bias decays as exp(-L/τ) where τ ≈ 14 for ρ=0.95.
         * 
         * Note: We don't copy noise for 0 to t_start because:
         * - Rejected: other buffer is discarded anyway
         * - Accepted: kernel_commit_accepted_noise only copies t_start onwards,
         *   preserving the old noise in indices 0 to t_start-1
         */
        z_tilde = d_checkpoint_z[global_idx];
        mu_h = d_checkpoint_mu_h[global_idx];
        var_h = d_checkpoint_var_h[global_idx];
        log_w = d_checkpoint_log_w[global_idx];
        t_start = t_checkpoint + 1;
    } else {
        /* Full history: Initialize from θ_prop's stationary */
        float one_minus_rho_sq = fmaxf(1.0f - rho * rho, 1e-6f);
        float z_tilde_stat_std = sigma_z / sqrtf(one_minus_rho_sq);
        
        /* Correlate t=0 noise */
        float z_noise_curr_0 = noise_load(d_z_noise_curr, z_noise_base + inner_idx);
        float z_noise_fresh_0 = curand_normal(&local_rng);
        float z_noise_prop_0 = cpmmh_rho * z_noise_curr_0 + scale * z_noise_fresh_0;
        noise_store(d_z_noise_other, z_noise_base + inner_idx, z_noise_prop_0);
        
        z_tilde = z_tilde_stat_std * z_noise_prop_0;
        float z_init = z_tilde_to_z(z_tilde);
        
        float theta_z_init = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z_init);
        float mu_z_init = eval_curve(mu_base, mu_scale, mu_rate, z_init);
        float sigma_h_init = eval_curve(sigma_base, sigma_scale, sigma_rate, z_init);
        float phi_init = 1.0f - theta_z_init;
        float h_stat_var = (sigma_h_init * sigma_h_init) / fmaxf(1.0f - phi_init * phi_init, 1e-6f);
        
        mu_h = mu_z_init;
        var_h = h_stat_var;
        log_w = -__logf((float)N_INNER);
        t_start = 0;
    }
    
    /* Process observations from t_start to t_current */
    for (int t = t_start; t <= t_current; t++) {
        float y_obs = y_history[t];
        
        /* Normalize weights */
        float log_max = block_reduce_max(log_w, s_reduction);
        if (inner_idx == 0) s_log_max = log_max;
        __syncthreads();
        log_max = s_log_max;
        
        float w_unnorm = __expf(log_w - log_max);
        float sum_w = block_reduce_sum(w_unnorm, s_reduction);
        if (inner_idx == 0) s_sum_w = sum_w;
        __syncthreads();
        sum_w = s_sum_w;
        
        float w_norm = w_unnorm / fmaxf(sum_w, 1e-30f);
        
        /* Store state for resampling */
        s_z[inner_idx] = z_tilde;
        s_mu[inner_idx] = mu_h;
        s_var[inner_idx] = var_h;
        s_cdf[inner_idx] = w_norm;
        __syncthreads();
        
        block_inclusive_scan(s_cdf, N_INNER);
        if (inner_idx == N_INNER - 1) s_cdf[N_INNER - 1] = 1.0f;
        __syncthreads();
        
        /* Generate correlated noise for t+1 */
        int64_t z_idx_t1 = z_noise_base + (int64_t)(t + 1) * N_INNER + inner_idx;
        float z_noise_curr_t1 = noise_load(d_z_noise_curr, z_idx_t1);
        float z_noise_fresh_t1 = curand_normal(&local_rng);
        float z_noise_prop_t1_raw = cpmmh_rho * z_noise_curr_t1 + scale * z_noise_fresh_t1;
        float z_noise_prop_t1 = noise_store_roundtrip(d_z_noise_other, z_idx_t1, z_noise_prop_t1_raw);
        
        /* Thread 0: correlate resampling noise */
        if (inner_idx == 0) {
            int64_t u0_idx_t1 = (int64_t)theta_idx * (noise_capacity + 1) + (t + 1);
            float u0_noise_curr = noise_load(d_u0_noise_curr, u0_idx_t1);
            float u0_noise_fresh = curand_normal(&local_rng);
            float u0_noise_prop_raw = cpmmh_rho * u0_noise_curr + scale * u0_noise_fresh;
            float u0_stored = noise_store_roundtrip(d_u0_noise_other, u0_idx_t1, u0_noise_prop_raw);
            s_u0_shared = u0_from_noise(u0_stored);
        }
        __syncthreads();
        
        /* Systematic resampling */
        float u = (s_u0_shared + (float)inner_idx) / (float)N_INNER;
        int lo = 0, hi = N_INNER - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (s_cdf[mid] < u) lo = mid + 1;
            else hi = mid;
        }
        
        z_tilde = s_z[lo];
        mu_h = s_mu[lo];
        var_h = s_var[lo];
        log_w = -__logf((float)N_INNER);
        __syncthreads();
        
        /* CPMMH sort (deterministic) */
        if ((t % SORT_EVERY_K) == 0) {
            s_z[inner_idx] = z_tilde;
            s_mu[inner_idx] = mu_h;
            s_var[inner_idx] = var_h;
            __syncthreads();
            
            cpmmh_sort<N_INNER>(s_z, s_mu, s_var, s_idx, s_cub_temp);
            
            z_tilde = s_z[inner_idx];
            mu_h = s_mu[inner_idx];
            var_h = s_var[inner_idx];
            __syncthreads();
        }
        
        /* Propagate */
        float z_tilde_new = rho * z_tilde + sigma_z * z_noise_prop_t1;
        float z = z_tilde_to_z(z_tilde_new);
        
        /* Kalman predict */
        float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z);
        float mu_z_val = eval_curve(mu_base, mu_scale, mu_rate, z);
        float sigma_h = eval_curve(sigma_base, sigma_scale, sigma_rate, z);
        float phi = 1.0f - theta_z;
        
        float mu_pred = phi * mu_h + theta_z * mu_z_val;
        float var_pred = phi * phi * var_h + sigma_h * sigma_h;
        var_pred = fmaxf(var_pred, 1e-8f);
        
        /* OCSN update */
        float mu_post, var_post, log_lik;
        ocsn_kalman_update(y_obs, mu_pred, var_pred, &mu_post, &var_post, &log_lik);
        
        log_w += log_lik;
        
        /* Accumulate likelihood */
        log_max = block_reduce_max(log_w, s_reduction);
        if (inner_idx == 0) s_log_max = log_max;
        __syncthreads();
        log_max = s_log_max;
        
        w_unnorm = __expf(log_w - log_max);
        sum_w = block_reduce_sum(w_unnorm, s_reduction);
        if (inner_idx == 0) s_sum_w = sum_w;
        __syncthreads();
        sum_w = s_sum_w;
        
        float ll_incr = log_max + __logf(fmaxf(sum_w, 1e-30f)) - __logf((float)N_INNER);
        ll_accum += ll_incr;
        
        z_tilde = z_tilde_new;
        mu_h = mu_post;
        var_h = var_post;
    }
    
    /* Final ESS */
    float w_norm = __expf(log_w - s_log_max) / fmaxf(s_sum_w, 1e-30f);
    float w_sq = w_norm * w_norm;
    float sum_w_sq = block_reduce_sum(w_sq, s_reduction);
    float ess = 1.0f / fmaxf(sum_w_sq, 1e-30f);
    
    /* Store proposed state */
    particles_scratch.inner_z[global_idx] = z_tilde;
    particles_scratch.inner_mu_h[global_idx] = mu_h;
    particles_scratch.inner_var_h[global_idx] = var_h;
    particles_scratch.inner_log_w[global_idx] = log_w;
    
    __shared__ float s_ll_base;
    
    if (inner_idx == 0) {
        /* For fixed-lag, we compare WINDOW likelihoods only:
         * - ll_window_prop = ll_accum (computed above from t_start to T)
         * - ll_window_curr = s_ll_curr - checkpoint_ll
         * 
         * The base likelihood (0 to t_start) cancels in the ratio.
         */
        float ll_base = (t_checkpoint >= 0 && d_checkpoint_ll != nullptr) 
                        ? d_checkpoint_ll[theta_idx] : 0.0f;
        s_ll_base = ll_base;
        s_ll_prop = ll_base + ll_accum;  /* Total: base + window */
        s_ess_prop = ess;
    }
    __syncthreads();
    
    /*─────────────────────────────────────────────────────────────────────────
     * MH ACCEPT/REJECT
     * 
     * For fixed-lag: compare window likelihoods (base cancels)
     *   log_alpha = (ll_window_prop + lp_prop) - (ll_window_curr + lp_curr)
     *   where ll_window_curr = s_ll_curr - s_ll_base
     *─────────────────────────────────────────────────────────────────────────*/
    if (inner_idx == 0) {
        /* For fixed-lag: subtract base from current to get window likelihood
         * For full history (t_checkpoint < 0): s_ll_base = 0, so no change */
        float ll_curr_effective = s_ll_curr - s_ll_base;
        float ll_prop_effective = ll_accum;  /* Just the window part */
        
        float log_alpha = (ll_prop_effective + s_lp_prop) - (ll_curr_effective + s_lp_curr);
        
        float u = curand_uniform(&local_rng);
        s_accept = (__logf(u) < log_alpha) ? 1 : 0;
        
        if (s_accept) {
            particles.rho[theta_idx] = s_rho_prop;
            particles.sigma_z[theta_idx] = s_sigma_z_prop;
            particles.mu_base[theta_idx] = s_mu_base_prop;
            particles.mu_scale[theta_idx] = s_mu_scale_prop;
            particles.mu_rate[theta_idx] = s_mu_rate_prop;
            particles.sigma_base[theta_idx] = s_sigma_base_prop;
            particles.sigma_scale[theta_idx] = s_sigma_scale_prop;
            particles.sigma_rate[theta_idx] = s_sigma_rate_prop;
            particles.log_likelihood[theta_idx] = s_ll_prop;  /* Store total for next time */
            particles.ess_inner[theta_idx] = s_ess_prop;
            atomicAdd(d_accepts, 1);
        }
        
        d_swap_flags[theta_idx] = s_accept;
    }
    __syncthreads();
    
    /* On accept: copy proposed state */
    if (s_accept) {
        particles.inner_z[global_idx] = particles_scratch.inner_z[global_idx];
        particles.inner_mu_h[global_idx] = particles_scratch.inner_mu_h[global_idx];
        particles.inner_var_h[global_idx] = particles_scratch.inner_var_h[global_idx];
        particles.inner_log_w[global_idx] = particles_scratch.inner_log_w[global_idx];
    }
    
    particles.rng_states[global_idx] = local_rng;
}

/* Wrapper for API compatibility */
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
    int block_id,
    int t_checkpoint,
    const float* d_checkpoint_z,
    const float* d_checkpoint_mu_h,
    const float* d_checkpoint_var_h,
    const float* d_checkpoint_log_w,
    const float* d_checkpoint_ll
) {
    /* Wrapper exists for API compatibility; host should use template version */
    (void)particles; (void)particles_scratch; (void)y_history;
    (void)d_z_noise_curr; (void)d_z_noise_other;
    (void)d_u0_noise_curr; (void)d_u0_noise_other;
    (void)t_current; (void)N_theta; (void)N_inner;
    (void)noise_capacity; (void)cpmmh_rho;
    (void)d_accepts; (void)d_swap_flags;
    (void)seed; (void)move_id; (void)block_id;
    (void)t_checkpoint; (void)d_checkpoint_z; (void)d_checkpoint_mu_h;
    (void)d_checkpoint_var_h; (void)d_checkpoint_log_w; (void)d_checkpoint_ll;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Commit Accepted Noise
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_commit_accepted_noise(
    noise_t* d_z_noise_0,
    noise_t* d_z_noise_1,
    noise_t* d_u0_noise_0,
    noise_t* d_u0_noise_1,
    const int* d_swap_flags,
    int N_theta, int N_inner,
    int t_current, int noise_capacity,
    int t_start  /* For fixed-lag: only commit from t_start onwards */
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    if (d_swap_flags[theta_idx] == 0) return;
    
    int64_t z_base = (int64_t)theta_idx * N_inner * (noise_capacity + 1);
    
    /* Only copy from t_start to t_current+1 (window that was modified) */
    for (int t = t_start; t <= t_current + 1; t++) {
        int64_t idx = z_base + t * N_inner + inner_idx;
        d_z_noise_0[idx] = d_z_noise_1[idx];
    }
    
    if (inner_idx == 0) {
        int64_t u0_base = (int64_t)theta_idx * (noise_capacity + 1);
        for (int t = t_start; t <= t_current + 1; t++) {
            d_u0_noise_0[u0_base + t] = d_u0_noise_1[u0_base + t];
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Save Checkpoint for Fixed-Lag PMMH
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_save_checkpoint(
    const ThetaParticlesSoA particles,
    float* d_checkpoint_z,
    float* d_checkpoint_mu_h,
    float* d_checkpoint_var_h,
    float* d_checkpoint_log_w,
    float* d_checkpoint_ll,
    int N_theta, int N_inner
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_inner + inner_idx;
    
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    
    /* Copy inner particle state */
    d_checkpoint_z[global_idx] = particles.inner_z[global_idx];
    d_checkpoint_mu_h[global_idx] = particles.inner_mu_h[global_idx];
    d_checkpoint_var_h[global_idx] = particles.inner_var_h[global_idx];
    d_checkpoint_log_w[global_idx] = particles.inner_log_w[global_idx];
    
    /* Thread 0 copies log-likelihood */
    if (inner_idx == 0) {
        d_checkpoint_ll[theta_idx] = particles.log_likelihood[theta_idx];
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * HOST API IMPLEMENTATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/* Fast xorshift64* for host-side uniform generation */
static inline uint64_t xorshift64star(uint64_t* state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float xorshift64star_uniform(uint64_t* state) {
    uint64_t r = xorshift64star(state);
    return (float)((r >> 11) + 1) * (1.0f / 9007199254740994.0f);
}

SMC2StateCUDA* smc2_cuda_alloc(int N_theta, int N_inner) {
    SMC2StateCUDA* state = (SMC2StateCUDA*)calloc(1, sizeof(SMC2StateCUDA));
    if (!state) return NULL;
    
    state->N_theta = N_theta;
    state->N_inner = N_inner;
    state->ess_threshold_outer = 0.3f;
    state->ess_threshold_inner = 0.5f;
    state->K_rejuv = 5;  /* 5 MH steps for adequate diversification */
    
    int N_total = N_theta * N_inner;
    
    /* Allocate main particles */
    CUDA_CHECK(cudaMalloc(&state->d_particles.rho, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.sigma_z, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.mu_base, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.mu_scale, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.mu_rate, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.sigma_base, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.sigma_scale, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.sigma_rate, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.inner_z, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.inner_mu_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.inner_var_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.inner_log_w, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.log_weight, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.weight, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.log_likelihood, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.ess_inner, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.rng_states, N_total * sizeof(curandState)));
    
    /* Allocate temp particles */
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.rho, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.sigma_z, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.mu_base, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.mu_scale, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.mu_rate, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.sigma_base, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.sigma_scale, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.sigma_rate, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.inner_z, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.inner_mu_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.inner_var_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.inner_log_w, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.log_weight, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.weight, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.log_likelihood, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.ess_inner, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.rng_states, N_total * sizeof(curandState)));
    
    /* Initialize RNG */
    kernel_init_rng<<<(N_total + 255) / 256, 256>>>(state->d_particles.rng_states, 12345ULL, N_total);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    /* Observation history */
    state->y_history_capacity = 8000;
    CUDA_CHECK(cudaMalloc(&state->d_y_history, state->y_history_capacity * sizeof(float)));
    state->y_history_len = 0;
    
    /* Scratch arrays */
    CUDA_CHECK(cudaMalloc(&state->d_ancestors, N_theta * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_uniform, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_ess, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_accepts, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_swap_flags, N_theta * sizeof(int)));
    
    /* CPMMH noise buffers */
    state->noise_capacity = 2048;
    state->cpmmh_rho = 0.99f;
    state->noise_buf = 0;
    state->user_seed = 0;
    state->host_rng_state = 0x853C49E6748FEA9BULL ^ (uint64_t)time(NULL);
    
    int64_t z_noise_size = (int64_t)N_theta * N_inner * (state->noise_capacity + 1);
    int64_t u0_noise_size = (int64_t)N_theta * (state->noise_capacity + 1);
    
    CUDA_CHECK(cudaMalloc(&state->d_z_noise[0], noise_array_bytes(z_noise_size)));
    CUDA_CHECK(cudaMalloc(&state->d_z_noise[1], noise_array_bytes(z_noise_size)));
    CUDA_CHECK(cudaMalloc(&state->d_u0_noise[0], noise_array_bytes(u0_noise_size)));
    CUDA_CHECK(cudaMalloc(&state->d_u0_noise[1], noise_array_bytes(u0_noise_size)));
    
    /* Default prior */
    state->prior.rho_mean = 0.95f; state->prior.rho_std = 0.02f;
    state->prior.sigma_z_mean = 0.1f; state->prior.sigma_z_std = 0.05f;
    state->prior.mu_base_mean = -1.0f; state->prior.mu_base_std = 0.5f;
    state->prior.mu_scale_mean = 0.5f; state->prior.mu_scale_std = 0.3f;
    state->prior.mu_rate_mean = 1.0f; state->prior.mu_rate_std = 0.5f;
    state->prior.sigma_base_mean = 0.15f; state->prior.sigma_base_std = 0.05f;
    state->prior.sigma_scale_mean = 0.1f; state->prior.sigma_scale_std = 0.05f;
    state->prior.sigma_rate_mean = 1.0f; state->prior.sigma_rate_std = 0.5f;
    
    /* Default bounds */
    state->bounds.rho_min = 0.8f; state->bounds.rho_max = 0.999f;
    state->bounds.sigma_z_min = 0.01f; state->bounds.sigma_z_max = 1.0f;
    state->bounds.mu_base_min = -10.0f; state->bounds.mu_base_max = 5.0f;
    state->bounds.mu_scale_min = -2.0f; state->bounds.mu_scale_max = 10.0f;
    state->bounds.mu_rate_min = 0.1f; state->bounds.mu_rate_max = 10.0f;
    state->bounds.sigma_base_min = 0.01f; state->bounds.sigma_base_max = 1.0f;
    state->bounds.sigma_scale_min = -0.5f; state->bounds.sigma_scale_max = 1.0f;
    state->bounds.sigma_rate_min = 0.1f; state->bounds.sigma_rate_max = 10.0f;
    
    /* Default theta curve */
    state->theta_curve.base = 0.02f;
    state->theta_curve.scale = 0.08f;
    state->theta_curve.rate = 1.5f;

    /* Proposal std */
    state->proposal_std[0] = 0.01f;
    state->proposal_std[1] = 0.02f;
    state->proposal_std[2] = 0.25f; // mu_base (was 0.1)
    state->proposal_std[3] = 0.25f; // mu_scale (was 0.1)
    state->proposal_std[4] = 0.35f; // mu_rate (was 0.15)

    state->proposal_std[5] = 0.02f;
    state->proposal_std[6] = 0.02f;
    state->proposal_std[7] = 0.35f; // sigma_rate (was 0.15)

    /* Fixed-lag checkpoint (disabled by default, L=0 means full history) */
    state->fixed_lag_L = 0;
    state->t_checkpoint = -1;
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_z, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_mu_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_var_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_log_w, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_ll, N_theta * sizeof(float)));
    
    return state;
}

void smc2_cuda_free(SMC2StateCUDA* state) {
    if (!state) return;
    
    /* Free main particles */
    cudaFree(state->d_particles.rho);
    cudaFree(state->d_particles.sigma_z);
    cudaFree(state->d_particles.mu_base);
    cudaFree(state->d_particles.mu_scale);
    cudaFree(state->d_particles.mu_rate);
    cudaFree(state->d_particles.sigma_base);
    cudaFree(state->d_particles.sigma_scale);
    cudaFree(state->d_particles.sigma_rate);
    cudaFree(state->d_particles.inner_z);
    cudaFree(state->d_particles.inner_mu_h);
    cudaFree(state->d_particles.inner_var_h);
    cudaFree(state->d_particles.inner_log_w);
    cudaFree(state->d_particles.log_weight);
    cudaFree(state->d_particles.weight);
    cudaFree(state->d_particles.log_likelihood);
    cudaFree(state->d_particles.ess_inner);
    cudaFree(state->d_particles.rng_states);
    
    /* Free temp particles */
    cudaFree(state->d_particles_temp.rho);
    cudaFree(state->d_particles_temp.sigma_z);
    cudaFree(state->d_particles_temp.mu_base);
    cudaFree(state->d_particles_temp.mu_scale);
    cudaFree(state->d_particles_temp.mu_rate);
    cudaFree(state->d_particles_temp.sigma_base);
    cudaFree(state->d_particles_temp.sigma_scale);
    cudaFree(state->d_particles_temp.sigma_rate);
    cudaFree(state->d_particles_temp.inner_z);
    cudaFree(state->d_particles_temp.inner_mu_h);
    cudaFree(state->d_particles_temp.inner_var_h);
    cudaFree(state->d_particles_temp.inner_log_w);
    cudaFree(state->d_particles_temp.log_weight);
    cudaFree(state->d_particles_temp.weight);
    cudaFree(state->d_particles_temp.log_likelihood);
    cudaFree(state->d_particles_temp.ess_inner);
    cudaFree(state->d_particles_temp.rng_states);
    
    cudaFree(state->d_y_history);
    cudaFree(state->d_ancestors);
    cudaFree(state->d_uniform);
    cudaFree(state->d_ess);
    cudaFree(state->d_accepts);
    cudaFree(state->d_swap_flags);
    
    cudaFree(state->d_z_noise[0]);
    cudaFree(state->d_z_noise[1]);
    cudaFree(state->d_u0_noise[0]);
    cudaFree(state->d_u0_noise[1]);
    
    /* Free fixed-lag checkpoint */
    cudaFree(state->d_checkpoint_z);
    cudaFree(state->d_checkpoint_mu_h);
    cudaFree(state->d_checkpoint_var_h);
    cudaFree(state->d_checkpoint_log_w);
    cudaFree(state->d_checkpoint_ll);
    
    free(state);
}

void smc2_cuda_set_seed(SMC2StateCUDA* state, uint64_t seed) {
    state->user_seed = seed;
    if (seed != 0) {
        state->host_rng_state = 0x853C49E6748FEA9BULL ^ seed;
    }
}

void smc2_cuda_set_noise_capacity(SMC2StateCUDA* state, int capacity) {
    if (capacity <= state->noise_capacity) return;
    
    int64_t new_z_size = (int64_t)state->N_theta * state->N_inner * (capacity + 1);
    int64_t old_z_size = (int64_t)state->N_theta * state->N_inner * (state->noise_capacity + 1);
    int64_t new_u0_size = (int64_t)state->N_theta * (capacity + 1);
    int64_t old_u0_size = (int64_t)state->N_theta * (state->noise_capacity + 1);
    
    noise_t *new_z_0, *new_z_1, *new_u0_0, *new_u0_1;
    CUDA_CHECK(cudaMalloc(&new_z_0, noise_array_bytes(new_z_size)));
    CUDA_CHECK(cudaMalloc(&new_z_1, noise_array_bytes(new_z_size)));
    CUDA_CHECK(cudaMalloc(&new_u0_0, noise_array_bytes(new_u0_size)));
    CUDA_CHECK(cudaMalloc(&new_u0_1, noise_array_bytes(new_u0_size)));
    
    if (state->d_z_noise[0] && old_z_size > 0) {
        CUDA_CHECK(cudaMemcpy(new_z_0, state->d_z_noise[0], 
                              noise_array_bytes(old_z_size), cudaMemcpyDeviceToDevice));
    }
    if (state->d_u0_noise[0] && old_u0_size > 0) {
        CUDA_CHECK(cudaMemcpy(new_u0_0, state->d_u0_noise[0],
                              noise_array_bytes(old_u0_size), cudaMemcpyDeviceToDevice));
    }
    
    cudaFree(state->d_z_noise[0]);
    cudaFree(state->d_z_noise[1]);
    cudaFree(state->d_u0_noise[0]);
    cudaFree(state->d_u0_noise[1]);
    
    state->d_z_noise[0] = new_z_0;
    state->d_z_noise[1] = new_z_1;
    state->d_u0_noise[0] = new_u0_0;
    state->d_u0_noise[1] = new_u0_1;
    state->noise_buf = 0;
    state->noise_capacity = capacity;
}

void smc2_cuda_set_fixed_lag(SMC2StateCUDA* state, int L) {
    /* 
     * Set fixed-lag window size for PMMH rejuvenation.
     * 
     * L = 0:   Full history replay (default, exact but O(T) variance)
     * L > 0:   Fixed-lag with window size L (bounded O(L) variance)
     * 
     * Recommended: L = 100-200 for ρ ≈ 0.95 (about 7 half-lives)
     */
    state->fixed_lag_L = L;
    state->t_checkpoint = -1;  /* Reset checkpoint */
}

void smc2_cuda_set_proposal_std(SMC2StateCUDA* state, const float* std) {
    if (std) {
        memcpy(state->proposal_std, std, 8 * sizeof(float));
    } else {
        /* Reset to defaults */
        state->proposal_std[0] = 0.01f;   /* rho */
        state->proposal_std[1] = 0.02f;   /* sigma_z */
        state->proposal_std[2] = 0.1f;    /* mu_base */
        state->proposal_std[3] = 0.1f;    /* mu_scale */
        state->proposal_std[4] = 0.15f;   /* mu_rate */
        state->proposal_std[5] = 0.02f;   /* sigma_base */
        state->proposal_std[6] = 0.02f;   /* sigma_scale */
        state->proposal_std[7] = 0.15f;   /* sigma_rate */
    }
    /* Update constant memory */
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_std, state->proposal_std, 8 * sizeof(float)));
}

void smc2_cuda_set_cpmmh_rho(SMC2StateCUDA* state, float rho) {
    state->cpmmh_rho = rho;
}

void smc2_cuda_init_from_prior(SMC2StateCUDA* state) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_prior, &state->prior, sizeof(SVPrior)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_bounds, &state->bounds, sizeof(SVBounds)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_theta_curve, &state->theta_curve, sizeof(SVCurve)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_std, state->proposal_std, 8 * sizeof(float)));
    
    int N_total = state->N_theta * state->N_inner;
    unsigned long long rng_seed = (state->user_seed != 0) ? state->user_seed : 12345ULL;
    kernel_init_rng<<<(N_total + 255) / 256, 256>>>(state->d_particles.rng_states, rng_seed, N_total);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    kernel_init_from_prior<<<state->N_theta, state->N_inner>>>(
        state->d_particles, state->N_theta, state->N_inner,
        state->d_z_noise[state->noise_buf], 
        state->d_u0_noise[state->noise_buf],
        state->noise_capacity
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    state->n_resamples = 0;
    state->n_rejuv_accepts = 0;
    state->n_rejuv_total = 0;
    state->y_history_len = 0;
    state->t_current = -1;
}

float smc2_cuda_update(SMC2StateCUDA* state, float y_obs) {
    /* Store observation */
    if (state->y_history_len >= state->y_history_capacity) {
        int new_cap = state->y_history_capacity * 2;
        float* new_hist;
        CUDA_CHECK(cudaMalloc(&new_hist, new_cap * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(new_hist, state->d_y_history,
                              state->y_history_len * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaFree(state->d_y_history);
        state->d_y_history = new_hist;
        state->y_history_capacity = new_cap;
    }
    CUDA_CHECK(cudaMemcpy(&state->d_y_history[state->y_history_len], &y_obs,
                          sizeof(float), cudaMemcpyHostToDevice));
    state->y_history_len++;
    state->t_current++;
    
    /* Grow noise if needed */
    if (state->t_current >= state->noise_capacity) {
        smc2_cuda_set_noise_capacity(state, state->noise_capacity * 2);
    }
    
    /* Dispatch based on N_inner - CUB requires compile-time block size */
    #define DISPATCH_RBPF_STEP(N) \
        kernel_rbpf_step_impl<N><<<state->N_theta, N, rbpf_shared_mem_size<N>()>>>( \
            state->d_particles, y_obs, \
            state->N_theta, \
            state->d_z_noise[state->noise_buf], \
            state->d_u0_noise[state->noise_buf], \
            state->t_current, state->noise_capacity)
    
    switch (state->N_inner) {
        case 64:  DISPATCH_RBPF_STEP(64);  break;
        case 128: DISPATCH_RBPF_STEP(128); break;
        case 256: DISPATCH_RBPF_STEP(256); break;
        case 512: DISPATCH_RBPF_STEP(512); break;
        default:
            fprintf(stderr, "Unsupported N_inner=%d. Must be 64, 128, 256, or 512.\n", state->N_inner);
            exit(EXIT_FAILURE);
    }
    #undef DISPATCH_RBPF_STEP
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    kernel_compute_outer_ess<<<1, state->N_theta, 32 * sizeof(float)>>>(
        state->d_particles, state->d_ess, state->N_theta
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float h_ess;
    CUDA_CHECK(cudaMemcpy(&h_ess, state->d_ess, sizeof(float), cudaMemcpyDeviceToHost));
    
    /* Resample and rejuvenate if needed */
    if (h_ess < state->ess_threshold_outer * state->N_theta) {
        state->n_resamples++;
        
        float h_uniform = xorshift64star_uniform(&state->host_rng_state);
        CUDA_CHECK(cudaMemcpy(state->d_uniform, &h_uniform, sizeof(float), cudaMemcpyHostToDevice));
        
        kernel_outer_resample<<<1, state->N_theta, state->N_theta * sizeof(float)>>>(
            state->d_particles, state->d_ancestors, state->d_uniform, state->N_theta
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        unsigned long long resample_seed = time(NULL) * 1000ULL + state->n_resamples * 12345ULL;
        kernel_copy_theta_particles<<<state->N_theta, state->N_inner>>>(
            state->d_particles, state->d_particles_temp, state->d_ancestors,
            state->N_theta, state->N_inner, resample_seed
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        int other_buf = 1 - state->noise_buf;
        kernel_copy_noise_arrays<<<state->N_theta, state->N_inner>>>(
            state->d_z_noise[state->noise_buf],
            state->d_z_noise[other_buf],
            state->d_u0_noise[state->noise_buf],
            state->d_u0_noise[other_buf],
            state->d_ancestors,
            state->N_theta, state->N_inner,
            state->t_current, state->noise_capacity
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        state->noise_buf = other_buf;
        
        ThetaParticlesSoA tmp = state->d_particles;
        state->d_particles = state->d_particles_temp;
        state->d_particles_temp = tmp;
        
        /* Copy checkpoint arrays according to ancestors (if fixed-lag enabled) */
        if (state->fixed_lag_L > 0 && state->t_checkpoint >= 0) {
            /* Use particles_temp inner arrays as scratch */
            kernel_copy_checkpoint<<<state->N_theta, state->N_inner>>>(
                state->d_checkpoint_z,
                state->d_checkpoint_mu_h,
                state->d_checkpoint_var_h,
                state->d_checkpoint_log_w,
                state->d_checkpoint_ll,
                state->d_particles_temp.inner_z,
                state->d_particles_temp.inner_mu_h,
                state->d_particles_temp.inner_var_h,
                state->d_particles_temp.inner_log_w,
                state->d_particles_temp.log_likelihood,
                state->d_ancestors,
                state->N_theta, state->N_inner
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            /* Copy back from scratch to checkpoint */
            int N_total = state->N_theta * state->N_inner;
            CUDA_CHECK(cudaMemcpy(state->d_checkpoint_z, state->d_particles_temp.inner_z,
                                  N_total * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(state->d_checkpoint_mu_h, state->d_particles_temp.inner_mu_h,
                                  N_total * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(state->d_checkpoint_var_h, state->d_particles_temp.inner_var_h,
                                  N_total * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(state->d_checkpoint_log_w, state->d_particles_temp.inner_log_w,
                                  N_total * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(state->d_checkpoint_ll, state->d_particles_temp.log_likelihood,
                                  state->N_theta * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        
        /* CPMMH rejuvenation - dispatch based on N_inner */
        /* Determine if we should use fixed-lag */
        int t_checkpoint_use = -1;
        const float* cp_z = nullptr;
        const float* cp_mu = nullptr;
        const float* cp_var = nullptr;
        const float* cp_logw = nullptr;
        const float* cp_ll = nullptr;
        
        if (state->fixed_lag_L > 0 && state->t_checkpoint >= 0) {
            /* Compute how many steps we'd replay with this checkpoint */
            int steps_to_replay = state->t_current - state->t_checkpoint;
            
            /* Only use checkpoint if:
             * 1. We have at least 1 step to replay (steps_to_replay > 0)
             * 2. Checkpoint is recent enough (within 2L window) */
            if (steps_to_replay > 0 && steps_to_replay <= 2 * state->fixed_lag_L) {
                t_checkpoint_use = state->t_checkpoint;
                cp_z = state->d_checkpoint_z;
                cp_mu = state->d_checkpoint_mu_h;
                cp_var = state->d_checkpoint_var_h;
                cp_logw = state->d_checkpoint_log_w;
                cp_ll = state->d_checkpoint_ll;
            }
        }
        
        #define DISPATCH_CPMMH(N) \
            kernel_cpmmh_rejuvenate_fused_impl<N><<<state->N_theta, N, cpmmh_shared_mem_size<N>()>>>( \
                state->d_particles, state->d_particles_temp, \
                state->d_y_history, \
                curr_noise, other_noise, curr_u0, other_u0, \
                state->t_current, \
                state->N_theta, \
                state->noise_capacity, state->cpmmh_rho, \
                state->d_accepts, state->d_swap_flags, \
                state->user_seed, state->n_rejuv_total / state->N_theta, k % 3, \
                t_checkpoint_use, cp_z, cp_mu, cp_var, cp_logw, cp_ll)
        
        for (int k = 0; k < state->K_rejuv; k++) {
            int h_accepts = 0;
            CUDA_CHECK(cudaMemcpy(state->d_accepts, &h_accepts, sizeof(int), cudaMemcpyHostToDevice));
            
            noise_t* curr_noise = state->d_z_noise[state->noise_buf];
            noise_t* other_noise = state->d_z_noise[1 - state->noise_buf];
            noise_t* curr_u0 = state->d_u0_noise[state->noise_buf];
            noise_t* other_u0 = state->d_u0_noise[1 - state->noise_buf];
            
            switch (state->N_inner) {
                case 64:  DISPATCH_CPMMH(64);  break;
                case 128: DISPATCH_CPMMH(128); break;
                case 256: DISPATCH_CPMMH(256); break;
                case 512: DISPATCH_CPMMH(512); break;
                default:
                    fprintf(stderr, "Unsupported N_inner=%d\n", state->N_inner);
                    exit(EXIT_FAILURE);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            
            /* Compute t_start for noise commit: only commit the modified window */
            int t_start_commit = (t_checkpoint_use >= 0) ? (t_checkpoint_use + 1) : 0;
            
            kernel_commit_accepted_noise<<<state->N_theta, state->N_inner>>>(
                curr_noise, other_noise, curr_u0, other_u0,
                state->d_swap_flags,
                state->N_theta, state->N_inner,
                state->t_current, state->noise_capacity,
                t_start_commit
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaMemcpy(&h_accepts, state->d_accepts, sizeof(int), cudaMemcpyDeviceToHost));
            state->n_rejuv_accepts += h_accepts;
            state->n_rejuv_total += state->N_theta;
        }
        #undef DISPATCH_CPMMH
        
        /*─────────────────────────────────────────────────────────────────────────
         * WEIGHT RESET (SMC² with Refresh)
         * 
         * After resample + MCMC moves, particles are approximate posterior samples.
         * Reset weights to uniform to prevent O(T) weight explosion.
         * 
         * The evidence from y_{1:t} is encoded in:
         *   - Which particles survived resampling
         *   - Where MCMC moved them
         * NOT in cumulative log-likelihoods (which would double-count).
         *─────────────────────────────────────────────────────────────────────────*/
        kernel_reset_outer_weights<<<(state->N_theta + 255) / 256, 256>>>(
            state->d_particles, state->N_theta
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        kernel_compute_outer_ess<<<1, state->N_theta, 32 * sizeof(float)>>>(
            state->d_particles, state->d_ess, state->N_theta
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_ess, state->d_ess, sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    /* Fixed-lag: save checkpoint AFTER CPMMH to ensure we use the previous checkpoint
     * during rejuvenation. This way:
     *   - At t=100: checkpoint saved at t=100 (for use at t=101-199)
     *   - At t=200: CPMMH uses checkpoint from t=100, THEN saves checkpoint at t=200
     */
    if (state->fixed_lag_L > 0) {
        int t_checkpoint_target = (state->t_current / state->fixed_lag_L) * state->fixed_lag_L;
        
        if (t_checkpoint_target > state->t_checkpoint && state->t_current > 0) {
            kernel_save_checkpoint<<<state->N_theta, state->N_inner>>>(
                state->d_particles,
                state->d_checkpoint_z,
                state->d_checkpoint_mu_h,
                state->d_checkpoint_var_h,
                state->d_checkpoint_log_w,
                state->d_checkpoint_ll,
                state->N_theta, state->N_inner
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            state->t_checkpoint = t_checkpoint_target;
        }
    }
    
    return h_ess;
}

void smc2_cuda_get_theta_mean(SMC2StateCUDA* state, float* theta_mean) {
    float* h_weight = (float*)malloc(state->N_theta * sizeof(float));
    float* h_params[8];
    for (int i = 0; i < 8; i++) h_params[i] = (float*)malloc(state->N_theta * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(h_weight, state->d_particles.weight, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[0], state->d_particles.rho, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[1], state->d_particles.sigma_z, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[2], state->d_particles.mu_base, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[3], state->d_particles.mu_scale, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[4], state->d_particles.mu_rate, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[5], state->d_particles.sigma_base, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[6], state->d_particles.sigma_scale, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[7], state->d_particles.sigma_rate, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < 8; i++) theta_mean[i] = 0.0f;
    for (int j = 0; j < state->N_theta; j++) {
        float w = h_weight[j];
        for (int i = 0; i < 8; i++) theta_mean[i] += w * h_params[i][j];
    }
    
    free(h_weight);
    for (int i = 0; i < 8; i++) free(h_params[i]);
}

void smc2_cuda_get_theta_std(SMC2StateCUDA* state, float* theta_std) {
    float theta_mean[8];
    smc2_cuda_get_theta_mean(state, theta_mean);
    
    float* h_weight = (float*)malloc(state->N_theta * sizeof(float));
    float* h_params[8];
    for (int i = 0; i < 8; i++) h_params[i] = (float*)malloc(state->N_theta * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(h_weight, state->d_particles.weight, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[0], state->d_particles.rho, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[1], state->d_particles.sigma_z, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[2], state->d_particles.mu_base, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[3], state->d_particles.mu_scale, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[4], state->d_particles.mu_rate, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[5], state->d_particles.sigma_base, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[6], state->d_particles.sigma_scale, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[7], state->d_particles.sigma_rate, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < 8; i++) theta_std[i] = 0.0f;
    for (int j = 0; j < state->N_theta; j++) {
        float w = h_weight[j];
        for (int i = 0; i < 8; i++) {
            float d = h_params[i][j] - theta_mean[i];
            theta_std[i] += w * d * d;
        }
    }
    for (int i = 0; i < 8; i++) theta_std[i] = sqrtf(theta_std[i]);
    
    free(h_weight);
    for (int i = 0; i < 8; i++) free(h_params[i]);
}

float smc2_cuda_get_outer_ess(SMC2StateCUDA* state) {
    float h_ess;
    CUDA_CHECK(cudaMemcpy(&h_ess, state->d_ess, sizeof(float), cudaMemcpyDeviceToHost));
    return h_ess;
}
