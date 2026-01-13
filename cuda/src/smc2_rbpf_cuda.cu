/**
 * @file smc2_rbpf_cuda.cu
 * @brief SMC² with RBPF Inner Filter - CUDA Implementation
 * 
 * Corrected version with:
 *   - Proper PMMH rejuvenation (O(t) full-history likelihood)
 *   - Fixed ESS calculation
 *   - Outer systematic resampling
 *   - Log-weight reset after resampling
 */

#include "smc2_rbpf_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>      /* For curandGenerator_t */
#include <cuda_fp16.h>   /* For half precision */

/*═══════════════════════════════════════════════════════════════════════════
 * OCSN Constants in Constant Memory
 *═══════════════════════════════════════════════════════════════════════════*/

__constant__ float d_OCSN_WEIGHTS[OCSN_K] = {
    0.00609f, 0.04775f, 0.13057f, 0.20674f, 0.22715f,
    0.18842f, 0.12047f, 0.05591f, 0.01575f, 0.00115f
};

__constant__ float d_OCSN_MEANS[OCSN_K] = {
    -10.12999f, -3.97281f, -8.56686f, 2.77786f, 0.61942f,
    1.79518f, -1.08819f, -5.21004f, 2.89816f, 1.55000f
};

__constant__ float d_OCSN_VARS[OCSN_K] = {
    5.79596f, 2.61369f, 5.17950f, 0.16735f, 0.64009f,
    0.34023f, 1.26261f, 2.61290f, 0.26460f, 0.17788f
};

__constant__ float d_OCSN_LOG_WEIGHTS[OCSN_K] = {
    -5.10100f, -3.04200f, -2.03500f, -1.57700f, -1.48200f,
    -1.66900f, -2.11600f, -2.88400f, -4.15100f, -6.76800f
};

__constant__ SVPrior d_prior;
__constant__ SVBounds d_bounds;
__constant__ SVCurve d_theta_curve;
__constant__ float d_proposal_std[8];

/*═══════════════════════════════════════════════════════════════════════════
 * Error Checking
 *═══════════════════════════════════════════════════════════════════════════*/

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/*═══════════════════════════════════════════════════════════════════════════
 * Warp/Block Reductions
 *═══════════════════════════════════════════════════════════════════════════*/

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float block_reduce_sum(float val, volatile float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    
    return shared[0];
}

__device__ float block_reduce_max(float val, volatile float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -1e30f;
    if (wid == 0) val = warp_reduce_max(val);
    
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    
    return shared[0];
}

/*═══════════════════════════════════════════════════════════════════════════
 * Helpers
 *═══════════════════════════════════════════════════════════════════════════*/

__device__ __forceinline__ float eval_curve(float base, float scale, float rate, float z) {
    return base + scale * (1.0f - __expf(-rate * z));
}

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Log Prior for θ parameters
 * Returns -INFINITY if out of bounds, otherwise sum of Gaussian log-densities
 *═══════════════════════════════════════════════════════════════════════════*/

__device__ float log_prior_theta(
    float rho, float sigma_z,
    float mu_base, float mu_scale, float mu_rate,
    float sigma_base, float sigma_scale, float sigma_rate
) {
    /* Bounds check - return -inf if any parameter out of bounds */
    if (rho < d_bounds.rho_min || rho > d_bounds.rho_max) return -INFINITY;
    if (sigma_z < d_bounds.sigma_z_min || sigma_z > d_bounds.sigma_z_max) return -INFINITY;
    if (mu_base < d_bounds.mu_base_min || mu_base > d_bounds.mu_base_max) return -INFINITY;
    if (mu_scale < d_bounds.mu_scale_min || mu_scale > d_bounds.mu_scale_max) return -INFINITY;
    if (mu_rate < d_bounds.mu_rate_min || mu_rate > d_bounds.mu_rate_max) return -INFINITY;
    if (sigma_base < d_bounds.sigma_base_min || sigma_base > d_bounds.sigma_base_max) return -INFINITY;
    if (sigma_scale < d_bounds.sigma_scale_min || sigma_scale > d_bounds.sigma_scale_max) return -INFINITY;
    if (sigma_rate < d_bounds.sigma_rate_min || sigma_rate > d_bounds.sigma_rate_max) return -INFINITY;
    
    /* Gaussian log-prior (ignoring normalization constants - they cancel in MH ratio) */
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

/*═══════════════════════════════════════════════════════════════════════════
 * Derive u0 from z_noise - eliminates separate u0 storage
 * Maps correlated Gaussian noise to [0,1) for systematic resampling
 *═══════════════════════════════════════════════════════════════════════════*/

__device__ __forceinline__ float u0_from_noise(float z_noise) {
    /* Use fractional part to map to [0,1) 
     * The 0.5 offset centers the distribution
     * The 0.1 scale ensures good spread */
    float u = 0.5f + 0.1f * z_noise;
    u = u - floorf(u);  /* fract() */
    return fmaxf(1e-7f, fminf(1.0f - 1e-7f, u));  /* Clamp away from 0,1 */
}

/*═══════════════════════════════════════════════════════════════════════════
 * OCSN Kalman Update
 *═══════════════════════════════════════════════════════════════════════════*/

__device__ void ocsn_kalman_update(
    float y, float mu_pred, float var_pred,
    float* mu_post_out, float* var_post_out, float* log_lik_out
) {
    float log_alpha_tilde[OCSN_K];
    float S[OCSN_K], K[OCSN_K], mu_k[OCSN_K], var_k[OCSN_K];
    float log_max = -1e30f;
    
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        S[k] = var_pred + d_OCSN_VARS[k];
        float innov = y - mu_pred - d_OCSN_MEANS[k] + OCSN_OFFSET;
        log_alpha_tilde[k] = d_OCSN_LOG_WEIGHTS[k] 
                           - 0.5f * __logf(S[k])
                           - 0.5f * innov * innov / S[k];
        log_max = fmaxf(log_max, log_alpha_tilde[k]);
    }
    
    float sum_exp = 0.0f;
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        sum_exp += __expf(log_alpha_tilde[k] - log_max);
    }
    float log_norm = log_max + __logf(sum_exp);
    
    float alpha[OCSN_K];
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        alpha[k] = __expf(log_alpha_tilde[k] - log_norm);
    }
    
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        K[k] = var_pred / S[k];
        float innov = y - mu_pred - d_OCSN_MEANS[k] + OCSN_OFFSET;
        mu_k[k] = mu_pred + K[k] * innov;
        var_k[k] = (1.0f - K[k]) * var_pred;
    }
    
    float mu_post = 0.0f, E_h_sq = 0.0f;
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        mu_post += alpha[k] * mu_k[k];
        E_h_sq += alpha[k] * (var_k[k] + mu_k[k] * mu_k[k]);
    }
    
    float var_post = fmaxf(E_h_sq - mu_post * mu_post, 1e-6f);
    
    *mu_post_out = mu_post;
    *var_post_out = var_post;
    *log_lik_out = log_norm;
}

/*═══════════════════════════════════════════════════════════════════════════
 * RNG Initialization
 *═══════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_init_rng(curandState* states, unsigned long long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * Initialize θ-particles from Prior (with noise storage for CPMMH)
 *═══════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_init_from_prior(
    ThetaParticlesSoA particles,
    int N_theta, int N_inner,
    half* d_z_noise,    /* Store t=0 z-noise for CPMMH (FP16) */
    int noise_capacity
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_inner + inner_idx;
    
    if (theta_idx >= N_theta) return;
    
    curandState* rng = &particles.rng_states[global_idx];
    
    __shared__ float s_rho, s_sigma_z;
    __shared__ float s_mu_base, s_mu_scale, s_mu_rate;
    __shared__ float s_sigma_base, s_sigma_scale, s_sigma_rate;
    
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
    
    float one_minus_rho_sq = fmaxf(1.0f - rho * rho, 1e-6f);
    float z_stat_std = sigma_z / sqrtf(one_minus_rho_sq);
    
    /* Generate and STORE t=0 z-noise (FP16) */
    float z_noise_init = curand_normal(rng);
    int64_t z_noise_idx = (int64_t)theta_idx * N_inner * (noise_capacity + 1) + inner_idx;  /* t=0 slot */
    d_z_noise[z_noise_idx] = __float2half(z_noise_init);
    
    float z = z_stat_std * z_noise_init;
    z = clampf(z, 0.0f, 3.0f);
    
    float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z);
    float mu_z = eval_curve(s_mu_base, s_mu_scale, s_mu_rate, z);
    float sigma_h = eval_curve(s_sigma_base, s_sigma_scale, s_sigma_rate, z);
    
    float phi = 1.0f - theta_z;
    float one_minus_phi_sq = fmaxf(1.0f - phi * phi, 1e-6f);
    float h_stat_var = (sigma_h * sigma_h) / one_minus_phi_sq;
    
    particles.inner_z[global_idx] = z;
    particles.inner_mu_h[global_idx] = mu_z;
    particles.inner_var_h[global_idx] = h_stat_var;
    particles.inner_log_w[global_idx] = -__logf((float)N_inner);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Inner RBPF Step - OPTIMIZED for CPMMH
 * 
 * Optimizations:
 *   1. FP16 noise storage (half bandwidth)
 *   2. u0 derived from z_noise (no separate storage)
 *   3. Always resample (R1) for deterministic replay coupling
 *═══════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_rbpf_step(
    ThetaParticlesSoA particles,
    float y_obs,
    int N_theta, int N_inner,
    half* d_z_noise,   /* [N_theta * N_inner * (T+1)] - FP16 z-innovations */
    int t_current,     /* Current timestep (0-indexed) */
    int noise_capacity /* Max T for noise arrays */
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_inner + inner_idx;
    
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    
    extern __shared__ float shared_mem[];
    float* s_reduction = shared_mem;
    float* s_weights = &shared_mem[32];
    float* s_cumsum = &shared_mem[32 + N_inner];
    
    __shared__ float s_rho, s_sigma_z;
    __shared__ float s_mu_base, s_mu_scale, s_mu_rate;
    __shared__ float s_sigma_base, s_sigma_scale, s_sigma_rate;
    __shared__ float s_log_max, s_sum_w, s_u0;
    __shared__ float s_z_noise_0;  /* Thread 0's noise for u0 derivation */
    
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
    
    float z = particles.inner_z[global_idx];
    float mu_h = particles.inner_mu_h[global_idx];
    float var_h = particles.inner_var_h[global_idx];
    float log_w = particles.inner_log_w[global_idx];
    
    /* Noise index: slot t+1 for propagation (slot 0 is init) */
    int64_t z_noise_base = (int64_t)theta_idx * N_inner * (noise_capacity + 1);
    int64_t z_noise_idx = z_noise_base + (int64_t)(t_current + 1) * N_inner + inner_idx;
    
    /* STEP 1: Generate z-noise FIRST, store it, derive u0 from thread 0's noise */
    float z_noise = curand_normal(&local_rng);
    d_z_noise[z_noise_idx] = __float2half(z_noise);
    
    /* Share thread 0's noise for u0 derivation */
    if (inner_idx == 0) {
        s_z_noise_0 = z_noise;
    }
    __syncthreads();
    
    /* STEP 2: ALWAYS RESAMPLE (R1) with u0 derived from z_noise */
    {
        float log_max = block_reduce_max(log_w, s_reduction);
        if (inner_idx == 0) s_log_max = log_max;
        __syncthreads();
        
        float w_unnorm = __expf(log_w - s_log_max);
        float sum_w = block_reduce_sum(w_unnorm, s_reduction);
        if (inner_idx == 0) s_sum_w = sum_w;
        __syncthreads();
        
        s_weights[inner_idx] = w_unnorm / s_sum_w;
        __syncthreads();
        
        if (inner_idx == 0) {
            s_cumsum[0] = s_weights[0];
            for (int i = 1; i < N_inner; i++) {
                s_cumsum[i] = s_cumsum[i-1] + s_weights[i];
            }
            s_cumsum[N_inner - 1] = 1.0f;
            
            /* Derive u0 from thread 0's z_noise - NO SEPARATE STORAGE */
            s_u0 = u0_from_noise(s_z_noise_0);
        }
        __syncthreads();
        
        float u = (s_u0 + (float)inner_idx) / (float)N_inner;
        int lo = 0, hi = N_inner - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (s_cumsum[mid] < u) lo = mid + 1;
            else hi = mid;
        }
        int ancestor = lo;
        
        z = particles.inner_z[theta_idx * N_inner + ancestor];
        mu_h = particles.inner_mu_h[theta_idx * N_inner + ancestor];
        var_h = particles.inner_var_h[theta_idx * N_inner + ancestor];
        log_w = -__logf((float)N_inner);
        
        __syncthreads();
    }
    
    /* STEP 3: Propagate z using the already-generated noise */
    float z_new = s_rho * z + s_sigma_z * z_noise;
    z_new = clampf(z_new, 0.0f, 3.0f);
    
    /* Kalman predict */
    float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z_new);
    float mu_z = eval_curve(s_mu_base, s_mu_scale, s_mu_rate, z_new);
    float sigma_h = eval_curve(s_sigma_base, s_sigma_scale, s_sigma_rate, z_new);
    
    float phi = 1.0f - theta_z;
    float mu_pred = phi * mu_h + theta_z * mu_z;
    float var_pred = phi * phi * var_h + sigma_h * sigma_h;
    var_pred = fmaxf(var_pred, 1e-8f);
    
    /* OCSN update */
    float mu_post, var_post, log_lik;
    ocsn_kalman_update(y_obs, mu_pred, var_pred, &mu_post, &var_post, &log_lik);
    
    log_w += log_lik;
    
    /* Normalize and compute ESS for outer level */
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
    
    float ll_incr = s_log_max + __logf(fmaxf(s_sum_w, 1e-30f)) - __logf((float)N_inner);
    
    /* Store */
    particles.inner_z[global_idx] = z_new;
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

/*═══════════════════════════════════════════════════════════════════════════
 * Compute Outer ESS
 *═══════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_compute_outer_ess(
    ThetaParticlesSoA particles,
    float* d_ess_out,
    int N_theta
) {
    extern __shared__ float s_data[];
    int idx = threadIdx.x;
    
    float log_w = (idx < N_theta) ? particles.log_weight[idx] : -1e30f;
    
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
    
    float w_sq = (idx < N_theta) ? w * w : 0.0f;
    float sum_w_sq = block_reduce_sum(w_sq, s_data);
    
    if (idx == 0) {
        *d_ess_out = 1.0f / fmaxf(sum_w_sq, 1e-30f);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * Outer Resampling
 *═══════════════════════════════════════════════════════════════════════════*/

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
    
    if (idx == 0) {
        for (int i = 1; i < N_theta; i++) s_cumsum[i] += s_cumsum[i-1];
        s_cumsum[N_theta - 1] = 1.0f;
    }
    __syncthreads();
    
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

/*═══════════════════════════════════════════════════════════════════════════
 * Copy θ-particles After Resampling
 *═══════════════════════════════════════════════════════════════════════════*/

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
    
    if (inner_idx == 0) {
        dst.rho[theta_idx] = src.rho[ancestor];
        dst.sigma_z[theta_idx] = src.sigma_z[ancestor];
        dst.mu_base[theta_idx] = src.mu_base[ancestor];
        dst.mu_scale[theta_idx] = src.mu_scale[ancestor];
        dst.mu_rate[theta_idx] = src.mu_rate[ancestor];
        dst.sigma_base[theta_idx] = src.sigma_base[ancestor];
        dst.sigma_scale[theta_idx] = src.sigma_scale[ancestor];
        dst.sigma_rate[theta_idx] = src.sigma_rate[ancestor];
        
        dst.log_weight[theta_idx] = 0.0f;  /* Reset after resampling */
        dst.weight[theta_idx] = 1.0f / N_theta;
        dst.log_likelihood[theta_idx] = src.log_likelihood[ancestor];
        dst.ess_inner[theta_idx] = src.ess_inner[ancestor];
    }
    
    if (inner_idx < N_inner) {
        int src_idx = ancestor * N_inner + inner_idx;
        int dst_idx = theta_idx * N_inner + inner_idx;
        
        dst.inner_z[dst_idx] = src.inner_z[src_idx];
        dst.inner_mu_h[dst_idx] = src.inner_mu_h[src_idx];
        dst.inner_var_h[dst_idx] = src.inner_var_h[src_idx];
        dst.inner_log_w[dst_idx] = src.inner_log_w[src_idx];
        
        /* CRITICAL: Re-initialize RNG with unique seed per particle
         * Otherwise all particles copied from same ancestor have identical RNG! */
        curand_init(resample_seed, dst_idx, 0, &dst.rng_states[dst_idx]);
    }
}

/* Copy noise arrays after outer resampling - PING-PONG VERSION
 * Reads from src buffer, writes to dst buffer based on ancestors
 * No temp storage needed - just swap buffer index after */
__global__ void kernel_copy_noise_arrays(
    const half* src_z_noise,  /* Read from current buffer */
    half* dst_z_noise,        /* Write to other buffer */
    const int* d_ancestors,
    int N_theta, int N_inner,
    int t_current, int noise_capacity
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    
    int ancestor = d_ancestors[theta_idx];
    
    int64_t dst_z_base = (int64_t)theta_idx * N_inner * (noise_capacity + 1);
    int64_t src_z_base = (int64_t)ancestor * N_inner * (noise_capacity + 1);
    
    /* Copy z-noise for all timesteps up to t_current+1 */
    for (int t = 0; t <= t_current + 1; t++) {
        int64_t src_idx = src_z_base + t * N_inner + inner_idx;
        int64_t dst_idx = dst_z_base + t * N_inner + inner_idx;
        dst_z_noise[dst_idx] = src_z_noise[src_idx];
    }
    /* u0 derived from z_noise - no separate copy needed */
}

/*═══════════════════════════════════════════════════════════════════════════
 * FUSED CPMMH Rejuvenation Kernel - ALL OPTIMIZATIONS
 * 
 * Optimizations applied:
 *   1. FUSED: Generate fresh noise + correlate + replay + MH in one kernel
 *   2. FP16: Half precision noise storage (half bandwidth)
 *   3. NO u0 ARRAYS: u0 derived from z_noise (zero extra storage)
 *   4. NO ESS in replay: Always resample (R1)
 *   5. PING-PONG: On accept, swap buffer index instead of copying
 * 
 * Cost: 1 kernel launch, 1 global memory pass for noise
 *═══════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_cpmmh_rejuvenate_fused(
    ThetaParticlesSoA particles,
    ThetaParticlesSoA particles_scratch,
    const float* y_history,
    half* d_z_noise_curr,      /* Current noise buffer (FP16) */
    half* d_z_noise_other,     /* Other ping-pong buffer (FP16) */
    int t_current,
    int N_theta, int N_inner,
    int noise_capacity,
    float cpmmh_rho,
    int* d_accepts,
    int* d_swap_flags          /* Per-particle: 1 if accepted (needs buffer swap) */
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_inner + inner_idx;
    
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    
    extern __shared__ float shared_mem[];
    float* s_reduction = shared_mem;
    float* s_z = &shared_mem[32];
    float* s_mu = &shared_mem[32 + N_inner];
    float* s_var = &shared_mem[32 + 2 * N_inner];
    float* s_cdf = &shared_mem[32 + 3 * N_inner];
    
    __shared__ float s_log_max, s_sum_w;
    __shared__ float s_ess_prop;
    
    __shared__ float s_rho_curr, s_sigma_z_curr;
    __shared__ float s_mu_base_curr, s_mu_scale_curr, s_mu_rate_curr;
    __shared__ float s_sigma_base_curr, s_sigma_scale_curr, s_sigma_rate_curr;
    
    __shared__ float s_rho_prop, s_sigma_z_prop;
    __shared__ float s_mu_base_prop, s_mu_scale_prop, s_mu_rate_prop;
    __shared__ float s_sigma_base_prop, s_sigma_scale_prop, s_sigma_rate_prop;
    
    __shared__ float s_ll_curr, s_ll_prop;
    __shared__ float s_lp_curr, s_lp_prop;
    __shared__ int s_accept, s_valid;
    __shared__ float s_u0_shared;  /* For derived u0 in replay */
    
    curandState local_rng = particles.rng_states[global_idx];
    
    /* Thread 0: Load current θ, propose θ', compute log priors */
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
        
        /* ONE-SHOT proposal */
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
    
    /* Noise base index for this θ-particle */
    int64_t z_noise_base = (int64_t)theta_idx * N_inner * (noise_capacity + 1);
    
    /* CPMMH correlation scale */
    float scale = sqrtf(1.0f - cpmmh_rho * cpmmh_rho);
    
    /*═══════════════════════════════════════════════════════════════════════
     * INLINE REPLAY with FUSED noise generation and correlation
     * Generate fresh → correlate → use immediately (all in registers)
     *═══════════════════════════════════════════════════════════════════════*/
    
    float rho = s_rho_prop;
    float sigma_z = s_sigma_z_prop;
    float mu_base = s_mu_base_prop;
    float mu_scale = s_mu_scale_prop;
    float mu_rate = s_mu_rate_prop;
    float sigma_base = s_sigma_base_prop;
    float sigma_scale = s_sigma_scale_prop;
    float sigma_rate = s_sigma_rate_prop;
    
    /* Initialize from stationary using t=0 noise (FUSED: generate + correlate) */
    float one_minus_rho_sq = fmaxf(1.0f - rho * rho, 1e-6f);
    float z_stat_std = sigma_z / sqrtf(one_minus_rho_sq);
    
    float z_noise_curr_0 = __half2float(d_z_noise_curr[z_noise_base + inner_idx]);
    float z_noise_fresh_0 = curand_normal(&local_rng);
    float z_noise_prop_0 = cpmmh_rho * z_noise_curr_0 + scale * z_noise_fresh_0;
    
    /* Store correlated noise to other buffer for potential swap */
    d_z_noise_other[z_noise_base + inner_idx] = __float2half(z_noise_prop_0);
    
    float z = z_stat_std * z_noise_prop_0;
    z = clampf(z, 0.0f, 3.0f);
    
    float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z);
    float mu_z_val = eval_curve(mu_base, mu_scale, mu_rate, z);
    float sigma_h = eval_curve(sigma_base, sigma_scale, sigma_rate, z);
    float phi = 1.0f - theta_z;
    float h_stat_var = (sigma_h * sigma_h) / fmaxf(1.0f - phi * phi, 1e-6f);
    
    float mu_h = mu_z_val;
    float var_h = h_stat_var;
    float log_w = -__logf((float)N_inner);
    float ll_accum = 0.0f;
    
    /* Process all observations */
    for (int t = 0; t <= t_current; t++) {
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
        
        /* ALWAYS RESAMPLE (R1) */
        s_z[inner_idx] = z;
        s_mu[inner_idx] = mu_h;
        s_var[inner_idx] = var_h;
        s_cdf[inner_idx] = w_norm;
        __syncthreads();
        
        if (inner_idx == 0) {
            for (int i = 1; i < N_inner; i++) {
                s_cdf[i] += s_cdf[i-1];
            }
            s_cdf[N_inner - 1] = 1.0f;
        }
        __syncthreads();
        
        /* FUSED: Generate fresh noise for timestep t+1, correlate, derive u0 */
        int64_t z_idx_t1 = z_noise_base + (int64_t)(t + 1) * N_inner + inner_idx;
        float z_noise_curr_t1 = __half2float(d_z_noise_curr[z_idx_t1]);
        float z_noise_fresh_t1 = curand_normal(&local_rng);
        float z_noise_prop_t1 = cpmmh_rho * z_noise_curr_t1 + scale * z_noise_fresh_t1;
        
        /* Store to other buffer */
        d_z_noise_other[z_idx_t1] = __float2half(z_noise_prop_t1);
        
        /* Derive u0 from thread 0's correlated noise (shared) */
        if (inner_idx == 0) {
            /* Use the first element's noise to derive u0 */
            float z0_for_u0 = z_noise_prop_t1;
            s_u0_shared = u0_from_noise(z0_for_u0);
        }
        __syncthreads();
        
        float u = (s_u0_shared + (float)inner_idx) / (float)N_inner;
        int lo = 0, hi = N_inner - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (s_cdf[mid] < u) lo = mid + 1;
            else hi = mid;
        }
        
        z = s_z[lo];
        mu_h = s_mu[lo];
        var_h = s_var[lo];
        log_w = -__logf((float)N_inner);
        __syncthreads();
        
        /* Propagate z using correlated noise */
        float z_new = rho * z + sigma_z * z_noise_prop_t1;
        z_new = clampf(z_new, 0.0f, 3.0f);
        
        /* Kalman predict */
        theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z_new);
        mu_z_val = eval_curve(mu_base, mu_scale, mu_rate, z_new);
        sigma_h = eval_curve(sigma_base, sigma_scale, sigma_rate, z_new);
        phi = 1.0f - theta_z;
        
        float mu_pred = phi * mu_h + theta_z * mu_z_val;
        float var_pred = phi * phi * var_h + sigma_h * sigma_h;
        var_pred = fmaxf(var_pred, 1e-8f);
        
        /* OCSN update */
        float mu_post, var_post, log_lik;
        ocsn_kalman_update(y_obs, mu_pred, var_pred, &mu_post, &var_post, &log_lik);
        
        log_w += log_lik;
        
        /* Accumulate log-likelihood */
        log_max = block_reduce_max(log_w, s_reduction);
        if (inner_idx == 0) s_log_max = log_max;
        __syncthreads();
        log_max = s_log_max;
        
        w_unnorm = __expf(log_w - log_max);
        sum_w = block_reduce_sum(w_unnorm, s_reduction);
        if (inner_idx == 0) s_sum_w = sum_w;
        __syncthreads();
        sum_w = s_sum_w;
        
        float ll_incr = log_max + __logf(fmaxf(sum_w, 1e-30f)) - __logf((float)N_inner);
        ll_accum += ll_incr;
        
        z = z_new;
        mu_h = mu_post;
        var_h = var_post;
    }
    
    /* Compute final ESS for diagnostics */
    float w_norm = __expf(log_w - s_log_max) / fmaxf(s_sum_w, 1e-30f);
    float w_sq = w_norm * w_norm;
    float sum_w_sq = block_reduce_sum(w_sq, s_reduction);
    float ess = 1.0f / fmaxf(sum_w_sq, 1e-30f);
    
    /* Store proposed PF state to scratch */
    particles_scratch.inner_z[global_idx] = z;
    particles_scratch.inner_mu_h[global_idx] = mu_h;
    particles_scratch.inner_var_h[global_idx] = var_h;
    particles_scratch.inner_log_w[global_idx] = log_w;
    
    if (inner_idx == 0) {
        s_ll_prop = ll_accum;
        s_ess_prop = ess;
    }
    __syncthreads();
    
    /*═══════════════════════════════════════════════════════════════════════
     * MH Accept/Reject
     *═══════════════════════════════════════════════════════════════════════*/
    if (inner_idx == 0) {
        float log_alpha = (s_ll_prop + s_lp_prop) - (s_ll_curr + s_lp_curr);
        
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
            particles.log_likelihood[theta_idx] = s_ll_prop;
            particles.ess_inner[theta_idx] = s_ess_prop;
            atomicAdd(d_accepts, 1);
        }
        
        /* Record whether this particle accepted (for buffer swap) */
        d_swap_flags[theta_idx] = s_accept;
    }
    __syncthreads();
    
    /* On accept: copy PF state from scratch */
    if (s_accept) {
        particles.inner_z[global_idx] = particles_scratch.inner_z[global_idx];
        particles.inner_mu_h[global_idx] = particles_scratch.inner_mu_h[global_idx];
        particles.inner_var_h[global_idx] = particles_scratch.inner_var_h[global_idx];
        particles.inner_log_w[global_idx] = particles_scratch.inner_log_w[global_idx];
        /* Noise already written to other buffer - swap handled by host */
    }
    
    particles.rng_states[global_idx] = local_rng;
}

/* Kernel to selectively swap noise buffers for accepted particles */
__global__ void kernel_swap_noise_for_accepted(
    half* d_z_noise_0,
    half* d_z_noise_1,
    const int* d_swap_flags,
    int N_theta, int N_inner,
    int t_current, int noise_capacity
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    if (d_swap_flags[theta_idx] == 0) return;  /* Not accepted, no swap */
    
    int64_t z_base = (int64_t)theta_idx * N_inner * (noise_capacity + 1);
    
    /* Swap by copying from buffer 1 to buffer 0 (or vice versa handled by caller) */
    for (int t = 0; t <= t_current + 1; t++) {
        int64_t idx = z_base + t * N_inner + inner_idx;
        d_z_noise_0[idx] = d_z_noise_1[idx];
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * Host API
 *═══════════════════════════════════════════════════════════════════════════*/

SMC2StateCUDA* smc2_cuda_alloc(int N_theta, int N_inner) {
    SMC2StateCUDA* state = (SMC2StateCUDA*)calloc(1, sizeof(SMC2StateCUDA));
    if (!state) return NULL;
    
    state->N_theta = N_theta;
    state->N_inner = N_inner;
    state->ess_threshold_outer = 0.5f;
    state->ess_threshold_inner = 0.5f;
    state->K_rejuv = 1;  /* One correct PMMH move per resample is sufficient */
    
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
    
    /* Init RNG */
    kernel_init_rng<<<(N_total + 255) / 256, 256>>>(state->d_particles.rng_states, 12345ULL, N_total);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    /* Observation history */
    state->y_history_capacity = 8000;
    CUDA_CHECK(cudaMalloc(&state->d_y_history, state->y_history_capacity * sizeof(float)));
    state->y_history_len = 0;
    
    /* Scratch */
    CUDA_CHECK(cudaMalloc(&state->d_ancestors, N_theta * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_uniform, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_ess, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_accepts, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_swap_flags, N_theta * sizeof(int)));
    
    /* CPMMH: Allocate PING-PONG noise buffers (FP16 for bandwidth reduction)
     * No u0 arrays - derived from z_noise */
    state->noise_capacity = 2048;
    state->cpmmh_rho = 0.99f;  /* High correlation for variance reduction */
    state->noise_buf = 0;     /* Start with buffer 0 */
    
    int64_t z_noise_size = (int64_t)N_theta * N_inner * (state->noise_capacity + 1);
    
    /* Two ping-pong buffers for z_noise (FP16) */
    CUDA_CHECK(cudaMalloc(&state->d_z_noise[0], z_noise_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&state->d_z_noise[1], z_noise_size * sizeof(half)));
    
    /* Initialize buffer 0 with N(0,1) noise - need to generate FP32 then convert */
    float* temp_noise;
    CUDA_CHECK(cudaMalloc(&temp_noise, z_noise_size * sizeof(float)));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 54321ULL);
    curandGenerateNormal(gen, temp_noise, z_noise_size, 0.0f, 1.0f);
    curandDestroyGenerator(gen);
    
    /* Convert FP32 to FP16 on host - use simple truncation */
    {
        uint16_t* h_temp = (uint16_t*)malloc(z_noise_size * sizeof(uint16_t));
        float* f_temp = (float*)malloc(z_noise_size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(f_temp, temp_noise, z_noise_size * sizeof(float), cudaMemcpyDeviceToHost));
        for (int64_t i = 0; i < z_noise_size; i++) {
            /* Simple FP32 to FP16 conversion (IEEE 754) */
            float val = f_temp[i];
            /* Clamp to FP16 range */
            if (val > 65504.0f) val = 65504.0f;
            if (val < -65504.0f) val = -65504.0f;
            /* Convert using union */
            union { float f; uint32_t u; } fu;
            fu.f = val;
            uint32_t f32 = fu.u;
            uint16_t sign = (f32 >> 16) & 0x8000;
            int32_t exp = ((f32 >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (f32 >> 13) & 0x3FF;
            if (exp <= 0) {
                h_temp[i] = sign;  /* Flush to zero */
            } else if (exp >= 31) {
                h_temp[i] = sign | 0x7C00;  /* Infinity */
            } else {
                h_temp[i] = sign | (exp << 10) | mant;
            }
        }
        CUDA_CHECK(cudaMemcpy(state->d_z_noise[0], h_temp, z_noise_size * sizeof(uint16_t), cudaMemcpyHostToDevice));
        free(h_temp);
        free(f_temp);
    }
    cudaFree(temp_noise);
    
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
    
    /* Proposal std - tuned for ~15-30% acceptance with correct likelihood */
    state->proposal_std[0] = 0.01f;   /* rho */
    state->proposal_std[1] = 0.02f;   /* sigma_z */
    state->proposal_std[2] = 0.1f;    /* mu_base */
    state->proposal_std[3] = 0.1f;    /* mu_scale */
    state->proposal_std[4] = 0.15f;   /* mu_rate */
    state->proposal_std[5] = 0.02f;   /* sigma_base */
    state->proposal_std[6] = 0.02f;   /* sigma_scale */
    state->proposal_std[7] = 0.15f;   /* sigma_rate */
    
    return state;
}

void smc2_cuda_free(SMC2StateCUDA* state) {
    if (!state) return;
    
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
    
    /* CPMMH ping-pong noise buffers (FP16) */
    cudaFree(state->d_z_noise[0]);
    cudaFree(state->d_z_noise[1]);
    
    free(state);
}

void smc2_cuda_set_noise_capacity(SMC2StateCUDA* state, int capacity) {
    if (capacity <= state->noise_capacity) return;  /* Already big enough */
    
    int64_t new_z_size = (int64_t)state->N_theta * state->N_inner * (capacity + 1);
    int64_t old_z_size = (int64_t)state->N_theta * state->N_inner * (state->noise_capacity + 1);
    
    half *new_z_0, *new_z_1;
    CUDA_CHECK(cudaMalloc(&new_z_0, new_z_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&new_z_1, new_z_size * sizeof(half)));
    
    /* Copy existing data if any */
    if (state->d_z_noise[0] && old_z_size > 0) {
        CUDA_CHECK(cudaMemcpy(new_z_0, state->d_z_noise[0], 
                              old_z_size * sizeof(half), cudaMemcpyDeviceToDevice));
    }
    
    cudaFree(state->d_z_noise[0]);
    cudaFree(state->d_z_noise[1]);
    
    state->d_z_noise[0] = new_z_0;
    state->d_z_noise[1] = new_z_1;
    state->noise_buf = 0;
    state->noise_capacity = capacity;
}

void smc2_cuda_init_from_prior(SMC2StateCUDA* state) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_prior, &state->prior, sizeof(SVPrior)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_bounds, &state->bounds, sizeof(SVBounds)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_theta_curve, &state->theta_curve, sizeof(SVCurve)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_std, state->proposal_std, 8 * sizeof(float)));
    
    kernel_init_from_prior<<<state->N_theta, state->N_inner>>>(
        state->d_particles, state->N_theta, state->N_inner,
        state->d_z_noise[state->noise_buf], state->noise_capacity
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
    
    /* Check if noise capacity needs to grow */
    if (state->t_current >= state->noise_capacity) {
        int new_cap = state->noise_capacity * 2;
        int64_t new_z_size = (int64_t)state->N_theta * state->N_inner * (new_cap + 1);
        int64_t old_z_size = (int64_t)state->N_theta * state->N_inner * (state->noise_capacity + 1);
        
        half *new_z_0, *new_z_1;
        CUDA_CHECK(cudaMalloc(&new_z_0, new_z_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&new_z_1, new_z_size * sizeof(half)));
        
        /* Copy old data from current buffer */
        CUDA_CHECK(cudaMemcpy(new_z_0, state->d_z_noise[state->noise_buf], 
                              old_z_size * sizeof(half), cudaMemcpyDeviceToDevice));
        
        cudaFree(state->d_z_noise[0]);
        cudaFree(state->d_z_noise[1]);
        
        state->d_z_noise[0] = new_z_0;
        state->d_z_noise[1] = new_z_1;
        state->noise_buf = 0;  /* Reset to buffer 0 */
        state->noise_capacity = new_cap;
    }
    
    size_t shared_size = (32 + 2 * state->N_inner) * sizeof(float);
    
    /* Forward filter step - uses current noise buffer */
    kernel_rbpf_step<<<state->N_theta, state->N_inner, shared_size>>>(
        state->d_particles, y_obs,
        state->N_theta, state->N_inner,
        state->d_z_noise[state->noise_buf],
        state->t_current, state->noise_capacity
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    kernel_compute_outer_ess<<<1, state->N_theta, 32 * sizeof(float)>>>(
        state->d_particles, state->d_ess, state->N_theta
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float h_ess;
    CUDA_CHECK(cudaMemcpy(&h_ess, state->d_ess, sizeof(float), cudaMemcpyDeviceToHost));
    
    if (h_ess < state->ess_threshold_outer * state->N_theta) {
        state->n_resamples++;
        
        /* Generate uniform for outer resampling */
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL) + state->n_resamples);
        curandGenerateUniform(gen, state->d_uniform, 1);
        curandDestroyGenerator(gen);
        
        /* Resample θ-particles */
        kernel_outer_resample<<<1, state->N_theta, state->N_theta * sizeof(float)>>>(
            state->d_particles, state->d_ancestors, state->d_uniform, state->N_theta
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        /* Copy particles - pass unique seed for RNG re-initialization */
        unsigned long long resample_seed = time(NULL) * 1000ULL + state->n_resamples * 12345ULL;
        kernel_copy_theta_particles<<<state->N_theta, state->N_inner>>>(
            state->d_particles, state->d_particles_temp, state->d_ancestors,
            state->N_theta, state->N_inner, resample_seed
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        /* Copy noise arrays using ping-pong: curr → other buffer */
        int other_buf = 1 - state->noise_buf;
        kernel_copy_noise_arrays<<<state->N_theta, state->N_inner>>>(
            state->d_z_noise[state->noise_buf],  /* Source */
            state->d_z_noise[other_buf],          /* Destination */
            state->d_ancestors,
            state->N_theta, state->N_inner,
            state->t_current, state->noise_capacity
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        /* Swap to the new buffer (noise is now in other_buf) */
        state->noise_buf = other_buf;
        
        /* Swap particle pointers */
        ThetaParticlesSoA tmp = state->d_particles;
        state->d_particles = state->d_particles_temp;
        state->d_particles_temp = tmp;
        
        /* FUSED CPMMH rejuvenation - ONE kernel does everything */
        size_t pmmh_shared_size = (32 + 4 * state->N_inner) * sizeof(float);
        
        for (int k = 0; k < state->K_rejuv; k++) {
            int h_accepts = 0;
            CUDA_CHECK(cudaMemcpy(state->d_accepts, &h_accepts, sizeof(int), cudaMemcpyHostToDevice));
            
            /* Current buffer and other buffer for ping-pong during CPMMH */
            half* curr_noise = state->d_z_noise[state->noise_buf];
            half* other_noise = state->d_z_noise[1 - state->noise_buf];
            
            kernel_cpmmh_rejuvenate_fused<<<state->N_theta, state->N_inner, pmmh_shared_size>>>(
                state->d_particles, 
                state->d_particles_temp,  /* Scratch for proposed PF state */
                state->d_y_history,
                curr_noise,               /* Current noise buffer */
                other_noise,              /* Other buffer for writing proposals */
                state->t_current,
                state->N_theta, state->N_inner,
                state->noise_capacity,
                state->cpmmh_rho,
                state->d_accepts,
                state->d_swap_flags       /* Per-particle accept flags */
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            /* For accepted particles, copy noise from other → curr 
             * (Alternative: could swap buffer indices per-particle but that's complex) */
            kernel_swap_noise_for_accepted<<<state->N_theta, state->N_inner>>>(
                curr_noise, other_noise,
                state->d_swap_flags,
                state->N_theta, state->N_inner,
                state->t_current, state->noise_capacity
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaMemcpy(&h_accepts, state->d_accepts, sizeof(int), cudaMemcpyDeviceToHost));
            state->n_rejuv_accepts += h_accepts;
            state->n_rejuv_total += state->N_theta;
        }
        
        kernel_compute_outer_ess<<<1, state->N_theta, 32 * sizeof(float)>>>(
            state->d_particles, state->d_ess, state->N_theta
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_ess, state->d_ess, sizeof(float), cudaMemcpyDeviceToHost));
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
