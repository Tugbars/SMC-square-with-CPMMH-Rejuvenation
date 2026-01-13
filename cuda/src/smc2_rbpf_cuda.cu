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
#include <curand.h>  /* For curandGenerator_t */

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
    float* d_z_noise,    /* Store t=0 z-noise for CPMMH */
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
    
    /* Generate and STORE t=0 z-noise */
    float z_noise_init = curand_normal(rng);
    int64_t z_noise_idx = (int64_t)theta_idx * N_inner * (noise_capacity + 1) + inner_idx;  /* t=0 slot */
    d_z_noise[z_noise_idx] = z_noise_init;
    
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
 * Inner RBPF Step - WITH NOISE STORAGE FOR CPMMH
 * 
 * The forward filter:
 *   1. Generates fresh noise via curand
 *   2. Stores it in d_z_noise and d_u0
 *   3. Uses stored noise for z propagation and resampling
 * 
 * This enables CPMMH rejuvenation to correlate against stored noise.
 *═══════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_rbpf_step(
    ThetaParticlesSoA particles,
    float y_obs,
    int N_theta, int N_inner,
    float ess_threshold_inner,
    float* d_z_noise,  /* [N_theta * N_inner * (T+1)] - stores z-innovations */
    float* d_u0,       /* [N_theta * T] - stores resampling uniforms */
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
    __shared__ float s_ess, s_log_max, s_sum_w, s_u0;
    
    if (inner_idx == 0) {
        s_rho = particles.rho[theta_idx];
        s_sigma_z = particles.sigma_z[theta_idx];
        s_mu_base = particles.mu_base[theta_idx];
        s_mu_scale = particles.mu_scale[theta_idx];
        s_mu_rate = particles.mu_rate[theta_idx];
        s_sigma_base = particles.sigma_base[theta_idx];
        s_sigma_scale = particles.sigma_scale[theta_idx];
        s_sigma_rate = particles.sigma_rate[theta_idx];
        s_ess = particles.ess_inner[theta_idx];
    }
    __syncthreads();
    
    curandState local_rng = particles.rng_states[global_idx];
    
    float z = particles.inner_z[global_idx];
    float mu_h = particles.inner_mu_h[global_idx];
    float var_h = particles.inner_var_h[global_idx];
    float log_w = particles.inner_log_w[global_idx];
    
    /* Noise index: slot t+1 for propagation (slot 0 is init) */
    int64_t z_noise_idx = (int64_t)theta_idx * N_inner * (noise_capacity + 1) 
                        + (int64_t)(t_current + 1) * N_inner + inner_idx;
    int64_t u0_idx = (int64_t)theta_idx * noise_capacity + t_current;
    
    /* Resampling */
    if (s_ess < ess_threshold_inner * N_inner) {
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
            
            /* Generate and STORE resampling uniform */
            float u0_fresh = curand_uniform(&local_rng);
            d_u0[u0_idx] = u0_fresh;
            s_u0 = u0_fresh;
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
    } else {
        /* No resampling - still store a dummy u0 for consistency */
        if (inner_idx == 0) {
            d_u0[u0_idx] = 0.5f;  /* Won't be used but needs to be defined */
        }
    }
    
    /* Generate and STORE z-innovation */
    float z_noise = curand_normal(&local_rng);
    d_z_noise[z_noise_idx] = z_noise;
    
    /* Propagate z using stored noise */
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
    
    /* Normalize and compute ESS */
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

/* Copy noise arrays after outer resampling */
__global__ void kernel_copy_noise_arrays(
    const float* src_z_noise,
    float* dst_z_noise,
    const float* src_u0,
    float* dst_u0,
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
    
    /* Thread 0: copy u0 */
    if (inner_idx == 0) {
        int64_t dst_u0_base = (int64_t)theta_idx * noise_capacity;
        int64_t src_u0_base = (int64_t)ancestor * noise_capacity;
        for (int t = 0; t <= t_current; t++) {
            dst_u0[dst_u0_base + t] = src_u0[src_u0_base + t];
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * CPMMH Rejuvenation - Correlated Pseudo-Marginal MH
 * 
 * Key insight: Use CORRELATED noise to reduce variance of likelihood ratio.
 *   noise_prop = rho * noise_curr + scale * noise_fresh
 *   where rho ≈ 0.99 gives ~100× variance reduction
 * 
 * Algorithm:
 *   1. Generate fresh noise
 *   2. Compute correlated: prop = rho * curr + scale * fresh
 *   3. Run replay with θ' and prop_noise → ll_prop
 *   4. Use cached ll_curr (from forward filter, NO replay!)
 *   5. MH accept/reject
 *   6. On accept: update θ, PF state, curr_noise ← prop_noise
 * 
 * Cost: 1 * O(T * N_inner) per θ-particle (half of standard PMMH!)
 *═══════════════════════════════════════════════════════════════════════════*/

/* Device function: RBPF replay using PRE-COMPUTED correlated noise */
__device__ float rbpf_replay_with_correlated_noise(
    /* θ parameters */
    float rho, float sigma_z,
    float mu_base, float mu_scale, float mu_rate,
    float sigma_base, float sigma_scale, float sigma_rate,
    /* Inputs */
    const float* y_history,
    int t_current,
    int N_inner,
    float ess_threshold_inner,
    /* CPMMH: Use pre-computed correlated noise instead of RNG */
    const float* z_noise_prop,    /* [N_inner * (T+1)] - correlated z-innovations */
    const float* u0_prop,         /* [T] - correlated resampling uniforms */
    int noise_capacity,
    /* Shared memory for reductions/resampling */
    float* s_reduction,
    float* s_z, float* s_mu, float* s_var, float* s_cdf,
    float* s_log_max_ptr, float* s_sum_w_ptr,
    /* Outputs: final PF state */
    float* z_out, float* mu_out, float* var_out, float* logw_out, float* ess_out
) {
    int inner_idx = threadIdx.x;
    
    /* Initialize from stationary using t=0 noise */
    float one_minus_rho_sq = fmaxf(1.0f - rho * rho, 1e-6f);
    float z_stat_std = sigma_z / sqrtf(one_minus_rho_sq);
    
    float z_noise_init = z_noise_prop[inner_idx];  /* t=0 slot */
    float z = z_stat_std * z_noise_init;
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
    float ess = (float)N_inner;
    
    /* Process all observations */
    for (int t = 0; t <= t_current; t++) {
        float y_obs = y_history[t];
        
        /* Compute ESS */
        float log_max = block_reduce_max(log_w, s_reduction);
        if (inner_idx == 0) *s_log_max_ptr = log_max;
        __syncthreads();
        log_max = *s_log_max_ptr;
        
        float w_unnorm = __expf(log_w - log_max);
        float sum_w = block_reduce_sum(w_unnorm, s_reduction);
        if (inner_idx == 0) *s_sum_w_ptr = sum_w;
        __syncthreads();
        sum_w = *s_sum_w_ptr;
        
        float w_norm = w_unnorm / fmaxf(sum_w, 1e-30f);
        float w_sq = w_norm * w_norm;
        float sum_w_sq = block_reduce_sum(w_sq, s_reduction);
        ess = 1.0f / fmaxf(sum_w_sq, 1e-30f);
        
        /* Resample if needed */
        if (ess < ess_threshold_inner * N_inner && t > 0) {
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
            
            /* Use stored correlated u0 */
            float u0 = u0_prop[t];
            u0 = fmaxf(1e-7f, fminf(1.0f - 1e-7f, u0));
            
            float u = (u0 + (float)inner_idx) / (float)N_inner;
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
        }
        
        /* Use stored correlated z-noise for propagation (t+1 slot) */
        float z_noise = z_noise_prop[(t + 1) * N_inner + inner_idx];
        float z_new = rho * z + sigma_z * z_noise;
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
        if (inner_idx == 0) *s_log_max_ptr = log_max;
        __syncthreads();
        log_max = *s_log_max_ptr;
        
        w_unnorm = __expf(log_w - log_max);
        sum_w = block_reduce_sum(w_unnorm, s_reduction);
        if (inner_idx == 0) *s_sum_w_ptr = sum_w;
        __syncthreads();
        sum_w = *s_sum_w_ptr;
        
        float ll_incr = log_max + __logf(fmaxf(sum_w, 1e-30f)) - __logf((float)N_inner);
        ll_accum += ll_incr;
        
        z = z_new;
        mu_h = mu_post;
        var_h = var_post;
    }
    
    /* Output final PF state */
    z_out[inner_idx] = z;
    mu_out[inner_idx] = mu_h;
    var_out[inner_idx] = var_h;
    logw_out[inner_idx] = log_w;
    if (inner_idx == 0) *ess_out = ess;
    
    return ll_accum;
}

/* Kernel: Generate fresh noise for CPMMH proposals */
__global__ void kernel_generate_fresh_noise(
    float* d_z_noise_fresh,
    float* d_u0_fresh,
    curandState* rng_states,
    int N_theta, int N_inner,
    int t_current,
    int noise_capacity
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_inner + inner_idx;
    
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    
    curandState local_rng = rng_states[global_idx];
    
    int64_t z_base = (int64_t)theta_idx * N_inner * (noise_capacity + 1);
    
    /* Generate fresh z-noise for all timesteps */
    for (int t = 0; t <= t_current + 1; t++) {
        d_z_noise_fresh[z_base + t * N_inner + inner_idx] = curand_normal(&local_rng);
    }
    
    /* Thread 0 generates fresh u0 for all timesteps */
    if (inner_idx == 0) {
        int64_t u0_base = (int64_t)theta_idx * noise_capacity;
        for (int t = 0; t <= t_current; t++) {
            d_u0_fresh[u0_base + t] = curand_uniform(&local_rng);
        }
    }
    
    rng_states[global_idx] = local_rng;
}

/* Kernel: Compute correlated noise: prop = rho * curr + scale * fresh */
__global__ void kernel_compute_correlated_noise(
    const float* d_z_noise_curr,
    const float* d_z_noise_fresh,
    float* d_z_noise_prop,
    const float* d_u0_curr,
    const float* d_u0_fresh,
    float* d_u0_prop,
    float cpmmh_rho,
    int N_theta, int N_inner,
    int t_current,
    int noise_capacity
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    
    float scale = sqrtf(1.0f - cpmmh_rho * cpmmh_rho);
    
    int64_t z_base = (int64_t)theta_idx * N_inner * (noise_capacity + 1);
    
    /* Correlate z-noise */
    for (int t = 0; t <= t_current + 1; t++) {
        int64_t idx = z_base + t * N_inner + inner_idx;
        d_z_noise_prop[idx] = cpmmh_rho * d_z_noise_curr[idx] + scale * d_z_noise_fresh[idx];
    }
    
    /* Thread 0: correlate u0 (wrap to [0,1]) */
    if (inner_idx == 0) {
        int64_t u0_base = (int64_t)theta_idx * noise_capacity;
        for (int t = 0; t <= t_current; t++) {
            float u = cpmmh_rho * d_u0_curr[u0_base + t] + scale * d_u0_fresh[u0_base + t];
            u = u - floorf(u);  /* Wrap to [0,1] */
            d_u0_prop[u0_base + t] = u;
        }
    }
}

/* Main CPMMH rejuvenation kernel */
__global__ void kernel_cpmmh_rejuvenate(
    ThetaParticlesSoA particles,
    ThetaParticlesSoA particles_scratch,
    const float* y_history,
    float* d_z_noise_curr,       /* Current stored noise */
    const float* d_z_noise_prop, /* Correlated proposal noise */
    float* d_u0_curr,
    const float* d_u0_prop,
    int t_current,
    int N_theta, int N_inner,
    float ess_threshold_inner,
    int noise_capacity,
    float cpmmh_rho,
    int* d_accepts
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
    __shared__ int s_accept, s_valid;
    
    curandState local_rng = particles.rng_states[global_idx];
    
    /* Thread 0: Load current θ, propose θ', get cached ll_curr */
    if (inner_idx == 0) {
        s_rho_curr = particles.rho[theta_idx];
        s_sigma_z_curr = particles.sigma_z[theta_idx];
        s_mu_base_curr = particles.mu_base[theta_idx];
        s_mu_scale_curr = particles.mu_scale[theta_idx];
        s_mu_rate_curr = particles.mu_rate[theta_idx];
        s_sigma_base_curr = particles.sigma_base[theta_idx];
        s_sigma_scale_curr = particles.sigma_scale[theta_idx];
        s_sigma_rate_curr = particles.sigma_rate[theta_idx];
        
        /* CRITICAL: ll_curr comes from forward filter, NO replay needed! */
        s_ll_curr = particles.log_likelihood[theta_idx];
        
        /* Propose θ' with random walk */
        s_valid = 1;
        int attempts = 0;
        int valid = 0;
        while (!valid && attempts < 100) {
            s_rho_prop = s_rho_curr + d_proposal_std[0] * curand_normal(&local_rng);
            s_sigma_z_prop = s_sigma_z_curr + d_proposal_std[1] * curand_normal(&local_rng);
            s_mu_base_prop = s_mu_base_curr + d_proposal_std[2] * curand_normal(&local_rng);
            s_mu_scale_prop = s_mu_scale_curr + d_proposal_std[3] * curand_normal(&local_rng);
            s_mu_rate_prop = s_mu_rate_curr + d_proposal_std[4] * curand_normal(&local_rng);
            s_sigma_base_prop = s_sigma_base_curr + d_proposal_std[5] * curand_normal(&local_rng);
            s_sigma_scale_prop = s_sigma_scale_curr + d_proposal_std[6] * curand_normal(&local_rng);
            s_sigma_rate_prop = s_sigma_rate_curr + d_proposal_std[7] * curand_normal(&local_rng);
            
            valid = (s_rho_prop >= d_bounds.rho_min && s_rho_prop <= d_bounds.rho_max &&
                     s_sigma_z_prop >= d_bounds.sigma_z_min && s_sigma_z_prop <= d_bounds.sigma_z_max &&
                     s_mu_base_prop >= d_bounds.mu_base_min && s_mu_base_prop <= d_bounds.mu_base_max &&
                     s_mu_scale_prop >= d_bounds.mu_scale_min && s_mu_scale_prop <= d_bounds.mu_scale_max &&
                     s_mu_rate_prop >= d_bounds.mu_rate_min && s_mu_rate_prop <= d_bounds.mu_rate_max &&
                     s_sigma_base_prop >= d_bounds.sigma_base_min && s_sigma_base_prop <= d_bounds.sigma_base_max &&
                     s_sigma_scale_prop >= d_bounds.sigma_scale_min && s_sigma_scale_prop <= d_bounds.sigma_scale_max &&
                     s_sigma_rate_prop >= d_bounds.sigma_rate_min && s_sigma_rate_prop <= d_bounds.sigma_rate_max);
            attempts++;
        }
        if (!valid) {
            s_valid = 0;  /* Invalid proposal - will reject */
        }
        s_accept = 0;
    }
    __syncthreads();
    
    /* Early exit for invalid proposals */
    if (s_valid == 0) {
        particles.rng_states[global_idx] = local_rng;
        return;
    }
    
    /* Get pointers to this θ-particle's noise */
    int64_t z_noise_base = (int64_t)theta_idx * N_inner * (noise_capacity + 1);
    int64_t u0_base = (int64_t)theta_idx * noise_capacity;
    
    /*═══════════════════════════════════════════════════════════════════════
     * CPMMH: Only replay θ' (proposed) with correlated noise
     * ll_curr is already cached from forward filter!
     *═══════════════════════════════════════════════════════════════════════*/
    float ll_prop = rbpf_replay_with_correlated_noise(
        s_rho_prop, s_sigma_z_prop,
        s_mu_base_prop, s_mu_scale_prop, s_mu_rate_prop,
        s_sigma_base_prop, s_sigma_scale_prop, s_sigma_rate_prop,
        y_history, t_current, N_inner, ess_threshold_inner,
        &d_z_noise_prop[z_noise_base],
        &d_u0_prop[u0_base],
        noise_capacity,
        s_reduction, s_z, s_mu, s_var, s_cdf,
        &s_log_max, &s_sum_w,
        &particles_scratch.inner_z[global_idx],
        &particles_scratch.inner_mu_h[global_idx],
        &particles_scratch.inner_var_h[global_idx],
        &particles_scratch.inner_log_w[global_idx],
        &s_ess_prop
    );
    
    if (inner_idx == 0) s_ll_prop = ll_prop;
    __syncthreads();
    
    /*═══════════════════════════════════════════════════════════════════════
     * MH Accept/Reject using CORRELATED likelihoods
     * Var(ll_prop - ll_curr) ≈ (1-rho²) * Var(ll) << 2*Var(ll)
     *═══════════════════════════════════════════════════════════════════════*/
    if (inner_idx == 0) {
        float log_alpha = s_ll_prop - s_ll_curr;
        
        float u = curand_uniform(&local_rng);
        s_accept = (__logf(u) < log_alpha) ? 1 : 0;
        
        if (s_accept) {
            /* Update θ to proposed */
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
    }
    __syncthreads();
    
    /*═══════════════════════════════════════════════════════════════════════
     * On accept: copy PF state AND update curr_noise ← prop_noise
     *═══════════════════════════════════════════════════════════════════════*/
    if (s_accept) {
        /* Copy PF state from scratch */
        particles.inner_z[global_idx] = particles_scratch.inner_z[global_idx];
        particles.inner_mu_h[global_idx] = particles_scratch.inner_mu_h[global_idx];
        particles.inner_var_h[global_idx] = particles_scratch.inner_var_h[global_idx];
        particles.inner_log_w[global_idx] = particles_scratch.inner_log_w[global_idx];
        
        /* CRITICAL: Update curr_noise ← prop_noise */
        for (int t = 0; t <= t_current + 1; t++) {
            int64_t idx = z_noise_base + t * N_inner + inner_idx;
            d_z_noise_curr[idx] = d_z_noise_prop[idx];
        }
        
        /* Thread 0: update u0 */
        if (inner_idx == 0) {
            for (int t = 0; t <= t_current; t++) {
                d_u0_curr[u0_base + t] = d_u0_prop[u0_base + t];
            }
        }
    }
    
    particles.rng_states[global_idx] = local_rng;
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
    
    /* CPMMH: Allocate noise storage for correlated proposals */
    /* Initial capacity for T=2000 (will grow if needed) */
    state->noise_capacity = 2048;
    state->cpmmh_rho = 0.99f;  /* High correlation for variance reduction */
    
    int64_t z_noise_size = (int64_t)N_theta * N_inner * (state->noise_capacity + 1);
    int64_t u0_size = (int64_t)N_theta * state->noise_capacity;
    
    CUDA_CHECK(cudaMalloc(&state->d_z_noise, z_noise_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_u0, u0_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_z_noise_fresh, z_noise_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_u0_fresh, u0_size * sizeof(float)));
    
    /* Initialize noise to N(0,1) */
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 54321ULL);
    curandGenerateNormal(gen, state->d_z_noise, z_noise_size, 0.0f, 1.0f);
    curandGenerateUniform(gen, state->d_u0, u0_size);
    curandDestroyGenerator(gen);
    
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
    
    /* CPMMH noise storage */
    cudaFree(state->d_z_noise);
    cudaFree(state->d_u0);
    cudaFree(state->d_z_noise_fresh);
    cudaFree(state->d_u0_fresh);
    
    free(state);
}

void smc2_cuda_init_from_prior(SMC2StateCUDA* state) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_prior, &state->prior, sizeof(SVPrior)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_bounds, &state->bounds, sizeof(SVBounds)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_theta_curve, &state->theta_curve, sizeof(SVCurve)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_std, state->proposal_std, 8 * sizeof(float)));
    
    kernel_init_from_prior<<<state->N_theta, state->N_inner>>>(
        state->d_particles, state->N_theta, state->N_inner,
        state->d_z_noise, state->noise_capacity
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
        int64_t new_u0_size = (int64_t)state->N_theta * new_cap;
        
        float *new_z_noise, *new_u0, *new_z_fresh, *new_u0_fresh;
        CUDA_CHECK(cudaMalloc(&new_z_noise, new_z_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&new_u0, new_u0_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&new_z_fresh, new_z_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&new_u0_fresh, new_u0_size * sizeof(float)));
        
        /* Copy old data */
        int64_t old_z_size = (int64_t)state->N_theta * state->N_inner * (state->noise_capacity + 1);
        int64_t old_u0_size = (int64_t)state->N_theta * state->noise_capacity;
        CUDA_CHECK(cudaMemcpy(new_z_noise, state->d_z_noise, old_z_size * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_u0, state->d_u0, old_u0_size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        cudaFree(state->d_z_noise);
        cudaFree(state->d_u0);
        cudaFree(state->d_z_noise_fresh);
        cudaFree(state->d_u0_fresh);
        
        state->d_z_noise = new_z_noise;
        state->d_u0 = new_u0;
        state->d_z_noise_fresh = new_z_fresh;
        state->d_u0_fresh = new_u0_fresh;
        state->noise_capacity = new_cap;
    }
    
    size_t shared_size = (32 + 2 * state->N_inner) * sizeof(float);
    
    kernel_rbpf_step<<<state->N_theta, state->N_inner, shared_size>>>(
        state->d_particles, y_obs,
        state->N_theta, state->N_inner,
        state->ess_threshold_inner,
        state->d_z_noise, state->d_u0,
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
        
        /* Generate uniform */
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL) + state->n_resamples);
        curandGenerateUniform(gen, state->d_uniform, 1);
        curandDestroyGenerator(gen);
        
        /* Resample */
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
        
        /* Copy noise arrays: use d_z_noise_fresh as temp buffer */
        kernel_copy_noise_arrays<<<state->N_theta, state->N_inner>>>(
            state->d_z_noise, state->d_z_noise_fresh,  /* src → fresh (temp) */
            state->d_u0, state->d_u0_fresh,
            state->d_ancestors,
            state->N_theta, state->N_inner,
            state->t_current, state->noise_capacity
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        /* Copy back from temp to main noise arrays */
        int64_t z_copy_size = (int64_t)state->N_theta * state->N_inner * (state->t_current + 2);
        int64_t u0_copy_size = (int64_t)state->N_theta * (state->t_current + 1);
        CUDA_CHECK(cudaMemcpy(state->d_z_noise, state->d_z_noise_fresh, 
                              z_copy_size * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->d_u0, state->d_u0_fresh,
                              u0_copy_size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        /* Swap particle pointers */
        ThetaParticlesSoA tmp = state->d_particles;
        state->d_particles = state->d_particles_temp;
        state->d_particles_temp = tmp;
        
        /* CPMMH rejuvenation - uses correlated noise for variance reduction */
        /* Shared memory: 32 (reduction) + 4*N_inner (z,mu,var,cdf for resampling) */
        size_t pmmh_shared_size = (32 + 4 * state->N_inner) * sizeof(float);
        
        for (int k = 0; k < state->K_rejuv; k++) {
            int h_accepts = 0;
            CUDA_CHECK(cudaMemcpy(state->d_accepts, &h_accepts, sizeof(int), cudaMemcpyHostToDevice));
            
            /* Step 1: Generate fresh noise */
            kernel_generate_fresh_noise<<<state->N_theta, state->N_inner>>>(
                state->d_z_noise_fresh, state->d_u0_fresh,
                state->d_particles.rng_states,
                state->N_theta, state->N_inner,
                state->t_current, state->noise_capacity
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            /* Step 2: Compute correlated noise: prop = rho * curr + scale * fresh */
            kernel_compute_correlated_noise<<<state->N_theta, state->N_inner>>>(
                state->d_z_noise, state->d_z_noise_fresh, state->d_z_noise_fresh,  /* Use fresh as prop buffer */
                state->d_u0, state->d_u0_fresh, state->d_u0_fresh,
                state->cpmmh_rho,
                state->N_theta, state->N_inner,
                state->t_current, state->noise_capacity
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            /* Step 3: CPMMH rejuvenate - only replays θ', uses cached ll_curr */
            kernel_cpmmh_rejuvenate<<<state->N_theta, state->N_inner, pmmh_shared_size>>>(
                state->d_particles, 
                state->d_particles_temp,  /* Scratch buffer for proposed PF state */
                state->d_y_history,
                state->d_z_noise,         /* Current noise (updated on accept) */
                state->d_z_noise_fresh,   /* Correlated proposal noise */
                state->d_u0,
                state->d_u0_fresh,
                state->t_current,
                state->N_theta, state->N_inner, state->ess_threshold_inner,
                state->noise_capacity,
                state->cpmmh_rho,
                state->d_accepts
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
