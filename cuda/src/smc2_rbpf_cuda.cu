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
 * Initialize θ-particles from Prior
 *═══════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_init_from_prior(
    ThetaParticlesSoA particles,
    int N_theta, int N_inner
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
                     s_mu_rate >= d_bounds.mu_rate_min && s_mu_rate <= d_bounds.mu_rate_max &&
                     s_sigma_base >= d_bounds.sigma_base_min && s_sigma_base <= d_bounds.sigma_base_max &&
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
    
    float z = z_stat_std * curand_normal(rng);
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
 * Inner RBPF Step
 *═══════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_rbpf_step(
    ThetaParticlesSoA particles,
    float y_obs,
    int N_theta, int N_inner,
    float ess_threshold_inner
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
            s_u0 = curand_uniform(&local_rng);
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
    
    /* Propagate z */
    float z_new = s_rho * z + s_sigma_z * curand_normal(&local_rng);
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

/*═══════════════════════════════════════════════════════════════════════════
 * PMMH Rejuvenation - Full O(t) Likelihood Computation
 *═══════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_pmmh_rejuvenate(
    ThetaParticlesSoA particles,
    const float* y_history,
    int t_current,
    int N_theta, int N_inner,
    float ess_threshold_inner,
    int* d_accepts
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_inner + inner_idx;
    
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    
    extern __shared__ float shared_mem[];
    float* s_reduction = shared_mem;
    float* s_weights = &shared_mem[32];
    float* s_cumsum = &shared_mem[32 + N_inner];
    
    __shared__ float s_rho_curr, s_sigma_z_curr;
    __shared__ float s_mu_base_curr, s_mu_scale_curr, s_mu_rate_curr;
    __shared__ float s_sigma_base_curr, s_sigma_scale_curr, s_sigma_rate_curr;
    __shared__ float s_ll_curr;
    
    __shared__ float s_rho_prop, s_sigma_z_prop;
    __shared__ float s_mu_base_prop, s_mu_scale_prop, s_mu_rate_prop;
    __shared__ float s_sigma_base_prop, s_sigma_scale_prop, s_sigma_rate_prop;
    
    __shared__ float s_log_max, s_sum_w, s_u0;
    __shared__ int s_accept;
    
    curandState local_rng = particles.rng_states[global_idx];
    
    /* Thread 0: Load current and propose */
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
        
        int valid = 0, attempts = 0;
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
                     s_mu_rate_prop >= d_bounds.mu_rate_min && s_mu_rate_prop <= d_bounds.mu_rate_max &&
                     s_sigma_base_prop >= d_bounds.sigma_base_min && s_sigma_base_prop <= d_bounds.sigma_base_max &&
                     s_sigma_rate_prop >= d_bounds.sigma_rate_min && s_sigma_rate_prop <= d_bounds.sigma_rate_max);
            attempts++;
        }
        if (!valid) {
            s_rho_prop = s_rho_curr;
            s_sigma_z_prop = s_sigma_z_curr;
            s_mu_base_prop = s_mu_base_curr;
            s_mu_scale_prop = s_mu_scale_curr;
            s_mu_rate_prop = s_mu_rate_curr;
            s_sigma_base_prop = s_sigma_base_curr;
            s_sigma_scale_prop = s_sigma_scale_curr;
            s_sigma_rate_prop = s_sigma_rate_curr;
        }
        s_accept = 0;
    }
    __syncthreads();
    
    /* Initialize from stationary under θ' */
    float rho = s_rho_prop;
    float sigma_z = s_sigma_z_prop;
    float one_minus_rho_sq = fmaxf(1.0f - rho * rho, 1e-6f);
    float z_stat_std = sigma_z / sqrtf(one_minus_rho_sq);
    
    float z = z_stat_std * curand_normal(&local_rng);
    z = clampf(z, 0.0f, 3.0f);
    
    float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z);
    float mu_z = eval_curve(s_mu_base_prop, s_mu_scale_prop, s_mu_rate_prop, z);
    float sigma_h = eval_curve(s_sigma_base_prop, s_sigma_scale_prop, s_sigma_rate_prop, z);
    float phi = 1.0f - theta_z;
    float h_stat_var = (sigma_h * sigma_h) / fmaxf(1.0f - phi * phi, 1e-6f);
    
    float mu_h = mu_z;
    float var_h = h_stat_var;
    float log_w = -__logf((float)N_inner);
    float ll_accum = 0.0f;
    
    /* Run RBPF from t=0 to t_current */
    for (int t = 0; t <= t_current; t++) {
        float y_obs = y_history[t];
        
        /* Check ESS for resampling */
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
        
        if (ess < ess_threshold_inner * N_inner && t > 0) {
            s_weights[inner_idx] = w_norm;
            __syncthreads();
            
            if (inner_idx == 0) {
                s_cumsum[0] = s_weights[0];
                for (int i = 1; i < N_inner; i++) s_cumsum[i] = s_cumsum[i-1] + s_weights[i];
                s_cumsum[N_inner - 1] = 1.0f;
                s_u0 = curand_uniform(&local_rng);
            }
            __syncthreads();
            
            float u = (s_u0 + (float)inner_idx) / (float)N_inner;
            int lo = 0, hi = N_inner - 1;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (s_cumsum[mid] < u) lo = mid + 1;
                else hi = mid;
            }
            
            /* Exchange via shared memory */
            s_weights[inner_idx] = z;
            s_cumsum[inner_idx] = mu_h;
            __syncthreads();
            z = s_weights[lo];
            mu_h = s_cumsum[lo];
            var_h = h_stat_var;
            log_w = -__logf((float)N_inner);
            __syncthreads();
        }
        
        /* Propagate */
        float z_new = s_rho_prop * z + s_sigma_z_prop * curand_normal(&local_rng);
        z_new = clampf(z_new, 0.0f, 3.0f);
        
        theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z_new);
        mu_z = eval_curve(s_mu_base_prop, s_mu_scale_prop, s_mu_rate_prop, z_new);
        sigma_h = eval_curve(s_sigma_base_prop, s_sigma_scale_prop, s_sigma_rate_prop, z_new);
        phi = 1.0f - theta_z;
        
        float mu_pred = phi * mu_h + theta_z * mu_z;
        float var_pred = phi * phi * var_h + sigma_h * sigma_h;
        var_pred = fmaxf(var_pred, 1e-8f);
        
        float mu_post, var_post, log_lik;
        ocsn_kalman_update(y_obs, mu_pred, var_pred, &mu_post, &var_post, &log_lik);
        
        log_w += log_lik;
        
        /* Accumulate log-likelihood */
        log_max = block_reduce_max(log_w, s_reduction);
        if (inner_idx == 0) s_log_max = log_max;
        __syncthreads();
        w_unnorm = __expf(log_w - s_log_max);
        sum_w = block_reduce_sum(w_unnorm, s_reduction);
        if (inner_idx == 0) s_sum_w = sum_w;
        __syncthreads();
        
        float ll_incr = s_log_max + __logf(fmaxf(s_sum_w, 1e-30f)) - __logf((float)N_inner);
        ll_accum += ll_incr;
        
        z = z_new;
        mu_h = mu_post;
        var_h = var_post;
    }
    
    /* MH Accept/Reject */
    if (inner_idx == 0) {
        float log_alpha = ll_accum - s_ll_curr;
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
            particles.log_likelihood[theta_idx] = ll_accum;
            atomicAdd(d_accepts, 1);
        }
    }
    __syncthreads();
    
    if (s_accept) {
        particles.inner_z[global_idx] = z;
        particles.inner_mu_h[global_idx] = mu_h;
        particles.inner_var_h[global_idx] = var_h;
        particles.inner_log_w[global_idx] = log_w;
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
    state->K_rejuv = 5;  /* More moves for better diversity after resample */
    
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
    state->y_history_capacity = 2000;
    CUDA_CHECK(cudaMalloc(&state->d_y_history, state->y_history_capacity * sizeof(float)));
    state->y_history_len = 0;
    
    /* Scratch */
    CUDA_CHECK(cudaMalloc(&state->d_ancestors, N_theta * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_uniform, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_ess, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_accepts, sizeof(int)));
    
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
    state->bounds.sigma_z_min = 0.01f; state->bounds.sigma_z_max = 0.5f;
    state->bounds.mu_base_min = -5.0f; state->bounds.mu_base_max = 2.0f;
    state->bounds.mu_scale_min = -1.0f; state->bounds.mu_scale_max = 2.0f;
    state->bounds.mu_rate_min = 0.1f; state->bounds.mu_rate_max = 5.0f;
    state->bounds.sigma_base_min = 0.01f; state->bounds.sigma_base_max = 0.5f;
    state->bounds.sigma_scale_min = -0.2f; state->bounds.sigma_scale_max = 0.5f;
    state->bounds.sigma_rate_min = 0.1f; state->bounds.sigma_rate_max = 5.0f;
    
    /* Default theta curve */
    state->theta_curve.base = 0.02f;
    state->theta_curve.scale = 0.08f;
    state->theta_curve.rate = 1.5f;
    
    /* Proposal std - tuned for ~25-35% acceptance */
    state->proposal_std[0] = 0.02f;   /* rho */
    state->proposal_std[1] = 0.05f;   /* sigma_z */
    state->proposal_std[2] = 0.2f;    /* mu_base */
    state->proposal_std[3] = 0.2f;    /* mu_scale */
    state->proposal_std[4] = 0.3f;    /* mu_rate */
    state->proposal_std[5] = 0.05f;   /* sigma_base */
    state->proposal_std[6] = 0.05f;   /* sigma_scale */
    state->proposal_std[7] = 0.3f;    /* sigma_rate */
    
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
    
    free(state);
}

void smc2_cuda_init_from_prior(SMC2StateCUDA* state) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_prior, &state->prior, sizeof(SVPrior)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_bounds, &state->bounds, sizeof(SVBounds)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_theta_curve, &state->theta_curve, sizeof(SVCurve)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_std, state->proposal_std, 8 * sizeof(float)));
    
    kernel_init_from_prior<<<state->N_theta, state->N_inner>>>(
        state->d_particles, state->N_theta, state->N_inner
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
    
    size_t shared_size = (32 + 2 * state->N_inner) * sizeof(float);
    
    kernel_rbpf_step<<<state->N_theta, state->N_inner, shared_size>>>(
        state->d_particles, y_obs,
        state->N_theta, state->N_inner,
        state->ess_threshold_inner
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
        
        /* Swap */
        ThetaParticlesSoA tmp = state->d_particles;
        state->d_particles = state->d_particles_temp;
        state->d_particles_temp = tmp;
        
        /* PMMH rejuvenation */
        for (int k = 0; k < state->K_rejuv; k++) {
            int h_accepts = 0;
            CUDA_CHECK(cudaMemcpy(state->d_accepts, &h_accepts, sizeof(int), cudaMemcpyHostToDevice));
            
            kernel_pmmh_rejuvenate<<<state->N_theta, state->N_inner, shared_size>>>(
                state->d_particles, state->d_y_history, state->t_current,
                state->N_theta, state->N_inner, state->ess_threshold_inner,
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
