/**
 * @file smc2_rbpf_cuda.cu
 * @brief SMC² with RBPF Inner Filter - CUDA Implementation
 */

#include "smc2_rbpf_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*═══════════════════════════════════════════════════════════════════════════
 * OCSN Constants in Constant Memory
 *═══════════════════════════════════════════════════════════════════════════*/

/* KSC/OCSN 10-component mixture for log(χ²_1) */
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

/* Prior and bounds in constant memory */
__constant__ SVPrior d_prior;
__constant__ SVBounds d_bounds;
__constant__ SVCurve d_theta_curve;
__constant__ float d_proposal_std[8];

/*═══════════════════════════════════════════════════════════════════════════
 * Error Checking Macro
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
 * Block-Level Reductions
 *═══════════════════════════════════════════════════════════════════════════*/

__device__ float block_reduce_max(float val, float* shared_data) {
    int lane = threadIdx.x % CUDA_WARP_SIZE;
    int warp_id = threadIdx.x / CUDA_WARP_SIZE;
    
    /* Warp-level reduction */
    val = warp_reduce_max(val);
    
    /* First thread of each warp writes to shared memory */
    if (lane == 0) {
        shared_data[warp_id] = val;
    }
    __syncthreads();
    
    /* First warp reduces across warps */
    if (warp_id == 0) {
        val = (threadIdx.x < CUDA_N_WARPS) ? shared_data[lane] : -1e30f;
        val = warp_reduce_max(val);
    }
    
    __syncthreads();
    return shared_data[0];  /* Broadcast result */
}

__device__ float block_reduce_sum(float val, float* shared_data) {
    int lane = threadIdx.x % CUDA_WARP_SIZE;
    int warp_id = threadIdx.x / CUDA_WARP_SIZE;
    
    /* Warp-level reduction */
    val = warp_reduce_sum(val);
    
    /* First thread of each warp writes to shared memory */
    if (lane == 0) {
        shared_data[warp_id] = val;
    }
    __syncthreads();
    
    /* First warp reduces across warps */
    if (warp_id == 0) {
        val = (threadIdx.x < CUDA_N_WARPS) ? shared_data[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            shared_data[0] = val;
        }
    }
    
    __syncthreads();
    return shared_data[0];
}

/*═══════════════════════════════════════════════════════════════════════════
 * OCSN Kalman Update (per-thread, branchless)
 *═══════════════════════════════════════════════════════════════════════════*/

__device__ void ocsn_kalman_update(
    float y, float mu_pred, float var_pred,
    float* mu_post_out, float* var_post_out, float* log_lik_out
) {
    float log_alpha_tilde[OCSN_K];
    float S[OCSN_K], K[OCSN_K], mu_k[OCSN_K], var_k[OCSN_K];
    float log_max = -1e30f;
    
    /* Step 1: Compute unnormalized log responsibilities */
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        S[k] = var_pred + d_OCSN_VARS[k];
        float innov = y - mu_pred - d_OCSN_MEANS[k] + OCSN_OFFSET;
        log_alpha_tilde[k] = d_OCSN_LOG_WEIGHTS[k] 
                           - 0.5f * __logf(S[k])
                           - 0.5f * innov * innov / S[k];
        log_max = fmaxf(log_max, log_alpha_tilde[k]);
    }
    
    /* Step 2: Normalize responsibilities */
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
    
    /* Step 3: Per-component Kalman updates */
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        K[k] = var_pred / S[k];
        float innov = y - mu_pred - d_OCSN_MEANS[k] + OCSN_OFFSET;
        mu_k[k] = mu_pred + K[k] * innov;
        var_k[k] = (1.0f - K[k]) * var_pred;
    }
    
    /* Step 4: Moment matching */
    float mu_post = 0.0f, E_h_sq = 0.0f;
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        mu_post += alpha[k] * mu_k[k];
        E_h_sq += alpha[k] * (var_k[k] + mu_k[k] * mu_k[k]);
    }
    
    float var_post = fmaxf(E_h_sq - mu_post * mu_post, 1e-6f);
    
    *mu_post_out = mu_post;
    *var_post_out = var_post;
    *log_lik_out = log_norm;  /* No Gaussian constant (cancels in ratios) */
}

/*═══════════════════════════════════════════════════════════════════════════
 * RNG Initialization Kernel
 *═══════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_init_rng(curandState* states, unsigned long long seed, int N_total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_total) {
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
    
    /* Thread 0 samples θ parameters */
    __shared__ float s_rho, s_sigma_z;
    __shared__ float s_mu_base, s_mu_scale, s_mu_rate;
    __shared__ float s_sigma_base, s_sigma_scale, s_sigma_rate;
    
    if (inner_idx == 0) {
        /* Sample from prior with rejection for bounds */
        int valid = 0;
        while (!valid) {
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
        }
        
        /* Store parameters */
        particles.rho[theta_idx] = s_rho;
        particles.sigma_z[theta_idx] = s_sigma_z;
        particles.mu_base[theta_idx] = s_mu_base;
        particles.mu_scale[theta_idx] = s_mu_scale;
        particles.mu_rate[theta_idx] = s_mu_rate;
        particles.sigma_base[theta_idx] = s_sigma_base;
        particles.sigma_scale[theta_idx] = s_sigma_scale;
        particles.sigma_rate[theta_idx] = s_sigma_rate;
        
        /* Initialize outer weights */
        particles.log_weight[theta_idx] = 0.0f;
        particles.weight[theta_idx] = 1.0f / N_theta;
        particles.log_likelihood[theta_idx] = 0.0f;
    }
    
    __syncthreads();
    
    /* All threads initialize inner particles from stationary distribution */
    float rho = s_rho;
    float sigma_z = s_sigma_z;
    
    /* z stationary: N(0, σ_z²/(1-ρ²)) */
    float one_minus_rho_sq = fmaxf(1.0f - rho * rho, 1e-6f);
    float z_stat_std = sigma_z / sqrtf(one_minus_rho_sq);
    
    float z = z_stat_std * curand_normal(rng);
    z = clampf(z, 0.0f, 3.0f);
    
    /* h stationary at this z */
    float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z);
    float mu_z = eval_curve(s_mu_base, s_mu_scale, s_mu_rate, z);
    float sigma_h = eval_curve(s_sigma_base, s_sigma_scale, s_sigma_rate, z);
    
    float phi = 1.0f - theta_z;
    float one_minus_phi_sq = fmaxf(1.0f - phi * phi, 1e-6f);
    float h_stat_var = (sigma_h * sigma_h) / one_minus_phi_sq;
    
    /* Store inner particle state */
    particles.inner_z[global_idx] = z;
    particles.inner_mu_h[global_idx] = mu_z;
    particles.inner_var_h[global_idx] = h_stat_var;
    particles.inner_weights[global_idx] = 1.0f / N_inner;
}

/*═══════════════════════════════════════════════════════════════════════════
 * Inner RBPF Step Kernel (one block per θ-particle)
 *═══════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_rbpf_step(
    ThetaParticlesSoA particles,
    float y_obs,
    int N_theta, int N_inner,
    float ess_threshold
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_inner + inner_idx;
    
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    
    /* Shared memory for reductions and resampling */
    __shared__ float s_reduction[CUDA_N_WARPS];
    __shared__ float s_weights[CUDA_N_INNER];
    __shared__ float s_cumsum[CUDA_N_INNER];
    __shared__ int s_ancestors[CUDA_N_INNER];
    __shared__ float s_ess;
    __shared__ float s_log_max;
    __shared__ float s_sum_w;
    
    /* Load parameters (broadcast within block) */
    __shared__ float s_rho, s_sigma_z;
    __shared__ float s_mu_base, s_mu_scale, s_mu_rate;
    __shared__ float s_sigma_base, s_sigma_scale, s_sigma_rate;
    
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
    
    curandState* rng = &particles.rng_states[global_idx];
    
    /* Load current particle state */
    float z = particles.inner_z[global_idx];
    float mu_h = particles.inner_mu_h[global_idx];
    float var_h = particles.inner_var_h[global_idx];
    float w = particles.inner_weights[global_idx];
    
    /*═══════════════════════════════════════════════════════════════════════
     * Resampling (if ESS < threshold)
     *═══════════════════════════════════════════════════════════════════════*/
    if (s_ess < ess_threshold * N_inner) {
        /* Build CDF */
        s_weights[inner_idx] = w;
        __syncthreads();
        
        /* Parallel prefix sum (simple sequential for now, can optimize) */
        if (inner_idx == 0) {
            s_cumsum[0] = s_weights[0];
            for (int i = 1; i < N_inner; i++) {
                s_cumsum[i] = s_cumsum[i-1] + s_weights[i];
            }
            float total = s_cumsum[N_inner - 1];
            if (total > 0) {
                for (int i = 0; i < N_inner; i++) {
                    s_cumsum[i] /= total;
                }
            }
            s_cumsum[N_inner - 1] = 1.0f;
        }
        __syncthreads();
        
        /* Systematic resampling - each thread finds its ancestor */
        float u0 = (inner_idx == 0) ? curand_uniform(rng) : 0.0f;
        if (inner_idx == 0) s_reduction[0] = u0;
        __syncthreads();
        u0 = s_reduction[0];
        
        float u = (u0 + (float)inner_idx) / (float)N_inner;
        
        /* Binary search for ancestor */
        int lo = 0, hi = N_inner - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (s_cumsum[mid] < u) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        s_ancestors[inner_idx] = lo;
        __syncthreads();
        
        /* Copy from ancestor */
        int ancestor = s_ancestors[inner_idx];
        z = particles.inner_z[theta_idx * N_inner + ancestor];
        mu_h = particles.inner_mu_h[theta_idx * N_inner + ancestor];
        var_h = particles.inner_var_h[theta_idx * N_inner + ancestor];
        w = 1.0f / N_inner;
        
        __syncthreads();
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * Propagate z (OU dynamics)
     *═══════════════════════════════════════════════════════════════════════*/
    float z_mean = s_rho * z;  /* Assuming z_floor = 0 */
    float z_new = z_mean + s_sigma_z * curand_normal(rng);
    z_new = clampf(z_new, 0.0f, 3.0f);
    
    /*═══════════════════════════════════════════════════════════════════════
     * Evaluate curves at z_new
     *═══════════════════════════════════════════════════════════════════════*/
    float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z_new);
    float mu_z = eval_curve(s_mu_base, s_mu_scale, s_mu_rate, z_new);
    float sigma_z_h = eval_curve(s_sigma_base, s_sigma_scale, s_sigma_rate, z_new);
    
    float phi = 1.0f - theta_z;
    
    /*═══════════════════════════════════════════════════════════════════════
     * Kalman predict for h
     *═══════════════════════════════════════════════════════════════════════*/
    float mu_pred = phi * mu_h + theta_z * mu_z;
    float var_pred = phi * phi * var_h + sigma_z_h * sigma_z_h;
    var_pred = fmaxf(var_pred, 1e-8f);
    
    /*═══════════════════════════════════════════════════════════════════════
     * OCSN Kalman update
     *═══════════════════════════════════════════════════════════════════════*/
    float mu_post, var_post, log_lik;
    ocsn_kalman_update(y_obs, mu_pred, var_pred, &mu_post, &var_post, &log_lik);
    
    /*═══════════════════════════════════════════════════════════════════════
     * Normalize weights (block-level log-sum-exp)
     *═══════════════════════════════════════════════════════════════════════*/
    
    /* Find max log-weight */
    float log_max = block_reduce_max(log_lik, s_reduction);
    if (inner_idx == 0) s_log_max = log_max;
    __syncthreads();
    log_max = s_log_max;
    
    /* Compute weights */
    float w_unnorm = __expf(log_lik - log_max);
    
    /* Sum weights */
    float sum_w = block_reduce_sum(w_unnorm, s_reduction);
    if (inner_idx == 0) s_sum_w = sum_w;
    __syncthreads();
    sum_w = s_sum_w;
    
    /* Normalize */
    float safe_sum = fmaxf(sum_w, 1e-30f);
    w = w_unnorm / safe_sum;
    
    /* Compute ESS */
    float w_sq = w * w;
    float sum_w_sq = block_reduce_sum(w_sq, s_reduction);
    float ess = 1.0f / fmaxf(sum_w_sq, 1e-30f);
    
    /* Compute log-likelihood increment */
    float ll_incr = log_max + __logf(safe_sum) - __logf((float)N_inner);
    
    /*═══════════════════════════════════════════════════════════════════════
     * Store results
     *═══════════════════════════════════════════════════════════════════════*/
    particles.inner_z[global_idx] = z_new;
    particles.inner_mu_h[global_idx] = mu_post;
    particles.inner_var_h[global_idx] = var_post;
    particles.inner_weights[global_idx] = w;
    
    if (inner_idx == 0) {
        particles.ess_inner[theta_idx] = ess;
        particles.log_weight[theta_idx] += ll_incr;
        particles.log_likelihood[theta_idx] += ll_incr;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * Outer θ-particle Weight Normalization Kernel
 *═══════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_normalize_theta_weights(
    ThetaParticlesSoA particles,
    float* d_ess_out,
    int N_theta
) {
    /* Single block kernel for N_theta particles */
    __shared__ float s_reduction[32];  /* Assume N_theta <= 1024 */
    __shared__ float s_log_max;
    __shared__ float s_sum_w;
    
    int idx = threadIdx.x;
    
    /* Find max log-weight */
    float my_log_w = (idx < N_theta) ? particles.log_weight[idx] : -1e30f;
    
    float log_max = my_log_w;
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        log_max = fmaxf(log_max, __shfl_down_sync(0xFFFFFFFF, log_max, offset));
    }
    if (threadIdx.x % 32 == 0) {
        s_reduction[threadIdx.x / 32] = log_max;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        log_max = (threadIdx.x < (blockDim.x + 31) / 32) ? s_reduction[threadIdx.x] : -1e30f;
        for (int offset = 16; offset > 0; offset /= 2) {
            log_max = fmaxf(log_max, __shfl_down_sync(0xFFFFFFFF, log_max, offset));
        }
        if (threadIdx.x == 0) s_log_max = log_max;
    }
    __syncthreads();
    log_max = s_log_max;
    
    /* Compute unnormalized weights */
    float w = (idx < N_theta) ? __expf(particles.log_weight[idx] - log_max) : 0.0f;
    
    /* Sum weights */
    float sum_w = w;
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum_w += __shfl_down_sync(0xFFFFFFFF, sum_w, offset);
    }
    if (threadIdx.x % 32 == 0) {
        s_reduction[threadIdx.x / 32] = sum_w;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        sum_w = (threadIdx.x < (blockDim.x + 31) / 32) ? s_reduction[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_w += __shfl_down_sync(0xFFFFFFFF, sum_w, offset);
        }
        if (threadIdx.x == 0) s_sum_w = sum_w;
    }
    __syncthreads();
    sum_w = s_sum_w;
    
    /* Normalize and compute ESS */
    if (idx < N_theta) {
        w /= sum_w;
        particles.weight[idx] = w;
    }
    
    /* Sum of squared weights for ESS */
    float w_sq = (idx < N_theta) ? w * w : 0.0f;
    float sum_w_sq = w_sq;
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum_w_sq += __shfl_down_sync(0xFFFFFFFF, sum_w_sq, offset);
    }
    if (threadIdx.x % 32 == 0) {
        s_reduction[threadIdx.x / 32] = sum_w_sq;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        sum_w_sq = (threadIdx.x < (blockDim.x + 31) / 32) ? s_reduction[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_w_sq += __shfl_down_sync(0xFFFFFFFF, sum_w_sq, offset);
        }
        if (threadIdx.x == 0) {
            *d_ess_out = 1.0f / sum_w_sq;
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * Host API Implementation
 *═══════════════════════════════════════════════════════════════════════════*/

SMC2StateCUDA* smc2_cuda_alloc(int N_theta, int N_inner) {
    SMC2StateCUDA* state = (SMC2StateCUDA*)calloc(1, sizeof(SMC2StateCUDA));
    if (!state) return NULL;
    
    state->N_theta = N_theta;
    state->N_inner = N_inner;
    state->ess_threshold_outer = 0.5f;
    state->ess_threshold_inner = 0.5f;
    state->K_rejuv = 3;
    
    int N_total = N_theta * N_inner;
    
    /* Allocate device memory - parameters */
    CUDA_CHECK(cudaMalloc(&state->d_particles.rho, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.sigma_z, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.mu_base, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.mu_scale, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.mu_rate, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.sigma_base, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.sigma_scale, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.sigma_rate, N_theta * sizeof(float)));
    
    /* Allocate device memory - inner particles */
    CUDA_CHECK(cudaMalloc(&state->d_particles.inner_z, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.inner_mu_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.inner_var_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.inner_weights, N_total * sizeof(float)));
    
    /* Allocate device memory - per θ-particle state */
    CUDA_CHECK(cudaMalloc(&state->d_particles.log_weight, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.weight, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.log_likelihood, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_particles.ess_inner, N_theta * sizeof(float)));
    
    /* Allocate RNG states */
    CUDA_CHECK(cudaMalloc(&state->d_particles.rng_states, N_total * sizeof(curandState)));
    
    /* Initialize RNG */
    int threads = 256;
    int blocks = (N_total + threads - 1) / threads;
    kernel_init_rng<<<blocks, threads>>>(state->d_particles.rng_states, 12345ULL, N_total);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    /* Allocate observation history */
    state->y_history_capacity = 1000;
    CUDA_CHECK(cudaMalloc(&state->d_y_history, state->y_history_capacity * sizeof(float)));
    state->y_history_len = 0;
    
    /* Allocate scratch space */
    CUDA_CHECK(cudaMalloc(&state->d_cumsum, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_ancestors, N_theta * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_uniform, sizeof(float)));
    
    /* Set default prior */
    state->prior.rho_mean = 0.95f; state->prior.rho_std = 0.02f;
    state->prior.sigma_z_mean = 0.1f; state->prior.sigma_z_std = 0.05f;
    state->prior.mu_base_mean = -1.0f; state->prior.mu_base_std = 0.5f;
    state->prior.mu_scale_mean = 0.5f; state->prior.mu_scale_std = 0.3f;
    state->prior.mu_rate_mean = 1.0f; state->prior.mu_rate_std = 0.5f;
    state->prior.sigma_base_mean = 0.15f; state->prior.sigma_base_std = 0.05f;
    state->prior.sigma_scale_mean = 0.1f; state->prior.sigma_scale_std = 0.05f;
    state->prior.sigma_rate_mean = 1.0f; state->prior.sigma_rate_std = 0.5f;
    
    /* Set default bounds */
    state->bounds.rho_min = 0.8f; state->bounds.rho_max = 0.999f;
    state->bounds.sigma_z_min = 0.01f; state->bounds.sigma_z_max = 0.5f;
    state->bounds.mu_base_min = -5.0f; state->bounds.mu_base_max = 2.0f;
    state->bounds.mu_scale_min = -1.0f; state->bounds.mu_scale_max = 2.0f;
    state->bounds.mu_rate_min = 0.1f; state->bounds.mu_rate_max = 5.0f;
    state->bounds.sigma_base_min = 0.01f; state->bounds.sigma_base_max = 0.5f;
    state->bounds.sigma_scale_min = -0.2f; state->bounds.sigma_scale_max = 0.5f;
    state->bounds.sigma_rate_min = 0.1f; state->bounds.sigma_rate_max = 5.0f;
    
    /* Set default theta curve (fixed) */
    state->theta_curve.base = 0.02f;
    state->theta_curve.scale = 0.08f;
    state->theta_curve.rate = 1.5f;
    
    /* Set default proposal std */
    state->proposal_std[0] = 0.01f;
    state->proposal_std[1] = 0.02f;
    state->proposal_std[2] = 0.1f;
    state->proposal_std[3] = 0.1f;
    state->proposal_std[4] = 0.2f;
    state->proposal_std[5] = 0.02f;
    state->proposal_std[6] = 0.02f;
    state->proposal_std[7] = 0.2f;
    
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
    cudaFree(state->d_particles.inner_weights);
    
    cudaFree(state->d_particles.log_weight);
    cudaFree(state->d_particles.weight);
    cudaFree(state->d_particles.log_likelihood);
    cudaFree(state->d_particles.ess_inner);
    cudaFree(state->d_particles.rng_states);
    
    cudaFree(state->d_y_history);
    cudaFree(state->d_cumsum);
    cudaFree(state->d_ancestors);
    cudaFree(state->d_uniform);
    
    free(state);
}

void smc2_cuda_init_from_prior(SMC2StateCUDA* state) {
    /* Copy prior/bounds/theta_curve to constant memory */
    CUDA_CHECK(cudaMemcpyToSymbol(d_prior, &state->prior, sizeof(SVPrior)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_bounds, &state->bounds, sizeof(SVBounds)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_theta_curve, &state->theta_curve, sizeof(SVCurve)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_std, state->proposal_std, 8 * sizeof(float)));
    
    /* Initialize ESS to N_inner (no resampling on first step) */
    float init_ess = (float)state->N_inner;
    for (int j = 0; j < state->N_theta; j++) {
        CUDA_CHECK(cudaMemcpy(&state->d_particles.ess_inner[j], &init_ess, 
                              sizeof(float), cudaMemcpyHostToDevice));
    }
    
    /* Launch initialization kernel */
    kernel_init_from_prior<<<state->N_theta, state->N_inner>>>(
        state->d_particles, state->N_theta, state->N_inner
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    state->n_resamples = 0;
    state->n_rejuv_accepts = 0;
    state->n_rejuv_total = 0;
    state->y_history_len = 0;
}

float smc2_cuda_update(SMC2StateCUDA* state, float y_obs) {
    /* Store observation in history */
    if (state->y_history_len >= state->y_history_capacity) {
        /* Grow capacity */
        int new_capacity = state->y_history_capacity * 2;
        float* new_history;
        CUDA_CHECK(cudaMalloc(&new_history, new_capacity * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(new_history, state->d_y_history, 
                              state->y_history_len * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaFree(state->d_y_history);
        state->d_y_history = new_history;
        state->y_history_capacity = new_capacity;
    }
    CUDA_CHECK(cudaMemcpy(&state->d_y_history[state->y_history_len], &y_obs, 
                          sizeof(float), cudaMemcpyHostToDevice));
    state->y_history_len++;
    
    /* Run inner RBPF step for all θ-particles */
    kernel_rbpf_step<<<state->N_theta, state->N_inner>>>(
        state->d_particles, y_obs,
        state->N_theta, state->N_inner,
        state->ess_threshold_inner
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    /* Normalize outer weights and compute ESS */
    float h_ess;
    float* d_ess;
    CUDA_CHECK(cudaMalloc(&d_ess, sizeof(float)));
    
    kernel_normalize_theta_weights<<<1, state->N_theta>>>(
        state->d_particles, d_ess, state->N_theta
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_ess, d_ess, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_ess);
    
    /* TODO: Implement outer resampling + PMMH rejuvenation */
    /* For now, just return ESS without resampling */
    if (h_ess < state->ess_threshold_outer * state->N_theta) {
        state->n_resamples++;
        /* Outer resampling and PMMH rejuvenation would go here */
        /* This requires re-running the inner filter from t=0, which is complex on GPU */
    }
    
    return h_ess;
}

void smc2_cuda_get_theta_mean(SMC2StateCUDA* state, float* theta_mean) {
    /* Download weights and parameters to host */
    float* h_weight = (float*)malloc(state->N_theta * sizeof(float));
    float* h_rho = (float*)malloc(state->N_theta * sizeof(float));
    float* h_sigma_z = (float*)malloc(state->N_theta * sizeof(float));
    float* h_mu_base = (float*)malloc(state->N_theta * sizeof(float));
    float* h_mu_scale = (float*)malloc(state->N_theta * sizeof(float));
    float* h_mu_rate = (float*)malloc(state->N_theta * sizeof(float));
    float* h_sigma_base = (float*)malloc(state->N_theta * sizeof(float));
    float* h_sigma_scale = (float*)malloc(state->N_theta * sizeof(float));
    float* h_sigma_rate = (float*)malloc(state->N_theta * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(h_weight, state->d_particles.weight, 
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rho, state->d_particles.rho,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sigma_z, state->d_particles.sigma_z,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mu_base, state->d_particles.mu_base,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mu_scale, state->d_particles.mu_scale,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mu_rate, state->d_particles.mu_rate,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sigma_base, state->d_particles.sigma_base,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sigma_scale, state->d_particles.sigma_scale,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sigma_rate, state->d_particles.sigma_rate,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    
    /* Compute weighted means */
    for (int i = 0; i < 8; i++) theta_mean[i] = 0.0f;
    
    for (int j = 0; j < state->N_theta; j++) {
        float w = h_weight[j];
        theta_mean[0] += w * h_rho[j];
        theta_mean[1] += w * h_sigma_z[j];
        theta_mean[2] += w * h_mu_base[j];
        theta_mean[3] += w * h_mu_scale[j];
        theta_mean[4] += w * h_mu_rate[j];
        theta_mean[5] += w * h_sigma_base[j];
        theta_mean[6] += w * h_sigma_scale[j];
        theta_mean[7] += w * h_sigma_rate[j];
    }
    
    free(h_weight);
    free(h_rho);
    free(h_sigma_z);
    free(h_mu_base);
    free(h_mu_scale);
    free(h_mu_rate);
    free(h_sigma_base);
    free(h_sigma_scale);
    free(h_sigma_rate);
}

void smc2_cuda_get_theta_std(SMC2StateCUDA* state, float* theta_std) {
    float theta_mean[8];
    smc2_cuda_get_theta_mean(state, theta_mean);
    
    /* Download weights and parameters */
    float* h_weight = (float*)malloc(state->N_theta * sizeof(float));
    float* h_params[8];
    for (int i = 0; i < 8; i++) {
        h_params[i] = (float*)malloc(state->N_theta * sizeof(float));
    }
    
    CUDA_CHECK(cudaMemcpy(h_weight, state->d_particles.weight,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[0], state->d_particles.rho,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[1], state->d_particles.sigma_z,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[2], state->d_particles.mu_base,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[3], state->d_particles.mu_scale,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[4], state->d_particles.mu_rate,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[5], state->d_particles.sigma_base,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[6], state->d_particles.sigma_scale,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[7], state->d_particles.sigma_rate,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    
    /* Compute weighted variance */
    for (int i = 0; i < 8; i++) theta_std[i] = 0.0f;
    
    for (int j = 0; j < state->N_theta; j++) {
        float w = h_weight[j];
        for (int i = 0; i < 8; i++) {
            float d = h_params[i][j] - theta_mean[i];
            theta_std[i] += w * d * d;
        }
    }
    
    for (int i = 0; i < 8; i++) {
        theta_std[i] = sqrtf(theta_std[i]);
    }
    
    free(h_weight);
    for (int i = 0; i < 8; i++) free(h_params[i]);
}

float smc2_cuda_get_outer_ess(SMC2StateCUDA* state) {
    float* h_weight = (float*)malloc(state->N_theta * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_weight, state->d_particles.weight,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    
    float sum_w_sq = 0.0f;
    for (int j = 0; j < state->N_theta; j++) {
        sum_w_sq += h_weight[j] * h_weight[j];
    }
    
    free(h_weight);
    return 1.0f / sum_w_sq;
}
