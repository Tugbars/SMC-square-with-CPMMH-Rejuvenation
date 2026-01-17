/*
 * smc2_prior_calibration_gpu.cuh
 * 
 * GPU-native prior calibration - zero CPU↔GPU transfers after warmup ingestion.
 * 
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  WHY GPU?                                                               │
 * │                                                                         │
 * │  CPU version requires:                                                  │
 * │    warmup[] ──► CPU calibrate() ──► prior ──► cudaMemcpy ──► GPU       │
 * │                                                                         │
 * │  GPU version:                                                           │
 * │    d_warmup[] ──► GPU calibrate_kernel() ──► d_prior (stays on GPU)    │
 * │                                                                         │
 * │  No CPU↔GPU boundary after initial data ingestion = fewer bugs.        │
 * └─────────────────────────────────────────────────────────────────────────┘
 * 
 * Usage:
 *   // Warmup buffer lives on GPU from the start
 *   float* d_warmup;
 *   cudaMalloc(&d_warmup, 100 * sizeof(float));
 *   
 *   // Stream returns directly to GPU as they arrive
 *   for (int i = 0; i < 100; i++) {
 *       float y = get_return();
 *       cudaMemcpyAsync(d_warmup + i, &y, sizeof(float), cudaMemcpyHostToDevice, stream);
 *   }
 *   
 *   // Calibrate entirely on GPU
 *   SMC2PriorGPU* d_prior;
 *   cudaMalloc(&d_prior, sizeof(SMC2PriorGPU));
 *   
 *   smc2_calibrate_prior_gpu(d_warmup, 100, d_prior, stream);
 *   
 *   // d_prior is now ready for filter initialization - never touches CPU
 */

#ifndef SMC2_PRIOR_CALIBRATION_GPU_CUH
#define SMC2_PRIOR_CALIBRATION_GPU_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

/* ============================================================================
 * GPU Prior Structure (mirrors CPU version)
 * ============================================================================ */

struct SMC2PriorGPU {
    /* Core SV parameters */
    float rho_mean, rho_std;
    float sigma_z_mean, sigma_z_std;
    float mu_base_mean, mu_base_std;
    
    /* Extended model parameters */
    float mu_scale_mean, mu_scale_std;
    float mu_rate_mean, mu_rate_std;
    float sigma_base_mean, sigma_base_std;
    float sigma_scale_mean, sigma_scale_std;
    float sigma_rate_mean, sigma_rate_std;
};

/* ============================================================================
 * GPU Historical Bounds (constant memory for fast access)
 * ============================================================================ */

struct HistoricalBoundsGPU {
    float rho_min, rho_max;
    float sigma_z_min, sigma_z_max;
    float mu_base_min, mu_base_max;
};

/* SPY bounds in constant memory - NVCC compatible initialization */
__constant__ HistoricalBoundsGPU c_spy_bounds = {
    0.665f,     /* rho_min */
    0.990f,     /* rho_max */
    0.031f,     /* sigma_z_min */
    0.650f,     /* sigma_z_max */
    -11.88f,    /* mu_base_min */
    -5.80f      /* mu_base_max */
};

/* Log(χ²(1)) constants */
__constant__ float c_log_chi2_mean = -1.2704f;
__constant__ float c_log_chi2_var = 4.9348f;

/* ============================================================================
 * Warp-level Reduction Primitives
 * ============================================================================ */

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  /* One slot per warp */
    
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    
    /* Warp-level reduction */
    val = warp_reduce_sum(val);
    
    /* Write warp results to shared memory */
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    /* First warp reduces across warps */
    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    
    return val;
}

/* ============================================================================
 * Main Calibration Kernel
 * 
 * Single block, handles up to 256 warmup observations.
 * Computes statistics and maps to prior in one launch.
 * ============================================================================ */

__global__ void smc2_calibrate_prior_kernel(
    const float* __restrict__ d_returns,    /* Raw returns */
    int n,                                   /* Number of observations */
    SMC2PriorGPU* __restrict__ d_prior       /* Output prior */
) {
    /* ─────────────────────────────────────────────────────────────────────
     * Step 1: Compute log(y²) and basic statistics
     * ───────────────────────────────────────────────────────────────────── */
    
    __shared__ float s_log_y2[256];  /* Shared storage for log(y²) */
    __shared__ float s_stats[8];      /* [sum, sum_sq, sum_lag, valid_n, realized_var, ...] */
    
    int tid = threadIdx.x;
    float log_y2 = 0.0f;
    float valid = 0.0f;
    
    /* Each thread computes log(y²) for its element */
    if (tid < n) {
        float y = d_returns[tid];
        if (y != 0.0f) {
            log_y2 = logf(y * y);
            valid = 1.0f;
            s_log_y2[tid] = log_y2;
        } else {
            s_log_y2[tid] = 0.0f;  /* Will be excluded */
        }
    } else {
        s_log_y2[tid] = 0.0f;
    }
    __syncthreads();
    
    /* Reduce to get count and sum */
    float local_sum = (tid < n && d_returns[tid] != 0.0f) ? log_y2 : 0.0f;
    float total_valid = block_reduce_sum(valid);
    float total_sum = block_reduce_sum(local_sum);
    
    if (tid == 0) {
        s_stats[0] = total_sum;
        s_stats[1] = total_valid;
    }
    __syncthreads();
    
    float mean_log_y2 = s_stats[0] / fmaxf(s_stats[1], 1.0f);
    int valid_n = (int)s_stats[1];
    
    /* ─────────────────────────────────────────────────────────────────────
     * Step 2: Compute variance and ACF at lag 1
     * ───────────────────────────────────────────────────────────────────── */
    
    /* Variance: sum of (x - mean)² */
    float local_sq = 0.0f;
    if (tid < n && d_returns[tid] != 0.0f) {
        float diff = s_log_y2[tid] - mean_log_y2;
        local_sq = diff * diff;
    }
    float total_sq = block_reduce_sum(local_sq);
    
    /* ACF lag 1: sum of (x_t - mean)(x_{t-1} - mean) */
    float local_lag = 0.0f;
    if (tid > 0 && tid < n && d_returns[tid] != 0.0f && d_returns[tid-1] != 0.0f) {
        float diff_t = s_log_y2[tid] - mean_log_y2;
        float diff_tm1 = s_log_y2[tid-1] - mean_log_y2;
        local_lag = diff_t * diff_tm1;
    }
    float total_lag = block_reduce_sum(local_lag);
    
    /* Realized variance (for volatility regime detection) */
    float local_ret_sq = 0.0f;
    if (tid < n) {
        float y = d_returns[tid];
        local_ret_sq = y * y;
    }
    float total_ret_sq = block_reduce_sum(local_ret_sq);
    
    if (tid == 0) {
        s_stats[2] = total_sq;      /* sum of squared deviations */
        s_stats[3] = total_lag;     /* sum of lagged products */
        s_stats[4] = total_ret_sq;  /* sum of return² */
    }
    __syncthreads();
    
    /* ─────────────────────────────────────────────────────────────────────
     * Step 3: Compute final statistics (thread 0 only)
     * ───────────────────────────────────────────────────────────────────── */
    
    if (tid == 0) {
        float var_log_y2 = s_stats[2] / fmaxf((float)(valid_n - 1), 1.0f);
        float std_log_y2 = sqrtf(var_log_y2);
        float cov_lag1 = s_stats[3] / fmaxf((float)(valid_n - 2), 1.0f);
        float acf1 = cov_lag1 / fmaxf(var_log_y2, 1e-6f);
        
        float realized_var = s_stats[4] / fmaxf((float)n, 1.0f);
        float realized_vol = sqrtf(realized_var);
        float annual_vol = realized_vol * sqrtf(252.0f);
        
        /* ─────────────────────────────────────────────────────────────────
         * Step 4: Map statistics to parameter estimates
         * ───────────────────────────────────────────────────────────────── */
        
        /* Estimate Var(h) by removing observation noise */
        float var_h_est = fmaxf(var_log_y2 - c_log_chi2_var, 0.1f);
        
        /* ρ from ACF1 with attenuation correction */
        float attenuation = var_h_est / fmaxf(var_h_est + c_log_chi2_var, 1e-6f);
        float rho_raw = acf1 / fmaxf(attenuation, 0.1f);
        
        /* Adjust ρ based on volatility regime */
        float vol_factor = fminf(annual_vol / 0.30f, 1.5f);  /* High vol → lower ρ */
        float rho_est = rho_raw * (1.0f - 0.1f * (vol_factor - 1.0f));
        rho_est = fmaxf(c_spy_bounds.rho_min, fminf(c_spy_bounds.rho_max, rho_est));
        
        /* σ_z from Var(h) and ρ */
        float one_minus_rho2 = 1.0f - rho_est * rho_est;
        float sigma_z_est = sqrtf(fmaxf(var_h_est * one_minus_rho2, 0.001f));
        
        /* Adjust σ_z based on volatility regime */
        if (annual_vol > 0.30f) {
            sigma_z_est = fmaxf(sigma_z_est, 0.15f);  /* Floor during crisis */
        }
        sigma_z_est = fmaxf(c_spy_bounds.sigma_z_min, fminf(c_spy_bounds.sigma_z_max, sigma_z_est));
        
        /* μ_base from mean */
        float mu_base_est = mean_log_y2 - c_log_chi2_mean;
        mu_base_est = fmaxf(c_spy_bounds.mu_base_min, fminf(c_spy_bounds.mu_base_max, mu_base_est));
        
        /* ─────────────────────────────────────────────────────────────────
         * Step 5: Set prior (center from warmup, width from history)
         * ───────────────────────────────────────────────────────────────── */
        
        /* Width = (historical_max - historical_min) / 4 → ±2σ covers range */
        float rho_std = (c_spy_bounds.rho_max - c_spy_bounds.rho_min) / 4.0f;
        float sigma_z_std = (c_spy_bounds.sigma_z_max - c_spy_bounds.sigma_z_min) / 4.0f;
        float mu_base_std = (c_spy_bounds.mu_base_max - c_spy_bounds.mu_base_min) / 4.0f;
        
        /* Write prior */
        d_prior->rho_mean = rho_est;
        d_prior->rho_std = rho_std;
        
        d_prior->sigma_z_mean = sigma_z_est;
        d_prior->sigma_z_std = sigma_z_std;
        
        d_prior->mu_base_mean = mu_base_est;
        d_prior->mu_base_std = mu_base_std;
        
        /* Extended parameters - use reasonable defaults */
        d_prior->mu_scale_mean = 0.5f;
        d_prior->mu_scale_std = 0.3f;
        
        d_prior->mu_rate_mean = 1.0f;
        d_prior->mu_rate_std = 0.5f;
        
        d_prior->sigma_base_mean = 0.15f;
        d_prior->sigma_base_std = 0.08f;
        
        d_prior->sigma_scale_mean = 0.10f;
        d_prior->sigma_scale_std = 0.06f;
        
        d_prior->sigma_rate_mean = 1.0f;
        d_prior->sigma_rate_std = 0.5f;
    }
}

/* ============================================================================
 * Host API
 * ============================================================================ */

/**
 * @brief Calibrate prior entirely on GPU
 * 
 * @param d_returns  Device pointer to warmup returns [n]
 * @param n          Number of observations (30-256)
 * @param d_prior    Device pointer to output prior
 * @param stream     CUDA stream (0 for default)
 * @return           cudaError_t
 */
inline cudaError_t smc2_calibrate_prior_gpu(
    const float* d_returns,
    int n,
    SMC2PriorGPU* d_prior,
    cudaStream_t stream = 0
) {
    if (n < 30) {
        fprintf(stderr, "[GPU Calibration] Error: need at least 30 observations, got %d\n", n);
        return cudaErrorInvalidValue;
    }
    
    if (n > 256) {
        fprintf(stderr, "[GPU Calibration] Warning: truncating to 256 observations\n");
        n = 256;
    }
    
    /* Single block, enough threads to cover warmup */
    int threads = ((n + 31) / 32) * 32;  /* Round up to warp multiple */
    threads = max(32, min(256, threads));
    
    smc2_calibrate_prior_kernel<<<1, threads, 0, stream>>>(d_returns, n, d_prior);
    
    return cudaGetLastError();
}

/**
 * @brief Calibrate and copy prior to host (for debugging/logging only)
 */
inline cudaError_t smc2_calibrate_prior_gpu_verbose(
    const float* d_returns,
    int n,
    SMC2PriorGPU* d_prior,
    SMC2PriorGPU* h_prior_out,  /* Optional: copy result to host */
    cudaStream_t stream = 0
) {
    cudaError_t err = smc2_calibrate_prior_gpu(d_returns, n, d_prior, stream);
    if (err != cudaSuccess) return err;
    
    if (h_prior_out) {
        err = cudaMemcpyAsync(h_prior_out, d_prior, sizeof(SMC2PriorGPU), 
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) return err;
        
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) return err;
        
        printf("\n");
        printf("┌─────────────────────────────────────────────────────────────────┐\n");
        printf("│  GPU Prior Calibration (n=%d)                                  │\n", n);
        printf("├─────────────────────────────────────────────────────────────────┤\n");
        printf("│  Parameter    │  Mean     │  Std      │  ±2σ Range            │\n");
        printf("├───────────────┼───────────┼───────────┼───────────────────────┤\n");
        printf("│  ρ            │  %7.4f  │  %7.4f  │  [%6.3f, %6.3f]      │\n",
               h_prior_out->rho_mean, h_prior_out->rho_std,
               h_prior_out->rho_mean - 2*h_prior_out->rho_std,
               h_prior_out->rho_mean + 2*h_prior_out->rho_std);
        printf("│  σ_z          │  %7.4f  │  %7.4f  │  [%6.3f, %6.3f]      │\n",
               h_prior_out->sigma_z_mean, h_prior_out->sigma_z_std,
               h_prior_out->sigma_z_mean - 2*h_prior_out->sigma_z_std,
               h_prior_out->sigma_z_mean + 2*h_prior_out->sigma_z_std);
        printf("│  μ_base       │  %7.3f  │  %7.4f  │  [%6.2f, %6.2f]      │\n",
               h_prior_out->mu_base_mean, h_prior_out->mu_base_std,
               h_prior_out->mu_base_mean - 2*h_prior_out->mu_base_std,
               h_prior_out->mu_base_mean + 2*h_prior_out->mu_base_std);
        printf("└─────────────────────────────────────────────────────────────────┘\n");
    }
    
    return cudaSuccess;
}

/* ============================================================================
 * Warmup Buffer Helper
 * 
 * Manages GPU-side warmup collection with ring buffer semantics.
 * ============================================================================ */

struct WarmupBufferGPU {
    float* d_buffer;        /* Device buffer */
    int capacity;           /* Max observations */
    int count;              /* Current count */
    cudaStream_t stream;    /* Associated stream */
};

/**
 * @brief Initialize warmup buffer on GPU
 */
inline cudaError_t warmup_buffer_init(WarmupBufferGPU* buf, int capacity, cudaStream_t stream = 0) {
    buf->capacity = capacity;
    buf->count = 0;
    buf->stream = stream;
    return cudaMalloc(&buf->d_buffer, capacity * sizeof(float));
}

/**
 * @brief Add one observation to warmup buffer
 */
inline cudaError_t warmup_buffer_add(WarmupBufferGPU* buf, float y) {
    if (buf->count >= buf->capacity) {
        return cudaErrorInvalidValue;  /* Buffer full */
    }
    
    cudaError_t err = cudaMemcpyAsync(
        buf->d_buffer + buf->count, 
        &y, 
        sizeof(float), 
        cudaMemcpyHostToDevice, 
        buf->stream
    );
    
    if (err == cudaSuccess) {
        buf->count++;
    }
    return err;
}

/**
 * @brief Check if warmup is complete
 */
inline bool warmup_buffer_ready(const WarmupBufferGPU* buf, int min_obs = 100) {
    return buf->count >= min_obs;
}

/**
 * @brief Free warmup buffer
 */
inline cudaError_t warmup_buffer_free(WarmupBufferGPU* buf) {
    if (buf->d_buffer) {
        cudaError_t err = cudaFree(buf->d_buffer);
        buf->d_buffer = nullptr;
        buf->count = 0;
        return err;
    }
    return cudaSuccess;
}

/* ============================================================================
 * Example Usage
 * ============================================================================
 *
 * int main() {
 *     // Initialize warmup buffer on GPU
 *     WarmupBufferGPU warmup;
 *     warmup_buffer_init(&warmup, 100, 0);
 *     
 *     // Allocate prior on GPU (stays there forever)
 *     SMC2PriorGPU* d_prior;
 *     cudaMalloc(&d_prior, sizeof(SMC2PriorGPU));
 *     
 *     // Collect warmup
 *     while (!warmup_buffer_ready(&warmup)) {
 *         float y = get_return_from_market();
 *         warmup_buffer_add(&warmup, y);
 *     }
 *     
 *     // Calibrate entirely on GPU
 *     smc2_calibrate_prior_gpu(warmup.d_buffer, warmup.count, d_prior);
 *     
 *     // Initialize filter with d_prior (no CPU round-trip!)
 *     smc2_cuda_init_from_prior_gpu(filter_state, d_prior);
 *     
 *     // Run filter...
 *     
 *     // Cleanup
 *     warmup_buffer_free(&warmup);
 *     cudaFree(d_prior);
 * }
 *
 * ============================================================================ */

#endif /* SMC2_PRIOR_CALIBRATION_GPU_CUH */
