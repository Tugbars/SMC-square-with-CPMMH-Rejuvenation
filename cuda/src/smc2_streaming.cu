/**
 * @file smc2_streaming.cu
 * @brief Streaming SMC² with Parameter Drift - CUDA Implementation
 * 
 * See smc2_streaming.cuh for algorithm documentation.
 */

#include "smc2_streaming.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <curand.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * CUDA ERROR CHECKING
 *═══════════════════════════════════════════════════════════════════════════════*/

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_ALLOC(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return NULL; \
    } \
} while(0)

/*═══════════════════════════════════════════════════════════════════════════════
 * OCSN CONSTANT MEMORY
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

/*═══════════════════════════════════════════════════════════════════════════════
 * INTERNAL STATE STRUCTURE
 *═══════════════════════════════════════════════════════════════════════════════*/

struct SMC2StreamState {
    /* Configuration */
    SMC2StreamConfig config;
    int buffer_mask;          /* buffer_size - 1, for fast modulo */
    
    /* Timing */
    int t_current;            /* Absolute time step (can exceed buffer_size) */
    int t_oldest;             /* Oldest valid time in buffer */
    
    /* θ-particle arrays (device, N_theta elements) */
    float* d_theta[SMC2_N_PARAMS];  /* Parameter values */
    float* d_log_weight;            /* Unnormalized log weights */
    float* d_weight;                /* Normalized weights */
    float* d_log_likelihood;        /* Accumulated log p̂(y | θ) */
    
    /* Inner RBPF arrays (device, N_theta × N_inner elements) */
    float* d_inner_z;               /* Regime z̃ (unconstrained) */
    float* d_inner_mu_h;            /* Kalman mean E[h | y] */
    float* d_inner_var_h;           /* Kalman variance */
    float* d_inner_log_w;           /* Inner particle weights */
    
    /* Circular noise buffers (device, N_theta × N_inner × buffer_size) */
    half* d_z_noise_circular;       /* z propagation noise */
    half* d_u0_noise_circular;      /* Resampling uniforms (derived from z) */
    
    /* Observation history (device, buffer_size) - for fixed-lag replay */
    float* d_y_history;
    
    /* RNG states */
    curandState* d_rng_states;      /* Per-particle RNG (N_theta × N_inner) */
    curandGenerator_t curand_gen;   /* Host-side generator for bulk noise */
    uint64_t seed;
    
    /* Posterior moments (host, updated at rejuvenation) */
    float theta_mean[SMC2_N_PARAMS];
    float theta_cov[SMC2_N_PARAMS * SMC2_N_PARAMS];
    float theta_cov_chol[SMC2_N_PARAMS * SMC2_N_PARAMS];  /* Lower triangular */
    
    /* Drift state */
    float Q_effective[SMC2_N_PARAMS];  /* Current drift (may be scaled) */
    float nll_excess_accum;            /* Accumulated NLL excess */
    float nll_sum;                     /* Sum of NLL for baseline calc */
    int nll_count;                     /* Count for baseline calc */
    
    /* Diagnostics from last update */
    SMC2StreamDiag last_diag;
    
    /* Scratch buffers */
    float* d_scratch;                  /* General purpose (N_theta floats) */
    int* d_ancestors;                  /* Resampling ancestors */
    int* d_accepts;                    /* CPMMH accept counts */
    
    /* Ping-pong buffers for outer resampling */
    float* d_theta_scratch[SMC2_N_PARAMS];
    float* d_inner_z_scratch;
    float* d_inner_mu_h_scratch;
    float* d_inner_var_h_scratch;
    float* d_inner_log_w_scratch;
};

/*═══════════════════════════════════════════════════════════════════════════════
 * HELPER: Next power of 2
 *═══════════════════════════════════════════════════════════════════════════════*/

static int next_pow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DEFAULT CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

SMC2StreamConfig smc2_stream_default_config(int N_theta, int N_inner) {
    SMC2StreamConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    
    cfg.N_theta = N_theta;
    cfg.N_inner = N_inner;
    cfg.fixed_lag = SMC2_DEFAULT_LAG;
    cfg.buffer_size = next_pow2(cfg.fixed_lag + 1);  /* Must be power of 2 */
    
    cfg.ess_threshold = 0.5f;
    cfg.cpmmh_rho = 0.99f;
    cfg.cpmmh_moves = 1;
    
    cfg.liu_west_a = SMC2_DEFAULT_LIU_WEST_A;
    
    /* Default drift Q (conservative) */
    cfg.Q.rho = 0.005f * 0.005f;          /* sqrt(Q) = 0.005 */
    cfg.Q.sigma_z = 0.01f * 0.01f;
    cfg.Q.mu_base = 0.1f * 0.1f;
    cfg.Q.mu_scale = 0.1f * 0.1f;
    cfg.Q.mu_rate = 0.05f * 0.05f;
    cfg.Q.sigma_base = 0.02f * 0.02f;
    cfg.Q.sigma_scale = 0.02f * 0.02f;
    cfg.Q.sigma_rate = 0.05f * 0.05f;
    
    cfg.enable_adaptive_Q = 0;
    cfg.adaptive_Q_scale = 3.0f;
    cfg.nll_baseline = 2.0f;  /* Typical for log χ²(1) */
    
    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE: CREATE
 *═══════════════════════════════════════════════════════════════════════════════*/

SMC2StreamState* smc2_stream_create(int N_theta, int N_inner) {
    SMC2StreamConfig cfg = smc2_stream_default_config(N_theta, N_inner);
    return smc2_stream_create_with_config(&cfg);
}

SMC2StreamState* smc2_stream_create_with_config(const SMC2StreamConfig* config) {
    SMC2StreamState* state = (SMC2StreamState*)calloc(1, sizeof(SMC2StreamState));
    if (!state) return NULL;
    
    state->config = *config;
    
    /* Ensure buffer_size is power of 2 and >= L+1 */
    int min_buffer = config->fixed_lag + 1;
    state->config.buffer_size = next_pow2(min_buffer);
    state->buffer_mask = state->config.buffer_size - 1;
    
    int N_theta = config->N_theta;
    int N_inner = config->N_inner;
    int buffer_size = state->config.buffer_size;
    int64_t N_total = (int64_t)N_theta * N_inner;
    int64_t noise_size = N_total * buffer_size;
    
    /* Allocate θ-particle arrays */
    for (int p = 0; p < SMC2_N_PARAMS; p++) {
        CUDA_CHECK_ALLOC(cudaMalloc(&state->d_theta[p], N_theta * sizeof(float)));
        CUDA_CHECK_ALLOC(cudaMalloc(&state->d_theta_scratch[p], N_theta * sizeof(float)));
    }
    
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_log_weight, N_theta * sizeof(float)));
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_weight, N_theta * sizeof(float)));
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_log_likelihood, N_theta * sizeof(float)));
    
    /* Allocate inner RBPF arrays */
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_inner_z, N_total * sizeof(float)));
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_inner_mu_h, N_total * sizeof(float)));
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_inner_var_h, N_total * sizeof(float)));
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_inner_log_w, N_total * sizeof(float)));
    
    /* Scratch for inner arrays */
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_inner_z_scratch, N_total * sizeof(float)));
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_inner_mu_h_scratch, N_total * sizeof(float)));
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_inner_var_h_scratch, N_total * sizeof(float)));
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_inner_log_w_scratch, N_total * sizeof(float)));
    
    /* Allocate circular noise buffers (FP16) */
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_z_noise_circular, noise_size * sizeof(half)));
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_u0_noise_circular, noise_size * sizeof(half)));
    
    /* Observation history */
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_y_history, buffer_size * sizeof(float)));
    
    /* RNG states */
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_rng_states, N_total * sizeof(curandState)));
    curandCreateGenerator(&state->curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    state->seed = 0;  /* Will be set by init */
    
    /* Scratch buffers */
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_scratch, N_theta * sizeof(float)));
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_ancestors, N_theta * sizeof(int)));
    CUDA_CHECK_ALLOC(cudaMalloc(&state->d_accepts, N_theta * sizeof(int)));
    
    /* Initialize timing */
    state->t_current = -1;  /* Will be 0 after first update */
    state->t_oldest = 0;
    
    /* Initialize drift */
    const SMC2DriftQ* Q = &config->Q;
    state->Q_effective[0] = Q->rho;
    state->Q_effective[1] = Q->sigma_z;
    state->Q_effective[2] = Q->mu_base;
    state->Q_effective[3] = Q->mu_scale;
    state->Q_effective[4] = Q->mu_rate;
    state->Q_effective[5] = Q->sigma_base;
    state->Q_effective[6] = Q->sigma_scale;
    state->Q_effective[7] = Q->sigma_rate;
    
    state->nll_excess_accum = 0.0f;
    state->nll_sum = 0.0f;
    state->nll_count = 0;
    
    printf("[SMC2 Stream] Created: N_theta=%d, N_inner=%d, L=%d, buffer=%d\n",
           N_theta, N_inner, config->fixed_lag, state->config.buffer_size);
    printf("[SMC2 Stream] Memory: %.1f MB (circular buffers)\n",
           (float)(2 * noise_size * sizeof(half) + buffer_size * sizeof(float)) / (1024 * 1024));
    
    return state;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE: FREE
 *═══════════════════════════════════════════════════════════════════════════════*/

void smc2_stream_free(SMC2StreamState* state) {
    if (!state) return;
    
    for (int p = 0; p < SMC2_N_PARAMS; p++) {
        if (state->d_theta[p]) cudaFree(state->d_theta[p]);
        if (state->d_theta_scratch[p]) cudaFree(state->d_theta_scratch[p]);
    }
    
    if (state->d_log_weight) cudaFree(state->d_log_weight);
    if (state->d_weight) cudaFree(state->d_weight);
    if (state->d_log_likelihood) cudaFree(state->d_log_likelihood);
    
    if (state->d_inner_z) cudaFree(state->d_inner_z);
    if (state->d_inner_mu_h) cudaFree(state->d_inner_mu_h);
    if (state->d_inner_var_h) cudaFree(state->d_inner_var_h);
    if (state->d_inner_log_w) cudaFree(state->d_inner_log_w);
    
    if (state->d_inner_z_scratch) cudaFree(state->d_inner_z_scratch);
    if (state->d_inner_mu_h_scratch) cudaFree(state->d_inner_mu_h_scratch);
    if (state->d_inner_var_h_scratch) cudaFree(state->d_inner_var_h_scratch);
    if (state->d_inner_log_w_scratch) cudaFree(state->d_inner_log_w_scratch);
    
    if (state->d_z_noise_circular) cudaFree(state->d_z_noise_circular);
    if (state->d_u0_noise_circular) cudaFree(state->d_u0_noise_circular);
    if (state->d_y_history) cudaFree(state->d_y_history);
    
    if (state->d_rng_states) cudaFree(state->d_rng_states);
    curandDestroyGenerator(state->curand_gen);
    
    if (state->d_scratch) cudaFree(state->d_scratch);
    if (state->d_ancestors) cudaFree(state->d_ancestors);
    if (state->d_accepts) cudaFree(state->d_accepts);
    
    free(state);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void smc2_stream_set_seed(SMC2StreamState* state, uint64_t seed) {
    state->seed = seed ? seed : (uint64_t)time(NULL);
    curandSetPseudoRandomGeneratorSeed(state->curand_gen, state->seed);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CIRCULAR BUFFER INDEXING (Device)
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ __forceinline__ 
int circular_idx(int t_abs, int buffer_mask) {
    return t_abs & buffer_mask;
}

__device__ __forceinline__
int64_t noise_offset(int theta_idx, int inner_idx, int circ_idx, 
                     int N_inner, int buffer_size) {
    return (int64_t)theta_idx * N_inner * buffer_size 
         + (int64_t)inner_idx * buffer_size 
         + circ_idx;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Initialize RNG states
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_stream_init_rng(
    curandState* states,
    uint64_t seed,
    int N_total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_total) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Initialize from prior (streaming version)
 *═══════════════════════════════════════════════════════════════════════════════*/

/* TODO: Implement kernel_stream_init_from_prior */
/* This will sample θ from prior and initialize inner filters */

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: RBPF forward step with circular buffer
 *═══════════════════════════════════════════════════════════════════════════════*/

/* TODO: Implement kernel_stream_rbpf_step */
/* Key difference: Uses circular_idx() for noise access */

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Compute posterior moments (μ_θ, Σ_θ)
 *═══════════════════════════════════════════════════════════════════════════════*/

/* TODO: Implement kernel_compute_theta_moments */
/* Computes weighted mean and covariance of θ-particles */

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Liu-West regularization
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_liu_west_regularize(
    float* d_theta[SMC2_N_PARAMS],
    const float* theta_mean,
    const float* theta_cov_chol,  /* Lower triangular, row-major */
    float a,                       /* Shrinkage factor */
    curandState* rng_states,
    int N_theta,
    int N_inner
) {
    int theta_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (theta_idx >= N_theta) return;
    
    /* Use first inner particle's RNG for this θ */
    curandState* rng = &rng_states[theta_idx * N_inner];
    
    float h = sqrtf(1.0f - a * a);  /* Jitter scale */
    
    /* Generate standard normal vector */
    float eps[SMC2_N_PARAMS];
    for (int p = 0; p < SMC2_N_PARAMS; p++) {
        eps[p] = curand_normal(rng);
    }
    
    /* Apply: θ' = a·θ + (1-a)·μ + h·L·ε */
    /* where L is Cholesky of Σ_θ */
    for (int p = 0; p < SMC2_N_PARAMS; p++) {
        float theta_old = d_theta[p][theta_idx];
        
        /* Shrink toward mean */
        float theta_new = a * theta_old + (1.0f - a) * theta_mean[p];
        
        /* Add jitter: h * (L * eps)[p] = h * sum_q L[p,q] * eps[q] */
        /* L is lower triangular, so L[p,q] = 0 for q > p */
        float jitter = 0.0f;
        for (int q = 0; q <= p; q++) {
            int L_idx = p * SMC2_N_PARAMS + q;  /* Row-major lower tri */
            jitter += theta_cov_chol[L_idx] * eps[q];
        }
        theta_new += h * jitter;
        
        d_theta[p][theta_idx] = theta_new;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: CPMMH with drift-augmented proposal
 *═══════════════════════════════════════════════════════════════════════════════*/

/* TODO: Implement kernel_stream_cpmmh_rejuvenate */
/* Key differences from batch:
 *   1. Circular buffer indexing for noise
 *   2. Fixed-lag replay window [t - L, t]
 *   3. Proposal uses Σ_inflated = Σ_θ + Q
 */

/*═══════════════════════════════════════════════════════════════════════════════
 * HOST: Compute Cholesky decomposition (simple, for 8×8)
 *═══════════════════════════════════════════════════════════════════════════════*/

static void cholesky_8x8(const float* A, float* L) {
    /* A and L are 8×8, row-major */
    const int n = SMC2_N_PARAMS;
    memset(L, 0, n * n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = A[i * n + j];
            for (int k = 0; k < j; k++) {
                sum -= L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                L[i * n + j] = sqrtf(fmaxf(sum, 1e-10f));
            } else {
                L[i * n + j] = sum / fmaxf(L[j * n + j], 1e-10f);
            }
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * HOST: Update posterior moments
 *═══════════════════════════════════════════════════════════════════════════════*/

static void update_posterior_moments(SMC2StreamState* state) {
    int N_theta = state->config.N_theta;
    
    /* Download weights and θ values */
    float* h_weight = (float*)malloc(N_theta * sizeof(float));
    float* h_theta[SMC2_N_PARAMS];
    for (int p = 0; p < SMC2_N_PARAMS; p++) {
        h_theta[p] = (float*)malloc(N_theta * sizeof(float));
        cudaMemcpy(h_theta[p], state->d_theta[p], N_theta * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(h_weight, state->d_weight, N_theta * sizeof(float), cudaMemcpyDeviceToHost);
    
    /* Compute weighted mean */
    for (int p = 0; p < SMC2_N_PARAMS; p++) {
        state->theta_mean[p] = 0.0f;
        for (int j = 0; j < N_theta; j++) {
            state->theta_mean[p] += h_weight[j] * h_theta[p][j];
        }
    }
    
    /* Compute weighted covariance */
    memset(state->theta_cov, 0, SMC2_N_PARAMS * SMC2_N_PARAMS * sizeof(float));
    for (int j = 0; j < N_theta; j++) {
        for (int p = 0; p < SMC2_N_PARAMS; p++) {
            float dp = h_theta[p][j] - state->theta_mean[p];
            for (int q = 0; q <= p; q++) {  /* Lower triangle */
                float dq = h_theta[q][j] - state->theta_mean[q];
                state->theta_cov[p * SMC2_N_PARAMS + q] += h_weight[j] * dp * dq;
            }
        }
    }
    
    /* Mirror to upper triangle */
    for (int p = 0; p < SMC2_N_PARAMS; p++) {
        for (int q = p + 1; q < SMC2_N_PARAMS; q++) {
            state->theta_cov[p * SMC2_N_PARAMS + q] = state->theta_cov[q * SMC2_N_PARAMS + p];
        }
    }
    
    /* Add drift Q to diagonal for inflated covariance, then Cholesky */
    float cov_inflated[SMC2_N_PARAMS * SMC2_N_PARAMS];
    memcpy(cov_inflated, state->theta_cov, sizeof(cov_inflated));
    for (int p = 0; p < SMC2_N_PARAMS; p++) {
        cov_inflated[p * SMC2_N_PARAMS + p] += state->Q_effective[p];
    }
    cholesky_8x8(cov_inflated, state->theta_cov_chol);
    
    /* Cleanup */
    free(h_weight);
    for (int p = 0; p < SMC2_N_PARAMS; p++) {
        free(h_theta[p]);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * HOST: Update adaptive Q scaling
 *═══════════════════════════════════════════════════════════════════════════════*/

static void update_adaptive_Q(SMC2StreamState* state, float nll_current) {
    if (!state->config.enable_adaptive_Q) return;
    
    /* Accumulate NLL excess */
    float excess = fmaxf(0.0f, nll_current - state->config.nll_baseline);
    state->nll_excess_accum += excess;
    
    /* Decay accumulated excess (forgetting factor) */
    state->nll_excess_accum *= 0.99f;
    
    /* Compute scaling factor */
    float scale = 1.0f + state->nll_excess_accum / 50.0f;
    scale = fminf(scale, state->config.adaptive_Q_scale);
    
    /* Apply to effective Q */
    const SMC2DriftQ* Q_base = &state->config.Q;
    state->Q_effective[0] = Q_base->rho * scale;
    state->Q_effective[1] = Q_base->sigma_z * scale;
    state->Q_effective[2] = Q_base->mu_base * scale;
    state->Q_effective[3] = Q_base->mu_scale * scale;
    state->Q_effective[4] = Q_base->mu_rate * scale;
    state->Q_effective[5] = Q_base->sigma_base * scale;
    state->Q_effective[6] = Q_base->sigma_scale * scale;
    state->Q_effective[7] = Q_base->sigma_rate * scale;
    
    state->last_diag.Q_scale_factor = scale;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN UPDATE FUNCTION (Skeleton)
 *═══════════════════════════════════════════════════════════════════════════════*/

float smc2_stream_update(SMC2StreamState* state, float y_obs) {
    state->t_current++;
    int t = state->t_current;
    int buffer_idx = t & state->buffer_mask;
    
    /* Store observation in circular buffer */
    cudaMemcpy(state->d_y_history + buffer_idx, &y_obs, sizeof(float), cudaMemcpyHostToDevice);
    
    /* TODO: Generate noise for this timestep in circular buffer */
    
    /* TODO: RBPF forward step (kernel_stream_rbpf_step) */
    
    /* TODO: Compute outer ESS */
    float outer_ess = 0.0f;  /* Placeholder */
    
    /* TODO: If ESS < threshold: resample + rejuvenate */
    int N_theta = state->config.N_theta;
    float ess_ratio = outer_ess / N_theta;
    
    if (ess_ratio < state->config.ess_threshold) {
        /* Update posterior moments (μ_θ, Σ_θ + Q) */
        update_posterior_moments(state);
        
        /* TODO: Outer resampling */
        
        /* TODO: CPMMH rejuvenation with drift-augmented proposal */
        
        /* Liu-West regularization */
        int block_size = 256;
        int n_blocks = (N_theta + block_size - 1) / block_size;
        
        /* Upload moments to device */
        float* d_theta_mean;
        float* d_theta_cov_chol;
        cudaMalloc(&d_theta_mean, SMC2_N_PARAMS * sizeof(float));
        cudaMalloc(&d_theta_cov_chol, SMC2_N_PARAMS * SMC2_N_PARAMS * sizeof(float));
        cudaMemcpy(d_theta_mean, state->theta_mean, SMC2_N_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_theta_cov_chol, state->theta_cov_chol, SMC2_N_PARAMS * SMC2_N_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
        
        /* Note: kernel_liu_west_regularize needs d_theta as device array of pointers */
        /* This requires some restructuring - leaving as TODO */
        
        cudaFree(d_theta_mean);
        cudaFree(d_theta_cov_chol);
        
        state->last_diag.did_resample = 1;
        state->last_diag.did_rejuvenate = 1;
    } else {
        state->last_diag.did_resample = 0;
        state->last_diag.did_rejuvenate = 0;
    }
    
    /* Update diagnostics */
    state->last_diag.t_current = t;
    state->last_diag.outer_ess = outer_ess;
    state->last_diag.outer_ess_ratio = ess_ratio;
    
    /* Update oldest valid time */
    if (t >= state->config.fixed_lag) {
        state->t_oldest = t - state->config.fixed_lag;
    }
    
    return outer_ess;
}

float smc2_stream_update_batch(SMC2StreamState* state, const float* y_obs, int n_obs) {
    float ess = 0.0f;
    for (int i = 0; i < n_obs; i++) {
        ess = smc2_stream_update(state, y_obs[i]);
    }
    return ess;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * POSTERIOR QUERY
 *═══════════════════════════════════════════════════════════════════════════════*/

void smc2_stream_get_theta_mean(SMC2StreamState* state, float* theta_mean) {
    /* Recompute from particles (more accurate than cached) */
    update_posterior_moments(state);
    memcpy(theta_mean, state->theta_mean, SMC2_N_PARAMS * sizeof(float));
}

void smc2_stream_get_theta_std(SMC2StreamState* state, float* theta_std) {
    update_posterior_moments(state);
    for (int p = 0; p < SMC2_N_PARAMS; p++) {
        theta_std[p] = sqrtf(state->theta_cov[p * SMC2_N_PARAMS + p]);
    }
}

void smc2_stream_get_theta_cov(SMC2StreamState* state, float* theta_cov) {
    update_posterior_moments(state);
    memcpy(theta_cov, state->theta_cov, SMC2_N_PARAMS * SMC2_N_PARAMS * sizeof(float));
}

float smc2_stream_get_outer_ess(SMC2StreamState* state) {
    return state->last_diag.outer_ess;
}

int smc2_stream_get_t_current(SMC2StreamState* state) {
    return state->t_current;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

SMC2StreamDiag smc2_stream_get_diag(SMC2StreamState* state) {
    return state->last_diag;
}

void smc2_stream_reset_nll_baseline(SMC2StreamState* state) {
    if (state->nll_count > 0) {
        state->config.nll_baseline = state->nll_sum / state->nll_count;
        printf("[SMC2 Stream] NLL baseline set to %.3f\n", state->config.nll_baseline);
    }
    state->nll_sum = 0.0f;
    state->nll_count = 0;
    state->nll_excess_accum = 0.0f;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * RUNTIME CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void smc2_stream_set_drift_Q(SMC2StreamState* state, const SMC2DriftQ* Q) {
    state->config.Q = *Q;
    /* Update effective Q (will be scaled if adaptive is on) */
    state->Q_effective[0] = Q->rho;
    state->Q_effective[1] = Q->sigma_z;
    state->Q_effective[2] = Q->mu_base;
    state->Q_effective[3] = Q->mu_scale;
    state->Q_effective[4] = Q->mu_rate;
    state->Q_effective[5] = Q->sigma_base;
    state->Q_effective[6] = Q->sigma_scale;
    state->Q_effective[7] = Q->sigma_rate;
}

void smc2_stream_get_drift_Q(SMC2StreamState* state, SMC2DriftQ* Q) {
    *Q = state->config.Q;
}

void smc2_stream_set_liu_west_a(SMC2StreamState* state, float a) {
    state->config.liu_west_a = a;
}

void smc2_stream_set_adaptive_Q(SMC2StreamState* state, int enable, float max_scale) {
    state->config.enable_adaptive_Q = enable;
    state->config.adaptive_Q_scale = max_scale;
}

void smc2_stream_force_rejuvenate(SMC2StreamState* state) {
    /* TODO: Trigger rejuvenation on next update */
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CHECKPOINT/RESTORE (Stubs)
 *═══════════════════════════════════════════════════════════════════════════════*/

size_t smc2_stream_checkpoint_size(SMC2StreamState* state) {
    /* TODO: Calculate actual size */
    return 0;
}

size_t smc2_stream_save_checkpoint(SMC2StreamState* state, void* buffer) {
    /* TODO: Implement */
    return 0;
}

int smc2_stream_load_checkpoint(SMC2StreamState* state, const void* buffer) {
    /* TODO: Implement */
    return -1;
}
