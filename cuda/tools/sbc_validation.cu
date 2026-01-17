/*
 * sbc_validation.cu
 * 
 * Simulation-Based Calibration (SBC) for SMC² validation.
 * 
 * The Gold Standard: If your implementation is correct, the true parameter
 * should fall uniformly across the posterior percentiles.
 * 
 * Method:
 *   1. Sample θ_true ~ Prior
 *   2. Simulate data y with θ_true
 *   3. Run SMC² on y to get posterior particles
 *   4. Compute rank of θ_true within particles
 *   5. Repeat N_sims times
 *   6. Rank histogram should be FLAT (uniform)
 * 
 * Diagnosis:
 *   - U-shape: Posterior too narrow (overconfident)
 *   - Dome-shape: Posterior too wide (underconfident)
 *   - Sloped: Systematic bias in code
 * 
 * Compile:
 *   nvcc -o sbc_validation sbc_validation.cu -std=c++17 -O3 -arch=sm_120
 * 
 * Run:
 *   ./sbc_validation [n_sims] [T]
 *   ./sbc_validation 100 500     # 100 simulations, T=500 each
 * 
 * Output:
 *   sbc_ranks.csv - Raw rank data for each parameter
 *   sbc_summary.txt - Chi-square test results
 */

#include "smc2_rbpf_cuda.cuh"
#include "smc2_prior_calibration_gpu.cuh"
#include "sv_data_generator.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>

/* ============================================================================
 * Configuration
 * ============================================================================ */

struct SBCConfig {
    int n_sims;         /* Number of simulations (default: 100) */
    int T;              /* Observations per simulation (default: 500) */
    int N_theta;        /* Outer particles (default: 256) */
    int N_inner;        /* Inner particles (default: 256) */
    int n_bins;         /* Histogram bins (default: 20) */
    uint64_t base_seed; /* RNG seed (default: time-based) */
    bool verbose;       /* Print progress */
};

static SBCConfig default_config() {
    SBCConfig cfg;
    cfg.n_sims = 100;
    cfg.T = 500;
    cfg.N_theta = 256;
    cfg.N_inner = 256;
    cfg.n_bins = 20;
    cfg.base_seed = 0;
    cfg.verbose = true;
    return cfg;
}

/* ============================================================================
 * Host RNG (xorshift64*)
 * ============================================================================ */

static uint64_t g_rng_state = 12345678901234567ULL;

static void seed_rng(uint64_t seed) {
    g_rng_state = seed ? seed : (uint64_t)time(NULL);
}

static uint64_t rand_u64() {
    g_rng_state ^= g_rng_state >> 12;
    g_rng_state ^= g_rng_state << 25;
    g_rng_state ^= g_rng_state >> 27;
    return g_rng_state * 0x2545F4914F6CDD1DULL;
}

static float rand_uniform() {
    return (rand_u64() >> 11) * (1.0f / 9007199254740992.0f);
}

static float rand_normal() {
    float u1 = rand_uniform();
    float u2 = rand_uniform();
    while (u1 < 1e-10f) u1 = rand_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

/* ============================================================================
 * Sample from Prior (with bounds clamping)
 * ============================================================================ */

struct TrueParams {
    float rho;
    float sigma_z;
    float mu_base;
    float mu_scale;
    float mu_rate;
    float sigma_base;
    float sigma_scale;
    float sigma_rate;
};

static TrueParams sample_from_prior(const SMC2PriorGPU& prior, const SVBounds& bounds) {
    TrueParams theta;
    
    /* Sample and clamp to bounds */
    theta.rho = prior.rho_mean + prior.rho_std * rand_normal();
    theta.rho = fmaxf(bounds.rho_min, fminf(bounds.rho_max, theta.rho));
    
    theta.sigma_z = prior.sigma_z_mean + prior.sigma_z_std * rand_normal();
    theta.sigma_z = fmaxf(bounds.sigma_z_min, fminf(bounds.sigma_z_max, theta.sigma_z));
    
    theta.mu_base = prior.mu_base_mean + prior.mu_base_std * rand_normal();
    theta.mu_base = fmaxf(bounds.mu_base_min, fminf(bounds.mu_base_max, theta.mu_base));
    
    theta.mu_scale = prior.mu_scale_mean + prior.mu_scale_std * rand_normal();
    theta.mu_scale = fmaxf(bounds.mu_scale_min, fminf(bounds.mu_scale_max, theta.mu_scale));
    
    theta.mu_rate = prior.mu_rate_mean + prior.mu_rate_std * rand_normal();
    theta.mu_rate = fmaxf(bounds.mu_rate_min, fminf(bounds.mu_rate_max, theta.mu_rate));
    
    theta.sigma_base = prior.sigma_base_mean + prior.sigma_base_std * rand_normal();
    theta.sigma_base = fmaxf(bounds.sigma_base_min, fminf(bounds.sigma_base_max, theta.sigma_base));
    
    theta.sigma_scale = prior.sigma_scale_mean + prior.sigma_scale_std * rand_normal();
    theta.sigma_scale = fmaxf(bounds.sigma_scale_min, fminf(bounds.sigma_scale_max, theta.sigma_scale));
    
    theta.sigma_rate = prior.sigma_rate_mean + prior.sigma_rate_std * rand_normal();
    theta.sigma_rate = fmaxf(bounds.sigma_rate_min, fminf(bounds.sigma_rate_max, theta.sigma_rate));
    
    return theta;
}

/* ============================================================================
 * Compute Rank of θ_true in Posterior Particles
 * ============================================================================ */

static int compute_rank(const float* particles, int n, float true_val) {
    int rank = 0;
    for (int i = 0; i < n; i++) {
        if (particles[i] < true_val) rank++;
    }
    return rank;
}

/* ============================================================================
 * Chi-Square Test for Uniformity
 * ============================================================================ */

static float chi_square_test(const int* histogram, int n_bins, int n_total) {
    float expected = (float)n_total / n_bins;
    float chi2 = 0.0f;
    
    for (int i = 0; i < n_bins; i++) {
        float diff = histogram[i] - expected;
        chi2 += diff * diff / expected;
    }
    
    return chi2;
}

/* Critical values for chi-square (df = n_bins - 1) at α = 0.05 */
static float chi_square_critical(int df) {
    /* Approximate critical values for common df */
    if (df <= 10) return 18.31f;
    if (df <= 15) return 25.00f;
    if (df <= 20) return 31.41f;
    if (df <= 25) return 37.65f;
    if (df <= 30) return 43.77f;
    return 1.5f * df;  /* Rough approximation */
}

/* ============================================================================
 * Main SBC Loop
 * ============================================================================ */

struct SBCResults {
    std::vector<int> ranks_rho;
    std::vector<int> ranks_sigma_z;
    std::vector<int> ranks_mu_base;
    
    int histogram_rho[50];
    int histogram_sigma_z[50];
    int histogram_mu_base[50];
    
    float chi2_rho;
    float chi2_sigma_z;
    float chi2_mu_base;
    
    int n_sims_completed;
};

static SBCResults run_sbc(const SBCConfig& cfg) {
    SBCResults results;
    memset(&results, 0, sizeof(results));
    
    seed_rng(cfg.base_seed);
    
    /* ─────────────────────────────────────────────────────────────────────
     * Setup: Allocate SMC² state and host buffers
     * ───────────────────────────────────────────────────────────────────── */
    
    SMC2StateCUDA* state = smc2_cuda_alloc(cfg.N_theta, cfg.N_inner);
    if (!state) {
        fprintf(stderr, "Failed to allocate SMC² state\n");
        return results;
    }
    
    /* Host buffers for particles */
    std::vector<float> h_rho(cfg.N_theta);
    std::vector<float> h_sigma_z(cfg.N_theta);
    std::vector<float> h_mu_base(cfg.N_theta);
    
    /* Data generator */
    SVDataGenerator gen;
    gen.T = cfg.T;
    
    /* Get prior and bounds from state */
    SMC2PriorGPU prior;
    prior.rho_mean = state->prior.rho_mean;
    prior.rho_std = state->prior.rho_std;
    prior.sigma_z_mean = state->prior.sigma_z_mean;
    prior.sigma_z_std = state->prior.sigma_z_std;
    prior.mu_base_mean = state->prior.mu_base_mean;
    prior.mu_base_std = state->prior.mu_base_std;
    prior.mu_scale_mean = state->prior.mu_scale_mean;
    prior.mu_scale_std = state->prior.mu_scale_std;
    prior.mu_rate_mean = state->prior.mu_rate_mean;
    prior.mu_rate_std = state->prior.mu_rate_std;
    prior.sigma_base_mean = state->prior.sigma_base_mean;
    prior.sigma_base_std = state->prior.sigma_base_std;
    prior.sigma_scale_mean = state->prior.sigma_scale_mean;
    prior.sigma_scale_std = state->prior.sigma_scale_std;
    prior.sigma_rate_mean = state->prior.sigma_rate_mean;
    prior.sigma_rate_std = state->prior.sigma_rate_std;
    
    if (cfg.verbose) {
        printf("\n");
        printf("╔═══════════════════════════════════════════════════════════════════╗\n");
        printf("║  Simulation-Based Calibration (SBC)                               ║\n");
        printf("╠═══════════════════════════════════════════════════════════════════╣\n");
        printf("║  Simulations: %-5d   T: %-5d   N_theta: %-5d   N_inner: %-5d   ║\n",
               cfg.n_sims, cfg.T, cfg.N_theta, cfg.N_inner);
        printf("╚═══════════════════════════════════════════════════════════════════╝\n");
        printf("\n");
    }
    
    /* ─────────────────────────────────────────────────────────────────────
     * Main SBC Loop
     * ───────────────────────────────────────────────────────────────────── */
    
    for (int sim = 0; sim < cfg.n_sims; sim++) {
        if (cfg.verbose && (sim % 10 == 0 || sim == cfg.n_sims - 1)) {
            printf("\r  [%3d / %3d] Running simulation...", sim + 1, cfg.n_sims);
            fflush(stdout);
        }
        
        /* 1. Sample θ_true from prior */
        TrueParams theta_true = sample_from_prior(prior, state->bounds);
        
        /* 2. Generate synthetic data with θ_true */
        gen.seed(rand_u64());
        gen.rho = theta_true.rho;
        gen.sigma_z = theta_true.sigma_z;
        gen.mu_base = theta_true.mu_base;
        gen.mu_scale = theta_true.mu_scale;
        gen.mu_rate = theta_true.mu_rate;
        gen.sigma_base = theta_true.sigma_base;
        gen.sigma_scale = theta_true.sigma_scale;
        gen.sigma_rate = theta_true.sigma_rate;
        gen.generate();
        
        /* 3. Run SMC² */
        smc2_cuda_set_seed(state, rand_u64());
        smc2_cuda_init_from_prior(state);
        
        for (int t = 0; t < cfg.T; t++) {
            smc2_cuda_update(state, gen.y[t]);
        }
        
        /* 4. Copy posterior particles to host */
        cudaMemcpy(h_rho.data(), state->d_particles.rho, 
                   cfg.N_theta * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sigma_z.data(), state->d_particles.sigma_z,
                   cfg.N_theta * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mu_base.data(), state->d_particles.mu_base,
                   cfg.N_theta * sizeof(float), cudaMemcpyDeviceToHost);
        
        /* 5. Compute ranks */
        int rank_rho = compute_rank(h_rho.data(), cfg.N_theta, theta_true.rho);
        int rank_sigma_z = compute_rank(h_sigma_z.data(), cfg.N_theta, theta_true.sigma_z);
        int rank_mu_base = compute_rank(h_mu_base.data(), cfg.N_theta, theta_true.mu_base);
        
        results.ranks_rho.push_back(rank_rho);
        results.ranks_sigma_z.push_back(rank_sigma_z);
        results.ranks_mu_base.push_back(rank_mu_base);
        
        /* 6. Update histograms */
        int bin_rho = (rank_rho * cfg.n_bins) / cfg.N_theta;
        int bin_sigma_z = (rank_sigma_z * cfg.n_bins) / cfg.N_theta;
        int bin_mu_base = (rank_mu_base * cfg.n_bins) / cfg.N_theta;
        
        bin_rho = (bin_rho >= cfg.n_bins) ? cfg.n_bins - 1 : bin_rho;
        bin_sigma_z = (bin_sigma_z >= cfg.n_bins) ? cfg.n_bins - 1 : bin_sigma_z;
        bin_mu_base = (bin_mu_base >= cfg.n_bins) ? cfg.n_bins - 1 : bin_mu_base;
        
        results.histogram_rho[bin_rho]++;
        results.histogram_sigma_z[bin_sigma_z]++;
        results.histogram_mu_base[bin_mu_base]++;
        
        results.n_sims_completed++;
    }
    
    if (cfg.verbose) {
        printf("\r  [%3d / %3d] Complete!                    \n", cfg.n_sims, cfg.n_sims);
    }
    
    /* ─────────────────────────────────────────────────────────────────────
     * Chi-Square Tests
     * ───────────────────────────────────────────────────────────────────── */
    
    results.chi2_rho = chi_square_test(results.histogram_rho, cfg.n_bins, cfg.n_sims);
    results.chi2_sigma_z = chi_square_test(results.histogram_sigma_z, cfg.n_bins, cfg.n_sims);
    results.chi2_mu_base = chi_square_test(results.histogram_mu_base, cfg.n_bins, cfg.n_sims);
    
    smc2_cuda_free(state);
    
    return results;
}

/* ============================================================================
 * Print Results
 * ============================================================================ */

static void print_histogram(const char* name, const int* hist, int n_bins, int n_sims) {
    int max_count = 0;
    for (int i = 0; i < n_bins; i++) {
        if (hist[i] > max_count) max_count = hist[i];
    }
    
    printf("\n  %s:\n", name);
    printf("  ");
    for (int i = 0; i < n_bins; i++) printf("─");
    printf("\n");
    
    int bar_height = 8;
    float expected = (float)n_sims / n_bins;
    
    for (int row = bar_height; row >= 1; row--) {
        printf("  │");
        for (int i = 0; i < n_bins; i++) {
            float threshold = (float)row / bar_height * (max_count + 1);
            if (hist[i] >= threshold) {
                printf("█");
            } else if (hist[i] >= threshold - expected * 0.3f) {
                printf("▄");
            } else {
                printf(" ");
            }
        }
        printf("│\n");
    }
    
    printf("  └");
    for (int i = 0; i < n_bins; i++) printf("─");
    printf("┘\n");
    
    printf("   0");
    for (int i = 0; i < n_bins - 4; i++) printf(" ");
    printf("N=%d\n", n_sims);
}

static void print_results(const SBCResults& results, const SBCConfig& cfg) {
    float critical = chi_square_critical(cfg.n_bins - 1);
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║  SBC Results                                                      ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Parameter  │  χ²      │  Critical │  Pass?                       ║\n");
    printf("╠═════════════╪══════════╪═══════════╪══════════════════════════════╣\n");
    
    const char* pass_rho = (results.chi2_rho < critical) ? "✓ PASS" : "✗ FAIL";
    const char* pass_sigma_z = (results.chi2_sigma_z < critical) ? "✓ PASS" : "✗ FAIL";
    const char* pass_mu_base = (results.chi2_mu_base < critical) ? "✓ PASS" : "✗ FAIL";
    
    printf("║  ρ          │  %7.2f │  %7.2f  │  %s                        ║\n",
           results.chi2_rho, critical, pass_rho);
    printf("║  σ_z        │  %7.2f │  %7.2f  │  %s                        ║\n",
           results.chi2_sigma_z, critical, pass_sigma_z);
    printf("║  μ_base     │  %7.2f │  %7.2f  │  %s                        ║\n",
           results.chi2_mu_base, critical, pass_mu_base);
    
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    printf("\n  Interpretation:\n");
    printf("    - χ² < critical: Histogram is uniform → code is likely correct\n");
    printf("    - χ² >> critical: Non-uniform → check for bugs\n");
    printf("\n  Histogram shapes:\n");
    printf("    - U-shape: Posterior too narrow (overconfident)\n");
    printf("    - Dome: Posterior too wide (underconfident)\n");
    printf("    - Sloped: Systematic bias\n");
    
    print_histogram("ρ (rho)", results.histogram_rho, cfg.n_bins, cfg.n_sims);
    print_histogram("σ_z (sigma_z)", results.histogram_sigma_z, cfg.n_bins, cfg.n_sims);
    print_histogram("μ_base", results.histogram_mu_base, cfg.n_bins, cfg.n_sims);
}

/* ============================================================================
 * Save Results to CSV
 * ============================================================================ */

static void save_results(const SBCResults& results, const SBCConfig& cfg, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }
    
    fprintf(f, "sim,rank_rho,rank_sigma_z,rank_mu_base\n");
    for (int i = 0; i < results.n_sims_completed; i++) {
        fprintf(f, "%d,%d,%d,%d\n", i,
                results.ranks_rho[i],
                results.ranks_sigma_z[i],
                results.ranks_mu_base[i]);
    }
    
    fclose(f);
    printf("\n  Ranks saved to: %s\n", filename);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    SBCConfig cfg = default_config();
    
    if (argc > 1) cfg.n_sims = atoi(argv[1]);
    if (argc > 2) cfg.T = atoi(argv[2]);
    if (argc > 3) cfg.base_seed = (uint64_t)atoll(argv[3]);
    
    /* Run SBC */
    SBCResults results = run_sbc(cfg);
    
    if (results.n_sims_completed == 0) {
        fprintf(stderr, "No simulations completed!\n");
        return 1;
    }
    
    /* Print and save results */
    print_results(results, cfg);
    save_results(results, cfg, "sbc_ranks.csv");
    
    /* Summary */
    float critical = chi_square_critical(cfg.n_bins - 1);
    bool all_pass = (results.chi2_rho < critical) &&
                    (results.chi2_sigma_z < critical) &&
                    (results.chi2_mu_base < critical);
    
    printf("\n");
    if (all_pass) {
        printf("  ✓ All parameters pass SBC validation!\n");
        printf("    Your SMC² implementation appears to be correct.\n");
    } else {
        printf("  ✗ Some parameters failed SBC validation.\n");
        printf("    Review the histograms above for diagnostic hints.\n");
    }
    printf("\n");
    
    return all_pass ? 0 : 1;
}
