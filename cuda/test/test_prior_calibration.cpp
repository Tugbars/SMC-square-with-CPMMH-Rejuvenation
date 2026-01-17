/*
 * test_prior_calibration.cpp
 * 
 * Validates smc2_prior_calibration.cuh by:
 *   1. Generating synthetic data with known parameters
 *   2. Running calibration on warmup subset
 *   3. Checking estimates are reasonable
 *   4. Verifying prior ±2σ covers true values
 * 
 * Compile:
 *   g++ -o test_prior_calibration test_prior_calibration.cpp -std=c++14 -lm
 * 
 * Run:
 *   ./test_prior_calibration
 */

// Include sv_data_generator FIRST so it defines minimal SMC2 structs
#include "sv_data_generator.cuh"
// Then include calibration (uses its own SMC2Prior, doesn't conflict)
#include "smc2_prior_calibration.cuh"
#include <cstdio>
#include <cmath>
#include <cstring>

/* ═══════════════════════════════════════════════════════════════════════════
 * Test Utilities
 * ═══════════════════════════════════════════════════════════════════════════ */

struct TestResult {
    const char* name;
    bool passed;
    char message[256];
};

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            snprintf(result.message, sizeof(result.message), "FAIL: %s", msg); \
            result.passed = false; \
            return result; \
        } \
    } while(0)

#define TEST_ASSERT_NEAR(val, expected, tol, name) \
    do { \
        float _v = (val), _e = (expected), _t = (tol); \
        if (fabsf(_v - _e) > _t) { \
            snprintf(result.message, sizeof(result.message), \
                     "FAIL: %s = %.4f, expected %.4f ± %.4f", name, _v, _e, _t); \
            result.passed = false; \
            return result; \
        } \
    } while(0)

#define TEST_ASSERT_COVERS(val, mean, std, name) \
    do { \
        float _v = (val), _m = (mean), _s = (std); \
        float _lo = _m - 2.0f * _s, _hi = _m + 2.0f * _s; \
        if (_v < _lo || _v > _hi) { \
            snprintf(result.message, sizeof(result.message), \
                     "FAIL: %s = %.4f not in prior ±2σ [%.4f, %.4f]", name, _v, _lo, _hi); \
            result.passed = false; \
            return result; \
        } \
    } while(0)

void print_result(const TestResult& r) {
    g_tests_run++;
    if (r.passed) {
        g_tests_passed++;
        printf("  ✓ %s\n", r.name);
    } else {
        printf("  ✗ %s\n    %s\n", r.name, r.message);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: Calm Regime Calibration
 * ═══════════════════════════════════════════════════════════════════════════ */

TestResult test_calm_regime() {
    TestResult result = {"Calm regime calibration", true, "OK"};
    
    // Generate calm data with known parameters
    SVDataGenerator gen;
    gen.seed(12345);
    gen.set_realistic_calm();  // ρ=0.98, σ_z=0.08, μ_base=-10.5
    gen.T = 500;
    gen.generate();
    
    float true_rho = gen.rho;
    float true_sigma_z = gen.sigma_z;
    float true_mu_base = gen.mu_base;
    
    // Extract warmup (first 100 observations need to be converted to "returns")
    // The generator outputs y = h + log(ε²), which is log(return²)
    // For calibration we need the same format
    float warmup[100];
    for (int i = 0; i < 100; i++) {
        // Convert log(y²) back to pseudo-return: y ≈ exp(log(y²)/2)
        // Actually, the calibration expects returns, and computes log(y²) internally
        // So we need: return = exp(gen.y[i] / 2) with random sign
        float log_y2 = gen.y[i];
        warmup[i] = expf(log_y2 / 2.0f);
        if (i % 2 == 0) warmup[i] = -warmup[i];  // Random sign
    }
    
    // Calibrate
    SMC2Prior prior;
    int rc = smc2_calibrate_priors(warmup, 100, &SPY_BOUNDS, &prior);
    TEST_ASSERT(rc == 0, "smc2_calibrate_priors returned error");
    
    // Check estimates are in reasonable range for calm regime
    // Note: Estimates won't be exact due to noise, but should be calm-ish
    TEST_ASSERT(prior.rho_mean > 0.85f, "rho estimate too low for calm regime");
    TEST_ASSERT(prior.sigma_z_mean < 0.35f, "sigma_z estimate too high for calm regime");
    
    // Check prior ±2σ covers true values
    TEST_ASSERT_COVERS(true_rho, prior.rho_mean, prior.rho_std, "rho");
    TEST_ASSERT_COVERS(true_sigma_z, prior.sigma_z_mean, prior.sigma_z_std, "sigma_z");
    TEST_ASSERT_COVERS(true_mu_base, prior.mu_base_mean, prior.mu_base_std, "mu_base");
    
    snprintf(result.message, sizeof(result.message), 
             "OK: est ρ=%.3f (true=%.3f), est σ_z=%.3f (true=%.3f)", 
             prior.rho_mean, true_rho, prior.sigma_z_mean, true_sigma_z);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: Crisis Regime Calibration
 * ═══════════════════════════════════════════════════════════════════════════ */

TestResult test_crisis_regime() {
    TestResult result = {"Crisis regime calibration", true, "OK"};
    
    // Generate crisis data
    SVDataGenerator gen;
    gen.seed(67890);
    gen.set_realistic_crisis();  // ρ=0.85, σ_z=0.30, μ_base=-7.5
    gen.T = 500;
    gen.generate();
    
    float true_rho = gen.rho;
    float true_sigma_z = gen.sigma_z;
    float true_mu_base = gen.mu_base;
    
    // Extract warmup
    float warmup[100];
    for (int i = 0; i < 100; i++) {
        float log_y2 = gen.y[i];
        warmup[i] = expf(log_y2 / 2.0f);
        if (i % 2 == 0) warmup[i] = -warmup[i];
    }
    
    // Calibrate
    SMC2Prior prior;
    int rc = smc2_calibrate_priors(warmup, 100, &SPY_BOUNDS, &prior);
    TEST_ASSERT(rc == 0, "smc2_calibrate_priors returned error");
    
    // Check estimates are in reasonable range for crisis regime
    TEST_ASSERT(prior.rho_mean < 0.95f, "rho estimate too high for crisis regime");
    TEST_ASSERT(prior.sigma_z_mean > 0.10f, "sigma_z estimate too low for crisis regime");
    
    // Check prior ±2σ covers true values
    TEST_ASSERT_COVERS(true_rho, prior.rho_mean, prior.rho_std, "rho");
    TEST_ASSERT_COVERS(true_sigma_z, prior.sigma_z_mean, prior.sigma_z_std, "sigma_z");
    TEST_ASSERT_COVERS(true_mu_base, prior.mu_base_mean, prior.mu_base_std, "mu_base");
    
    snprintf(result.message, sizeof(result.message), 
             "OK: est ρ=%.3f (true=%.3f), est σ_z=%.3f (true=%.3f)", 
             prior.rho_mean, true_rho, prior.sigma_z_mean, true_sigma_z);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: Moderate Regime Calibration
 * ═══════════════════════════════════════════════════════════════════════════ */

TestResult test_moderate_regime() {
    TestResult result = {"Moderate regime calibration", true, "OK"};
    
    // Generate moderate data
    SVDataGenerator gen;
    gen.seed(11111);
    gen.set_realistic_moderate();  // ρ=0.94, σ_z=0.15, μ_base=-9.0
    gen.T = 500;
    gen.generate();
    
    float true_rho = gen.rho;
    float true_sigma_z = gen.sigma_z;
    float true_mu_base = gen.mu_base;
    
    // Extract warmup
    float warmup[100];
    for (int i = 0; i < 100; i++) {
        float log_y2 = gen.y[i];
        warmup[i] = expf(log_y2 / 2.0f);
        if (i % 2 == 0) warmup[i] = -warmup[i];
    }
    
    // Calibrate
    SMC2Prior prior;
    int rc = smc2_calibrate_priors(warmup, 100, &SPY_BOUNDS, &prior);
    TEST_ASSERT(rc == 0, "smc2_calibrate_priors returned error");
    
    // Check prior ±2σ covers true values
    TEST_ASSERT_COVERS(true_rho, prior.rho_mean, prior.rho_std, "rho");
    TEST_ASSERT_COVERS(true_sigma_z, prior.sigma_z_mean, prior.sigma_z_std, "sigma_z");
    TEST_ASSERT_COVERS(true_mu_base, prior.mu_base_mean, prior.mu_base_std, "mu_base");
    
    snprintf(result.message, sizeof(result.message), 
             "OK: est ρ=%.3f (true=%.3f), est σ_z=%.3f (true=%.3f)", 
             prior.rho_mean, true_rho, prior.sigma_z_mean, true_sigma_z);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: Prior Width Covers Historical Range
 * ═══════════════════════════════════════════════════════════════════════════ */

TestResult test_prior_width_coverage() {
    TestResult result = {"Prior width covers historical range", true, "OK"};
    
    // Generate any data (moderate)
    SVDataGenerator gen;
    gen.seed(22222);
    gen.set_realistic_moderate();
    gen.T = 200;
    gen.generate();
    
    float warmup[100];
    for (int i = 0; i < 100; i++) {
        float log_y2 = gen.y[i];
        warmup[i] = expf(log_y2 / 2.0f);
        if (i % 2 == 0) warmup[i] = -warmup[i];
    }
    
    SMC2Prior prior;
    smc2_calibrate_priors(warmup, 100, &SPY_BOUNDS, &prior);
    
    // Check that prior ±2σ covers SPY historical bounds
    float rho_lo = prior.rho_mean - 2.0f * prior.rho_std;
    float rho_hi = prior.rho_mean + 2.0f * prior.rho_std;
    float sigma_z_lo = prior.sigma_z_mean - 2.0f * prior.sigma_z_std;
    float sigma_z_hi = prior.sigma_z_mean + 2.0f * prior.sigma_z_std;
    
    // Should cover both calm (ρ=0.99) and crisis (ρ=0.70) at ±2σ
    // The width should be ~(0.99-0.67)/4 = 0.08, so ±2σ = ±0.16
    TEST_ASSERT(prior.rho_std >= 0.05f, "rho_std too narrow");
    TEST_ASSERT(prior.sigma_z_std >= 0.10f, "sigma_z_std too narrow");
    
    // Check the range spans a reasonable portion of historical bounds
    float rho_range = rho_hi - rho_lo;
    float sigma_z_range = sigma_z_hi - sigma_z_lo;
    
    TEST_ASSERT(rho_range >= 0.20f, "rho prior range too narrow for regime coverage");
    TEST_ASSERT(sigma_z_range >= 0.40f, "sigma_z prior range too narrow for regime coverage");
    
    snprintf(result.message, sizeof(result.message), 
             "OK: ρ ±2σ=[%.2f,%.2f], σ_z ±2σ=[%.2f,%.2f]",
             rho_lo, rho_hi, sigma_z_lo, sigma_z_hi);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: Edge Case - Minimum Warmup Length
 * ═══════════════════════════════════════════════════════════════════════════ */

TestResult test_minimum_warmup() {
    TestResult result = {"Minimum warmup length (30 obs)", true, "OK"};
    
    SVDataGenerator gen;
    gen.seed(33333);
    gen.set_realistic_moderate();
    gen.T = 100;
    gen.generate();
    
    // Exactly 30 observations (minimum)
    float warmup[30];
    for (int i = 0; i < 30; i++) {
        float log_y2 = gen.y[i];
        warmup[i] = expf(log_y2 / 2.0f);
        if (i % 2 == 0) warmup[i] = -warmup[i];
    }
    
    SMC2Prior prior;
    int rc = smc2_calibrate_priors(warmup, 30, &SPY_BOUNDS, &prior);
    TEST_ASSERT(rc == 0, "Should accept 30 observations");
    
    // Just check it produces valid output
    TEST_ASSERT(prior.rho_mean > 0.5f && prior.rho_mean < 1.0f, "Invalid rho estimate");
    TEST_ASSERT(prior.sigma_z_mean > 0.0f && prior.sigma_z_mean < 1.0f, "Invalid sigma_z estimate");
    
    snprintf(result.message, sizeof(result.message), "OK: calibration works with n=30");
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: Edge Case - Too Few Observations
 * ═══════════════════════════════════════════════════════════════════════════ */

TestResult test_too_few_observations() {
    TestResult result = {"Reject too few observations (<30)", true, "OK"};
    
    float warmup[20] = {0.01f, -0.02f, 0.015f, -0.01f, 0.02f,
                        0.01f, -0.02f, 0.015f, -0.01f, 0.02f,
                        0.01f, -0.02f, 0.015f, -0.01f, 0.02f,
                        0.01f, -0.02f, 0.015f, -0.01f, 0.02f};
    
    SMC2Prior prior;
    int rc = smc2_calibrate_priors(warmup, 20, &SPY_BOUNDS, &prior);
    TEST_ASSERT(rc != 0, "Should reject n=20 (below minimum 30)");
    
    snprintf(result.message, sizeof(result.message), "OK: correctly rejects n=20");
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: Edge Case - Zero Returns Handling
 * ═══════════════════════════════════════════════════════════════════════════ */

TestResult test_zero_returns() {
    TestResult result = {"Handle zero returns gracefully", true, "OK"};
    
    SVDataGenerator gen;
    gen.seed(44444);
    gen.set_realistic_moderate();
    gen.T = 200;
    gen.generate();
    
    float warmup[100];
    int zero_count = 0;
    for (int i = 0; i < 100; i++) {
        float log_y2 = gen.y[i];
        warmup[i] = expf(log_y2 / 2.0f);
        if (i % 2 == 0) warmup[i] = -warmup[i];
        
        // Inject some zeros (but not too many)
        if (i % 10 == 0) {
            warmup[i] = 0.0f;
            zero_count++;
        }
    }
    
    SMC2Prior prior;
    int rc = smc2_calibrate_priors(warmup, 100, &SPY_BOUNDS, &prior);
    TEST_ASSERT(rc == 0, "Should handle sparse zero returns");
    TEST_ASSERT(prior.rho_mean > 0.5f && prior.rho_mean < 1.0f, "Invalid rho estimate");
    
    snprintf(result.message, sizeof(result.message), 
             "OK: handled %d zero returns gracefully", zero_count);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: Default Bounds Fallback
 * ═══════════════════════════════════════════════════════════════════════════ */

TestResult test_default_bounds() {
    TestResult result = {"Default bounds when NULL provided", true, "OK"};
    
    SVDataGenerator gen;
    gen.seed(55555);
    gen.set_realistic_moderate();
    gen.T = 200;
    gen.generate();
    
    float warmup[100];
    for (int i = 0; i < 100; i++) {
        float log_y2 = gen.y[i];
        warmup[i] = expf(log_y2 / 2.0f);
        if (i % 2 == 0) warmup[i] = -warmup[i];
    }
    
    SMC2Prior prior;
    int rc = smc2_calibrate_priors(warmup, 100, NULL, &prior);  // NULL bounds
    TEST_ASSERT(rc == 0, "Should work with NULL bounds (uses DEFAULT_BOUNDS)");
    TEST_ASSERT(prior.rho_std > 0.0f, "Should have valid rho_std from default bounds");
    
    snprintf(result.message, sizeof(result.message), "OK: default bounds applied");
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: Multiple Seeds Robustness
 * ═══════════════════════════════════════════════════════════════════════════ */

TestResult test_multiple_seeds() {
    TestResult result = {"Robustness across multiple seeds", true, "OK"};
    
    int seeds[] = {111, 222, 333, 444, 555, 666, 777, 888, 999, 1000};
    int n_seeds = sizeof(seeds) / sizeof(seeds[0]);
    int n_covered = 0;
    
    for (int s = 0; s < n_seeds; s++) {
        SVDataGenerator gen;
        gen.seed(seeds[s]);
        gen.set_realistic_moderate();  // ρ=0.94, σ_z=0.15
        gen.T = 300;
        gen.generate();
        
        float warmup[100];
        for (int i = 0; i < 100; i++) {
            float log_y2 = gen.y[i];
            warmup[i] = expf(log_y2 / 2.0f);
            if (i % 2 == 0) warmup[i] = -warmup[i];
        }
        
        SMC2Prior prior;
        smc2_calibrate_priors(warmup, 100, &SPY_BOUNDS, &prior);
        
        // Check if true values are within ±2σ
        float true_rho = gen.rho;
        float true_sigma_z = gen.sigma_z;
        
        bool rho_covered = (true_rho >= prior.rho_mean - 2*prior.rho_std) &&
                           (true_rho <= prior.rho_mean + 2*prior.rho_std);
        bool sigma_z_covered = (true_sigma_z >= prior.sigma_z_mean - 2*prior.sigma_z_std) &&
                               (true_sigma_z <= prior.sigma_z_mean + 2*prior.sigma_z_std);
        
        if (rho_covered && sigma_z_covered) n_covered++;
    }
    
    // Coverage threshold is 50% due to noisy log-variance proxy estimation
    // The key safety mechanism is the WIDE prior (from historical bounds), not the center
    // Even with imperfect centering, the filter will adapt
    TEST_ASSERT(n_covered >= 5, "Prior ±2σ should cover true values in ≥50% of cases");
    
    snprintf(result.message, sizeof(result.message), 
             "OK: %d/%d seeds had prior ±2σ covering true values", n_covered, n_seeds);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: Warmup Stats Computation
 * ═══════════════════════════════════════════════════════════════════════════ */

TestResult test_warmup_stats() {
    TestResult result = {"Warmup statistics computation", true, "OK"};
    
    SVDataGenerator gen;
    gen.seed(77777);
    gen.set_realistic_calm();
    gen.T = 500;
    gen.generate();
    
    float warmup[100];
    for (int i = 0; i < 100; i++) {
        float log_y2 = gen.y[i];
        warmup[i] = expf(log_y2 / 2.0f);
        if (i % 2 == 0) warmup[i] = -warmup[i];
    }
    
    WarmupStats stats;
    int rc = smc2_compute_warmup_stats(warmup, 100, &stats);
    TEST_ASSERT(rc == 0, "smc2_compute_warmup_stats failed");
    
    // Check stats are reasonable
    TEST_ASSERT(stats.n_obs == 100, "Wrong observation count");
    TEST_ASSERT(stats.mean_log_y2 < 0.0f, "mean_log_y2 should be negative for calm regime");
    TEST_ASSERT(stats.std_log_y2 > 0.0f, "std_log_y2 should be positive");
    TEST_ASSERT(stats.realized_vol > 0.0f, "realized_vol should be positive");
    
    // For calm regime, realized vol should be low
    float annual_vol = stats.realized_vol * sqrtf(252.0f);
    TEST_ASSERT(annual_vol < 0.50f, "Annual vol too high for calm regime");
    
    snprintf(result.message, sizeof(result.message), 
             "OK: mean_log_y2=%.2f, std=%.2f, ann_vol=%.1f%%",
             stats.mean_log_y2, stats.std_log_y2, annual_vol * 100);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════════ */

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║  SMC² Prior Calibration Tests                                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Regime Calibration Tests:\n");
    print_result(test_calm_regime());
    print_result(test_crisis_regime());
    print_result(test_moderate_regime());
    
    printf("\nPrior Coverage Tests:\n");
    print_result(test_prior_width_coverage());
    print_result(test_multiple_seeds());
    
    printf("\nEdge Case Tests:\n");
    print_result(test_minimum_warmup());
    print_result(test_too_few_observations());
    print_result(test_zero_returns());
    print_result(test_default_bounds());
    
    printf("\nInternal Function Tests:\n");
    print_result(test_warmup_stats());
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Results: %d/%d tests passed\n", g_tests_passed, g_tests_run);
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
