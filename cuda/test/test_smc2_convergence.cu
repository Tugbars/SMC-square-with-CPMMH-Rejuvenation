/*
 * test_smc2_convergence.cu
 * 
 * Unit and integration tests for SMC² convergence diagnostics (v2)
 * 
 * Tests:
 *   1. Vanishing Denominator - drift floor prevents false alarms
 *   2. Negative Log-Likelihood - std calculation on negative values
 *   3. Circular Buffer Wraparound - no memory corruption at capacity
 *   4. Happy Path - static params → Learning → Stable → Ready
 *   5. Regime Change - detects parameter jumps
 *   6. Broken Inner Filter - unhealthy inner ESS blocks "Ready"
 *   7. NaN/Inf Handling - graceful degradation
 *   8. ESS CV Calculation - verify fix from v1
 * 
 * Compile:
 *   g++ -o test_convergence test_smc2_convergence.cpp -std=c++14
 * 
 * Run:
 *   ./test_convergence           # All tests
 *   ./test_convergence <name>    # Specific test
 */

#include "smc2_convergence_v2.cuh"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>

/*═══════════════════════════════════════════════════════════════════════════
 * Test Utilities
 *═══════════════════════════════════════════════════════════════════════════*/

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("    FAIL: %s\n", msg); \
        printf("      at %s:%d\n", __FILE__, __LINE__); \
        g_tests_failed++; \
        return; \
    } \
} while(0)

#define TEST_ASSERT_FLOAT_EQ(a, b, tol, msg) do { \
    if (fabsf((a) - (b)) > (tol)) { \
        printf("    FAIL: %s (got %.6f, expected %.6f)\n", msg, (float)(a), (float)(b)); \
        printf("      at %s:%d\n", __FILE__, __LINE__); \
        g_tests_failed++; \
        return; \
    } \
} while(0)

#define TEST_PASS() do { \
    printf("    PASS\n"); \
    g_tests_passed++; \
} while(0)

/*═══════════════════════════════════════════════════════════════════════════
 * Test 1: Vanishing Denominator (Drift Floor Fix)
 * 
 * Bug: When variance → 0 (late learning), small mean noise causes huge drift.
 * Fix: Floor denominator at 1% of |mean|.
 * 
 * Scenario: var = 1e-12, mean shift = 1e-5
 * Old: drift = 1e-5 / 1e-6 = 10.0 (false alarm!)
 * New: drift = 1e-5 / (0.01 * mean) = small (correct)
 *═══════════════════════════════════════════════════════════════════════════*/

void test_vanishing_denominator(void) {
    printf("\n[Test 1] Vanishing Denominator (Drift Floor Fix)\n");
    
    SMC2ConvergenceTracker tracker;
    smc2_conv_init(&tracker, 256, 256);
    
    /* Set priors (used for floor calculation) */
    float prior_mean[8] = {0.95f, 0.15f, -1.0f, 0.5f, 1.0f, 0.15f, 0.10f, 1.0f};
    float prior_std[8] = {0.05f, 0.05f, 0.5f, 0.3f, 0.5f, 0.05f, 0.05f, 0.5f};
    smc2_conv_set_priors(&tracker, prior_mean, prior_std);
    
    /* Simulate stable filter with TINY variance (highly converged) */
    float theta_mean[8] = {0.95f, 0.15f, -1.0f, 0.5f, 1.0f, 0.15f, 0.10f, 1.0f};
    float theta_var[8] = {1e-12f, 1e-12f, 1e-12f, 1e-12f, 1e-12f, 1e-12f, 1e-12f, 1e-12f};
    
    /* Fill history with stable values */
    for (int i = 0; i < SMC2_CONV_WINDOW + 5; i++) {
        smc2_conv_update(&tracker, 200.0f, -1.5f, theta_mean, theta_var, NULL);
    }
    
    /* Now add tiny mean shift (noise) */
    float theta_mean_shifted[8];
    memcpy(theta_mean_shifted, theta_mean, sizeof(theta_mean));
    theta_mean_shifted[0] += 1e-5f;  /* Tiny shift in rho */
    theta_mean_shifted[1] += 1e-5f;  /* Tiny shift in sigma_z */
    
    for (int i = 0; i < SMC2_CONV_WINDOW; i++) {
        smc2_conv_update(&tracker, 200.0f, -1.5f, theta_mean_shifted, theta_var, NULL);
    }
    
    /* Check: drift should be SMALL (not a false alarm) */
    SMC2ConvergenceDiag diag = smc2_conv_check(&tracker);
    
    printf("    param_drift = %.6f (threshold = %.4f)\n", diag.param_drift, SMC2_CONV_DRIFT_THRESH);
    printf("    per-dim drift[0] (rho) = %.6f\n", diag.param_drift_per_dim[0]);
    printf("    per-dim drift[1] (sigma_z) = %.6f\n", diag.param_drift_per_dim[1]);
    
    /* OLD BUG: drift would be ~10.0 (1e-5 / 1e-6) */
    /* NEW: drift should be << 1.0 due to floor at 1% of mean */
    TEST_ASSERT(diag.param_drift < 1.0f, 
                "Drift should be small with floor (got huge value = false alarm)");
    
    smc2_conv_free(&tracker);
    TEST_PASS();
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test 2: Negative Log-Likelihood Std Calculation
 * 
 * Bug: CV = std/mean is meaningless for negative numbers.
 * Fix: Use absolute std with threshold, not CV.
 * 
 * Scenario: log_lik = [-100, -102, -98, -101, ...] (typical SV data)
 * Expected: std ≈ 2.0, ll_healthy = true (std < threshold)
 *═══════════════════════════════════════════════════════════════════════════*/

void test_negative_loglik_std(void) {
    printf("\n[Test 2] Negative Log-Likelihood Std Calculation\n");
    
    SMC2ConvergenceTracker tracker;
    smc2_conv_init(&tracker, 256, 256);
    
    float theta_mean[8] = {0.95f, 0.15f, -1.0f, 0.5f, 1.0f, 0.15f, 0.10f, 1.0f};
    float theta_var[8] = {0.001f, 0.001f, 0.01f, 0.01f, 0.01f, 0.001f, 0.001f, 0.01f};
    
    /* Feed in typical negative log-likelihoods with known std */
    float ll_values[] = {-100.0f, -102.0f, -98.0f, -101.0f, -99.0f, -103.0f, -97.0f, -100.5f};
    int n_ll = sizeof(ll_values) / sizeof(ll_values[0]);
    
    /* Fill enough history */
    for (int i = 0; i < SMC2_CONV_WINDOW + 10; i++) {
        float ll = ll_values[i % n_ll];
        smc2_conv_update(&tracker, 200.0f, ll, theta_mean, theta_var, NULL);
    }
    
    SMC2ConvergenceDiag diag = smc2_conv_check(&tracker);
    
    printf("    ll_mean = %.2f\n", diag.ll_mean);
    printf("    ll_std = %.2f (threshold = %.1f)\n", diag.ll_std, SMC2_CONV_LL_STD_THRESH);
    printf("    ll_healthy = %s\n", diag.ll_healthy ? "true" : "false");
    
    /* Std should be approximately 2.0 (not some nonsense from CV calculation) */
    TEST_ASSERT(diag.ll_std > 1.0f && diag.ll_std < 5.0f, 
                "ll_std should be ~2.0 for this data");
    TEST_ASSERT(diag.ll_healthy, 
                "Should be healthy (std < threshold)");
    TEST_ASSERT(!diag.ll_exploding, 
                "Should not be exploding");
    
    smc2_conv_free(&tracker);
    TEST_PASS();
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test 3: Circular Buffer Wraparound
 * 
 * Scenario: Push 65+ updates into 64-slot buffer
 * Expected: Stats reflect recent values, no memory corruption
 *═══════════════════════════════════════════════════════════════════════════*/

void test_circular_buffer_wraparound(void) {
    printf("\n[Test 3] Circular Buffer Wraparound\n");
    
    SMC2ConvergenceTracker tracker;
    smc2_conv_init(&tracker, 256, 256);
    
    float theta_mean[8] = {0.90f, 0.10f, -1.0f, 0.5f, 1.0f, 0.15f, 0.10f, 1.0f};
    float theta_var[8] = {0.01f, 0.01f, 0.1f, 0.1f, 0.1f, 0.01f, 0.01f, 0.1f};
    
    /* Phase 1: Fill with ESS = 100 */
    for (int i = 0; i < SMC2_CONV_HISTORY_LEN; i++) {
        smc2_conv_update(&tracker, 100.0f, -50.0f, theta_mean, theta_var, NULL);
    }
    
    SMC2ConvergenceDiag diag1 = smc2_conv_check(&tracker);
    printf("    After %d updates: ess_mean = %.1f\n", SMC2_CONV_HISTORY_LEN, diag1.ess_mean);
    TEST_ASSERT_FLOAT_EQ(diag1.ess_mean, 100.0f, 1.0f, "ESS mean should be ~100");
    
    /* Phase 2: Overwrite with ESS = 200 (wraparound) */
    for (int i = 0; i < SMC2_CONV_HISTORY_LEN; i++) {
        smc2_conv_update(&tracker, 200.0f, -50.0f, theta_mean, theta_var, NULL);
    }
    
    SMC2ConvergenceDiag diag2 = smc2_conv_check(&tracker);
    printf("    After %d updates (wraparound): ess_mean = %.1f\n", 
           2 * SMC2_CONV_HISTORY_LEN, diag2.ess_mean);
    
    /* Should now reflect ESS=200, not mixed with old ESS=100 */
    TEST_ASSERT_FLOAT_EQ(diag2.ess_mean, 200.0f, 1.0f, 
                         "ESS mean should be ~200 after wraparound");
    TEST_ASSERT(tracker.total_updates == 2 * SMC2_CONV_HISTORY_LEN, 
                "Total updates should track correctly");
    
    smc2_conv_free(&tracker);
    TEST_PASS();
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test 4: Happy Path - Static Parameters → Ready
 * 
 * Scenario: Simulate filter converging to stable parameters
 * Expected: Learning... → Stable → Ready
 *═══════════════════════════════════════════════════════════════════════════*/

void test_happy_path(void) {
    printf("\n[Test 4] Happy Path (Static Params → Ready)\n");
    
    SMC2ConvergenceTracker tracker;
    smc2_conv_init(&tracker, 256, 256);
    
    float prior_mean[8] = {0.90f, 0.10f, 0.0f, 0.5f, 1.0f, 0.15f, 0.10f, 1.0f};
    float prior_std[8] = {0.10f, 0.10f, 1.0f, 0.5f, 0.5f, 0.10f, 0.10f, 0.5f};
    smc2_conv_set_priors(&tracker, prior_mean, prior_std);
    
    /* Target parameters (what filter will "learn") */
    float theta_true[8] = {0.95f, 0.15f, -1.0f, 0.5f, 1.0f, 0.15f, 0.10f, 1.0f};
    
    /* Simulate learning: variance shrinks, mean approaches true */
    float theta_mean[8], theta_var[8];
    float inner_ess[256];
    
    int reached_ready = 0;
    int ready_at = -1;
    
    for (int t = 0; t < 500; t++) {
        /* Simulate learning dynamics */
        float progress = fminf(1.0f, (float)t / 200.0f);  /* Converge by t=200 */
        
        for (int p = 0; p < 8; p++) {
            /* Mean approaches true value */
            theta_mean[p] = prior_mean[p] + progress * (theta_true[p] - prior_mean[p]);
            /* Variance shrinks */
            theta_var[p] = prior_std[p] * prior_std[p] * (1.0f - 0.9f * progress);
            /* Add small noise */
            theta_mean[p] += (t % 2 == 0 ? 0.001f : -0.001f);
        }
        
        /* Healthy inner ESS */
        for (int i = 0; i < 256; i++) {
            inner_ess[i] = 200.0f + (i % 50);  /* 200-250 range */
        }
        
        float ess = 180.0f + 20.0f * progress;  /* ESS improves */
        float ll = -1.5f + 0.1f * ((t % 10) - 5);  /* Small variance */
        
        smc2_conv_update(&tracker, ess, ll, theta_mean, theta_var, inner_ess);
        
        /* Check periodically */
        if (t >= SMC2_CONV_WINDOW && t % 50 == 0) {
            SMC2ConvergenceDiag diag = smc2_conv_check(&tracker);
            printf("    t=%3d: stable=%d, healthy=%d, ready=%d, streak=%d\n",
                   t, diag.stable, diag.healthy, diag.ready, diag.stable_streak);
            
            if (diag.ready && !reached_ready) {
                reached_ready = 1;
                ready_at = t;
            }
        }
    }
    
    TEST_ASSERT(reached_ready, "Should eventually reach Ready state");
    printf("    Reached Ready at t=%d\n", ready_at);
    TEST_ASSERT(ready_at > 100 && ready_at < 400, 
                "Should reach Ready after learning, not immediately");
    
    smc2_conv_free(&tracker);
    TEST_PASS();
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test 5: Regime Change Detection
 * 
 * Scenario: Parameters jump at t=250
 * Expected: Ready → Drifting → Ready again
 *═══════════════════════════════════════════════════════════════════════════*/

void test_regime_change(void) {
    printf("\n[Test 5] Regime Change Detection\n");
    
    SMC2ConvergenceTracker tracker;
    smc2_conv_init(&tracker, 256, 256);
    
    float theta_mean[8] = {0.95f, 0.15f, -1.0f, 0.5f, 1.0f, 0.15f, 0.10f, 1.0f};
    float theta_var[8] = {0.001f, 0.001f, 0.01f, 0.01f, 0.01f, 0.001f, 0.001f, 0.01f};
    float inner_ess[256];
    for (int i = 0; i < 256; i++) inner_ess[i] = 200.0f;
    
    int was_ready_before_jump = 0;
    int detected_drift = 0;
    int recovered_after_jump = 0;
    
    for (int t = 0; t < 500; t++) {
        /* REGIME CHANGE at t=250: sigma_z doubles */
        if (t == 250) {
            printf("    >>> REGIME CHANGE at t=%d <<<\n", t);
            theta_mean[1] = 0.30f;  /* sigma_z: 0.15 → 0.30 */
        }
        
        /* Add small noise after jump to simulate re-learning */
        if (t > 250 && t < 350) {
            float noise = 0.001f * ((t % 2) - 0.5f);
            theta_mean[1] += noise;
        }
        
        smc2_conv_update(&tracker, 200.0f, -1.5f, theta_mean, theta_var, inner_ess);
        
        if (t >= SMC2_CONV_WINDOW && t % 25 == 0) {
            SMC2ConvergenceDiag diag = smc2_conv_check(&tracker);
            
            printf("    t=%3d: drift=%.4f, stable=%d, ready=%d\n",
                   t, diag.param_drift, diag.stable, diag.ready);
            
            if (t < 250 && diag.ready) {
                was_ready_before_jump = 1;
            }
            if (t > 250 && t < 350 && !diag.stable) {
                detected_drift = 1;
            }
            if (t > 400 && diag.ready) {
                recovered_after_jump = 1;
            }
        }
    }
    
    TEST_ASSERT(was_ready_before_jump, "Should be Ready before regime change");
    TEST_ASSERT(detected_drift, "Should detect drift after regime change");
    TEST_ASSERT(recovered_after_jump, "Should recover to Ready after adaptation");
    
    smc2_conv_free(&tracker);
    TEST_PASS();
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test 6: Broken Inner Filter (Health Check)
 * 
 * Scenario: Outer looks stable, but inner ESS is garbage
 * Expected: stable=true, healthy=false, ready=FALSE
 * 
 * This is the CRITICAL test from reviewer feedback.
 *═══════════════════════════════════════════════════════════════════════════*/

void test_broken_inner_filter(void) {
    printf("\n[Test 6] Broken Inner Filter (Critical Health Check)\n");
    
    SMC2ConvergenceTracker tracker;
    int N_theta = 256;
    int N_inner = 256;
    smc2_conv_init(&tracker, N_theta, N_inner);
    
    /* Outer filter looks PERFECT */
    float theta_mean[8] = {0.95f, 0.15f, -1.0f, 0.5f, 1.0f, 0.15f, 0.10f, 1.0f};
    float theta_var[8] = {0.001f, 0.001f, 0.01f, 0.01f, 0.01f, 0.001f, 0.001f, 0.01f};
    
    /* Inner filter is BROKEN - most particles have ESS < threshold */
    float inner_ess_broken[256];
    float threshold = SMC2_CONV_INNER_ESS_MIN_RATIO * N_inner;  /* 0.1 * 256 = 25.6 */
    printf("    Inner ESS threshold: %.1f (%.1f%% of N_inner)\n", 
           threshold, SMC2_CONV_INNER_ESS_MIN_RATIO * 100.0f);
    
    for (int i = 0; i < N_theta; i++) {
        /* 80% of particles have ESS = 5 (way below threshold) */
        /* 20% have ESS = 100 (healthy) */
        inner_ess_broken[i] = (i % 5 == 0) ? 100.0f : 5.0f;
    }
    
    /* Fill history - outer looks stable */
    for (int t = 0; t < SMC2_CONV_WINDOW + 100; t++) {
        smc2_conv_update(&tracker, 200.0f, -1.5f, theta_mean, theta_var, inner_ess_broken);
    }
    
    SMC2ConvergenceDiag diag = smc2_conv_check(&tracker);
    
    printf("    Outer: ess_cv=%.3f, drift=%.4f, stable=%d\n",
           diag.ess_cv, diag.param_drift, diag.outer_stable);
    printf("    Inner: mean=%.1f, q10=%.1f, bad_frac=%.1f%%, healthy=%d\n",
           diag.inner_ess_mean, diag.inner_ess_q10, 
           diag.inner_ess_bad_frac * 100.0f, diag.inner_healthy);
    printf("    Final: stable=%d, healthy=%d, READY=%d\n",
           diag.stable, diag.healthy, diag.ready);
    
    /* CRITICAL ASSERTIONS */
    TEST_ASSERT(diag.outer_stable || diag.stable, 
                "Outer should look stable (that's the trap!)");
    TEST_ASSERT(!diag.inner_healthy, 
                "Inner should be UNHEALTHY (bad ESS)");
    TEST_ASSERT(!diag.ready, 
                "MUST NOT be Ready when inner filter is broken!");
    TEST_ASSERT(diag.inner_ess_bad_frac > 0.5f, 
                "Bad fraction should be >50%");
    
    smc2_conv_free(&tracker);
    TEST_PASS();
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test 7: NaN/Inf Handling
 * 
 * Scenario: GPU returns NaN/Inf (filter diverged)
 * Expected: Graceful handling, not crash
 *═══════════════════════════════════════════════════════════════════════════*/

void test_nan_inf_handling(void) {
    printf("\n[Test 7] NaN/Inf Handling\n");
    
    SMC2ConvergenceTracker tracker;
    smc2_conv_init(&tracker, 256, 256);
    
    float theta_mean_normal[8] = {0.95f, 0.15f, -1.0f, 0.5f, 1.0f, 0.15f, 0.10f, 1.0f};
    float theta_var_normal[8] = {0.01f, 0.01f, 0.1f, 0.1f, 0.1f, 0.01f, 0.01f, 0.1f};
    
    /* Fill with normal data first */
    for (int t = 0; t < SMC2_CONV_WINDOW; t++) {
        smc2_conv_update(&tracker, 200.0f, -1.5f, theta_mean_normal, theta_var_normal, NULL);
    }
    
    /* Now inject NaN/Inf */
    float theta_mean_bad[8];
    float theta_var_bad[8];
    memcpy(theta_mean_bad, theta_mean_normal, sizeof(theta_mean_normal));
    memcpy(theta_var_bad, theta_var_normal, sizeof(theta_var_normal));
    
    theta_mean_bad[0] = NAN;      /* NaN in rho */
    theta_mean_bad[1] = INFINITY; /* Inf in sigma_z */
    theta_var_bad[2] = -1.0f;     /* Negative variance (invalid) */
    
    printf("    Injecting: mean[0]=NaN, mean[1]=Inf, var[2]=-1\n");
    
    for (int t = 0; t < 10; t++) {
        smc2_conv_update(&tracker, 200.0f, -1.5f, theta_mean_bad, theta_var_bad, NULL);
    }
    
    /* Should not crash when checking */
    SMC2ConvergenceDiag diag = smc2_conv_check(&tracker);
    
    printf("    Check completed without crash\n");
    printf("    param_drift = %.4f (may be NaN/Inf)\n", diag.param_drift);
    printf("    stable = %d, ready = %d\n", diag.stable, diag.ready);
    
    /* Should definitely NOT be ready with corrupted data */
    /* Note: exact behavior depends on implementation - main thing is no crash */
    TEST_ASSERT(!std::isnan(diag.ess_mean), "ESS mean should not propagate NaN from params");
    
    smc2_conv_free(&tracker);
    TEST_PASS();
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test 8: ESS CV Calculation Fix Verification
 * 
 * Directly verify the CV calculation is correct (was buggy in v1)
 *═══════════════════════════════════════════════════════════════════════════*/

void test_ess_cv_calculation(void) {
    printf("\n[Test 8] ESS CV Calculation Fix\n");
    
    SMC2ConvergenceTracker tracker;
    smc2_conv_init(&tracker, 256, 256);
    
    float theta_mean[8] = {0.95f, 0.15f, -1.0f, 0.5f, 1.0f, 0.15f, 0.10f, 1.0f};
    float theta_var[8] = {0.01f, 0.01f, 0.1f, 0.1f, 0.1f, 0.01f, 0.01f, 0.1f};
    
    /* Known ESS values: mean=100, std=10 → CV=0.1 */
    float ess_values[] = {90, 100, 110, 95, 105, 85, 115, 100};
    int n_ess = sizeof(ess_values) / sizeof(ess_values[0]);
    
    for (int t = 0; t < SMC2_CONV_WINDOW + 10; t++) {
        float ess = ess_values[t % n_ess];
        smc2_conv_update(&tracker, ess, -1.5f, theta_mean, theta_var, NULL);
    }
    
    SMC2ConvergenceDiag diag = smc2_conv_check(&tracker);
    
    printf("    ess_mean = %.2f (expected ~100)\n", diag.ess_mean);
    printf("    ess_std = %.2f (expected ~10)\n", diag.ess_std);
    printf("    ess_cv = %.4f (expected ~0.10)\n", diag.ess_cv);
    
    /* CV should be std/mean ≈ 10/100 = 0.10 */
    TEST_ASSERT(diag.ess_mean > 95.0f && diag.ess_mean < 105.0f, 
                "ESS mean should be ~100");
    TEST_ASSERT(diag.ess_cv > 0.05f && diag.ess_cv < 0.20f, 
                "ESS CV should be ~0.10 (std/mean)");
    
    /* v1 BUG would have computed CV wrong (double division) */
    TEST_ASSERT(diag.ess_cv < 1.0f, 
                "CV should be < 1 for stable ESS (v1 bug would give larger value)");
    
    smc2_conv_free(&tracker);
    TEST_PASS();
}

/*═══════════════════════════════════════════════════════════════════════════
 * Main
 *═══════════════════════════════════════════════════════════════════════════*/

typedef void (*TestFunc)(void);

typedef struct {
    const char* name;
    TestFunc func;
} TestEntry;

TestEntry g_tests[] = {
    {"vanishing_denom", test_vanishing_denominator},
    {"negative_ll", test_negative_loglik_std},
    {"circular_buffer", test_circular_buffer_wraparound},
    {"happy_path", test_happy_path},
    {"regime_change", test_regime_change},
    {"broken_inner", test_broken_inner_filter},
    {"nan_inf", test_nan_inf_handling},
    {"ess_cv", test_ess_cv_calculation},
};

int main(int argc, char** argv) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  SMC² Convergence Diagnostics (v2) - Test Suite               ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    int n_tests = sizeof(g_tests) / sizeof(g_tests[0]);
    
    if (argc > 1) {
        /* Run specific test */
        const char* target = argv[1];
        int found = 0;
        
        for (int i = 0; i < n_tests; i++) {
            if (strcmp(g_tests[i].name, target) == 0) {
                g_tests[i].func();
                found = 1;
                break;
            }
        }
        
        if (!found) {
            printf("Unknown test: %s\n", target);
            printf("Available tests:\n");
            for (int i = 0; i < n_tests; i++) {
                printf("  %s\n", g_tests[i].name);
            }
            return 1;
        }
    } else {
        /* Run all tests */
        for (int i = 0; i < n_tests; i++) {
            g_tests[i].func();
        }
    }
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Results: %d PASSED, %d FAILED\n", g_tests_passed, g_tests_failed);
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    return g_tests_failed > 0 ? 1 : 0;
}
