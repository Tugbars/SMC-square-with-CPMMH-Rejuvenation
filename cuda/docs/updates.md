This is a classic phenomenon in Sequential Monte Carlo (SMC) methods. The root cause is the **Variance of the Log-Likelihood Estimator**.

As  increases, the variance of the likelihood estimate provided by your inner particle filter grows linearly. When this variance exceeds a critical threshold, the PMMH rejuvenation step breaks down, leading to "sticky" chains and parameter degeneration.

### The Problem: Variance Scaling

In a particle filter with  particles, the variance of the estimated log-likelihood  typically scales as:

* **At T=500, N=256:** The variance is manageable. Your PMMH acceptance rate is healthy (~50%).
* **At T=2000, N=256:** The variance is  higher.

### Why High Variance "Breaks" PMMH

PMMH relies on the ratio of two likelihood estimates:

When the variance of these estimates is high, two pathologies occur:

1. **The "Sticky" Particle:** Occasionally, the filter for the *current* parameter  will get "lucky" and produce a massive **over-estimate** of the likelihood (e.g., due to a few particles navigating a difficult observation perfectly).
2. **Rejection trap:** Once  is overestimated, it becomes statistically impossible for any proposal  to beat it. The acceptance probability  drops to near zero.
3. **Result:** The parameters stop moving during the rejuvenation step.
4. **Final Death:** Since the parameters are stuck, the outer SMC² resampling step eventually kills off the diversity. You are left with copies of a single "lucky" parameter that isn't necessarily the ground truth—it's just the one that got the highest variance "spike" in its likelihood estimate.

### Visualizing the Collapse

The diagram above illustrates how the "Estimated Likelihood" (y-axis) widens as "Time T" (x-axis) increases. The green zone represents the healthy acceptance region. As T grows, the noise pushes estimates into the red "Sticky/Reject" zones.

### Solutions

You have three options, ordered from easiest to hardest:

#### 1. The "Brute Force" Fix (Recommended for GPU)

Since the variance scales as , you can counteract the growth of  by increasing .

* **Action:** Increase `N_inner` from 256 to **512** or **1024**.
* **Trade-off:** This increases memory usage and compute time linearly, but it is the only way to maintain the validity of the full-history PMMH on a longer dataset without changing the algorithm.

#### 2. The "Stop Moving" Heuristic

In Bayesian learning, parameter learning typically saturates. After observing 1,000 data points, the posterior  narrows significantly, and new data points shift the mean very little.

* **Action:** Disable the PMMH rejuvenation step after a certain time threshold (e.g., ).
* **Logic:** If you stop rejuvenating, you avoid the high-variance rejection trap. You simply trust the population you built during the first 1000 steps and let the outer filter re-weight them.

#### 3. Fixed-Lag Rejuvenation (Complex Implementation)

Instead of calculating the likelihood for , you only calculate it for a window .

* **Why:** This bounds the variance (it scales with , not ).
* **Difficulty:** Your current implementation overwrites the inner particle state `inner_z` at every step. To run a fixed-lag filter, you would need to store a "snapshot" of the particle states at time  to restart the filter from there. This would require significant memory changes to your `SMC2StateCUDA` struct.

### Summary Recommendation

Since you are running on a GPU with high parallelism capabilities:

1. **Increase `N_inner` to 512.**
2. If that is not enough for , **increase to 1024.**

This will lower the variance constant enough to prevent the "sticky" behavior for the duration of your 2000-step experiment.