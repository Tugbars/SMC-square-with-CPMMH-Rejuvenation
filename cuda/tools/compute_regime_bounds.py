#!/usr/bin/env python3
"""
compute_regime_bounds.py

Compute historical regime bounds for SMC² prior calibration.
Uses SPY data from yfinance covering calm and crisis periods.

Output: C struct with min/max bounds for each parameter.

Usage:
    pip install yfinance pandas numpy
    python compute_regime_bounds.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, Tuple

# ============================================================================
# Regime Period Definitions
# ============================================================================

REGIME_PERIODS = {
    # Crisis periods
    "2008_crisis": ("2008-09-01", "2009-03-31"),  # Lehman collapse
    "2011_eurozone": ("2011-08-01", "2011-10-31"),  # EU debt crisis
    "2015_china": ("2015-08-01", "2015-09-30"),  # China devaluation
    "2018_volmageddon": ("2018-02-01", "2018-02-28"),  # VIX spike
    "2018_q4": ("2018-10-01", "2018-12-31"),  # Fed tightening selloff
    "2020_covid": ("2020-02-15", "2020-04-15"),  # COVID crash
    "2022_hiking": ("2022-01-01", "2022-10-31"),  # Rate hiking cycle
    
    # Calm periods
    "2013_calm": ("2013-01-01", "2013-12-31"),  # Low vol grind
    "2017_calm": ("2017-01-01", "2017-12-31"),  # Extremely low VIX
    "2019_calm": ("2019-04-01", "2019-12-31"),  # Post Q4-2018 recovery
    "2021_bull": ("2021-04-01", "2021-11-30"),  # Low vol bull run
}

# ============================================================================
# Statistics Computation
# ============================================================================

def compute_log_variance_stats(returns: pd.Series) -> Dict[str, float]:
    """
    Compute statistics from log(y²) as proxy for latent volatility.
    
    For SV model: y_t = exp(h_t/2) * ε_t
    So: log(y²) = h_t + log(ε²)
    
    The noise term log(ε²) has variance π²/2 ≈ 4.93, which washes out
    autocorrelation at daily frequency. Need adjusted mapping.
    """
    # Remove zeros/NaN
    returns = returns.dropna()
    returns = returns[returns != 0]
    
    if len(returns) < 20:
        return None
    
    # Log squared returns (proxy for latent vol)
    log_y2 = np.log(returns.values ** 2)
    
    # Basic statistics
    mean_log_y2 = np.mean(log_y2)
    std_log_y2 = np.std(log_y2)
    var_log_y2 = std_log_y2 ** 2
    
    # ACF(1) - autocorrelation at lag 1
    n = len(log_y2)
    centered = log_y2 - mean_log_y2
    acf1_num = np.sum(centered[:-1] * centered[1:])
    acf1_den = np.sum(centered ** 2)
    acf1 = acf1_num / acf1_den if acf1_den > 0 else 0
    
    # Also compute ACF at longer lags for robustness
    acf5_num = np.sum(centered[:-5] * centered[5:])
    acf5 = acf5_num / acf1_den if acf1_den > 0 else 0
    
    # =========================================================================
    # Improved mapping to SV parameters
    # 
    # Theory: Var(log y²) = Var(h) + π²/2, where π²/2 ≈ 4.93
    # And: Var(h) = σ_z² / (1 - ρ²)
    # And: ACF_log_y2(1) ≈ ρ × Var(h) / Var(log y²)
    #
    # For typical SV: ρ ≈ 0.95-0.99, σ_z ≈ 0.05-0.20
    # =========================================================================
    
    NOISE_VAR = np.pi**2 / 2  # ≈ 4.93, variance of log(χ²(1))
    
    # Estimate Var(h) from Var(log y²)
    var_h_est = max(var_log_y2 - NOISE_VAR, 0.1)  # Floor at 0.1
    
    # Realized volatility (annualized)
    realized_vol = np.std(returns) * np.sqrt(252)
    
    # ρ estimation from ACF - but ACF is attenuated by noise
    # ACF_log_y2 ≈ ρ × Var(h) / (Var(h) + NOISE_VAR)
    # So: ρ ≈ ACF_log_y2 × (Var(h) + NOISE_VAR) / Var(h)
    if var_h_est > 0.1:
        rho_factor = (var_h_est + NOISE_VAR) / var_h_est
        est_rho_raw = acf1 * rho_factor
    else:
        est_rho_raw = 0.95  # Default to high persistence if uncertain
    
    # Use realized vol as additional signal
    # High realized vol → likely lower persistence (crisis)
    # Low realized vol → likely higher persistence (calm)
    if realized_vol > 0.40:  # Crisis
        rho_adjustment = -0.10
    elif realized_vol > 0.25:
        rho_adjustment = -0.05
    elif realized_vol < 0.10:  # Very calm
        rho_adjustment = 0.03
    else:
        rho_adjustment = 0.0
    
    est_rho = np.clip(est_rho_raw + 0.95 + rho_adjustment, 0.70, 0.99)
    
    # σ_z estimation
    # Var(h) = σ_z² / (1 - ρ²)
    # So: σ_z = sqrt(Var(h) × (1 - ρ²))
    one_minus_rho_sq = 1 - est_rho**2
    est_sigma_z = np.sqrt(var_h_est * one_minus_rho_sq)
    est_sigma_z = np.clip(est_sigma_z, 0.02, 0.50)
    
    # Adjust σ_z based on realized vol
    if realized_vol > 0.40:
        est_sigma_z = max(est_sigma_z, 0.20)
    elif realized_vol > 0.25:
        est_sigma_z = max(est_sigma_z, 0.12)
    
    # μ_base is roughly mean(log y²) adjusted for E[log ε²] ≈ -1.27
    est_mu_base = mean_log_y2 + 1.27
    
    return {
        "mean_log_y2": mean_log_y2,
        "std_log_y2": std_log_y2,
        "var_log_y2": var_log_y2,
        "var_h_est": var_h_est,
        "acf1_log_y2": acf1,
        "acf5_log_y2": acf5,
        "est_rho": est_rho,
        "est_sigma_z": est_sigma_z,
        "est_mu_base": est_mu_base,
        "realized_vol_ann": realized_vol,
        "n_obs": len(returns),
    }


def fetch_spy_data(start: str = "2007-01-01", end: str = "2024-12-31") -> pd.Series:
    """Fetch SPY daily returns from yfinance."""
    print(f"Fetching SPY data from {start} to {end}...")
    
    spy = yf.download("SPY", start=start, end=end, progress=False)
    
    if spy.empty:
        raise ValueError("Failed to fetch SPY data")
    
    # Handle both single and multi-level columns (yfinance API varies)
    if isinstance(spy.columns, pd.MultiIndex):
        # New yfinance format: ('Close', 'SPY')
        if ('Close', 'SPY') in spy.columns:
            close = spy[('Close', 'SPY')]
        elif ('Adj Close', 'SPY') in spy.columns:
            close = spy[('Adj Close', 'SPY')]
        else:
            # Fallback: first column with 'Close'
            close_cols = [c for c in spy.columns if 'Close' in c[0]]
            close = spy[close_cols[0]]
    else:
        # Old format: single level
        close = spy['Adj Close'] if 'Adj Close' in spy.columns else spy['Close']
    
    returns = close.pct_change().dropna()
    print(f"  Retrieved {len(returns)} daily returns")
    
    return returns


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_regimes(returns: pd.Series) -> Tuple[Dict, Dict]:
    """Analyze each regime period and compute statistics."""
    
    results = {}
    
    print("\n" + "=" * 70)
    print("REGIME ANALYSIS")
    print("=" * 70)
    
    for name, (start, end) in REGIME_PERIODS.items():
        # Filter to period
        mask = (returns.index >= start) & (returns.index <= end)
        period_returns = returns[mask]
        
        if len(period_returns) < 20:
            print(f"\n{name}: Insufficient data ({len(period_returns)} obs)")
            continue
        
        stats = compute_log_variance_stats(period_returns)
        if stats is None:
            continue
            
        results[name] = stats
        
        # Classify as crisis or calm
        is_crisis = "crisis" in name or "covid" in name or "volmageddon" in name or "hiking" in name or "eurozone" in name or "china" in name or "q4" in name
        regime_type = "CRISIS" if is_crisis else "CALM"
        
        print(f"\n{name} [{regime_type}] ({start} to {end})")
        print(f"  Observations: {stats['n_obs']}")
        print(f"  Realized Vol (ann): {stats['realized_vol_ann']:.1%}")
        print(f"  log(y²): mean={stats['mean_log_y2']:.2f}, std={stats['std_log_y2']:.2f}, ACF1={stats['acf1_log_y2']:.3f}")
        print(f"  → est_rho={stats['est_rho']:.3f}, est_sigma_z={stats['est_sigma_z']:.3f}")
    
    # Separate into crisis and calm
    crisis_results = {k: v for k, v in results.items() 
                      if "crisis" in k or "covid" in k or "volmageddon" in k or "hiking" in k or "eurozone" in k or "china" in k or "q4" in k}
    calm_results = {k: v for k, v in results.items() if k not in crisis_results}
    
    return crisis_results, calm_results


def compute_bounds(crisis_results: Dict, calm_results: Dict) -> Dict[str, Tuple[float, float]]:
    """Compute min/max bounds from regime analysis."""
    
    all_results = {**crisis_results, **calm_results}
    
    if not all_results:
        raise ValueError("No regime results to compute bounds")
    
    # Extract arrays
    rhos = [r["est_rho"] for r in all_results.values()]
    sigma_zs = [r["est_sigma_z"] for r in all_results.values()]
    mu_bases = [r["est_mu_base"] for r in all_results.values()]
    
    # Compute bounds with some margin
    bounds = {
        "rho": (min(rhos) * 0.95, min(max(rhos) * 1.02, 0.99)),
        "sigma_z": (min(sigma_zs) * 0.7, max(sigma_zs) * 1.3),
        "mu_base": (min(mu_bases) - 0.5, max(mu_bases) + 0.5),
    }
    
    return bounds


def print_cpp_struct(bounds: Dict, crisis_results: Dict, calm_results: Dict):
    """Print C struct for use in SMC² code."""
    
    print("\n" + "=" * 70)
    print("C STRUCT OUTPUT")
    print("=" * 70)
    
    # Get typical calm and crisis values
    calm_rhos = [r["est_rho"] for r in calm_results.values()]
    calm_sigma_zs = [r["est_sigma_z"] for r in calm_results.values()]
    crisis_rhos = [r["est_rho"] for r in crisis_results.values()]
    crisis_sigma_zs = [r["est_sigma_z"] for r in crisis_results.values()]
    
    print("""
/*
 * Historical regime bounds for SPY
 * Generated from yfinance data (2007-2024)
 * 
 * Use these to set wide priors that cover both calm and crisis regimes.
 */
typedef struct {
    /* Persistence parameter rho */
    float rho_min;          /* Crisis: faster mean reversion */
    float rho_max;          /* Calm: high persistence */
    float rho_calm;         /* Typical calm value */
    float rho_crisis;       /* Typical crisis value */
    
    /* Vol-of-vol parameter sigma_z */
    float sigma_z_min;      /* Calm: low vol-of-vol */
    float sigma_z_max;      /* Crisis: high vol-of-vol */
    float sigma_z_calm;     /* Typical calm value */
    float sigma_z_crisis;   /* Typical crisis value */
    
    /* Log-variance level mu_base */
    float mu_base_min;
    float mu_base_max;
} HistoricalRegimeBounds;
""")
    
    print(f"""static const HistoricalRegimeBounds SPY_BOUNDS = {{
    /* rho */
    .rho_min = {bounds['rho'][0]:.3f}f,
    .rho_max = {bounds['rho'][1]:.3f}f,
    .rho_calm = {np.mean(calm_rhos):.3f}f,
    .rho_crisis = {np.mean(crisis_rhos):.3f}f,
    
    /* sigma_z */
    .sigma_z_min = {bounds['sigma_z'][0]:.3f}f,
    .sigma_z_max = {bounds['sigma_z'][1]:.3f}f,
    .sigma_z_calm = {np.mean(calm_sigma_zs):.3f}f,
    .sigma_z_crisis = {np.mean(crisis_sigma_zs):.3f}f,
    
    /* mu_base */
    .mu_base_min = {bounds['mu_base'][0]:.2f}f,
    .mu_base_max = {bounds['mu_base'][1]:.2f}f,
}};""")
    
    print("\n/* Recommended prior widths (covers range at ~2σ) */")
    rho_range = bounds['rho'][1] - bounds['rho'][0]
    sigma_z_range = bounds['sigma_z'][1] - bounds['sigma_z'][0]
    mu_base_range = bounds['mu_base'][1] - bounds['mu_base'][0]
    
    print(f"#define PRIOR_RHO_STD      {rho_range/4:.3f}f  /* (max-min)/4 */")
    print(f"#define PRIOR_SIGMA_Z_STD  {sigma_z_range/4:.3f}f")
    print(f"#define PRIOR_MU_BASE_STD  {mu_base_range/4:.2f}f")


def print_summary(bounds: Dict, crisis_results: Dict, calm_results: Dict):
    """Print summary statistics."""
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nAnalyzed {len(crisis_results)} crisis periods, {len(calm_results)} calm periods")
    
    print(f"\nParameter Bounds:")
    print(f"  rho:      [{bounds['rho'][0]:.3f}, {bounds['rho'][1]:.3f}]")
    print(f"  sigma_z:  [{bounds['sigma_z'][0]:.3f}, {bounds['sigma_z'][1]:.3f}]")
    print(f"  mu_base:  [{bounds['mu_base'][0]:.2f}, {bounds['mu_base'][1]:.2f}]")
    
    # Widths needed to cover range at 2σ
    print(f"\nRecommended prior std (range/4):")
    print(f"  rho_std:      {(bounds['rho'][1] - bounds['rho'][0])/4:.3f}")
    print(f"  sigma_z_std:  {(bounds['sigma_z'][1] - bounds['sigma_z'][0])/4:.3f}")
    print(f"  mu_base_std:  {(bounds['mu_base'][1] - bounds['mu_base'][0])/4:.2f}")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    print("=" * 70)
    print("SMC² Prior Calibration - Historical Regime Bounds")
    print("=" * 70)
    
    # Fetch data
    returns = fetch_spy_data()
    
    # Analyze regimes
    crisis_results, calm_results = analyze_regimes(returns)
    
    # Compute bounds
    bounds = compute_bounds(crisis_results, calm_results)
    
    # Print summary
    print_summary(bounds, crisis_results, calm_results)
    
    # Print C struct
    print_cpp_struct(bounds, crisis_results, calm_results)
    
    print("\n" + "=" * 70)
    print("Done. Copy the struct above into smc2_prior_calibration.cuh")
    print("=" * 70)


if __name__ == "__main__":
    main()
