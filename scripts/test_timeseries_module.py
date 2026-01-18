"""
Test the timeseries_tests module with actual data.

This script validates that our statistical testing functionality works correctly
with real return data, specifically testing the researcher-focused concerns about
stationarity, autocorrelation, and heteroscedasticity.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gcapm
import pandas as pd

def main():
    print("=" * 80)
    print("Testing Time Series Diagnostics Module")
    print("=" * 80)
    print()

    # Load daily returns data
    print("Loading returns_daily dataset...")
    df = gcapm.load_data('returns_daily')
    print(f"Loaded {len(df)} observations from {df['date'].min()} to {df['date'].max()}")
    print()

    # Test 1: Stationarity test on Bitcoin returns
    print("-" * 80)
    print("Test 1: Stationarity of Bitcoin Returns (ADF Test)")
    print("-" * 80)

    btc_returns = df['return_btc'].dropna()
    result = gcapm.test_stationarity(btc_returns, test='adf', name='Bitcoin Returns')

    print(f"Test Statistic: {result.statistic:.4f}")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Is Stationary: {result.is_stationary}")
    print(f"Interpretation: {result.interpretation}")
    print()

    # Test 2: Stationarity test on S&P 500 returns
    print("-" * 80)
    print("Test 2: Stationarity of S&P 500 Returns (ADF Test)")
    print("-" * 80)

    sp500_returns = df['return_sp500'].dropna()
    result = gcapm.test_stationarity(sp500_returns, test='adf', name='S&P 500 Returns')

    print(f"Test Statistic: {result.statistic:.4f}")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Is Stationary: {result.is_stationary}")
    print(f"Interpretation: {result.interpretation}")
    print()

    # Test 3: Comprehensive diagnostics for BTC vs S&P 500 regression
    print("-" * 80)
    print("Test 3: Full Regression Diagnostics (BTC ~ S&P 500)")
    print("-" * 80)
    print("This checks ALL assumptions before running a regression:")
    print("  - Stationarity of both series")
    print("  - Autocorrelation in residuals")
    print("  - Heteroscedasticity in residuals")
    print()

    # Align the series
    valid_mask = btc_returns.notna() & sp500_returns.notna()
    y = btc_returns[valid_mask]
    X = sp500_returns[valid_mask]

    diagnostics = gcapm.check_all_assumptions(
        y=y,
        X=X,
        name_y='Bitcoin Returns',
        name_x='S&P 500 Returns'
    )

    # The check_all_assumptions function already printed a detailed report above
    # Let's just add a quick programmatic summary here
    print("Programmatic Access to Results:")
    print(f"  All tests passed: {diagnostics['all_passed']}")
    print(f"  Number of warnings: {len(diagnostics['warnings'])}")
    if diagnostics['warnings']:
        print("  Warnings:")
        for warning in diagnostics['warnings']:
            print(f"    - {warning}")
    print()

    # Show we can access individual results
    print("Detailed Results (programmatic):")
    y_stat = diagnostics['stationarity']['y']
    x_stat = diagnostics['stationarity']['x']
    print(f"  Y stationarity: {y_stat.is_stationary} (p={y_stat.p_value:.4f})")
    print(f"  X stationarity: {x_stat.is_stationary} (p={x_stat.p_value:.4f})")
    print(f"  Regression beta: {diagnostics['regression'].params.iloc[1]:.4f}")
    print(f"  R-squared: {diagnostics['regression'].rsquared:.4f}")
    print()

    print("=" * 80)
    print("Test Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
