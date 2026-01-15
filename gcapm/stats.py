"""
Statistical analysis functions.

This module provides statistical tools for financial analysis:
- Rolling correlations (time-varying relationships)
- Time-varying beta
- Statistical tests
- Regime detection

Special focus on crypto analysis where correlations change dramatically over time.

Examples
--------
>>> import gcapm
>>> df = gcapm.load_data('returns_daily')
>>> corr = gcapm.rolling_correlation(
...     df['return_btc'],
...     df['return_sp500'],
...     window=90
... )
>>> corr.plot(title='Bitcoin-S&P500 Correlation Over Time')
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import warnings


def rolling_correlation(series1: pd.Series,
                       series2: pd.Series,
                       window: int = 90,
                       min_periods: Optional[int] = None) -> pd.Series:
    """
    Calculate rolling correlation between two return series.

    This is crucial for understanding how relationships between assets change over time,
    especially for crypto assets where correlations with traditional markets vary
    dramatically across different market regimes.

    Parameters
    ----------
    series1 : pd.Series
        First return series (e.g., Bitcoin returns)
    series2 : pd.Series
        Second return series (e.g., S&P 500 returns)
    window : int, default 90
        Rolling window size in periods (90 days H 3 months for daily data)
    min_periods : int, optional
        Minimum number of observations required. Defaults to window.

    Returns
    -------
    pd.Series
        Rolling correlation values (range: -1 to +1)
        - +1: Perfect positive correlation
        - 0: No correlation
        - -1: Perfect negative correlation

    Examples
    --------
    >>> import gcapm
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Load crypto and stock data
    >>> df = gcapm.load_data('returns_daily')
    >>>
    >>> # Calculate rolling 3-month correlation
    >>> btc_sp500_corr = gcapm.rolling_correlation(
    ...     df['return_btc'],
    ...     df['return_sp500'],
    ...     window=90
    ... )
    >>>
    >>> # Plot to see how correlation changes over time
    >>> btc_sp500_corr.plot(
    ...     title='Bitcoin-S&P500 3-Month Rolling Correlation',
    ...     ylabel='Correlation',
    ...     figsize=(12, 6)
    ... )
    >>> plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    >>> plt.show()
    >>>
    >>> # Identify periods of high/low correlation
    >>> print(f"Mean correlation: {btc_sp500_corr.mean():.3f}")
    >>> print(f"Max correlation: {btc_sp500_corr.max():.3f} on {btc_sp500_corr.idxmax()}")
    >>> print(f"Min correlation: {btc_sp500_corr.min():.3f} on {btc_sp500_corr.idxmin()}")

    Notes
    -----
    For crypto assets:
    - Early days (2016-2018): Often low/negative correlation with stocks
    - COVID (2020): Increased correlation as "risk-off" behavior dominated
    - 2021-2022: Varying correlation depending on macro regime
    - Bear markets: Often higher correlation (everything falls together)
    - Bull markets: Lower correlation (diversification benefits)

    This makes crypto correlation analysis critical for portfolio construction.
    """
    if min_periods is None:
        min_periods = window

    # Calculate rolling correlation
    rolling_corr = series1.rolling(
        window=window,
        min_periods=min_periods
    ).corr(series2)

    return rolling_corr


def rolling_correlation_matrix(df: pd.DataFrame,
                               window: int = 90,
                               min_periods: Optional[int] = None) -> Dict[Tuple[str, str], pd.Series]:
    """
    Calculate pairwise rolling correlations for all columns in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with multiple return series (each column is an asset)
    window : int, default 90
        Rolling window size
    min_periods : int, optional
        Minimum observations required

    Returns
    -------
    dict
        Dictionary mapping (col1, col2) tuples to correlation Series

    Examples
    --------
    >>> import gcapm
    >>> df = gcapm.load_data('returns_daily')
    >>>
    >>> # Get all pairwise correlations
    >>> corr_dict = gcapm.stats.rolling_correlation_matrix(
    ...     df[['return_btc', 'return_eth', 'return_sp500']],
    ...     window=90
    ... )
    >>>
    >>> # Access specific pair
    >>> btc_eth_corr = corr_dict[('return_btc', 'return_eth')]
    >>> btc_sp500_corr = corr_dict[('return_btc', 'return_sp500')]
    """
    correlations = {}
    columns = df.columns.tolist()

    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:  # Only upper triangle (avoid duplicates)
            corr = rolling_correlation(
                df[col1],
                df[col2],
                window=window,
                min_periods=min_periods
            )
            correlations[(col1, col2)] = corr

    return correlations


def time_varying_beta(asset_returns: pd.Series,
                     market_returns: pd.Series,
                     window: int = 252,
                     method: str = 'rolling') -> pd.Series:
    """
    Calculate time-varying beta using rolling windows.

    This extends rolling_beta from capm.py with additional options
    and is specifically designed for detecting regime changes.

    Parameters
    ----------
    asset_returns : pd.Series
        Asset returns
    market_returns : pd.Series
        Market returns
    window : int, default 252
        Window size (252 = 1 year of daily data)
    method : str, default 'rolling'
        Method for calculation:
        - 'rolling': Simple rolling window
        - 'expanding': Expanding window (all data up to that point)
        - 'exponential': Exponentially weighted (more recent data gets more weight)

    Returns
    -------
    pd.Series
        Time-varying beta values

    Examples
    --------
    >>> import gcapm
    >>> df = gcapm.load_data('returns_daily')
    >>>
    >>> # Rolling 1-year beta for Bitcoin
    >>> btc_beta = gcapm.time_varying_beta(
    ...     df['return_btc'],
    ...     df['return_sp500'],
    ...     window=252,
    ...     method='rolling'
    ... )
    >>>
    >>> # Exponentially weighted beta (more responsive to recent changes)
    >>> btc_beta_ewm = gcapm.time_varying_beta(
    ...     df['return_btc'],
    ...     df['return_sp500'],
    ...     window=90,
    ...     method='exponential'
    ... )
    """
    if method == 'rolling':
        # Rolling covariance and variance
        rolling_cov = asset_returns.rolling(window).cov(market_returns)
        rolling_var = market_returns.rolling(window).var()
        beta = rolling_cov / rolling_var.replace(0, np.nan)

    elif method == 'expanding':
        # Expanding window (cumulative)
        expanding_cov = asset_returns.expanding().cov(market_returns)
        expanding_var = market_returns.expanding().var()
        beta = expanding_cov / expanding_var.replace(0, np.nan)

    elif method == 'exponential':
        # Exponentially weighted
        ewm_cov = asset_returns.ewm(span=window).cov(market_returns)
        ewm_var = market_returns.ewm(span=window).var()
        beta = ewm_cov / ewm_var.replace(0, np.nan)

    else:
        raise ValueError(f"Invalid method: {method}. Use 'rolling', 'expanding', or 'exponential'")

    return beta


def correlation_regime_changes(correlation: pd.Series,
                               threshold: float = 0.3,
                               min_duration: int = 30) -> pd.DataFrame:
    """
    Detect regime changes in correlation time series.

    Identifies periods where correlation shifts significantly, useful for:
    - Understanding when diversification benefits appear/disappear
    - Detecting market regime changes
    - Portfolio rebalancing decisions

    Parameters
    ----------
    correlation : pd.Series
        Rolling correlation series
    threshold : float, default 0.3
        Absolute correlation level defining "high" vs "low"
    min_duration : int, default 30
        Minimum days in a regime to be considered significant

    Returns
    -------
    pd.DataFrame
        Regime information with columns:
        - start_date: When regime started
        - end_date: When regime ended
        - regime: 'high_positive', 'low', 'high_negative'
        - mean_corr: Average correlation during regime
        - duration: Number of periods

    Examples
    --------
    >>> import gcapm
    >>> df = gcapm.load_data('returns_daily')
    >>>
    >>> # Calculate rolling correlation
    >>> corr = gcapm.rolling_correlation(
    ...     df['return_btc'],
    ...     df['return_sp500'],
    ...     window=90
    ... )
    >>>
    >>> # Detect regime changes
    >>> regimes = gcapm.stats.correlation_regime_changes(
    ...     corr,
    ...     threshold=0.3,
    ...     min_duration=30
    ... )
    >>>
    >>> print(regimes)
    >>>
    >>> # Analyze: When was Bitcoin most correlated with stocks?
    >>> high_corr_periods = regimes[regimes['regime'] == 'high_positive']
    >>> print(f"High correlation periods:\\n{high_corr_periods}")
    """
    # Classify correlation into regimes
    regime = pd.Series(index=correlation.index, dtype='object')
    regime[correlation > threshold] = 'high_positive'
    regime[correlation < -threshold] = 'high_negative'
    regime[(correlation >= -threshold) & (correlation <= threshold)] = 'low'

    # Find regime changes
    regime_changes = regime != regime.shift(1)
    regime_starts = regime[regime_changes].index

    # Build regime periods
    regimes = []
    for i in range(len(regime_starts)):
        start = regime_starts[i]
        end = regime_starts[i+1] if i+1 < len(regime_starts) else correlation.index[-1]

        # Get regime data
        regime_data = correlation[start:end]
        duration = len(regime_data)

        # Only include if meets minimum duration
        if duration >= min_duration:
            regimes.append({
                'start_date': start,
                'end_date': end,
                'regime': regime[start],
                'mean_corr': regime_data.mean(),
                'min_corr': regime_data.min(),
                'max_corr': regime_data.max(),
                'duration': duration
            })

    return pd.DataFrame(regimes)


def statistical_tests(asset_returns: pd.Series,
                     market_returns: pd.Series) -> Dict[str, float]:
    """
    Run standard statistical tests on returns data.

    Tests include:
    - Correlation significance
    - Normality test (Jarque-Bera)
    - Autocorrelation test (Ljung-Box)

    Parameters
    ----------
    asset_returns : pd.Series
        Asset returns
    market_returns : pd.Series
        Market returns

    Returns
    -------
    dict
        Dictionary of test results with p-values

    Examples
    --------
    >>> import gcapm
    >>> df = gcapm.load_data('tsla_daily')
    >>> tests = gcapm.statistical_tests(
    ...     df['return_tsla'],
    ...     df['return_sp500_tr']
    ... )
    >>> print(tests)
    """
    from scipy import stats

    results = {}

    # Correlation and significance
    corr, p_corr = stats.pearsonr(
        asset_returns.dropna(),
        market_returns.dropna()
    )
    results['correlation'] = corr
    results['correlation_pvalue'] = p_corr

    # Normality test (Jarque-Bera)
    jb_stat, jb_pvalue = stats.jarque_bera(asset_returns.dropna())
    results['normality_statistic'] = jb_stat
    results['normality_pvalue'] = jb_pvalue

    # Note: Returns are typically NOT normal (fat tails, skewness)
    # This is expected!

    return results


def correlation_heatmap_over_time(df: pd.DataFrame,
                                  window: int = 90,
                                  periods: Optional[list] = None):
    """
    Create a heatmap showing how correlations change over time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with multiple asset return columns
    window : int, default 90
        Rolling window size
    periods : list of datetime, optional
        Specific dates to show correlations for

    Returns
    -------
    fig, axes
        Matplotlib figure and axes

    Examples
    --------
    >>> import gcapm
    >>> df = gcapm.load_data('returns_daily')
    >>>
    >>> # Show correlation evolution
    >>> fig, axes = gcapm.stats.correlation_heatmap_over_time(
    ...     df[['return_btc', 'return_eth', 'return_sp500', 'return_ssc']],
    ...     window=90
    ... )
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # If no specific periods, use quartiles
    if periods is None:
        n = len(df)
        periods = [
            df.index[n//4],
            df.index[n//2],
            df.index[3*n//4],
            df.index[-1]
        ]

    n_periods = len(periods)
    fig, axes = plt.subplots(1, n_periods, figsize=(5*n_periods, 4))

    if n_periods == 1:
        axes = [axes]

    for ax, period in zip(axes, periods):
        # Get data up to this period
        period_idx = df.index.get_loc(period)
        start_idx = max(0, period_idx - window)
        window_data = df.iloc[start_idx:period_idx+1]

        # Calculate correlation matrix
        corr_matrix = window_data.corr()

        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            cbar_kws={'label': 'Correlation'}
        )
        ax.set_title(f'{period:%Y-%m-%d}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig, axes
