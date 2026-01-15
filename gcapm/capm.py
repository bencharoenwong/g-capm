"""
CAPM (Capital Asset Pricing Model) analysis functions.

This module provides functions for:
- Beta calculation (covariance method)
- CAPM regression with full statistics
- Rolling beta (time-varying)
- Visualization

Examples
--------
>>> import gcapm
>>> df = gcapm.load_data('tsla_daily')
>>> beta = gcapm.calculate_beta(df['return_tsla'], df['return_sp500_tr'])
>>> print(f"Tesla beta: {beta:.4f}")
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CAPMResults:
    """
    Container for CAPM regression results.

    Attributes
    ----------
    beta : float
        Beta coefficient (market sensitivity)
    alpha : float
        Alpha (intercept, excess return)
    r_squared : float
        R-squared (proportion of variance explained)
    std_error : float
        Standard error of beta estimate
    p_value : float
        P-value for beta (H0: beta = 0)
    conf_int_lower : float
        Lower bound of 95% confidence interval for beta
    conf_int_upper : float
        Upper bound of 95% confidence interval for beta
    n_observations : int
        Number of observations used

    Methods
    -------
    summary()
        Print formatted summary of results
    """
    beta: float
    alpha: float
    r_squared: float
    std_error: float
    p_value: float
    conf_int_lower: float
    conf_int_upper: float
    n_observations: int

    def summary(self) -> str:
        """
        Generate formatted summary string.

        Returns
        -------
        str
            Multi-line summary of CAPM results
        """
        return f"""
CAPM Results
============
Beta:        {self.beta:>10.4f} (SE: {self.std_error:.4f})
Alpha:       {self.alpha:>10.6f} ({self.alpha*100:.4f}%)
R-squared:   {self.r_squared:>10.4f} ({self.r_squared*100:.2f}%)
P-value:     {self.p_value:>10.6f}
95% CI:      [{self.conf_int_lower:.4f}, {self.conf_int_upper:.4f}]
N:           {self.n_observations:>10}
"""

    def __repr__(self):
        return f"CAPMResults(beta={self.beta:.4f}, alpha={self.alpha:.6f}, R2={self.r_squared:.4f})"


def calculate_beta(asset_returns: pd.Series,
                  market_returns: pd.Series) -> float:
    """
    Calculate beta using the covariance method.

    Beta measures how much an asset moves with the market.
    Formula: � = Cov(R_asset, R_market) / Var(R_market)

    Parameters
    ----------
    asset_returns : pd.Series
        Asset returns (decimal format, e.g., 0.01 = 1%)
    market_returns : pd.Series
        Market returns (decimal format)

    Returns
    -------
    float
        Beta coefficient

    Raises
    ------
    ValueError
        If market variance is zero (no market movement)

    Examples
    --------
    >>> import gcapm
    >>> df = gcapm.load_data('tsla_daily')
    >>> beta = gcapm.calculate_beta(df['return_tsla'], df['return_sp500_tr'])
    >>> print(f"Beta: {beta:.4f}")

    Notes
    -----
    - Beta > 1: Asset is more volatile than market
    - Beta = 1: Asset moves with market
    - Beta < 1: Asset is less volatile than market
    - Beta < 0: Asset moves opposite to market
    """
    # Calculate covariance and variance
    covariance = asset_returns.cov(market_returns)
    market_variance = market_returns.var()

    # Check for zero variance
    if market_variance == 0:
        raise ValueError("Market variance is zero - cannot calculate beta")

    return covariance / market_variance


def estimate_capm(asset_returns: pd.Series,
                 market_returns: pd.Series) -> CAPMResults:
    """
    Estimate CAPM using OLS regression with full statistics.

    Model: R_asset = � + � * R_market + �

    This provides more information than simple covariance method:
    - Standard errors
    - Confidence intervals
    - Statistical significance
    - R-squared

    Parameters
    ----------
    asset_returns : pd.Series
        Asset returns
    market_returns : pd.Series
        Market returns

    Returns
    -------
    CAPMResults
        Complete regression results

    Examples
    --------
    >>> import gcapm
    >>> df = gcapm.load_data('tsla_daily')
    >>> results = gcapm.estimate_capm(df['return_tsla'], df['return_sp500_tr'])
    >>> print(results.summary())
    >>>
    >>> # Check if beta is significantly different from 1
    >>> if results.conf_int_lower < 1 < results.conf_int_upper:
    ...     print("Beta not significantly different from 1")

    Notes
    -----
    The model assumes:
    - Linear relationship between asset and market
    - Homoscedasticity (constant variance)
    - No autocorrelation in residuals
    - Normal distribution of errors (for inference)

    For financial data, these assumptions often don't hold perfectly.
    Consider using robust standard errors or other methods for production use.
    """
    # Prepare data for regression
    y = asset_returns.values
    X = market_returns.values
    X = sm.add_constant(X)  # Add intercept

    # Run OLS regression
    model = sm.OLS(y, X, missing='drop').fit()

    # Extract results
    return CAPMResults(
        beta=float(model.params[1]),
        alpha=float(model.params[0]),
        r_squared=float(model.rsquared),
        std_error=float(model.bse[1]),
        p_value=float(model.pvalues[1]),
        conf_int_lower=float(model.conf_int()[1][0]),
        conf_int_upper=float(model.conf_int()[1][1]),
        n_observations=int(model.nobs)
    )


def rolling_beta(asset_returns: pd.Series,
                market_returns: pd.Series,
                window: int = 252) -> pd.Series:
    """
    Calculate rolling (time-varying) beta.

    This shows how beta changes over time, useful for:
    - Detecting regime changes
    - Understanding stability of market relationship
    - Risk management

    Parameters
    ----------
    asset_returns : pd.Series
        Asset returns
    market_returns : pd.Series
        Market returns
    window : int, default 252
        Rolling window size (252 = 1 year of daily data)

    Returns
    -------
    pd.Series
        Rolling beta values (NaN for first window-1 observations)

    Examples
    --------
    >>> import gcapm
    >>> df = gcapm.load_data('tsla_daily')
    >>> rolling_b = gcapm.rolling_beta(
    ...     df['return_tsla'],
    ...     df['return_sp500_tr'],
    ...     window=252
    ... )
    >>> rolling_b.plot(title='Tesla Rolling 1-Year Beta')

    See Also
    --------
    calculate_beta : Single beta estimate
    time_varying_beta : More sophisticated time-varying estimation
    """
    if len(asset_returns) != len(market_returns):
        raise ValueError("Return series must have same length")

    if window < 30:
        import warnings
        warnings.warn(f"Window size {window} is small, results may be unstable")

    # Calculate rolling covariance and variance
    rolling_cov = asset_returns.rolling(window).cov(market_returns)
    rolling_var = market_returns.rolling(window).var()

    # Calculate beta (handle division by zero)
    rolling_beta = rolling_cov / rolling_var.replace(0, np.nan)

    return rolling_beta


class CAPMAnalyzer:
    """
    High-level interface for CAPM analysis.

    This class provides a convenient way to run complete CAPM analysis
    including estimation, visualization, and diagnostics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing returns data
    asset_col : str
        Name of column with asset returns
    market_col : str
        Name of column with market returns
    date_col : str, default 'date'
        Name of date column

    Attributes
    ----------
    results : CAPMResults or None
        Results from estimation (None until estimate() is called)

    Examples
    --------
    >>> import gcapm
    >>> df = gcapm.load_data('tsla_daily')
    >>> analyzer = gcapm.CAPMAnalyzer(
    ...     df,
    ...     asset_col='return_tsla',
    ...     market_col='return_sp500_tr'
    ... )
    >>> results = analyzer.estimate()
    >>> print(results.summary())
    >>> analyzer.plot()  # Create scatter plot with regression line
    """

    def __init__(self,
                 data: pd.DataFrame,
                 asset_col: str,
                 market_col: str,
                 date_col: str = 'date'):
        self.data = data
        self.asset_col = asset_col
        self.market_col = market_col
        self.date_col = date_col
        self.results = None

    def estimate(self) -> CAPMResults:
        """
        Estimate CAPM.

        Returns
        -------
        CAPMResults
            Regression results
        """
        self.results = estimate_capm(
            self.data[self.asset_col],
            self.data[self.market_col]
        )
        return self.results

    def plot(self, figsize=(10, 6)):
        """
        Create scatter plot with regression line.

        Parameters
        ----------
        figsize : tuple, default (10, 6)
            Figure size (width, height) in inches

        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt

        # Estimate if not done yet
        if self.results is None:
            self.estimate()

        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot
        ax.scatter(
            self.data[self.market_col],
            self.data[self.asset_col],
            alpha=0.5,
            s=20,
            label='Observations'
        )

        # Regression line
        x_range = np.linspace(
            self.data[self.market_col].min(),
            self.data[self.market_col].max(),
            100
        )
        y_pred = self.results.alpha + self.results.beta * x_range
        ax.plot(
            x_range,
            y_pred,
            'r-',
            linewidth=2,
            label=f'�={self.results.beta:.4f}'
        )

        # Reference line (beta = 1)
        ax.plot(
            x_range,
            x_range,
            'g--',
            linewidth=1,
            alpha=0.5,
            label='�=1 Reference'
        )

        ax.set_xlabel(f'{self.market_col}', fontsize=11)
        ax.set_ylabel(f'{self.asset_col}', fontsize=11)
        ax.set_title(
            f'CAPM Regression: �={self.results.beta:.4f}, R2={self.results.r_squared:.4f}',
            fontsize=13,
            fontweight='bold'
        )
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def rolling_analysis(self, window=252):
        """
        Calculate and plot rolling beta.

        Parameters
        ----------
        window : int, default 252
            Rolling window size

        Returns
        -------
        pd.Series
            Rolling beta values
        """
        return rolling_beta(
            self.data[self.asset_col],
            self.data[self.market_col],
            window=window
        )
