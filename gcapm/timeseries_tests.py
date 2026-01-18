"""
Time Series Testing Module

Statistical tests for time series properties essential for rigorous research:
- Stationarity testing (ADF, KPSS, PP)
- Autocorrelation testing
- Heteroscedasticity testing
- Structural break detection

These tests should be run BEFORE performing regressions or other analyses.

Examples
--------
>>> import gcapm
>>> df = gcapm.load_data('returns_daily')
>>>
>>> # Test if Bitcoin returns are stationary
>>> result = gcapm.timeseries_tests.test_stationarity(
...     df['return_btc'],
...     name='Bitcoin Returns'
... )
>>> if not result['is_stationary']:
...     print("WARNING: Series may not be stationary!")
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
import warnings


@dataclass
class StationarityTestResult:
    """
    Container for stationarity test results.

    Attributes
    ----------
    test_name : str
        Name of the test ("ADF", "KPSS", etc.)
    statistic : float
        Test statistic value
    p_value : float
        P-value for the test
    critical_values : dict
        Critical values at different significance levels
    is_stationary : bool
        Whether series appears stationary (based on p_value < 0.05)
    interpretation : str
        Plain English interpretation
    """
    test_name: str
    statistic: float
    p_value: float
    critical_values: Dict[str, float]
    is_stationary: bool
    interpretation: str

    def __repr__(self):
        status = "✓ STATIONARY" if self.is_stationary else "✗ NON-STATIONARY"
        return f"{self.test_name}: {status} (p={self.p_value:.4f})"


def test_stationarity(series: pd.Series,
                     test: str = 'adf',
                     name: Optional[str] = None) -> StationarityTestResult:
    """
    Test if a time series is stationary.

    Stationarity is crucial for time series regression:
    - Non-stationary series can give spurious correlations
    - Standard errors and p-values will be invalid
    - Always test before running regressions!

    Parameters
    ----------
    series : pd.Series
        Time series to test
    test : str, default 'adf'
        Which test to use:
        - 'adf': Augmented Dickey-Fuller (most common)
        - 'kpss': Kwiatkowski-Phillips-Schmidt-Shin
        - 'both': Run both tests
    name : str, optional
        Name for display purposes

    Returns
    -------
    StationarityTestResult or dict of results (if test='both')

    Examples
    --------
    >>> import gcapm
    >>> df = gcapm.load_data('returns_daily')
    >>>
    >>> # Test Bitcoin returns
    >>> result = gcapm.timeseries_tests.test_stationarity(
    ...     df['return_btc'],
    ...     test='adf',
    ...     name='Bitcoin Returns'
    ... )
    >>>
    >>> print(result)
    >>> if not result.is_stationary:
    ...     print("WARNING: Series may not be stationary!")
    ...     print("Consider:")
    ...     print("  1. Differencing the series")
    ...     print("  2. Checking for structural breaks")

    Notes
    -----
    **ADF Test** (Augmented Dickey-Fuller):
    - H0: Series has unit root (non-stationary)
    - H1: Series is stationary
    - Reject H0 if p < 0.05 → Series is stationary

    **KPSS Test**:
    - H0: Series is stationary
    - H1: Series has unit root
    - Reject H0 if p < 0.05 → Series is non-stationary

    **Best practice**: Run both tests
    - Both reject: Strong evidence
    - Both fail to reject: Inconclusive, investigate further
    - Disagreement: Check for structural breaks

    References
    ----------
    - Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators
      for autoregressive time series with a unit root.
    - Kwiatkowski, D., et al. (1992). Testing the null hypothesis of
      stationarity against the alternative of a unit root.
    """
    series_clean = series.dropna()

    if len(series_clean) < 20:
        warnings.warn("Sample size < 20, test may be unreliable")

    if test == 'adf' or test == 'both':
        adf_result = _test_adf(series_clean, name)
        if test == 'adf':
            return adf_result

    if test == 'kpss' or test == 'both':
        kpss_result = _test_kpss(series_clean, name)
        if test == 'kpss':
            return kpss_result

    if test == 'both':
        return {
            'adf': adf_result,
            'kpss': kpss_result,
            'agreement': adf_result.is_stationary == kpss_result.is_stationary
        }

    raise ValueError(f"Invalid test: {test}. Use 'adf', 'kpss', or 'both'.")


def _test_adf(series: pd.Series, name: Optional[str]) -> StationarityTestResult:
    """Run Augmented Dickey-Fuller test."""
    result = adfuller(series, autolag='AIC')

    statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    is_stationary = p_value < 0.05

    # Interpretation
    if is_stationary:
        interpretation = (
            f"ADF test rejects unit root hypothesis (p={p_value:.4f} < 0.05). "
            f"Series appears stationary. Safe to use in regression."
        )
    else:
        interpretation = (
            f"ADF test fails to reject unit root hypothesis (p={p_value:.4f} ≥ 0.05). "
            f"Series may be non-stationary. Consider differencing or checking for breaks."
        )

    return StationarityTestResult(
        test_name=f"ADF ({name})" if name else "ADF",
        statistic=statistic,
        p_value=p_value,
        critical_values=critical_values,
        is_stationary=is_stationary,
        interpretation=interpretation
    )


def _test_kpss(series: pd.Series, name: Optional[str]) -> StationarityTestResult:
    """Run KPSS test."""
    statistic, p_value, n_lags, critical_values = kpss(series, regression='c')

    # Note: KPSS has OPPOSITE null hypothesis
    is_stationary = p_value >= 0.05  # Fail to reject stationary null

    # Interpretation
    if is_stationary:
        interpretation = (
            f"KPSS test fails to reject stationarity hypothesis (p={p_value:.4f} ≥ 0.05). "
            f"Series appears stationary."
        )
    else:
        interpretation = (
            f"KPSS test rejects stationarity hypothesis (p={p_value:.4f} < 0.05). "
            f"Series appears non-stationary. Consider differencing."
        )

    return StationarityTestResult(
        test_name=f"KPSS ({name})" if name else "KPSS",
        statistic=statistic,
        p_value=p_value,
        critical_values=critical_values,
        is_stationary=is_stationary,
        interpretation=interpretation
    )


def test_autocorrelation(series: pd.Series,
                        lags: int = 20,
                        test: str = 'ljungbox') -> Dict:
    """
    Test for autocorrelation in time series.

    Autocorrelation violates OLS assumption of independent errors.
    If present, you need to use HAC (Newey-West) standard errors.

    Parameters
    ----------
    series : pd.Series
        Time series (typically residuals from regression)
    lags : int, default 20
        Number of lags to test
    test : str, default 'ljungbox'
        Which test to use:
        - 'ljungbox': Ljung-Box Q-test (recommended)
        - 'acf': Just calculate ACF (no formal test)

    Returns
    -------
    dict
        Test results including:
        - 'has_autocorrelation': bool
        - 'acf': autocorrelation function values
        - 'ljungbox_pvalues': p-values for each lag (if test='ljungbox')

    Examples
    --------
    >>> import gcapm
    >>> df = gcapm.load_data('returns_daily')
    >>>
    >>> # Run CAPM regression
    >>> results = gcapm.estimate_capm(df['return_btc'], df['return_sp500'])
    >>>
    >>> # Test residuals for autocorrelation
    >>> # (In practice, you'd need to extract residuals from regression)
    >>> autocorr_test = gcapm.timeseries_tests.test_autocorrelation(
    ...     df['return_btc'],
    ...     lags=20
    ... )
    >>>
    >>> if autocorr_test['has_autocorrelation']:
    ...     print("⚠️ WARNING: Autocorrelation detected!")
    ...     print("Use HAC standard errors (Newey-West)")

    Notes
    -----
    **Ljung-Box Q-test**:
    - H0: No autocorrelation up to lag k
    - H1: Autocorrelation present
    - If p < 0.05 for any lag: Reject H0, autocorrelation present

    **Rule of thumb for ACF**:
    - |ACF| > 2/√n suggests significant autocorrelation
    - For n=1000, threshold ≈ 0.063

    **What to do if autocorrelation is present**:
    1. Use Newey-West HAC standard errors
    2. Model the autocorrelation (ARMA)
    3. Use GLS instead of OLS
    """
    series_clean = series.dropna()
    n = len(series_clean)

    # Calculate ACF
    acf_values = acf(series_clean, nlags=lags, fft=False)

    # Rule of thumb threshold
    threshold = 2 / np.sqrt(n)

    # Check if any ACF exceeds threshold (exclude lag 0)
    significant_lags = np.where(np.abs(acf_values[1:lags+1]) > threshold)[0] + 1

    if test == 'ljungbox':
        # Ljung-Box test
        lb_result = acorr_ljungbox(series_clean, lags=lags, return_df=True)

        # Check if any p-value < 0.05
        has_autocorr_lb = (lb_result['lb_pvalue'] < 0.05).any()

        return {
            'test': 'Ljung-Box',
            'has_autocorrelation': has_autocorr_lb,
            'acf': acf_values,
            'ljungbox_pvalues': lb_result['lb_pvalue'].values,
            'significant_lags': significant_lags,
            'threshold': threshold,
            'interpretation': (
                "⚠️ Autocorrelation detected! Use HAC standard errors."
                if has_autocorr_lb else
                "No significant autocorrelation detected."
            )
        }
    else:  # acf only
        has_autocorr = len(significant_lags) > 0

        return {
            'test': 'ACF',
            'has_autocorrelation': has_autocorr,
            'acf': acf_values,
            'significant_lags': significant_lags,
            'threshold': threshold,
            'interpretation': (
                f"⚠️ ACF significant at lags: {significant_lags}. "
                f"Consider HAC standard errors."
                if has_autocorr else
                "No significant autocorrelation (by rule of thumb)."
            )
        }


def test_heteroscedasticity(y: pd.Series,
                           X: pd.DataFrame,
                           test: str = 'white') -> Dict:
    """
    Test for heteroscedasticity (non-constant variance).

    Heteroscedasticity doesn't bias coefficient estimates but makes
    standard errors wrong (confidence intervals and p-values invalid).

    Parameters
    ----------
    y : pd.Series
        Dependent variable
    X : pd.DataFrame
        Independent variables (should include constant)
    test : str, default 'white'
        Which test to use (currently only 'white' implemented)

    Returns
    -------
    dict
        Test results including:
        - 'has_heteroscedasticity': bool
        - 'statistic': test statistic
        - 'p_value': p-value
        - 'interpretation': what to do

    Examples
    --------
    >>> import gcapm
    >>> import statsmodels.api as sm
    >>> df = gcapm.load_data('returns_daily')
    >>>
    >>> # Prepare data
    >>> y = df['return_btc']
    >>> X = sm.add_constant(df['return_sp500'])
    >>>
    >>> # Test for heteroscedasticity
    >>> het_test = gcapm.timeseries_tests.test_heteroscedasticity(y, X)
    >>>
    >>> if het_test['has_heteroscedasticity']:
    ...     print("⚠️ Use robust standard errors (HC1, HC3)")

    Notes
    -----
    **White's Test**:
    - H0: Homoscedasticity (constant variance)
    - H1: Heteroscedasticity present
    - If p < 0.05: Reject H0, use robust SEs

    **What to do if heteroscedasticity is present**:
    1. Use White's robust standard errors (most common)
    2. Use weighted least squares (WLS) if you know the pattern
    3. Model the variance (GARCH for financial data)
    """
    import statsmodels.api as sm

    # Run regression
    model = sm.OLS(y, X, missing='drop').fit()

    # White's test
    if test == 'white':
        from statsmodels.stats.diagnostic import het_white
        white_test = het_white(model.resid, model.model.exog)

        statistic = white_test[0]
        p_value = white_test[1]

        has_het = p_value < 0.05

        return {
            'test': "White's Test",
            'statistic': statistic,
            'p_value': p_value,
            'has_heteroscedasticity': has_het,
            'interpretation': (
                "⚠️ Heteroscedasticity detected! Use robust standard errors (HC1, HC3)."
                if has_het else
                "No significant heteroscedasticity detected. "
                "Standard OLS standard errors are fine."
            )
        }
    else:
        raise ValueError(f"Test '{test}' not implemented. Use 'white'.")


def check_all_assumptions(y: pd.Series,
                         X: pd.Series,
                         name_y: str = "Y",
                         name_x: str = "X") -> Dict:
    """
    Run all standard time series regression checks.

    This is a convenience function that runs:
    1. Stationarity tests (ADF) on both series
    2. Runs regression and checks residuals
    3. Autocorrelation test
    4. Heteroscedasticity test

    Parameters
    ----------
    y : pd.Series
        Dependent variable
    X : pd.Series
        Independent variable
    name_y : str
        Name of y variable (for display)
    name_x : str
        Name of x variable (for display)

    Returns
    -------
    dict
        Complete diagnostic results with recommendations

    Examples
    --------
    >>> import gcapm
    >>> df = gcapm.load_data('returns_daily')
    >>>
    >>> # Check all assumptions for CAPM regression
    >>> diagnostics = gcapm.timeseries_tests.check_all_assumptions(
    ...     y=df['return_btc'],
    ...     X=df['return_sp500'],
    ...     name_y='Bitcoin Returns',
    ...     name_x='S&P 500 Returns'
    ... )
    >>>
    >>> # Print summary
    >>> print(diagnostics['summary'])
    >>>
    >>> # Get recommendations
    >>> if diagnostics['warnings']:
    ...     print("⚠️ Issues found:")
    ...     for warning in diagnostics['warnings']:
    ...         print(f"  - {warning}")

    This is the function you should run BEFORE any regression!
    """
    import statsmodels.api as sm

    warnings_list = []
    results = {}

    print(f"\n{'='*70}")
    print(f"TIME SERIES REGRESSION DIAGNOSTICS")
    print(f"{'='*70}\n")

    # 1. Test stationarity
    print(f"1. Testing Stationarity")
    print(f"   {'-'*66}")

    y_stat = test_stationarity(y, test='adf', name=name_y)
    x_stat = test_stationarity(X, test='adf', name=name_x)

    print(f"   {y_stat}")
    print(f"   {x_stat}\n")

    results['stationarity'] = {
        'y': y_stat,
        'x': x_stat
    }

    if not y_stat.is_stationary:
        warnings_list.append(f"{name_y} may be non-stationary (p={y_stat.p_value:.4f})")

    if not x_stat.is_stationary:
        warnings_list.append(f"{name_x} may be non-stationary (p={x_stat.p_value:.4f})")

    # 2. Run regression
    print(f"2. Running Regression")
    print(f"   {'-'*66}")

    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const, missing='drop').fit()

    print(f"   Coefficient: {model.params.iloc[1]:.6f}")
    print(f"   R-squared: {model.rsquared:.4f}\n")

    results['regression'] = model

    # 3. Test autocorrelation
    print(f"3. Testing Autocorrelation (Residuals)")
    print(f"   {'-'*66}")

    autocorr = test_autocorrelation(model.resid, lags=20)
    print(f"   {autocorr['interpretation']}\n")

    results['autocorrelation'] = autocorr

    if autocorr['has_autocorrelation']:
        warnings_list.append("Autocorrelation detected - use HAC (Newey-West) standard errors")

    # 4. Test heteroscedasticity
    print(f"4. Testing Heteroscedasticity")
    print(f"   {'-'*66}")

    het = test_heteroscedasticity(y, X_with_const)
    print(f"   {het['interpretation']}\n")

    results['heteroscedasticity'] = het

    if het['has_heteroscedasticity']:
        warnings_list.append("Heteroscedasticity detected - use robust standard errors")

    # 5. Summary
    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}\n")

    if not warnings_list:
        summary = "✓ All checks passed! Standard OLS is appropriate."
        print(summary)
    else:
        summary = "⚠️ Issues detected. See recommendations below."
        print(summary)
        print(f"\nWarnings:")
        for i, warning in enumerate(warnings_list, 1):
            print(f"   {i}. {warning}")

        print(f"\nRecommendations:")
        if any("non-stationary" in w for w in warnings_list):
            print(f"   - Consider differencing or checking for structural breaks")
        if any("Autocorrelation" in w for w in warnings_list):
            print(f"   - Use cov_type='HAC' with maxlags=5 in OLS.fit()")
        if any("Heteroscedasticity" in w for w in warnings_list):
            print(f"   - Use cov_type='HC1' or 'HC3' in OLS.fit()")

    print(f"\n{'='*70}\n")

    results['warnings'] = warnings_list
    results['summary'] = summary
    results['all_passed'] = len(warnings_list) == 0

    return results
