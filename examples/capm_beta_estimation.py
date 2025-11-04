#!/usr/bin/env python3
"""
CAPM Beta Estimation Script

This script calculates the Capital Asset Pricing Model (CAPM) beta coefficient
for securities using market returns. It demonstrates robust statistical analysis
with comprehensive error handling and validation.

CAPM Formula:
    E(R_i) = R_f + β_i * (E(R_m) - R_f)

Where:
    E(R_i) = Expected return of asset i
    R_f = Risk-free rate
    β_i = Beta of asset i
    E(R_m) = Expected market return

Beta Calculation:
    β = Cov(R_i, R_m) / Var(R_m)

Author: g-capm Project
Date: 2025-11-04
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Optional
import warnings

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class CAPMAnalyzer:
    """
    A comprehensive class for CAPM beta estimation and analysis.

    Attributes:
        data (pd.DataFrame): The loaded return data
        asset_col (str): Name of the asset return column
        market_col (str): Name of the market return column
        beta (float): Calculated beta coefficient
        alpha (float): Calculated alpha (intercept)
        r_squared (float): R-squared of the regression
        std_error (float): Standard error of beta estimate
    """

    def __init__(self, data_path: str, asset_col: str, market_col: str,
                 date_col: str = 'Date', date_format: str = None):
        """
        Initialize the CAPM analyzer.

        Args:
            data_path: Path to the CSV file containing returns data
            asset_col: Name of the column containing asset returns
            market_col: Name of the column containing market returns
            date_col: Name of the date column (default: 'Date')
            date_format: Date format string for parsing (optional)

        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If required columns are missing
        """
        self.data_path = Path(data_path)
        self.asset_col = asset_col
        self.market_col = market_col
        self.date_col = date_col

        # Initialize result attributes
        self.beta = None
        self.alpha = None
        self.r_squared = None
        self.std_error = None
        self.model = None

        # Load and validate data
        self._load_data(date_format)
        self._validate_data()

    def _load_data(self, date_format: Optional[str] = None):
        """Load data from CSV file with error handling."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        try:
            if date_format:
                self.data = pd.read_csv(self.data_path,
                                       parse_dates=[self.date_col],
                                       date_format=date_format)
            else:
                self.data = pd.read_csv(self.data_path,
                                       parse_dates=[self.date_col])

            print(f"✓ Loaded {len(self.data)} observations from {self.data_path.name}")

        except Exception as e:
            raise IOError(f"Error loading data: {str(e)}")

    def _validate_data(self):
        """Validate that required columns exist and contain valid data."""
        # Check for required columns
        required_cols = [self.asset_col, self.market_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}\n"
                           f"Available columns: {list(self.data.columns)}")

        # Check for missing values
        n_missing_asset = self.data[self.asset_col].isna().sum()
        n_missing_market = self.data[self.market_col].isna().sum()

        if n_missing_asset > 0 or n_missing_market > 0:
            warnings.warn(f"Found missing values: {n_missing_asset} in asset returns, "
                         f"{n_missing_market} in market returns. These will be dropped.")

        # Drop rows with missing values
        self.data = self.data.dropna(subset=[self.asset_col, self.market_col])

        # Check for infinite values
        if np.isinf(self.data[self.asset_col]).any() or \
           np.isinf(self.data[self.market_col]).any():
            warnings.warn("Found infinite values in data. These will be removed.")
            self.data = self.data.replace([np.inf, -np.inf], np.nan).dropna()

        print(f"✓ Validated {len(self.data)} clean observations")

    def calculate_beta_covariance(self) -> float:
        """
        Calculate beta using the covariance method.

        Formula: β = Cov(R_asset, R_market) / Var(R_market)

        Returns:
            Beta coefficient
        """
        covariance = self.data[self.asset_col].cov(self.data[self.market_col])
        market_variance = self.data[self.market_col].var()

        if market_variance == 0:
            raise ValueError("Market variance is zero. Cannot calculate beta.")

        beta = covariance / market_variance
        return beta

    def calculate_beta_regression(self) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Calculate beta using OLS regression with comprehensive statistics.

        Model: R_asset = α + β * R_market + ε

        Returns:
            Statsmodels regression results object
        """
        # Prepare data for regression
        y = self.data[self.asset_col].values
        X = self.data[self.market_col].values
        X = sm.add_constant(X)  # Add intercept

        # Run OLS regression
        model = sm.OLS(y, X, missing='drop').fit()

        # Store results
        self.alpha = model.params[0]
        self.beta = model.params[1]
        self.r_squared = model.rsquared
        self.std_error = model.bse[1]
        self.model = model

        return model

    def calculate_statistics(self) -> dict:
        """
        Calculate comprehensive statistics for the returns.

        Returns:
            Dictionary containing statistical measures
        """
        stats = {
            'asset_mean': self.data[self.asset_col].mean(),
            'asset_std': self.data[self.asset_col].std(),
            'asset_min': self.data[self.asset_col].min(),
            'asset_max': self.data[self.asset_col].max(),
            'market_mean': self.data[self.market_col].mean(),
            'market_std': self.data[self.market_col].std(),
            'correlation': self.data[self.asset_col].corr(self.data[self.market_col]),
            'n_observations': len(self.data)
        }

        return stats

    def print_results(self):
        """Print formatted results of the CAPM analysis."""
        print("\n" + "="*70)
        print("CAPM BETA ESTIMATION RESULTS")
        print("="*70)

        print(f"\nAsset: {self.asset_col}")
        print(f"Market: {self.market_col}")
        print(f"Period: {self.data[self.date_col].min()} to {self.data[self.date_col].max()}")
        print(f"Observations: {len(self.data)}")

        print("\n" + "-"*70)
        print("REGRESSION RESULTS")
        print("-"*70)

        print(f"Beta (β):        {self.beta:>10.4f}  (Std. Error: {self.std_error:.4f})")
        print(f"Alpha (α):       {self.alpha:>10.6f}  ({self.alpha*100:.4f}%)")
        print(f"R-squared:       {self.r_squared:>10.4f}  ({self.r_squared*100:.2f}%)")

        # Calculate confidence interval
        conf_int = self.model.conf_int()[1]
        print(f"95% CI for β:    [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")

        # Statistical significance
        p_value = self.model.pvalues[1]
        print(f"P-value:         {p_value:>10.6f}  ", end="")
        if p_value < 0.01:
            print("(Highly significant ***)")
        elif p_value < 0.05:
            print("(Significant **)")
        elif p_value < 0.10:
            print("(Marginally significant *)")
        else:
            print("(Not significant)")

        print("\n" + "-"*70)
        print("RETURN STATISTICS")
        print("-"*70)

        stats = self.calculate_statistics()

        print(f"\nAsset Returns:")
        print(f"  Mean:          {stats['asset_mean']:>10.6f}  ({stats['asset_mean']*100:.4f}%)")
        print(f"  Std Dev:       {stats['asset_std']:>10.6f}  ({stats['asset_std']*100:.4f}%)")
        print(f"  Min:           {stats['asset_min']:>10.6f}  ({stats['asset_min']*100:.4f}%)")
        print(f"  Max:           {stats['asset_max']:>10.6f}  ({stats['asset_max']*100:.4f}%)")

        print(f"\nMarket Returns:")
        print(f"  Mean:          {stats['market_mean']:>10.6f}  ({stats['market_mean']*100:.4f}%)")
        print(f"  Std Dev:       {stats['market_std']:>10.6f}  ({stats['market_std']*100:.4f}%)")

        print(f"\nCorrelation:     {stats['correlation']:>10.4f}")

        print("\n" + "-"*70)
        print("INTERPRETATION")
        print("-"*70)

        self._print_interpretation()

        print("\n" + "="*70 + "\n")

    def _print_interpretation(self):
        """Print interpretation of the beta coefficient."""
        print(f"\n• Beta = {self.beta:.4f} means:")

        if self.beta > 1:
            print(f"  → Asset is MORE VOLATILE than market (amplifies by {(self.beta-1)*100:.1f}%)")
            print(f"  → When market rises 1%, asset expected to rise {self.beta:.2f}%")
        elif self.beta < 1 and self.beta > 0:
            print(f"  → Asset is LESS VOLATILE than market (dampens by {(1-self.beta)*100:.1f}%)")
            print(f"  → When market rises 1%, asset expected to rise {self.beta:.2f}%")
        elif self.beta < 0:
            print(f"  → Asset moves INVERSELY to market")
            print(f"  → When market rises 1%, asset expected to fall {abs(self.beta):.2f}%")
        else:
            print(f"  → Asset has NO systematic relationship with market")

        if abs(self.alpha) > 0.001:
            if self.alpha > 0:
                print(f"\n• Alpha = {self.alpha:.6f} ({self.alpha*100:.4f}%)")
                print(f"  → Asset generates EXCESS returns (outperforms risk-adjusted expectation)")
            else:
                print(f"\n• Alpha = {self.alpha:.6f} ({self.alpha*100:.4f}%)")
                print(f"  → Asset generates NEGATIVE excess returns (underperforms)")

        print(f"\n• R² = {self.r_squared:.4f} means:")
        print(f"  → {self.r_squared*100:.2f}% of asset variance explained by market")
        print(f"  → {(1-self.r_squared)*100:.2f}% due to idiosyncratic (firm-specific) risk")

    def plot_scatter_with_regression(self, save_path: Optional[str] = None):
        """
        Create scatter plot of asset vs market returns with regression line.

        Args:
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Scatter plot
        ax.scatter(self.data[self.market_col], self.data[self.asset_col],
                   alpha=0.5, s=30, label='Observations')

        # Regression line
        x_range = np.linspace(self.data[self.market_col].min(),
                             self.data[self.market_col].max(), 100)
        y_pred = self.alpha + self.beta * x_range
        ax.plot(x_range, y_pred, 'r-', linewidth=2,
                label=f'Regression Line (β={self.beta:.4f})')

        # Reference line (β = 1)
        y_reference = x_range  # Slope of 1
        ax.plot(x_range, y_reference, 'g--', linewidth=1, alpha=0.5,
                label='Reference Line (β=1)')

        ax.set_xlabel(f'{self.market_col} (Market Returns)', fontsize=12)
        ax.set_ylabel(f'{self.asset_col} (Asset Returns)', fontsize=12)
        ax.set_title(f'CAPM Beta Estimation: {self.asset_col}\n'
                    f'β={self.beta:.4f}, α={self.alpha:.6f}, R²={self.r_squared:.4f}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add text box with statistics
        textstr = f'Observations: {len(self.data)}\n'
        textstr += f'Correlation: {self.data[self.asset_col].corr(self.data[self.market_col]):.4f}\n'
        textstr += f'95% CI: [{self.model.conf_int()[1][0]:.4f}, {self.model.conf_int()[1][1]:.4f}]'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")

        plt.show()

    def plot_time_series(self, save_path: Optional[str] = None):
        """
        Plot time series of asset and market returns.

        Args:
            save_path: Optional path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Asset returns
        ax1.plot(self.data[self.date_col], self.data[self.asset_col],
                linewidth=0.8, label=self.asset_col)
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.set_ylabel('Returns', fontsize=12)
        ax1.set_title(f'{self.asset_col} Returns Over Time', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Market returns
        ax2.plot(self.data[self.date_col], self.data[self.market_col],
                linewidth=0.8, label=self.market_col, color='orange')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Returns', fontsize=12)
        ax2.set_title(f'{self.market_col} Returns Over Time', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")

        plt.show()


def main():
    """
    Main function demonstrating CAPM beta estimation for Tesla.
    """
    print("\n" + "="*70)
    print("CAPM BETA ESTIMATION - TESLA EXAMPLE")
    print("="*70 + "\n")

    # Define paths
    data_path = '../TSLA_daily.csv'

    # Check if file exists
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure TSLA_daily.csv is in the repository root.")
        return

    try:
        # Initialize analyzer
        print("Initializing CAPM Analyzer...")
        analyzer = CAPMAnalyzer(
            data_path=data_path,
            asset_col='TSLA',
            market_col='SPX_TR',
            date_col='Date'
        )

        # Calculate beta using regression
        print("Calculating beta using OLS regression...")
        analyzer.calculate_beta_regression()

        # Print results
        analyzer.print_results()

        # Create visualizations
        print("Generating visualizations...")
        analyzer.plot_scatter_with_regression()
        analyzer.plot_time_series()

        # Compare with covariance method
        beta_cov = analyzer.calculate_beta_covariance()
        print(f"Beta (covariance method): {beta_cov:.4f}")
        print(f"Beta (regression method):  {analyzer.beta:.4f}")
        print(f"Difference: {abs(beta_cov - analyzer.beta):.6f} (should be negligible)")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
