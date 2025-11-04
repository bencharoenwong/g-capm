#!/usr/bin/env python3
"""
Fama-French Factor Model Analysis

This script performs multi-factor regression analysis using the Fama-French
three-factor model and Carhart four-factor model. It provides comprehensive
statistical analysis with robust error handling.

Three-Factor Model (Fama-French):
    R_i - R_f = α + β₁(R_m - R_f) + β₂(SMB) + β₃(HML) + ε

Four-Factor Model (Carhart):
    R_i - R_f = α + β₁(R_m - R_f) + β₂(SMB) + β₃(HML) + β₄(WML) + ε

Where:
    R_i = Portfolio/asset return
    R_f = Risk-free rate
    R_m = Market return
    SMB = Small Minus Big (size factor)
    HML = High Minus Low (value factor)
    WML = Winners Minus Losers (momentum factor)
    α = Alpha (excess return)

Author: g-capm Project
Date: 2025-11-04
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class FactorModelAnalyzer:
    """
    Comprehensive class for Fama-French factor model analysis.

    Attributes:
        data (pd.DataFrame): Factor and portfolio return data
        portfolio_cols (List[str]): List of portfolio columns to analyze
        results (Dict): Dictionary storing regression results for each portfolio
    """

    def __init__(self, data_path: str, portfolio_cols: List[str] = None):
        """
        Initialize the factor model analyzer.

        Args:
            data_path: Path to the CSV file containing factor data
            portfolio_cols: List of portfolio column names (default: A-H)

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If required factor columns are missing
        """
        self.data_path = Path(data_path)
        self.portfolio_cols = portfolio_cols or ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.results = {}

        # Load and validate data
        self._load_data()
        self._validate_data()
        self._calculate_excess_returns()

    def _load_data(self):
        """Load factor data from CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        try:
            self.data = pd.read_csv(self.data_path, parse_dates=['Date'])
            print(f"✓ Loaded {len(self.data)} observations from {self.data_path.name}")
            print(f"  Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        except Exception as e:
            raise IOError(f"Error loading data: {str(e)}")

    def _validate_data(self):
        """Validate that required factor columns exist."""
        required_factors = ['Mkt_RF', 'SMB', 'HML', 'RF', 'WML']
        missing = [col for col in required_factors if col not in self.data.columns]

        if missing:
            raise ValueError(f"Missing required factor columns: {missing}\n"
                           f"Available columns: {list(self.data.columns)}")

        # Check for missing portfolios
        missing_portfolios = [col for col in self.portfolio_cols
                             if col not in self.data.columns]
        if missing_portfolios:
            warnings.warn(f"Some portfolios not found: {missing_portfolios}")
            self.portfolio_cols = [col for col in self.portfolio_cols
                                   if col in self.data.columns]

        # Check for missing values
        for factor in required_factors + self.portfolio_cols:
            n_missing = self.data[factor].isna().sum()
            if n_missing > 0:
                warnings.warn(f"Found {n_missing} missing values in {factor}")

        print(f"✓ Validated {len(self.portfolio_cols)} portfolios and factor data")

    def _calculate_excess_returns(self):
        """Calculate excess returns for all portfolios."""
        for portfolio in self.portfolio_cols:
            excess_col = f"{portfolio}_excess"
            self.data[excess_col] = self.data[portfolio] - self.data['RF']

        print(f"✓ Calculated excess returns for all portfolios")

    def run_three_factor_model(self, portfolio: str) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Run Fama-French three-factor model regression.

        Model: R_i - R_f = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + ε

        Args:
            portfolio: Name of portfolio column

        Returns:
            Statsmodels regression results
        """
        # Prepare data
        y = self.data[f"{portfolio}_excess"].dropna()
        X = self.data.loc[y.index, ['Mkt_RF', 'SMB', 'HML']]
        X = sm.add_constant(X)

        # Run regression
        model = sm.OLS(y, X, missing='drop').fit()

        return model

    def run_four_factor_model(self, portfolio: str) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Run Carhart four-factor model regression (adds momentum).

        Model: R_i - R_f = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + β₄(WML) + ε

        Args:
            portfolio: Name of portfolio column

        Returns:
            Statsmodels regression results
        """
        # Prepare data
        y = self.data[f"{portfolio}_excess"].dropna()
        X = self.data.loc[y.index, ['Mkt_RF', 'SMB', 'HML', 'WML']]
        X = sm.add_constant(X)

        # Run regression
        model = sm.OLS(y, X, missing='drop').fit()

        return model

    def analyze_portfolio(self, portfolio: str, verbose: bool = True) -> Dict:
        """
        Perform comprehensive factor analysis for a single portfolio.

        Args:
            portfolio: Name of portfolio column
            verbose: Whether to print detailed results

        Returns:
            Dictionary containing analysis results
        """
        if portfolio not in self.portfolio_cols:
            raise ValueError(f"Portfolio {portfolio} not in available portfolios: "
                           f"{self.portfolio_cols}")

        # Run both models
        model_3f = self.run_three_factor_model(portfolio)
        model_4f = self.run_four_factor_model(portfolio)

        # Store results
        results = {
            'portfolio': portfolio,
            'model_3f': model_3f,
            'model_4f': model_4f,
            'alpha_3f': model_3f.params['const'],
            'alpha_4f': model_4f.params['const'],
            'beta_market_3f': model_3f.params['Mkt_RF'],
            'beta_market_4f': model_4f.params['Mkt_RF'],
            'beta_smb_3f': model_3f.params['SMB'],
            'beta_smb_4f': model_4f.params['SMB'],
            'beta_hml_3f': model_3f.params['HML'],
            'beta_hml_4f': model_4f.params['HML'],
            'beta_wml_4f': model_4f.params['WML'],
            'r_squared_3f': model_3f.rsquared,
            'r_squared_4f': model_4f.rsquared,
            'adj_r_squared_3f': model_3f.rsquared_adj,
            'adj_r_squared_4f': model_4f.rsquared_adj,
        }

        self.results[portfolio] = results

        if verbose:
            self._print_portfolio_results(results)

        return results

    def analyze_all_portfolios(self, verbose: bool = False) -> pd.DataFrame:
        """
        Analyze all portfolios and return summary DataFrame.

        Args:
            verbose: Whether to print results for each portfolio

        Returns:
            DataFrame with factor loadings and statistics for all portfolios
        """
        print(f"\nAnalyzing {len(self.portfolio_cols)} portfolios...")

        for portfolio in self.portfolio_cols:
            self.analyze_portfolio(portfolio, verbose=verbose)

        # Create summary DataFrame
        summary_df = self._create_summary_dataframe()

        return summary_df

    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame from all portfolio results."""
        summary_data = []

        for portfolio, results in self.results.items():
            row = {
                'Portfolio': portfolio,
                'Alpha_3F': results['alpha_3f'],
                'Alpha_4F': results['alpha_4f'],
                'Mkt_Beta_3F': results['beta_market_3f'],
                'Mkt_Beta_4F': results['beta_market_4f'],
                'SMB_3F': results['beta_smb_3f'],
                'SMB_4F': results['beta_smb_4f'],
                'HML_3F': results['beta_hml_3f'],
                'HML_4F': results['beta_hml_4f'],
                'WML_4F': results['beta_wml_4f'],
                'R²_3F': results['r_squared_3f'],
                'R²_4F': results['r_squared_4f'],
                'Adj_R²_3F': results['adj_r_squared_3f'],
                'Adj_R²_4F': results['adj_r_squared_4f'],
            }
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        return summary_df

    def _print_portfolio_results(self, results: Dict):
        """Print formatted results for a single portfolio."""
        portfolio = results['portfolio']

        print("\n" + "="*70)
        print(f"FACTOR ANALYSIS: PORTFOLIO {portfolio}")
        print("="*70)

        print("\n" + "-"*70)
        print("THREE-FACTOR MODEL (Fama-French)")
        print("-"*70)
        print(f"Model: R_{portfolio} - R_f = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + ε\n")

        print(f"Alpha (α):         {results['alpha_3f']:>10.6f}  ({results['alpha_3f']*100:.4f}%)")
        print(f"Market Beta (β₁):  {results['beta_market_3f']:>10.4f}")
        print(f"SMB Beta (β₂):     {results['beta_smb_3f']:>10.4f}")
        print(f"HML Beta (β₃):     {results['beta_hml_3f']:>10.4f}")
        print(f"R-squared:         {results['r_squared_3f']:>10.4f}  ({results['r_squared_3f']*100:.2f}%)")
        print(f"Adj. R-squared:    {results['adj_r_squared_3f']:>10.4f}  ({results['adj_r_squared_3f']*100:.2f}%)")

        print("\n" + "-"*70)
        print("FOUR-FACTOR MODEL (Carhart)")
        print("-"*70)
        print(f"Model: R_{portfolio} - R_f = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + β₄(WML) + ε\n")

        print(f"Alpha (α):         {results['alpha_4f']:>10.6f}  ({results['alpha_4f']*100:.4f}%)")
        print(f"Market Beta (β₁):  {results['beta_market_4f']:>10.4f}")
        print(f"SMB Beta (β₂):     {results['beta_smb_4f']:>10.4f}")
        print(f"HML Beta (β₃):     {results['beta_hml_4f']:>10.4f}")
        print(f"WML Beta (β₄):     {results['beta_wml_4f']:>10.4f}")
        print(f"R-squared:         {results['r_squared_4f']:>10.4f}  ({results['r_squared_4f']*100:.2f}%)")
        print(f"Adj. R-squared:    {results['adj_r_squared_4f']:>10.4f}  ({results['adj_r_squared_4f']*100:.2f}%)")

        print("\n" + "-"*70)
        print("INTERPRETATION")
        print("-"*70)
        self._print_interpretation(results)

        print("\n" + "="*70)

    def _print_interpretation(self, results: Dict):
        """Print interpretation of factor loadings."""
        portfolio = results['portfolio']

        # Market beta interpretation
        beta_mkt = results['beta_market_4f']
        print(f"\n• Market Beta = {beta_mkt:.4f}")
        if beta_mkt > 1:
            print(f"  → Portfolio is MORE VOLATILE than market")
        elif beta_mkt < 1:
            print(f"  → Portfolio is LESS VOLATILE than market")

        # SMB interpretation
        beta_smb = results['beta_smb_4f']
        print(f"\n• SMB Beta = {beta_smb:.4f}")
        if beta_smb > 0:
            print(f"  → Portfolio tilts toward SMALL-CAP stocks")
        else:
            print(f"  → Portfolio tilts toward LARGE-CAP stocks")

        # HML interpretation
        beta_hml = results['beta_hml_4f']
        print(f"\n• HML Beta = {beta_hml:.4f}")
        if beta_hml > 0:
            print(f"  → Portfolio tilts toward VALUE stocks (high book-to-market)")
        else:
            print(f"  → Portfolio tilts toward GROWTH stocks (low book-to-market)")

        # WML interpretation
        beta_wml = results['beta_wml_4f']
        print(f"\n• WML Beta = {beta_wml:.4f}")
        if beta_wml > 0:
            print(f"  → Portfolio exhibits MOMENTUM (follows recent winners)")
        else:
            print(f"  → Portfolio exhibits REVERSAL (contrarian strategy)")

        # Alpha interpretation
        alpha = results['alpha_4f']
        p_value = results['model_4f'].pvalues['const']
        print(f"\n• Alpha = {alpha:.6f} ({alpha*100:.4f}%/month, p={p_value:.4f})")
        if p_value < 0.05:
            if alpha > 0:
                print(f"  → Portfolio generates SIGNIFICANT positive excess returns")
                print(f"  → Outperforms after adjusting for factor exposures")
            else:
                print(f"  → Portfolio generates SIGNIFICANT negative excess returns")
                print(f"  → Underperforms after adjusting for factor exposures")
        else:
            print(f"  → Alpha is NOT statistically significant")
            print(f"  → Returns are explained by factor exposures")

        # Model fit
        r2_improvement = (results['r_squared_4f'] - results['r_squared_3f']) * 100
        print(f"\n• Model Fit:")
        print(f"  → R² improves by {r2_improvement:.2f}% when adding momentum factor")

    def plot_factor_loadings(self, save_path: Optional[str] = None):
        """
        Create visualization of factor loadings across portfolios.

        Args:
            save_path: Optional path to save the figure
        """
        if not self.results:
            raise ValueError("No results to plot. Run analyze_all_portfolios() first.")

        # Prepare data for plotting
        portfolios = list(self.results.keys())
        loadings = {
            'Market': [self.results[p]['beta_market_4f'] for p in portfolios],
            'SMB': [self.results[p]['beta_smb_4f'] for p in portfolios],
            'HML': [self.results[p]['beta_hml_4f'] for p in portfolios],
            'WML': [self.results[p]['beta_wml_4f'] for p in portfolios],
        }

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Factor Loadings Across Portfolios (Four-Factor Model)',
                    fontsize=16, fontweight='bold')

        factors = ['Market', 'SMB', 'HML', 'WML']
        titles = ['Market Beta', 'Size Factor (SMB)', 'Value Factor (HML)',
                 'Momentum Factor (WML)']

        for idx, (factor, title) in enumerate(zip(factors, titles)):
            ax = axes[idx // 2, idx % 2]

            bars = ax.bar(portfolios, loadings[factor], alpha=0.7)

            # Color bars based on positive/negative
            for i, bar in enumerate(bars):
                if loadings[factor][i] >= 0:
                    bar.set_color('steelblue')
                else:
                    bar.set_color('coral')

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_ylabel('Factor Loading', fontsize=11)
            ax.set_xlabel('Portfolio', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")

        plt.show()

    def plot_alpha_comparison(self, save_path: Optional[str] = None):
        """
        Plot comparison of alphas from three-factor and four-factor models.

        Args:
            save_path: Optional path to save the figure
        """
        if not self.results:
            raise ValueError("No results to plot. Run analyze_all_portfolios() first.")

        portfolios = list(self.results.keys())
        alpha_3f = [self.results[p]['alpha_3f'] * 100 for p in portfolios]  # Convert to %
        alpha_4f = [self.results[p]['alpha_4f'] * 100 for p in portfolios]

        x = np.arange(len(portfolios))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 7))

        bars1 = ax.bar(x - width/2, alpha_3f, width, label='3-Factor Model', alpha=0.8)
        bars2 = ax.bar(x + width/2, alpha_4f, width, label='4-Factor Model', alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('Alpha (%/month)', fontsize=12)
        ax.set_xlabel('Portfolio', fontsize=12)
        ax.set_title('Alpha Comparison: Three-Factor vs Four-Factor Model',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(portfolios)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")

        plt.show()


def main():
    """
    Main function demonstrating Fama-French factor analysis.
    """
    print("\n" + "="*70)
    print("FAMA-FRENCH FACTOR MODEL ANALYSIS")
    print("="*70 + "\n")

    # Define path
    data_path = '../benchmarking.csv'

    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure benchmarking.csv is in the repository root.")
        return

    try:
        # Initialize analyzer
        print("Initializing Factor Model Analyzer...")
        analyzer = FactorModelAnalyzer(
            data_path=data_path,
            portfolio_cols=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        )

        # Analyze all portfolios
        summary_df = analyzer.analyze_all_portfolios(verbose=False)

        # Print summary table
        print("\n" + "="*70)
        print("SUMMARY: ALL PORTFOLIOS")
        print("="*70 + "\n")

        print("Four-Factor Model Results:")
        print(summary_df[['Portfolio', 'Alpha_4F', 'Mkt_Beta_4F', 'SMB_4F',
                          'HML_4F', 'WML_4F', 'R²_4F']].to_string(index=False))

        # Detailed analysis of one portfolio
        print("\n\nDetailed Analysis of Portfolio A:")
        analyzer.analyze_portfolio('A', verbose=True)

        # Create visualizations
        print("\nGenerating visualizations...")
        analyzer.plot_factor_loadings()
        analyzer.plot_alpha_comparison()

        print("\n✓ Analysis complete!")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
