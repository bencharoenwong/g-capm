"""
g-capm: Tools for CAPM and Factor Model Analysis

A Python package for quantitative finance education, specifically focused on
CAPM beta estimation, factor models (Fama-French, Carhart), and portfolio analysis.

Key features:
- Clean data loading with validation
- CAPM beta estimation with confidence intervals
- Factor model regression (3-factor, 4-factor)
- Time-varying correlation analysis (crypto focus)
- Portfolio optimization
- AI-enhanced learning helpers

Author: g-capm Project
Date: 2025-01-14
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "g-capm Project"

# Import key functions for easy access
from .data import load_data, list_datasets, DataLoader
from .capm import calculate_beta, estimate_capm, CAPMAnalyzer
from .stats import rolling_correlation, time_varying_beta, statistical_tests
from .timeseries_tests import (
    test_stationarity,
    test_autocorrelation,
    test_heteroscedasticity,
    check_all_assumptions,
    StationarityTestResult,
)

# Factors and portfolio coming in later phases
# from .factors import FactorAnalyzer, fama_french_3factor, carhart_4factor

__all__ = [
    # Data
    'load_data',
    'list_datasets',
    'DataLoader',
    # CAPM
    'calculate_beta',
    'estimate_capm',
    'CAPMAnalyzer',
    # Stats
    'rolling_correlation',
    'time_varying_beta',
    'statistical_tests',
    # Time series tests (for research rigor)
    'test_stationarity',
    'test_autocorrelation',
    'test_heteroscedasticity',
    'check_all_assumptions',
    'StationarityTestResult',
]
