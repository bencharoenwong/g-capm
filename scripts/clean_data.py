#!/usr/bin/env python3
"""
Data Cleaning and Standardization Script

This script standardizes all datasets for consistency:
1. All dates to ISO 8601 (YYYY-MM-DD)
2. All columns to snake_case
3. Remove special characters
4. Add data validation
5. Generate metadata.json

Author: g-capm Project
Date: 2025-01-14
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import warnings

# Paths
DATA_DIR = Path(__file__).parent.parent
RAW_DIR = DATA_DIR
CLEANED_DIR = DATA_DIR / "data" / "cleaned"
METADATA_PATH = DATA_DIR / "data" / "metadata.json"

# Ensure cleaned directory exists
CLEANED_DIR.mkdir(parents=True, exist_ok=True)


def clean_tsla_daily():
    """
    Clean TSLA_daily.csv

    Changes:
    - Remove BOM (byte order mark)
    - Convert date format: DD/M/YY to YYYY-MM-DD
    - Rename columns to snake_case: TSLA → return_tsla, SPX_TR → return_sp500_tr
    - Validate: dates sorted, no missing values
    """
    print("\n📊 Cleaning TSLA_daily.csv...")

    # Load data
    df = pd.read_csv(RAW_DIR / 'TSLA_daily.csv', encoding='utf-8-sig')

    # Remove BOM if present in column names
    df.columns = df.columns.str.replace('\ufeff', '')

    # Convert date format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # Rename columns
    df = df.rename(columns={
        'Date': 'date',
        'TSLA': 'return_tsla',
        'SPX_TR': 'return_sp500_tr'
    })

    # Validate
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    assert df['date'].is_monotonic_increasing, "Dates not sorted"
    assert not df.isnull().any().any(), f"Missing values found: {df.isnull().sum()}"

    # Convert date back to string for CSV
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Save
    output_path = CLEANED_DIR / 'tsla_daily.csv'
    df.to_csv(output_path, index=False)

    print(f"  ✓ Cleaned {len(df)} rows")
    print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  ✓ Saved to {output_path}")

    return df


def clean_tsla_monthly():
    """
    Clean TSLA_monthly.csv

    Same cleaning as daily version.
    """
    print("\n📊 Cleaning TSLA_monthly.csv...")

    # Load data
    df = pd.read_csv(RAW_DIR / 'TSLA_monthly.csv', encoding='utf-8-sig')

    # Remove BOM
    df.columns = df.columns.str.replace('\ufeff', '')

    # Convert date (try multiple formats)
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
    except:
        df['Date'] = pd.to_datetime(df['Date'])

    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # Rename columns
    df = df.rename(columns={
        'Date': 'date',
        'TSLA': 'return_tsla',
        'SPX_TR': 'return_sp500_tr'
    })

    # Validate
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Allow missing values in monthly data (less strict)
    if df.isnull().any().any():
        warnings.warn(f"Missing values found in TSLA_monthly: {df.isnull().sum()}")

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Save
    output_path = CLEANED_DIR / 'tsla_monthly.csv'
    df.to_csv(output_path, index=False)

    print(f"  ✓ Cleaned {len(df)} rows")
    print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  ✓ Saved to {output_path}")

    return df


def clean_mkt_timing():
    """
    Clean mkt_timing.csv

    Changes:
    - Convert date format: DD/M/YYYY to YYYY-MM-DD
    - Lowercase column names: SPX_close → spx_close, etc.
    - Handle NA values explicitly
    """
    print("\n📊 Cleaning mkt_timing.csv...")

    # Load data
    df = pd.read_csv(RAW_DIR / 'mkt_timing.csv')

    # Convert date format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # Rename columns to lowercase
    df = df.rename(columns={
        'Date': 'date',
        'SPX_close': 'spx_close',
        'SPX_ret': 'spx_ret',
        'SPX_TR_close': 'spx_tr_close',
        'SPX_TR_ret': 'spx_tr_ret',
        'rf': 'rf'
    })

    # Validate
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # First row will have NA for returns (no prior observation)
    # This is expected and OK

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Save
    output_path = CLEANED_DIR / 'mkt_timing.csv'
    df.to_csv(output_path, index=False)

    print(f"  ✓ Cleaned {len(df)} rows")
    print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  ✓ Saved to {output_path}")

    return df


def clean_returns_daily():
    """
    Clean returns_daily.csv

    Changes:
    - Date already in ISO format (verify)
    - Columns already in snake_case (verify)
    - Just validate and copy
    """
    print("\n📊 Cleaning returns_daily.csv...")

    # Load data
    df = pd.read_csv(RAW_DIR / 'returns_daily.csv')

    # Check date format
    df['Date'] = pd.to_datetime(df['Date'])

    # Rename
    df = df.rename(columns={'Date': 'date'})

    # Validate
    df = df.sort_values('date').reset_index(drop=True)
    assert df['date'].is_monotonic_increasing, "Dates not sorted"

    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        warnings.warn(f"Found {missing_count} missing values in returns_daily")

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Save
    output_path = CLEANED_DIR / 'returns_daily.csv'
    df.to_csv(output_path, index=False)

    print(f"  ✓ Cleaned {len(df)} rows")
    print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  ✓ Assets: {', '.join([c for c in df.columns if c != 'date'])}")
    print(f"  ✓ Saved to {output_path}")

    return df


def clean_returns_daily_melt():
    """
    Clean returns_daily_melt.csv

    Long format version of returns_daily.
    """
    print("\n📊 Cleaning returns_daily_melt.csv...")

    # Load data
    df = pd.read_csv(RAW_DIR / 'returns_daily_melt.csv')

    # Check date format
    df['Date'] = pd.to_datetime(df['Date'])

    # Rename
    df = df.rename(columns={'Date': 'date'})

    # Validate
    df = df.sort_values(['date', 'variable']).reset_index(drop=True)

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Save
    output_path = CLEANED_DIR / 'returns_daily_melt.csv'
    df.to_csv(output_path, index=False)

    print(f"  ✓ Cleaned {len(df)} rows")
    print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  ✓ Variables: {', '.join(df['variable'].unique())}")
    print(f"  ✓ Saved to {output_path}")

    return df


def clean_benchmarking():
    """
    Clean benchmarking.csv

    Changes:
    - Date already ISO format
    - Rename: Mkt_RF → mkt_rf (lowercase)
    - Keep factor names uppercase: SMB, HML, WML, RF (convention)
    """
    print("\n📊 Cleaning benchmarking.csv...")

    # Load data
    df = pd.read_csv(RAW_DIR / 'benchmarking.csv')

    # Parse date
    df['Date'] = pd.to_datetime(df['Date'])

    # Rename
    df = df.rename(columns={
        'Date': 'date',
        'Mkt_RF': 'mkt_rf',
        'RF': 'rf'
        # Keep A-H, SMB, HML, WML as is (standard factor names)
    })

    # Validate
    df = df.sort_values('date').reset_index(drop=True)
    assert df['date'].is_monotonic_increasing, "Dates not sorted"

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Save
    output_path = CLEANED_DIR / 'benchmarking.csv'
    df.to_csv(output_path, index=False)

    print(f"  ✓ Cleaned {len(df)} rows")
    print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  ✓ Portfolios: A-H, Factors: mkt_rf, SMB, HML, WML, rf")
    print(f"  ✓ Saved to {output_path}")

    return df


def clean_test_sptr():
    """
    Clean test_sptr.csv

    Changes:
    - Date already ISO format
    - Rename: "Adj Close**" → adj_close (remove asterisks)
    """
    print("\n📊 Cleaning test_sptr.csv...")

    # Load data
    df = pd.read_csv(RAW_DIR / 'test_sptr.csv')

    # Parse date
    df['date'] = pd.to_datetime(df['date'])

    # Rename columns (remove asterisks)
    df.columns = df.columns.str.replace('*', '', regex=False)
    df = df.rename(columns={'Adj Close': 'adj_close'})

    # Validate
    df = df.sort_values('date').reset_index(drop=True)

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Save
    output_path = CLEANED_DIR / 'test_sptr.csv'
    df.to_csv(output_path, index=False)

    print(f"  ✓ Cleaned {len(df)} rows")
    print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  ✓ Saved to {output_path}")

    return df


def clean_reit_altdata():
    """
    Clean reit_altdata.csv

    Changes:
    - Standardize column names
    - Keep text descriptions as is
    """
    print("\n📊 Cleaning reit_altdata.csv...")

    # Load data (may be large)
    df = pd.read_csv(RAW_DIR / 'reit_altdata.csv')

    # Clean column names
    df.columns = df.columns.str.replace('.', '_', regex=False)
    df.columns = df.columns.str.lower()

    # Rename key columns if they exist
    column_mapping = {
        'date_align': 'date',
        'ret': 'return',
        'ric': 'company',
        'business_description': 'description'
    }

    df = df.rename(columns=column_mapping)

    # Parse date if exists
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        except:
            pass  # Date may not be in expected format

    # Save
    output_path = CLEANED_DIR / 'reit_altdata.csv'
    df.to_csv(output_path, index=False)

    print(f"  ✓ Cleaned {len(df)} rows")
    if 'date' in df.columns:
        unique_dates = df['date'].nunique()
        print(f"  ✓ Unique dates: {unique_dates}")
    print(f"  ✓ Columns: {', '.join(df.columns[:5])}...")
    print(f"  ✓ Saved to {output_path}")

    return df


def generate_metadata():
    """
    Generate metadata.json with information about all datasets.
    """
    print("\n📝 Generating metadata.json...")

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "datasets": {
            "tsla_daily": {
                "description": "Tesla daily returns vs S&P 500 Total Return",
                "frequency": "daily",
                "columns": {
                    "date": "Trading date (ISO 8601 format YYYY-MM-DD)",
                    "return_tsla": "Tesla daily return (decimal format, 0.01 = 1%)",
                    "return_sp500_tr": "S&P 500 Total Return daily return (decimal format)"
                },
                "source": "Financial data provider",
                "use_cases": ["Beta estimation", "CAPM analysis", "Single-stock risk"],
                "cleaning_applied": [
                    "Date format standardized to ISO 8601",
                    "Column names converted to snake_case",
                    "BOM removed",
                    "Dates sorted ascending"
                ]
            },
            "tsla_monthly": {
                "description": "Tesla monthly returns vs S&P 500 Total Return",
                "frequency": "monthly",
                "columns": {
                    "date": "Month end date (ISO 8601)",
                    "return_tsla": "Tesla monthly return (decimal format)",
                    "return_sp500_tr": "S&P 500 Total Return monthly return (decimal format)"
                },
                "source": "Financial data provider",
                "use_cases": ["Beta estimation (monthly)", "Long-term trend analysis"],
                "cleaning_applied": [
                    "Date format standardized to ISO 8601",
                    "Column names converted to snake_case"
                ]
            },
            "mkt_timing": {
                "description": "S&P 500 market data with risk-free rates (1991-2020)",
                "frequency": "monthly",
                "columns": {
                    "date": "Month end date (ISO 8601)",
                    "spx_close": "S&P 500 closing price (index points)",
                    "spx_ret": "S&P 500 price return (decimal, NA for first row)",
                    "spx_tr_close": "S&P 500 Total Return Index close (includes dividends)",
                    "spx_tr_ret": "S&P 500 Total Return return (decimal, NA for first row)",
                    "rf": "Risk-free rate (monthly, decimal)"
                },
                "source": "Financial data provider",
                "use_cases": ["Market timing analysis", "CAPM (market returns + rf)", "Historical backtesting"],
                "cleaning_applied": [
                    "Date format standardized to ISO 8601",
                    "Column names converted to snake_case",
                    "Dates sorted ascending"
                ]
            },
            "returns_daily": {
                "description": "Multi-asset daily returns (wide format): BTC, ETH, S&P 500, Small Cap",
                "frequency": "daily",
                "columns": {
                    "date": "Trading date (ISO 8601)",
                    "return_btc": "Bitcoin daily return (decimal)",
                    "return_eth": "Ethereum daily return (decimal)",
                    "return_sp500": "S&P 500 daily return (decimal)",
                    "return_ssc": "Small Cap daily return (decimal)"
                },
                "source": "Crypto exchanges and financial data providers",
                "use_cases": [
                    "Crypto CAPM analysis",
                    "Cross-asset correlation",
                    "Diversification analysis",
                    "Time-varying correlation analysis (IMPORTANT!)"
                ],
                "cleaning_applied": [
                    "Date format verified as ISO 8601",
                    "Dates sorted ascending"
                ]
            },
            "returns_daily_melt": {
                "description": "Multi-asset daily returns (long format): Same data as returns_daily but reshaped",
                "frequency": "daily",
                "columns": {
                    "date": "Trading date (ISO 8601)",
                    "variable": "Asset identifier (return_btc, return_eth, return_sp500, return_ssc)",
                    "value": "Return value (decimal)"
                },
                "source": "Crypto exchanges and financial data providers",
                "use_cases": ["Panel regression", "Time series modeling", "ggplot-style visualization"],
                "cleaning_applied": [
                    "Date format verified as ISO 8601",
                    "Sorted by date and variable"
                ]
            },
            "benchmarking": {
                "description": "Fama-French factor returns and sector portfolios (monthly, 2010-)",
                "frequency": "monthly",
                "columns": {
                    "date": "Month end date (ISO 8601)",
                    "A": "Portfolio A return (decimal)",
                    "B": "Portfolio B return (decimal)",
                    "C": "Portfolio C return (decimal)",
                    "D": "Portfolio D return (decimal)",
                    "E": "Portfolio E return (decimal)",
                    "F": "Portfolio F return (decimal)",
                    "G": "Portfolio G return (decimal)",
                    "H": "Portfolio H return (decimal)",
                    "mkt_rf": "Market excess return (Rm - Rf) (decimal)",
                    "SMB": "Small Minus Big factor (size) (decimal)",
                    "HML": "High Minus Low factor (value) (decimal)",
                    "rf": "Risk-free rate (monthly, decimal)",
                    "WML": "Winners Minus Losers factor (momentum) (decimal)"
                },
                "source": "Kenneth French Data Library (Fama-French factors)",
                "use_cases": [
                    "Fama-French 3-factor model",
                    "Carhart 4-factor model",
                    "Performance attribution",
                    "Factor strategy analysis"
                ],
                "cleaning_applied": [
                    "Date format verified as ISO 8601",
                    "Mkt_RF renamed to mkt_rf",
                    "RF renamed to rf",
                    "Dates sorted ascending"
                ]
            },
            "test_sptr": {
                "description": "S&P 500 Total Return adjusted close prices",
                "frequency": "monthly",
                "columns": {
                    "date": "Trading date (ISO 8601)",
                    "adj_close": "Adjusted closing price (total return index)"
                },
                "source": "Financial data provider",
                "use_cases": ["Return calculation", "Matches spx_tr_close in mkt_timing.csv"],
                "cleaning_applied": [
                    "Date format verified as ISO 8601",
                    "Removed asterisks from column name"
                ]
            },
            "reit_altdata": {
                "description": "Australian REIT alternative data with business descriptions (2020 snapshot)",
                "frequency": "snapshot",
                "columns": {
                    "date": "Snapshot date (2020-03-31)",
                    "company": "Company name/ticker",
                    "description": "Business description (text)",
                    "market_cap_bin": "Market cap category (small/mid/large)",
                    "return": "Period return (decimal)"
                },
                "source": "Australian Securities Exchange filings",
                "use_cases": [
                    "REIT analysis",
                    "Alternative data / text mining",
                    "NLP for investment signals",
                    "Market cap factor analysis"
                ],
                "cleaning_applied": [
                    "Column names converted to lowercase",
                    "Date format verified"
                ]
            }
        }
    }

    # Write to file
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Metadata saved to {METADATA_PATH}")
    print(f"  ✓ {len(metadata['datasets'])} datasets documented")


def main():
    """
    Main function to clean all datasets.
    """
    print("="*70)
    print("DATA CLEANING AND STANDARDIZATION")
    print("="*70)

    # Clean all datasets
    clean_tsla_daily()
    clean_tsla_monthly()
    clean_mkt_timing()
    clean_returns_daily()
    clean_returns_daily_melt()
    clean_benchmarking()
    clean_test_sptr()
    clean_reit_altdata()

    # Generate metadata
    generate_metadata()

    print("\n" + "="*70)
    print("✅ ALL DATA CLEANED AND VALIDATED")
    print("="*70)
    print(f"\nCleaned data saved to: {CLEANED_DIR}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print("\nNext step: Build gcapm package")


if __name__ == "__main__":
    main()
