# g-capm: Capital Asset Pricing Model Data Repository

A comprehensive collection of financial datasets for Capital Asset Pricing Model (CAPM) analysis, factor modeling, and portfolio performance evaluation.

## 📋 Overview

This repository contains curated financial datasets covering various asset classes, market indices, and risk factors. The data is organized to support:

- **CAPM Analysis**: Beta estimation and expected return calculations
- **Factor Modeling**: Fama-French three-factor and momentum factor analysis
- **Market Timing**: Risk-free rate analysis and market return evaluation
- **Alternative Assets**: Cryptocurrency returns and REIT portfolio analysis

## 📊 Datasets

### Core Market Data

| Dataset | Description | Date Range | Use Case |
|---------|-------------|------------|----------|
| `mkt_timing.csv` | S&P 500 prices, returns, and risk-free rates | 1991-2020 | Market timing, CAPM analysis |
| `test_sptr.csv` | S&P 500 total return adjusted close | 1991-07-01 onwards | Benchmark returns |

### Equity Data

| Dataset | Description | Date Range | Use Case |
|---------|-------------|------------|----------|
| `TSLA_daily.csv` | Tesla daily returns vs S&P 500 | Recent history | Beta estimation, single-stock CAPM |
| `TSLA_monthly.csv` | Tesla monthly returns | 60 months | Monthly beta estimation |

### Multi-Asset Returns

| Dataset | Description | Date Range | Use Case |
|---------|-------------|------------|----------|
| `returns_daily.csv` | Daily returns: BTC, ETH, S&P 500, SSC (wide format) | 2016-03-10 onwards | Cross-asset analysis |
| `returns_daily_melt.csv` | Same as above (long format) | 2016-03-10 onwards | Statistical modeling, time series analysis |

### Factor Data

| Dataset | Description | Date Range | Use Case |
|---------|-------------|------------|----------|
| `benchmarking.csv` | Fama-French factors (Mkt-RF, SMB, HML, WML) + sector portfolios | 2010-04-01 onwards | Multi-factor modeling, performance attribution |

### Alternative Data

| Dataset | Description | Date Range | Use Case |
|---------|-------------|------------|----------|
| `reit_altdata.csv` | Australian REIT data with business descriptions | 2020-03-31 snapshot | Alternative data analysis, text mining |

## 🚀 Quick Start

### Prerequisites

```bash
# Install required Python packages
pip install -r requirements.txt
```

### Example Usage

```python
import pandas as pd
import numpy as np

# Load market timing data
mkt_data = pd.read_csv('mkt_timing.csv', parse_dates=['Date'])

# Load Tesla daily returns for beta estimation
tsla_data = pd.read_csv('TSLA_daily.csv', parse_dates=['Date'])

# Calculate beta using covariance method
covariance = tsla_data['TSLA'].cov(tsla_data['SPX_TR'])
market_variance = tsla_data['SPX_TR'].var()
beta = covariance / market_variance

print(f"Tesla Beta: {beta:.4f}")
```

See the `examples/` directory for more detailed analysis scripts.

## 📁 Data Dictionary

For detailed information about each dataset including:
- Column definitions
- Data types
- Units and formats
- Data sources
- Known limitations

Please refer to [DATA_DICTIONARY.md](DATA_DICTIONARY.md)

## 🔧 Analysis Scripts

This repository includes example analysis scripts demonstrating:

1. **`capm_beta_estimation.py`**: Calculate security betas and expected returns
2. **`factor_analysis.py`**: Perform Fama-French factor model analysis
3. **`market_timing_analysis.py`**: Evaluate market timing strategies
4. **`crypto_capm.py`**: Apply CAPM to cryptocurrency returns

Each script includes:
- Comprehensive documentation
- Error handling and data validation
- Visualization outputs
- Statistical test results

## 📈 Methodology Notes

### Beta Estimation
- Uses ordinary least squares (OLS) regression
- Handles missing data with forward/backward fill
- Includes statistical significance tests

### Factor Models
- Implements Fama-French three-factor model
- Includes momentum factor (WML)
- Provides alpha and factor loading estimates

### Risk-Free Rate
- Uses standard proxies (Treasury rates)
- Annualized rates provided in `mkt_timing.csv`

## 🛠️ Data Format Standards

All CSV files follow these conventions:
- **Date columns**: Format varies (ISO 8601 or DD/MM/YY)
- **Returns**: Decimal format (0.01 = 1%)
- **Prices**: Nominal values
- **Missing data**: Handled explicitly in analysis scripts

## 📝 Contributing

When adding new datasets:
1. Update this README with dataset description
2. Add detailed documentation to DATA_DICTIONARY.md
3. Include data sources and collection methodology
4. Provide example usage code

## 🔍 Data Sources

The data in this repository comes from various financial data providers and public sources:
- Market indices: Standard financial data providers
- Factor data: Kenneth French Data Library (Fama-French factors)
- Cryptocurrency: Major exchange APIs
- REIT data: Australian Securities Exchange and company filings

## ⚠️ Disclaimer

This data is provided for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always verify data accuracy and consult with qualified financial professionals before making investment decisions.

## 📧 Contact

For questions or issues regarding the data, please open an issue in this repository.

## 📄 License

This repository is provided for educational purposes. Please respect the terms of use for underlying data sources.

---

**Last Updated**: 2025-11-04
**Repository Status**: Active Development
