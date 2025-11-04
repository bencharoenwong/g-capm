# Data Dictionary

This document provides detailed information about each dataset in the g-capm repository, including column definitions, data types, units, and usage notes.

---

## Table of Contents

1. [mkt_timing.csv](#mkt_timingcsv)
2. [test_sptr.csv](#test_sptrcsv)
3. [TSLA_daily.csv](#tsla_dailycsv)
4. [TSLA_monthly.csv](#tsla_monthlycsv)
5. [returns_daily.csv](#returns_dailycsv)
6. [returns_daily_melt.csv](#returns_daily_meltcsv)
7. [benchmarking.csv](#benchmarkingcsv)
8. [reit_altdata.csv](#reit_altdatacsv)

---

## mkt_timing.csv

**Purpose**: Market timing analysis with S&P 500 data and risk-free rates

**Date Range**: 1991-07-01 to 2020

**Rows**: 363

**Size**: 21 KB

### Columns

| Column | Data Type | Description | Units | Notes |
|--------|-----------|-------------|-------|-------|
| `Date` | Date | Trading date | DD/M/YYYY | Monthly frequency |
| `SPX_close` | Float | S&P 500 closing price | USD | Price index level |
| `SPX_ret` | Float | S&P 500 price return | Decimal | 0.01 = 1%, NA for first observation |
| `SPX_TR_close` | Float | S&P 500 Total Return Index close | Index points | Includes dividends |
| `SPX_TR_ret` | Float | S&P 500 Total Return return | Decimal | 0.01 = 1%, NA for first observation |
| `rf` | Float | Risk-free rate | Decimal (monthly) | Typically 1-month Treasury rate |

### Usage Notes
- **SPX_ret**: Calculated as (SPX_close[t] - SPX_close[t-1]) / SPX_close[t-1]
- **SPX_TR_ret**: More accurate for CAPM as it includes dividend reinvestment
- **rf**: Use this as the risk-free rate in CAPM calculations
- **Missing Values**: First row has NA for returns (no prior observation)

### Example Usage
```python
# Calculate excess returns for CAPM
df = pd.read_csv('mkt_timing.csv', parse_dates=['Date'])
df['excess_return'] = df['SPX_TR_ret'] - df['rf']
```

---

## test_sptr.csv

**Purpose**: S&P 500 Total Return adjusted closing prices

**Date Range**: 1991-07-01 onwards

**Rows**: 363

**Size**: 6.7 KB

### Columns

| Column | Data Type | Description | Units | Notes |
|--------|-----------|-------------|-------|-------|
| `date` | Date | Trading date | ISO 8601 | Format varies |
| `Adj Close**` | Float | Adjusted closing price | Index points | Total return index |

### Usage Notes
- Matches the SPX_TR_close column in mkt_timing.csv
- Use for calculating returns manually
- Double asterisk in column name is intentional (from source data)

---

## TSLA_daily.csv

**Purpose**: Tesla daily returns vs S&P 500 for beta estimation

**Date Range**: 2016-09-14 onwards

**Rows**: 1,259

**Size**: 42 KB

### Columns

| Column | Data Type | Description | Units | Notes |
|--------|-----------|-------------|-------|-------|
| `Date` | Date | Trading date | DD/M/YY | Daily frequency |
| `TSLA` | Float | Tesla daily return | Decimal | 0.01 = 1% |
| `SPX_TR` | Float | S&P 500 Total Return daily return | Decimal | Market benchmark |

### Usage Notes
- Pre-aligned dates for easy regression analysis
- Returns are already calculated (not prices)
- Use for calculating Tesla's beta: β = Cov(R_TSLA, R_Market) / Var(R_Market)
- Note the date format differs from other files

### Example Usage
```python
# Calculate beta
df = pd.read_csv('TSLA_daily.csv')
beta = df['TSLA'].cov(df['SPX_TR']) / df['SPX_TR'].var()
print(f"Tesla Beta: {beta:.4f}")
```

---

## TSLA_monthly.csv

**Purpose**: Tesla monthly returns aggregated from daily data

**Date Range**: 60 months of data

**Rows**: 60

**Size**: 2.2 KB

### Columns

| Column | Data Type | Description | Units | Notes |
|--------|-----------|-------------|-------|-------|
| `Date` | Date | Month end date | Various formats | Monthly frequency |
| `TSLA` | Float | Tesla monthly return | Decimal | Compounded from daily |
| `SPX_TR` | Float | S&P 500 Total Return monthly return | Decimal | Market benchmark |

### Usage Notes
- Longer time period (monthly) may provide more stable beta estimates
- Lower frequency reduces noise but loses information
- Better for long-term strategic analysis

---

## returns_daily.csv

**Purpose**: Multi-asset daily returns (wide format)

**Date Range**: 2016-03-10 onwards

**Rows**: 1,296

**Size**: 69 KB

### Columns

| Column | Data Type | Description | Units | Notes |
|--------|-----------|-------------|-------|-------|
| `Date` | Date | Trading date | ISO 8601 | Daily frequency |
| `return_btc` | Float | Bitcoin daily return | Decimal | Crypto asset |
| `return_eth` | Float | Ethereum daily return | Decimal | Crypto asset |
| `return_sp500` | Float | S&P 500 daily return | Decimal | Traditional market |
| `return_ssc` | Float | SSC (Small Cap) daily return | Decimal | Size factor proxy |

### Usage Notes
- **Wide format**: Each asset is a separate column
- Useful for correlation analysis and portfolio optimization
- Crypto returns exhibit much higher volatility than equity returns
- All assets aligned by date for direct comparison

### Example Usage
```python
# Calculate correlation matrix
df = pd.read_csv('returns_daily.csv', parse_dates=['Date'])
returns = df[['return_btc', 'return_eth', 'return_sp500', 'return_ssc']]
correlation_matrix = returns.corr()
```

---

## returns_daily_melt.csv

**Purpose**: Multi-asset daily returns (long/melted format)

**Date Range**: 2016-03-10 onwards

**Rows**: 5,181 (1,296 dates × 4 assets + header)

**Size**: 169 KB

### Columns

| Column | Data Type | Description | Units | Notes |
|--------|-----------|-------------|-------|-------|
| `Date` | Date | Trading date | ISO 8601 | Daily frequency |
| `variable` | String | Asset identifier | Categorical | return_btc, return_eth, return_sp500, return_ssc |
| `value` | Float | Return value | Decimal | The actual return for the asset on that date |

### Usage Notes
- **Long format**: Each row is a date-asset-return combination
- Ideal for panel regression, mixed models, and ggplot-style visualization
- Same data as returns_daily.csv, just restructured
- Preferred format for time series modeling in R and statistical packages

### Example Usage
```python
# Filter to single asset for time series analysis
df = pd.read_csv('returns_daily_melt.csv', parse_dates=['Date'])
btc_returns = df[df['variable'] == 'return_btc']
```

---

## benchmarking.csv

**Purpose**: Fama-French factor returns and sector portfolios

**Date Range**: 2010-04-01 onwards

**Rows**: 136

**Size**: 19 KB

### Columns

| Column | Data Type | Description | Units | Notes |
|--------|-----------|-------------|-------|-------|
| `Date` | Date | Month end date | ISO 8601 | Monthly frequency |
| `A` | Float | Portfolio A return | Decimal | Sector/style portfolio |
| `B` | Float | Portfolio B return | Decimal | Sector/style portfolio |
| `C` | Float | Portfolio C return | Decimal | Sector/style portfolio |
| `D` | Float | Portfolio D return | Decimal | Sector/style portfolio |
| `E` | Float | Portfolio E return | Decimal | Sector/style portfolio |
| `F` | Float | Portfolio F return | Decimal | Sector/style portfolio |
| `G` | Float | Portfolio G return | Decimal | Sector/style portfolio |
| `H` | Float | Portfolio H return | Decimal | Sector/style portfolio |
| `Mkt_RF` | Float | Market excess return | Decimal | Market return minus risk-free rate |
| `SMB` | Float | Small Minus Big | Decimal | Size factor |
| `HML` | Float | High Minus Low | Decimal | Value factor (Book-to-Market) |
| `RF` | Float | Risk-free rate | Decimal | Monthly rate |
| `WML` | Float | Winners Minus Losers | Decimal | Momentum factor |

### Usage Notes
- **Fama-French Factors**: Core factors for multi-factor models
  - **Mkt_RF**: Market risk premium (already excess return)
  - **SMB**: Small cap premium over large cap
  - **HML**: Value premium over growth
  - **WML**: Momentum factor (Carhart four-factor model)
- **RF**: Risk-free rate for calculating excess returns
- **Portfolios A-H**: Likely represent sector or style portfolios for benchmarking

### Factor Model Equations

**Three-Factor Model**:
```
R_i - R_f = α + β₁(R_m - R_f) + β₂(SMB) + β₃(HML) + ε
```

**Four-Factor Model (Carhart)**:
```
R_i - R_f = α + β₁(R_m - R_f) + β₂(SMB) + β₃(HML) + β₄(WML) + ε
```

### Example Usage
```python
# Run Fama-French three-factor regression
import statsmodels.api as sm

df = pd.read_csv('benchmarking.csv', parse_dates=['Date'])

# Prepare data
X = df[['Mkt_RF', 'SMB', 'HML']]
X = sm.add_constant(X)
y = df['A'] - df['RF']  # Portfolio A excess return

# Run regression
model = sm.OLS(y, X).fit()
print(model.summary())
print(f"Alpha: {model.params['const']:.6f}")
```

---

## reit_altdata.csv

**Purpose**: Australian REIT alternative data with business descriptions

**Date Range**: 2020-03-31 snapshot

**Rows**: 2,864

**Size**: 2.2 MB

### Columns

| Column | Data Type | Description | Units | Notes |
|--------|-----------|-------------|-------|-------|
| `Date` | Date | Snapshot date | ISO 8601 | All rows are 2020-03-31 |
| `Company` | String | Company name/ticker | Text | REIT identifier |
| `Description` | String | Business description | Text | Detailed company operations |
| `Market_Cap_Bin` | String | Market cap category | Categorical | Small/Mid/Large |
| `Return` | Float | Period return | Decimal | Return for the period |
| Additional columns | Various | Company-specific metrics | Various | May include sector, geography, etc. |

### Usage Notes
- **Alternative Data**: Rich text descriptions suitable for NLP analysis
- **Text Mining**: Can extract themes, sentiment, and business model characteristics
- **Snapshot Data**: Point-in-time, not a time series
- Large file size due to extensive text descriptions
- Useful for:
  - Topic modeling
  - Sentiment analysis
  - Classification by property type
  - Relating business descriptions to returns

### Example Usage
```python
# Text analysis of REIT descriptions
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('reit_altdata.csv')

# Extract key terms from descriptions
vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])
feature_names = vectorizer.get_feature_names_out()

# Analyze relationship between description topics and returns
```

---

## Data Quality Notes

### Missing Values
- **Returns**: First observation in time series typically has NA/missing return
- **Handling**: Use `.dropna()` or `.fillna()` appropriately for your analysis

### Date Format Variations
Different files use different date formats:
- **ISO 8601** (YYYY-MM-DD): returns_daily.csv, benchmarking.csv
- **DD/M/YYYY**: mkt_timing.csv
- **DD/M/YY**: TSLA_daily.csv

Always use `parse_dates=` parameter when reading with pandas.

### Return Conventions
- All returns in **decimal format**: 0.01 = 1%
- To convert to percentage: multiply by 100
- To annualize daily returns: (1 + daily_return)^252 - 1
- To annualize monthly returns: (1 + monthly_return)^12 - 1

### Risk-Free Rate
- Provided in **same frequency** as returns (daily/monthly)
- Already in decimal format
- For CAPM, subtract rf from both asset and market returns

---

## Common Calculations

### Beta Estimation
```python
beta = cov(return_asset, return_market) / var(return_market)
```

### CAPM Expected Return
```python
E(R_i) = R_f + β_i * (E(R_m) - R_f)
```

### Sharpe Ratio
```python
sharpe_ratio = (mean_return - risk_free_rate) / std_dev_return
```

### Excess Returns
```python
excess_return = asset_return - risk_free_rate
```

---

## Data Sources

### Methodology
- **Market Data**: Sourced from standard financial data providers
- **Factor Data**: Kenneth French Data Library (Dartmouth/Chicago)
- **Cryptocurrency**: Major exchange APIs with volume-weighted pricing
- **REIT Data**: Australian Securities Exchange (ASX) filings

### Update Frequency
This is a snapshot repository. Data ranges are fixed:
- Market timing data: Through 2020
- Daily returns: 2016-03-10 onwards
- Factor data: 2010-04-01 onwards

For live/updated data, please refer to original sources.

---

## Citation

If using this data in research or publications, please cite:
- Fama-French factors: Kenneth R. French Data Library
- Market data: Specify your data provider
- This repository: Include GitHub repository link

---

**Last Updated**: 2025-11-04
**Version**: 1.0
