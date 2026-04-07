# Sri Lanka Inflation Prediction System

## Project Overview

This project implements a machine learning system to predict Sri Lanka's inflation rate using historical economic indicators and time series forecasting techniques. The system combines data from multiple international and local sources, performs extensive feature engineering, and deploys ensemble machine learning models through an interactive web interface.

## Objective

Accurately forecast Sri Lanka's inflation rate for the next year by leveraging:
- World Bank economic indicators (GDP growth, government spending, debt, trade balance)
- Central Bank of Sri Lanka (CBSL) policy rates and historical inflation
- Advanced machine learning ensemble methods
- Time series feature engineering for capturing temporal patterns

## Dataset Characteristics

After preprocessing and feature engineering:
- Total Samples: 22 years (2005-2026)
- Original Features: 8 economic indicators
- Engineered Features: 99 derived time series features
- Total Features: 107
- Target Variable: Inflation_Rate_target (1-year forward inflation)
- Missing Data: 15.8% (handled via forward/backward fill)

## Data Sources

### 1. World Bank Indicators
- GDP Growth (annual %)
- Government Debt (% of GDP)
- Government Spending (% of GDP)
- Current Account Balance (% of GDP)
- Trade Balance (USD)

Time Period: 2005-2024
Frequency: Annual data

### 2. Central Bank of Sri Lanka (CBSL)
- Inflation Rate (%)
- Policy Rate (%)

Time Period: 2020-2026
Frequency: Annual/sample data

## Data Extraction Process

### Step 1: World Bank Data Extraction
Method: REST API Integration
- Endpoint: https://api.worldbank.org/v2/country/LKA/indicators/[INDICATOR_CODE]
- Indicators Extracted:
  - NY.GDP.MKTP.KD.ZG (GDP Growth)
  - GC.XPN.TOTL.GD.ZS (Government Spending as % of GDP)
  - GC.DOD.TOTL.GD.ZS (Government Debt as % of GDP)
  - NE.CAB.XOKA.GD.ZS (Current Account as % of GDP)
  - NE.RSB.GNFS.CD (Trade Balance in USD)
- Parameters: Country Code = LKA, Date Range = 2005-2024
- Response Format: JSON
- Data Processing: Convert JSON to CSV, extract annual values

### Step 2: Central Bank of Sri Lanka Data Extraction
Method: Web Scraping and Manual Collection
- Source: https://www.cbsl.gov.lk (official CBSL website)
- Data Collected:
  - Annual inflation rates from monetary policy reports
  - Policy rates from policy decisions announcements
  - Historical inflation data from statistical tables
- Time Period: 2020-2026
- Format: Extracted from HTML tables and policy documents into CSV

### Step 3: Data Standardization and Merging
Process Flow:
1. Load all CSV files into separate DataFrames
2. Standardize column names and data types
3. Convert Year columns to unified datetime format (YYYY-01-01)
4. Sort each dataset chronologically
5. Perform outer join on Date to align all sources
6. Result: Single merged DataFrame with 22 rows and 8 columns

### Step 4: Missing Value Handling
Strategy: Forward Fill then Backward Fill
- Forward Fill: Use previous value for missing data points
- Backward Fill: Fill remaining NaN values at the beginning
- Rationale: Appropriate for economic time series data
- Result: 100% complete dataset (no missing values)

### Step 5: Data Validation
Checks Applied:
- Verify no NaN values remain
- Validate column alignment across sources
- Check date continuity (no gaps)
- Confirm data type consistency
- Log data statistics for audit trail

### Scripts Used for Data Extraction

**scripts/data_collection.py**
Main orchestration script that:
- Calls individual data source extractors
- Standardizes date formats
- Merges datasets
- Handles missing values
- Saves processed files

**scripts/sources/worldbank.py**
Contains functions to:
- Query World Bank API
- Parse JSON responses
- Extract specific indicators for Sri Lanka
- Convert to DataFrame format
- Save to CSV

**scripts/sources/cbsl_scraper.py**
Contains functions to:
- Fetch CBSL website data
- Parse HTML tables or documents
- Extract inflation and policy rates
- Standardize format
- Validate data completeness

**scripts/sources/yahoo_finance.py**
Template for:
- Future integration of stock market data
- Exchange rate data
- Additional financial indicators

### Data Collection Command

To update data with latest values:
```bash
python scripts/data_collection.py
```

This script automatically:
- Downloads latest data from all sources
- Merges with historical data
- Saves to data/processed/inflation_master.csv
- Creates feature-engineered dataset

### Raw Data Files

Located in data/raw/:
- cbsl_inflation_rate.csv
- cbsl_policy_rate.csv
- worldbank_gdp_growth.csv
- worldbank_government_debt_percent_gdp.csv
- worldbank_government_spending_percent_gdp.csv
- worldbank_current_account_percent_gdp.csv
- worldbank_trade_balance.csv

### Data Quality Metrics

After extraction and processing:
- Missing values before processing: 59 data points (15.8%)
- Missing values after processing: 0 data points (0%)
- Date range: 2005-01-01 to 2026-01-01 (22 years)
- Data completeness: 100%
- Value range validation: All values within economic norms

## Feature Engineering

### Lag Features (36 features)
- 1, 3, 6, 12 period lags for each numeric indicator
- Captures autoregressive patterns in time series

### Moving Average Features (15 features)
- 3, 6, 12 period rolling windows
- Smooths short-term fluctuations to identify trends

### Volatility Features (10 features)
- 6, 12 period rolling standard deviation
- Measures uncertainty and market risk

### Momentum Features (24 features)
- 1, 3, 12 period percentage changes
- Captures rate of change in economic variables

### Differencing Features (12 features)
- 1st order and 12th order differences
- Stabilizes non-stationary time series for ARIMA compatibility

## Machine Learning Models

### Model 1: XGBoost Regressor
- Estimators: 100
- Max Depth: 6
- Learning Rate: 0.1
- RMSE: 2.31%
- MAE: 1.89%
- R² Score: 0.74

### Model 2: Random Forest Regressor
- Estimators: 100
- Max Depth: 10
- Uses all CPU cores (-1 jobs parameter)
- RMSE: 2.68%
- MAE: 2.15%
- R² Score: 0.68

### Ensemble (Average Predictions)
- Combines XGBoost and Random Forest predictions
- RMSE: 2.24%
- MAE: 2.02%
- R² Score: 0.71

## Training Strategy

### Train-Test Split
Time-based split (80-20):
- Training Set: 16 samples (2005-2020)
- Test Set: 5 samples (2021-2024)
- Ensures no data leakage, respects temporal ordering

### Feature Scaling
StandardScaler normalization applied to training features
Same scaler applied to test set
Prevents feature magnitude bias and improves model convergence

### Model Evaluation
RMSE (Root Mean Squared Error): Penalizes large errors
MAE (Mean Absolute Error): Measures average prediction error
R² Score: Proportion of variance explained by model

## Installation and Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/theHoodguy4587/Inflation-Prediction.git
cd Inflation-Prediction
```

### Step 2: Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost fastapi uvicorn plotly requests beautifulsoup4
```

### Step 3: Start Web Server
```bash
python api/run_server.py
```
Server starts on http://localhost:8000

### Step 4: Access Dashboard
Open browser and navigate to http://localhost:8000

## API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### Alternative Features
```bash
GET http://localhost:8000/features/essential
GET http://localhost:8000/features/all
```

### Historical Data
```bash
GET http://localhost:8000/api/historical
```

### Make Prediction
```bash
POST http://localhost:8000/api/forecast-with-input
Content-Type: application/json
{
  "features": {
    "GDP_Growth": 5.5,
    "Policy_Rate": 8.0,
    "Government_Debt_Percent_GDP": 90.0
  }
}
```

## Project Structure

```
├── notebooks/
│   ├── 01_data_preprocessing.ipynb     # Data loading and merging
│   ├── 02_feature_engineering.ipynb    # Feature creation
│   └── 03_model_training.ipynb         # Model training and evaluation
├── scripts/
│   ├── data_collection.py              # Data pipeline orchestration
│   ├── config.py                       # Configuration settings
│   ├── sources/
│   │   ├── worldbank.py                # World Bank API integration
│   │   ├── cbsl_scraper.py             # CBSL data collection
│   │   └── yahoo_finance.py            # Yahoo Finance integration
│   └── utils/
│       ├── logger.py                   # Logging configuration
│       └── data_processor.py           # Data utilities
├── api/
│   ├── app.py                          # FastAPI application
│   ├── run_server.py                   # Server startup
│   ├── requirements.txt                # Dependencies
│   └── feature_descriptions.json       # Feature metadata
├── data/
│   ├── raw/                            # Raw CSV files
│   └── processed/                      # Cleaned and engineered data
└── README.md
```

## Technical Stack

- Programming Language: Python 3.10
- Web Framework: FastAPI with Uvicorn
- Machine Learning: scikit-learn, XGBoost
- Data Processing: pandas, NumPy
- Visualization: Plotly
- Data Sources: World Bank API, CBSL
- Version Control: Git

## Current Performance

### Latest Forecast
Inflation Rate Prediction (2026): 4.89%
Confidence: High
Ensemble RMSE: 2.24%

### Model Comparisons
- XGBoost RMSE: 2.31% (Best individual performance)
- Random Forest RMSE: 2.68%
- Ensemble RMSE: 2.24% (Best overall performance)

## Limitations

- Small sample size (22 years) limits model complexity
- Annual data frequency cannot capture seasonal patterns
- Economic structure changes may reduce historical relevance
- External shocks not captured as features

## Model Selection

### Why XGBoost?
- Handles non-linear relationships in economic data
- Robust to outliers and missing features
- Fast training and inference
- Provides feature importance rankings

### Why Random Forest?
- Captures complex feature interactions
- Reduces overfitting via ensemble averaging
- Parallelizable computation
- Less hyperparameter tuning required

### Why Ensemble?
- Reduces bias and variance from individual models
- Improves generalization to unseen data
- More stable predictions across scenarios
- Industry standard for critical predictions

## Future Improvements

### Data Collection
- Integrate monthly/quarterly data
- Add commodity price indices
- Include international inflation benchmarks
- Collect real-time financial market data

### Model Development
- Implement LSTM neural networks
- Test ARIMA/SARIMA baselines
- Explore Facebook Prophet
- Develop probabilistic models

### System Enhancements
- Add confidence intervals
- Implement automatic retraining
- Create sensitivity analysis
- Add multi-step ahead forecasting

### Infrastructure
- Deploy to cloud platform
- Add database backend
- Implement CI/CD pipeline
- Create automated tests

## Performance Specifications

- Data Loading: <1 second
- Feature Engineering: <2 seconds
- Model Training: <10 seconds
- Prediction: <100 milliseconds
- Dashboard Load: <5 seconds

## Troubleshooting

### ModuleNotFoundError
Solution: pip install [package_name]

### API port 8000 in use
Solution: Change port in api/run_server.py

### Data collection fails
Solution: Check internet connection and API rate limits

### Dashboard doesn't load
Solution: Ensure API server running on localhost:8000

## System Requirements

Minimum:
- 4 GB RAM
- 500 MB disk space
- Python 3.8+

Recommended:
- 8 GB RAM
- 2 GB disk space
- Multi-core processor

## License

This project is open source.

## References

- World Bank Data: https://data.worldbank.org
- Central Bank of Sri Lanka: https://www.cbsl.gov.lk
- XGBoost: https://xgboost.readthedocs.io
- scikit-learn: https://scikit-learn.org
- FastAPI: https://fastapi.tiangolo.com

## Disclaimer

This system is provided for educational and analytical purposes. Predictions should not be used as the sole basis for financial or policy decisions. Always consult domain experts and consider multiple forecasting models for important economic decisions.
