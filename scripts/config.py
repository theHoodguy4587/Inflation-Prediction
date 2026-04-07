"""
Configuration for Inflation Prediction Data Collection System
Defines data sources, indicators, and settings
"""

import os
from datetime import datetime

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, "processed")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Date range for data collection
START_YEAR = 2005
END_YEAR = 2026

# World Bank API configuration
WORLD_BANK_CONFIG = {
    "base_url": "https://api.worldbank.org/v2/country/LK/indicators",
    "indicators": {
        "NY.GDP.MKTP.KD.ZG": "GDP_Growth",  # GDP growth (annual %)
        "NE.IMP.GNFS.CD.ZS": "Imports_Percent_GDP",  # Imports of goods/services (% of GDP)
        "NE.EXP.GNFS.CD.ZS": "Exports_Percent_GDP",  # Exports of goods/services (% of GDP)
        "NE.RSB.GNFS.CD": "Trade_Balance",  # Trade balance (current US$)
        "GC.XPN.TOTL.GD.ZS": "Government_Spending_Percent_GDP",  # General govt final consumption (% of GDP)
        "GC.DOD.TOTL.GD.ZS": "Government_Debt_Percent_GDP",  # Central government debt (% of GDP)
        "BN.CAB.XOKA.GD.ZS": "Current_Account_Percent_GDP",  # Current account balance (% of GDP)
    },
    "format": "json",
    "per_page": 100,
}

# Yahoo Finance configuration
YAHOO_FINANCE_CONFIG = {
    "symbols": {
        "LKRUSD=X": "Exchange_Rate_USD_LKR",  # USD/LKR
        "CL=F": "Oil_Price_USD_Per_Barrel",  # Crude Oil
        "EURUSD=X": "Exchange_Rate_USD_EUR",  # For reference
    },
    "interval": "1mo",  # Monthly data
}

# Central Bank of Sri Lanka configuration
CBSL_CONFIG = {
    "base_url": "https://www.cbsl.gov.lk",
    "inflation_url": "https://www.cbsl.gov.lk/en/statistics/consultations",
    "policy_rate_url": "https://www.cbsl.gov.lk/en/monetary-policy",
    "key_indicators": [
        "Inflation_Rate",
        "Policy_Rate",
        "Lending_Rate",
        "Deposit_Rate",
    ]
}

# IMF configuration
IMF_CONFIG = {
    "base_url": "https://www.imf.org/external/datamapper/api/v1",
    "indicators": {
        "NGDP_RPCH": "Real_GDP_Growth",
        "PPPPC": "PPP_Per_Capita",
        "BCA": "Current_Account_Balance",
    },
    "note": "IMF data may require manual export or API key - check IMF data portal"
}

# Global commodity price configuration
COMMODITIES_CONFIG = {
    "sources": [
        {
            "name": "Gasoline_Price",
            "url": "https://www.globalpetrolprices.com/Sri-Lanka/gasoline_prices/",
            "unit": "USD per liter"
        },
        {
            "name": "Diesel_Price",
            "url": "https://www.globalpetrolprices.com/Sri-Lanka/diesel_prices/",
            "unit": "USD per liter"
        }
    ]
}

# Data validation rules
VALIDATION_CONFIG = {
    "missing_threshold": 0.3,  # Warn if > 30% missing values
    "range_checks": {
        "GDP_Growth": (-10, 15),  # Reasonable GDP growth range
        "Inflation_Rate": (-5, 30),  # Reasonable inflation range
        "Exchange_Rate_USD_LKR": (150, 400),  # Reasonable LKR/USD range
    }
}

# CSV output configuration
CSV_CONFIG = {
    "master_file": os.path.join(PROCESSED_DATA_DIR, "inflation_master.csv"),
    "encoding": "utf-8",
    "date_format": "%Y-%m-%d",
}

# Logging configuration
LOG_CONFIG = {
    "log_dir": LOGS_DIR,
    "log_level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# API retry configuration
RETRY_CONFIG = {
    "max_retries": 3,
    "retry_delay": 5,  # seconds
    "timeout": 30,  # seconds
}

# Sample master dataframe columns
MASTER_COLUMNS = [
    "Date",
    "Indicator",
    "Value",
    "Unit",
    "Country",
    "Source",
    "Last_Updated"
]

# Indicator reference table
INDICATOR_METADATA = {
    "GDP_Growth": {"unit": "%", "frequency": "annual", "source": "World Bank"},
    "Inflation_Rate": {"unit": "%", "frequency": "monthly", "source": "CBSL"},
    "Exchange_Rate_USD_LKR": {"unit": "LKR per USD", "frequency": "daily", "source": "Yahoo Finance"},
    "Imports_Percent_GDP": {"unit": "% of GDP", "frequency": "annual", "source": "World Bank"},
    "Exports_Percent_GDP": {"unit": "% of GDP", "frequency": "annual", "source": "World Bank"},
    "Trade_Balance": {"unit": "USD", "frequency": "annual", "source": "World Bank"},
    "Current_Account_Percent_GDP": {"unit": "% of GDP", "frequency": "annual", "source": "World Bank"},
    "Government_Spending_Percent_GDP": {"unit": "% of GDP", "frequency": "annual", "source": "World Bank"},
    "Government_Debt_Percent_GDP": {"unit": "% of GDP", "frequency": "annual", "source": "World Bank"},
    "Policy_Rate": {"unit": "%", "frequency": "monthly", "source": "CBSL"},
    "Lending_Rate": {"unit": "%", "frequency": "monthly", "source": "CBSL"},
}
