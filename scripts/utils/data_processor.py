"""
Data processing utilities for cleaning and merging data
"""

import pandas as pd
import os
from datetime import datetime
import config
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file safely"""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {filepath}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return pd.DataFrame()


def save_csv(df: pd.DataFrame, filepath: str) -> bool:
    """Save DataFrame to CSV"""
    try:
        df.to_csv(filepath, index=False, encoding=config.CSV_CONFIG["encoding"])
        logger.info(f"Saved {filepath}, rows: {len(df)}")
        return True
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")
        return False


def standardize_date(date_col) -> pd.Series:
    """Standardize date format"""
    try:
        return pd.to_datetime(date_col).dt.strftime(config.CSV_CONFIG["date_format"])
    except Exception as e:
        logger.warning(f"Error standardizing dates: {e}")
        return date_col


def validate_data(df: pd.DataFrame, indicator: str) -> bool:
    """Validate data completeness and ranges"""
    if df.empty:
        logger.warning(f"Empty dataframe for {indicator}")
        return False

    # Check missing values
    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if missing_pct > config.VALIDATION_CONFIG["missing_threshold"]:
        logger.warning(
            f"{indicator}: {missing_pct:.1%} missing values exceeds threshold"
        )

    # Check value ranges if defined
    if indicator in config.VALIDATION_CONFIG["range_checks"]:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            min_val, max_val = config.VALIDATION_CONFIG["range_checks"][indicator]
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if not out_of_range.empty:
                logger.warning(
                    f"{indicator}/{col}: {len(out_of_range)} values out of range [{min_val}, {max_val}]"
                )

    return True


def merge_all_data() -> pd.DataFrame:
    """Merge all raw CSV files into master CSV"""
    logger.info("Starting data merge process...")

    all_files = []
    raw_files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith(".csv")]

    if not raw_files:
        logger.warning("No raw CSV files found in data/raw/")
        return pd.DataFrame()

    for filename in raw_files:
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        df = load_csv(filepath)

        if df.empty:
            continue

        # Extract indicator name from filename
        indicator = filename.replace(".csv", "").replace("_", " ").title()

        # Standardize structure: Date, Value, Unit columns
        if "Date" in df.columns or "date" in df.columns or "Year" in df.columns:
            date_col = next(
                (col for col in df.columns if col.lower() in ["date", "year"]), None
            )
            if date_col:
                df["Date"] = standardize_date(df[date_col])
            else:
                logger.warning(f"No date column found in {filename}")
                continue
        else:
            logger.warning(f"No date column found in {filename}")
            continue

        # Get value columns (exclude date)
        value_cols = [col for col in df.columns if col.lower() not in ["date", "year"]]
        if not value_cols:
            logger.warning(f"No value columns found in {filename}")
            continue

        # Melt into long format
        for value_col in value_cols:
            melt_df = df[["Date", value_col]].copy()
            melt_df["Indicator"] = f"{indicator}_{value_col}"
            melt_df["Value"] = melt_df[value_col]
            melt_df["Unit"] = "Unknown"
            melt_df["Country"] = "Sri Lanka"
            melt_df["Source"] = filename.split("_")[0].title()
            melt_df["Last_Updated"] = datetime.now().strftime(
                config.CSV_CONFIG["date_format"]
            )

            # Keep only needed columns
            melt_df = melt_df[config.MASTER_COLUMNS]
            all_files.append(melt_df)

    if not all_files:
        logger.error("No data to merge!")
        return pd.DataFrame()

    master_df = pd.concat(all_files, ignore_index=True)
    master_df = master_df.drop_duplicates(subset=["Date", "Indicator"])
    master_df = master_df.sort_values(by=["Date", "Indicator"])

    logger.info(f"Merged data shape: {master_df.shape}")
    return master_df


def create_master_csv() -> bool:
    """Create and save master CSV"""
    master_df = merge_all_data()

    if master_df.empty:
        logger.error("Master dataframe is empty!")
        return False

    success = save_csv(master_df, config.CSV_CONFIG["master_file"])
    if success:
        logger.info(f"[SUCCESS] Master CSV created: {config.CSV_CONFIG['master_file']}")
    return success
