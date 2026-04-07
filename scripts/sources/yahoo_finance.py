"""
Yahoo Finance data fetcher
Fetches exchange rates and commodity prices
"""

import pandas as pd
import os
import time
from utils.logger import setup_logger
import config

logger = setup_logger(__name__)


def fetch_yahoo_finance_data() -> dict:
    """
    Fetch Yahoo Finance data for currency and commodity indicators

    Returns:
        Dictionary of {indicator_name: DataFrame}
    """
    logger.info("Starting Yahoo Finance data fetch...")

    try:
        import yfinance as yf
    except ImportError:
        logger.error(
            "yfinance not installed. Install with: pip install yfinance"
        )
        return {}

    results = {}

    for symbol, indicator_name in config.YAHOO_FINANCE_CONFIG["symbols"].items():
        try:
            logger.info(f"Fetching {indicator_name} ({symbol})...")
            df = _fetch_symbol(symbol, indicator_name, yf)

            if df is not None and not df.empty:
                results[indicator_name] = df
                logger.info(f"✅ Fetched {indicator_name}: {len(df)} records")
            else:
                logger.warning(f"No data for {indicator_name}")

        except Exception as e:
            logger.error(f"Error fetching {indicator_name}: {e}")

        time.sleep(1)

    return results


def _fetch_symbol(symbol: str, indicator_name: str, yf) -> pd.DataFrame:
    """
    Fetch a single symbol from Yahoo Finance

    Args:
        symbol: Yahoo Finance ticker symbol
        indicator_name: Human-readable name
        yf: yfinance module

    Returns:
        DataFrame with Date and Close columns
    """
    try:
        # Fetch data
        data = yf.download(
            symbol,
            start=f"{config.START_YEAR}-01-01",
            end=f"{config.END_YEAR}-12-31",
            interval=config.YAHOO_FINANCE_CONFIG["interval"],
            progress=False
        )

        if data.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Reset index to convert date from index to column
        data = data.reset_index()

        # Rename columns for consistency
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"])
        if "Close" in data.columns:
            data = data[["Date", "Close"]].rename(columns={"Close": indicator_name})
        elif "Adj Close" in data.columns:
            data = data[["Date", "Adj Close"]].rename(
                columns={"Adj Close": indicator_name}
            )
        else:
            logger.warning(f"No price column found for {symbol}")
            return pd.DataFrame()

        data = data.drop_duplicates(subset=["Date"])
        data = data.sort_values("Date")

        return data

    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


def save_yahoo_finance_data(results: dict) -> bool:
    """Save Yahoo Finance data to CSV files"""
    success_count = 0

    for indicator_name, df in results.items():
        if df.empty:
            continue

        # Format date for consistency
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

        filename = f"yahoo_{indicator_name.lower()}.csv"
        filepath = os.path.join(config.RAW_DATA_DIR, filename)

        try:
            df.to_csv(filepath, index=False, encoding=config.CSV_CONFIG["encoding"])
            logger.info(f"Saved: {filename}")
            success_count += 1
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")

    logger.info(f"Yahoo Finance: {success_count}/{len(results)} files saved")
    return success_count > 0


if __name__ == "__main__":
    data = fetch_yahoo_finance_data()
    save_yahoo_finance_data(data)
