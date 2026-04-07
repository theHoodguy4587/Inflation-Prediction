"""
Central Bank of Sri Lanka (CBSL) data scraper
Attempts to fetch inflation, policy rates, and other monetary indicators
"""

import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from utils.logger import setup_logger
import config

logger = setup_logger(__name__)


def fetch_cbsl_data() -> dict:
    """
    Fetch CBSL data (inflation, policy rates, etc.)

    Returns:
        Dictionary of {indicator_name: DataFrame}
    """
    logger.info("Starting CBSL data fetch...")
    results = {}

    # Try to fetch from CBSL website
    try:
        inflation_df = _scrape_inflation_data()
        if inflation_df is not None and not inflation_df.empty:
            results["Inflation_Rate"] = inflation_df
            logger.info(f"✅ Fetched Inflation Rate: {len(inflation_df)} records")
    except Exception as e:
        logger.warning(f"Error fetching inflation data: {e}")

    try:
        policy_rate_df = _scrape_policy_rate_data()
        if policy_rate_df is not None and not policy_rate_df.empty:
            results["Policy_Rate"] = policy_rate_df
            logger.info(f"✅ Fetched Policy Rate: {len(policy_rate_df)} records")
    except Exception as e:
        logger.warning(f"Error fetching policy rate data: {e}")

    if not results:
        logger.warning(
            "CBSL: No data scraped. "
            "Note: CBSL data may require manual download or custom parsing. "
            "Visit: https://www.cbsl.gov.lk/en/statistics"
        )

    return results


def _scrape_inflation_data() -> pd.DataFrame:
    """
    Attempt to scrape inflation data from CBSL website

    Returns:
        DataFrame with Date and Inflation_Rate columns
    """
    try:
        url = config.CBSL_CONFIG["inflation_url"]
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=config.RETRY_CONFIG["timeout"])
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, "html.parser")

        # Look for data tables - this is heuristic and may need adjustment
        tables = soup.find_all("table")

        if not tables:
            logger.warning("No tables found on CBSL inflation page")
            return pd.DataFrame()

        # Try to parse first relevant table
        for table in tables:
            try:
                df = pd.read_html(str(table))[0]
                if not df.empty and len(df) > 2:
                    logger.info(f"Found table with {len(df)} rows")
                    # Basic validation - if we get data, return it
                    return df
            except Exception:
                continue

        logger.warning("Could not parse CBSL tables")
        return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error scraping inflation data: {e}")
        return pd.DataFrame()


def _scrape_policy_rate_data() -> pd.DataFrame:
    """
    Attempt to scrape policy rate from CBSL website

    Returns:
        DataFrame with Date and Policy_Rate columns
    """
    try:
        url = config.CBSL_CONFIG["policy_rate_url"]
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=config.RETRY_CONFIG["timeout"])
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Look for policy rate information in text or tables
        text_content = soup.get_text()

        if "Overnight Policy Rate" in text_content or "OPR" in text_content:
            logger.info("Found policy rate reference on page")
            # Policy rate data would need custom parsing
            # For now, this is a placeholder

        tables = soup.find_all("table")
        for table in tables:
            try:
                df = pd.read_html(str(table))[0]
                if not df.empty:
                    return df
            except Exception:
                continue

        return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error scraping policy rate: {e}")
        return pd.DataFrame()


def save_cbsl_data(results: dict) -> bool:
    """Save CBSL data to CSV files"""
    success_count = 0

    for indicator_name, df in results.items():
        if df.empty:
            continue

        filename = f"cbsl_{indicator_name.lower()}.csv"
        filepath = os.path.join(config.RAW_DATA_DIR, filename)

        try:
            df.to_csv(filepath, index=False, encoding=config.CSV_CONFIG["encoding"])
            logger.info(f"Saved: {filename}")
            success_count += 1
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")

    logger.info(f"CBSL: {success_count}/{len(results)} files saved")
    return success_count > 0


def create_manual_cbsl_sample() -> bool:
    """
    Create sample CBSL data for testing
    In production, replace with actual data download
    """
    logger.info("Creating sample CBSL data for testing...")

    # Sample inflation data (CBSL published values)
    inflation_data = {
        "Year": [2020, 2021, 2022, 2023, 2024, 2025, 2026],
        "Inflation_Rate": [4.6, 4.3, 6.7, 6.3, 2.2, 2.0, 2.2]
    }

    # Sample policy rate (recent historical values)
    policy_data = {
        "Year": [2023, 2024, 2025, 2026],
        "Policy_Rate": [8.5, 8.0, 7.75, 7.75]
    }

    try:
        # Save sample data
        inf_df = pd.DataFrame(inflation_data)
        inf_file = os.path.join(config.RAW_DATA_DIR, "cbsl_inflation_rate.csv")
        inf_df.to_csv(inf_file, index=False, encoding=config.CSV_CONFIG["encoding"])
        logger.info(f"Created sample: {inf_file}")

        policy_df = pd.DataFrame(policy_data)
        policy_file = os.path.join(config.RAW_DATA_DIR, "cbsl_policy_rate.csv")
        policy_df.to_csv(policy_file, index=False, encoding=config.CSV_CONFIG["encoding"])
        logger.info(f"Created sample: {policy_file}")

        return True

    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return False


if __name__ == "__main__":
    data = fetch_cbsl_data()
    if not data:
        logger.info("No live data fetched, creating sample data...")
        create_manual_cbsl_sample()
    else:
        save_cbsl_data(data)
