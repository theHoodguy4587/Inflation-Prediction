"""
World Bank API data fetcher
Fetches economic indicators for Sri Lanka
"""

import pandas as pd
import requests
import time
import os
from utils.logger import setup_logger
import config

logger = setup_logger(__name__)


def fetch_worldbank_data() -> dict:
    """
    Fetch World Bank data for all configured indicators

    Returns:
        Dictionary of {indicator_name: DataFrame}
    """
    logger.info("Starting World Bank data fetch...")
    results = {}

    for wb_code, indicator_name in config.WORLD_BANK_CONFIG["indicators"].items():
        try:
            df = _fetch_indicator(wb_code, indicator_name)
            if df is not None and not df.empty:
                results[indicator_name] = df
                logger.info(f"[SUCCESS] Fetched {indicator_name}: {len(df)} records")
            else:
                logger.warning(f"No data returned for {indicator_name}")
        except Exception as e:
            logger.error(f"Error fetching {indicator_name}: {e}")
            continue

        # Rate limiting - be respectful to API
        time.sleep(1)

    return results


def _fetch_indicator(indicator_code: str, indicator_name: str) -> pd.DataFrame:
    """
    Fetch a single indicator from World Bank API

    Args:
        indicator_code: World Bank indicator code (e.g., 'NY.GDP.MKTP.KD.ZG')
        indicator_name: Human-readable name

    Returns:
        DataFrame with Year and Value columns
    """
    url = f"{config.WORLD_BANK_CONFIG['base_url']}/{indicator_code}"
    params = {
        "format": config.WORLD_BANK_CONFIG["format"],
        "per_page": config.WORLD_BANK_CONFIG["per_page"],
        "date": f"{config.START_YEAR}:{config.END_YEAR}"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=config.RETRY_CONFIG["timeout"]
        )
        response.raise_for_status()

        data = response.json()

        # Parse response
        if len(data) < 2:
            logger.warning(f"Unexpected API response for {indicator_code}")
            return pd.DataFrame()

        records = data[1]
        if not records:
            logger.warning(f"No records found for {indicator_code}")
            return pd.DataFrame()

        # Extract year and value
        values = []
        for record in records:
            if record.get("value") is not None and record.get("date") is not None:
                try:
                    year = int(record["date"])
                    value = float(record["value"])
                    values.append({"Year": year, indicator_name: value})
                except (ValueError, TypeError):
                    continue

        if not values:
            return pd.DataFrame()

        df = pd.DataFrame(values)
        df = df.drop_duplicates(subset=["Year"])
        df = df.sort_values("Year")

        return df

    except requests.RequestException as e:
        logger.error(f"API request failed for {indicator_code}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing {indicator_code}: {e}")
        return pd.DataFrame()


def save_worldbank_data(results: dict) -> bool:
    """Save World Bank data to CSV files"""
    success_count = 0

    for indicator_name, df in results.items():
        if df.empty:
            continue

        filename = f"worldbank_{indicator_name.lower()}.csv"
        filepath = os.path.join(config.RAW_DATA_DIR, filename)

        try:
            df.to_csv(filepath, index=False, encoding=config.CSV_CONFIG["encoding"])
            logger.info(f"Saved: {filename}")
            success_count += 1
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")

    logger.info(f"World Bank: {success_count}/{len(results)} files saved")
    return success_count > 0


if __name__ == "__main__":
    data = fetch_worldbank_data()
    save_worldbank_data(data)
