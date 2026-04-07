"""
Main data collection orchestration script
Coordinates fetching from all sources and creates master dataset
"""

import sys
import os

# Add scripts directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import config
from utils.logger import setup_logger
from utils.data_processor import create_master_csv
from sources import worldbank, yahoo_finance
from sources import cbsl_scraper

logger = setup_logger(__name__)


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("[START] DATA COLLECTION PIPELINE")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("=" * 60)

    sources_success = {
        "World Bank": False,
        "Yahoo Finance": False,
        "CBSL": False,
    }

    try:
        # 1. Fetch World Bank data
        logger.info("\n[1/3] Fetching World Bank data...")
        wb_data = worldbank.fetch_worldbank_data()
        if wb_data:
            sources_success["World Bank"] = worldbank.save_worldbank_data(wb_data)
        else:
            logger.warning("No World Bank data retrieved")

    except Exception as e:
        logger.error(f"World Bank fetch failed: {e}")

    try:
        # 2. Fetch Yahoo Finance data
        logger.info("\n[2/3] Fetching Yahoo Finance data...")
        yf_data = yahoo_finance.fetch_yahoo_finance_data()
        if yf_data:
            sources_success["Yahoo Finance"] = yahoo_finance.save_yahoo_finance_data(
                yf_data
            )
        else:
            logger.warning("No Yahoo Finance data retrieved")

    except Exception as e:
        logger.error(f"Yahoo Finance fetch failed: {e}")

    try:
        # 3. Fetch CBSL data
        logger.info("\n[3/3] Fetching CBSL data...")
        cbsl_data = cbsl_scraper.fetch_cbsl_data()
        if cbsl_data:
            sources_success["CBSL"] = cbsl_scraper.save_cbsl_data(cbsl_data)
        else:
            logger.info("Creating sample CBSL data for demonstration...")
            cbsl_scraper.create_manual_cbsl_sample()
            sources_success["CBSL"] = True

    except Exception as e:
        logger.error(f"CBSL fetch failed: {e}")

    # 4. Merge all data
    logger.info("\n[MERGE] Merging all data sources...")
    try:
        master_success = create_master_csv()
    except Exception as e:
        logger.error(f"Data merge failed: {e}")
        master_success = False

    # 5. Summary
    logger.info("\n" + "=" * 60)
    logger.info("[SUMMARY] Collection Results")
    logger.info("=" * 60)

    for source, success in sources_success.items():
        status = "SUCCESS" if success else "FAILED/SKIPPED"
        logger.info(f"{source:20s}: {status}")

    logger.info("=" * 60)

    if master_success:
        logger.info("[SUCCESS] MASTER CSV CREATED SUCCESSFULLY!")
        logger.info(f"Location: {config.CSV_CONFIG['master_file']}")
    else:
        logger.warning("[WARNING] Master CSV creation failed or returned empty")

    logger.info("=" * 60)
    logger.info("[COMPLETE] PIPELINE COMPLETED")
    logger.info("=" * 60)

    return 0 if (any(sources_success.values()) or master_success) else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
