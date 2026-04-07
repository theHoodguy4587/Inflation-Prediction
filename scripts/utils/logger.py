"""
Logging utility for data collection pipeline
"""

import logging
import os
from datetime import datetime
from scripts.config import LOG_CONFIG

def setup_logger(name: str):
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_CONFIG["log_level"]))

    # Skip if handlers already exist
    if logger.handlers:
        return logger

    # Create formatters and handlers
    formatter = logging.Formatter(LOG_CONFIG["format"])

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    os.makedirs(LOG_CONFIG["log_dir"], exist_ok=True)
    log_file = os.path.join(
        LOG_CONFIG["log_dir"],
        f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
