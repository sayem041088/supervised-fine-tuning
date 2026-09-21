"""
Structured logger setup for supervised fine-tuning pipeline.
"""

import logging
import os
import sys
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get configured application logger.

    Args:
        name: Name of the logger module.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name or "gemini_sft")

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
