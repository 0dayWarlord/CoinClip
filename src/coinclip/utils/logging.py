#logging utilities

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "coinclip",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    set up a logger with console and optional file output

    Args:
        name: logger name
        log_file: optional path to log file
        level: logging level

    Returns:
        configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    #remove existing handlers to avoid duplicates
    logger.handlers.clear()

    #console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    #file handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

