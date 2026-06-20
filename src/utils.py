"""
Utility module for logging configuration and helper functions.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Configure and return a logger instance with UTF-8 safe handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Attempt to make stdout/stderr UTF-8 safe on Windows
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        # Best-effort; continue if not supported
        pass

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified) - ensure UTF-8 encoding
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class SpectraError(Exception):
    """Base exception for spectral processing errors."""
    pass


class CleaningError(SpectraError):
    """Exception for cleaning stage errors."""
    pass


class StandardizationError(SpectraError):
    """Exception for standardization stage errors."""
    pass


class ContinuumError(SpectraError):
    """Exception for continuum removal errors."""
    pass


class DerivativeError(SpectraError):
    """Exception for derivative calculation errors."""
    pass
