"""
Logging utilities for Tabular Transformer.

This module provides a simple logging configuration for the package.
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "tabular_transformer", 
                level: int = logging.INFO,
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
        
    Returns:
        The configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger instance
logger = setup_logger()


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(
                f"tabular_transformer.{self.__class__.__name__}"
            )
        return self._logger
