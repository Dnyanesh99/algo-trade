"""
Production-grade centralized logging configuration using Loguru.
Provides thread-safe, high-performance logging with proper configuration management.
"""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

from src.utils.config_loader import config_loader

if TYPE_CHECKING:
    from types import FrameType

# Load configuration
config = config_loader.get_config()


class LoggerSetup:
    """Centralized logger setup with production-grade configuration."""

    _initialized: bool = False

    @classmethod
    def setup_logger(cls) -> None:
        """Setup logger with configuration-driven settings."""
        if cls._initialized:
            return

        # Remove default logger to prevent duplicate logs
        logger.remove()

        # Ensure log directory exists
        if config.logging:
            log_file_path = Path(config.logging.file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Add file logging with comprehensive settings
            # For production, consider serialize=True for structured JSON logs
            # which are easier for log aggregation systems (e.g., ELK, Splunk).
            logger.add(
                config.logging.file,
                level=config.logging.level,
                format=config.logging.format,
                rotation=config.logging.rotation,
                compression=config.logging.compression,
                retention=config.logging.retention,
                enqueue=True,
                backtrace=True,
                diagnose=True,
                catch=True,
                serialize=False,  # Changed to False for readable logs
            )

            # Add console logging for development/debugging
            logger.add(
                sys.stderr,
                level=config.logging.level,
                format=config.logging.format,
                colorize=True,
                enqueue=True,
                backtrace=False,
                diagnose=False,
                catch=True,
            )

        # Configure standard logging to work with Loguru
        # This ensures third-party libraries using standard logging are captured
        class InterceptHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                # Get corresponding Loguru level if it exists
                level: str
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = str(record.levelno)

                # Find caller from where record originated
                frame: Optional[FrameType] = sys._getframe(6)
                depth: int = 6
                while frame and frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1

                logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

        # Replace standard logging
        if config.logging:
            logging.basicConfig(handlers=[InterceptHandler()], level=getattr(logging, config.logging.level), force=True)
        else:
            logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)

        # Disable some noisy loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)

        cls._initialized = True
        logger.info("Centralized logging initialized successfully")


# Initialize logger on module import
LoggerSetup.setup_logger()

# Export the configured logger instance
LOGGER = logger
