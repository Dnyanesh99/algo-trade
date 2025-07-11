"""
Production-grade centralized logging configuration using Loguru.
Provides thread-safe, high-performance logging with proper configuration management.
"""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

if TYPE_CHECKING:
    from types import FrameType

    from src.utils.config_loader import LoggingConfig


class LoggerSetup:
    """Centralized logger setup with production-grade configuration."""

    _initialized: bool = False

    @classmethod
    def setup_logger(cls, logging_config: "LoggingConfig") -> None:
        """Setup logger with configuration-driven settings."""
        if cls._initialized:
            return

        # Remove default logger to prevent duplicate logs
        logger.remove()

        # Ensure log directory exists
        if logging_config:
            log_file_path = Path(logging_config.file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Add file logging with comprehensive settings
            # For production, consider serialize=True for structured JSON logs
            # which are easier for log aggregation systems (e.g., ELK, Splunk).
            # For production, backtrace and diagnose should be False to avoid leaking sensitive data.
            # These could be enabled in a development environment via environment variables.
            logger.add(
                logging_config.file,
                level=logging_config.level,
                format=logging_config.format,
                rotation=logging_config.rotation,
                compression=logging_config.compression,
                retention=logging_config.retention,
                enqueue=True,
                backtrace=False,
                diagnose=False,
                catch=True,
                serialize=False,  # Changed to False for readable logs
            )

            # Add console logging for development/debugging
            logger.add(
                sys.stderr,
                level=logging_config.level,
                format=logging_config.format,
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
                # Start with depth=1 and walk up the stack to find the actual caller
                depth: int = 1
                frame: Optional[FrameType] = sys._getframe(depth)
                while frame and (frame.f_code.co_filename == logging.__file__ or frame.f_code.co_filename == __file__):
                    depth += 1
                    try:
                        frame = sys._getframe(depth)
                    except ValueError:
                        break

                logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

        # Replace standard logging
        if logging_config:
            logging.basicConfig(handlers=[InterceptHandler()], level=getattr(logging, logging_config.level), force=True)
        else:
            logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)

        # Disable some noisy loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)

        cls._initialized = True
        logger.info("Centralized logging initialized successfully")


# Export the logger instance (initialize in main application)
LOGGER = logger
# Hot reload test comment
