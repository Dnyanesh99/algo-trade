"""
Integration utilities for connecting the FastAPI backend with the existing trading application
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Optional

from src.auth.token_manager import TokenManager
from src.database.db_utils import db_manager
from src.database.feature_repo import FeatureRepository
from src.database.instrument_repo import InstrumentRepository
from src.database.label_repo import LabelRepository
from src.database.ohlcv_repo import OHLCVRepository
from src.database.signal_repo import SignalRepository
from src.state.system_state import SystemState
from src.utils.config_loader import AppConfig, ConfigLoader
from src.utils.logger import logger


class TradingSystemIntegration:
    """
    Integration layer to provide shared, initialized components to the FastAPI backend.
    """

    def __init__(self) -> None:
        self.config: Optional[AppConfig] = None
        self.system_state: Optional[SystemState] = None
        self.repositories: dict[str, Any] = {}
        self.components: dict[str, Any] = {}
        self.background_tasks: set[asyncio.Task[Any]] = set()

    async def initialize(self) -> None:
        """
        Initializes shared components that are safe to be used across the application.
        This should be called once at application startup.
        """
        try:
            logger.info("Initializing trading system integration...")

            # Load configuration
            config_loader = ConfigLoader()
            self.config = config_loader.get_config()
            logger.info(f"Configuration loaded successfully (mode: {self.config.system.mode})")

            # Initialize database
            await db_manager.initialize()
            logger.info("Database initialized successfully")

            # Initialize repositories
            self.repositories = {
                "instrument": InstrumentRepository(),
                "ohlcv": OHLCVRepository(),
                "feature": FeatureRepository(),
                "signal": SignalRepository(),
                "label": LabelRepository(),
            }
            logger.info("Repositories initialized successfully")

            # Initialize system state
            self.system_state = SystemState()
            logger.info("SystemState initialized successfully")

            # Initialize other shared components
            self.components["token_manager"] = TokenManager()

            logger.info("Trading system integration initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize trading system integration: {e}", exc_info=True)
            raise

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            logger.info("Cleaning up trading system integration...")

            # Cancel all background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)

            # Close database connections
            await db_manager.close()

            logger.info("Trading system integration cleanup completed")

        except Exception as e:
            logger.error(f"Failed to cleanup: {e}", exc_info=True)
            raise


# Global integration instance
_integration_instance: Optional[TradingSystemIntegration] = None


async def get_integration() -> TradingSystemIntegration:
    """Get the global integration instance, initializing it if necessary."""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = TradingSystemIntegration()
        await _integration_instance.initialize()
    return _integration_instance


@asynccontextmanager
async def integration_context() -> Any:
    """Context manager for the integration lifecycle."""
    integration = await get_integration()
    try:
        yield integration
    finally:
        await integration.cleanup()
