import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

import psutil
import pytz

from src.broker.websocket_client import KiteWebSocketClient
from src.database.db_utils import DatabaseManager
from src.metrics import metrics_registry
from src.state.system_state import SystemState
from src.utils.config_loader import HealthMonitorConfig, SystemConfig
from src.utils.logger import LOGGER as logger


class HealthMonitor:
    """
    Actively checks the status of various system components like WebSocket connection,
    database connectivity, and data freshness.
    """

    def __init__(
        self,
        system_state: SystemState,
        ws_client: Optional[KiteWebSocketClient],
        db_manager: DatabaseManager,
        health_monitor_config: HealthMonitorConfig,
        system_config: SystemConfig,
    ):
        self.system_state = system_state
        self.ws_client = ws_client
        self.db_manager = db_manager
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_data_timestamp: dict[int, datetime] = {}
        self._monitor_interval = health_monitor_config.monitor_interval
        self.data_freshness_threshold = timedelta(minutes=health_monitor_config.data_freshness_threshold_minutes)
        self.system_timezone = system_config.timezone

    def update_last_data_timestamp(self, instrument_token: int, timestamp: datetime) -> None:
        """
        Updates the timestamp of the last received data for a given instrument.
        """
        self._last_data_timestamp[instrument_token] = timestamp
        logger.debug(f"Last data timestamp updated for {instrument_token}: {timestamp}")

    async def _check_websocket_health(self) -> None:
        """
        Checks the health of the WebSocket connection.
        """
        if self.ws_client is None:
            health_info = {"status": "not_available"}
            self.system_state.update_component_health("WebSocket", health_info)
            return

        is_connected = self.ws_client.is_connected()
        health_info = {"status": "connected" if is_connected else "disconnected"}
        self.system_state.update_component_health("WebSocket", health_info)
        if not is_connected:
            logger.warning("WebSocket is disconnected.")

    async def _check_database_health(self) -> None:
        """
        Checks the health of the database connection.
        """
        try:
            # Attempt to acquire and release a connection to test connectivity
            async with self.db_manager.get_connection():
                health_info = {"status": "connected"}
                self.system_state.update_component_health("Database", health_info)
                logger.debug("Database is connected.")
        except Exception as e:
            health_info = {"status": "disconnected", "error": str(e)}
            self.system_state.update_component_health("Database", health_info)
            logger.error(f"Database connection failed: {e}")

    async def _check_data_freshness(self) -> None:
        """
        Checks the freshness of the last received data for all tracked instruments.
        """
        current_time = datetime.now(pytz.timezone(self.system_timezone))

        for instrument_token, last_ts in self._last_data_timestamp.items():
            if current_time - last_ts > self.data_freshness_threshold:
                logger.warning(f"Data for instrument {instrument_token} is stale. Last update: {last_ts}")
                self.system_state.update_component_health(
                    f"DataFreshness_{instrument_token}", {"status": "stale", "last_update": last_ts}
                )
            else:
                self.system_state.update_component_health(
                    f"DataFreshness_{instrument_token}", {"status": "fresh", "last_update": last_ts}
                )

    async def _monitor_health(self) -> None:
        """
        Main monitoring loop.
        """
        process = psutil.Process()
        while True:
            try:
                logger.debug("Running health checks...")
                metrics_registry.set_gauge("memory_usage_bytes", process.memory_info().rss, {"component": "system"})
                await self._check_websocket_health()
                await self._check_database_health()
                await self._check_data_freshness()
            except asyncio.CancelledError:
                logger.info("Health monitoring loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Unhandled exception in health monitoring loop: {e}", exc_info=True)

            await asyncio.sleep(self._monitor_interval)

    def start_monitoring(self) -> None:
        """
        Starts the background health monitoring task.
        """
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_health())
            logger.info("Health monitoring started.")

    async def stop_monitoring(self) -> None:
        """
        Stops the background health monitoring task gracefully.
        """
        if self._monitor_task:
            logger.info("Stopping health monitor...")
            self._monitor_task.cancel()

            try:
                await self._monitor_task
            except asyncio.CancelledError:
                logger.info("Health monitor stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping health monitor: {e}")
            finally:
                self._monitor_task = None

    def record_successful_operation(self, operation_name: str, **kwargs: Any) -> None:
        """
        Records a successful operation for monitoring purposes.
        """
        logger.debug(f"Successful operation recorded: {operation_name}")
        # Update system state with successful operation
        self.system_state.update_component_health(
            operation_name, {"status": "success", "timestamp": datetime.now(), **kwargs}
        )
