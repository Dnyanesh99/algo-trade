import asyncio
from datetime import datetime
from typing import Any, Callable, Optional

from src.broker.websocket_client import KiteWebSocketClient
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger  # Centralized logger

# Load configuration
config = ConfigLoader().get_config()


class ConnectionManager:
    """
    Manages the WebSocket connection to KiteConnect, handling disconnections,
    reconnection attempts with exponential backoff, and heartbeat processing.
    """

    def __init__(
        self,
        websocket_client: KiteWebSocketClient,
        on_data_gap_detected: Callable[[], Any],
        on_noreconnect_critical: Callable[[], Any],
    ):
        # Type hints ensure correct types for websocket_client and callback functions

        self.websocket_client = websocket_client
        self.on_data_gap_detected = on_data_gap_detected
        self.on_noreconnect_critical = on_noreconnect_critical
        self._monitor_task: Optional[asyncio.Task[None]] = None

        conn_manager_config = config.broker.connection_manager
        if not conn_manager_config or not conn_manager_config.monitor_interval:
            logger.critical(
                "CRITICAL: Broker connection manager configuration ('broker.connection_manager') is missing or incomplete in config.yaml."
            )
            raise ValueError("ConnectionManager configuration is missing or invalid.")

        self._monitor_interval = conn_manager_config.monitor_interval
        if not isinstance(self._monitor_interval, (int, float)) or self._monitor_interval <= 0:
            logger.critical(
                f"CRITICAL: Invalid monitor_interval '{self._monitor_interval}'. Must be a positive number."
            )
            raise ValueError("monitor_interval must be a positive number.")
        # Assign KiteTicker callbacks to be handled by ConnectionManager
        self.websocket_client.on_reconnect_callback = self._on_reconnect_from_kws
        self.websocket_client.on_noreconnect_callback = self._on_noreconnect_from_kws
        logger.info(f"ConnectionManager initialized with a monitor interval of {self._monitor_interval} seconds.")

    async def _monitor_connection(self) -> None:
        """
        Monitors the WebSocket connection status periodically.
        If a disconnection is detected and not handled by the underlying client,
        it triggers a critical failure workflow.
        """
        logger.info("Connection monitoring loop started.")
        while True:
            try:
                await asyncio.sleep(self._monitor_interval)
                if not self.websocket_client.is_connected():
                    logger.critical(
                        "CRITICAL: WebSocket is disconnected. The underlying client has failed to maintain the connection. "
                        "Triggering the critical 'no-reconnect' workflow."
                    )
                    self.on_noreconnect_critical()
                    # Stop monitoring after a critical failure is confirmed and handled.
                    break
                logger.debug("Connection status: OK")
            except asyncio.CancelledError:
                logger.info("Connection monitoring task was cancelled.")
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred in the connection monitor: {e}", exc_info=True)
                # In case of an unexpected error, we should probably stop monitoring to avoid error loops.
                break
        logger.info("Connection monitoring loop has terminated.")

    def start_monitoring(self) -> None:
        """
        Starts the background task for connection monitoring if it is not already running.
        """
        if self._monitor_task and not self._monitor_task.done():
            logger.warning("Attempted to start connection monitoring, but it is already running.")
            return

        self._monitor_task = asyncio.create_task(self._monitor_connection())
        logger.info(
            f"Connection monitoring task has been started. Will check status every {self._monitor_interval} seconds."
        )

    async def stop_monitoring(self) -> None:
        """
        Stops the background task for connection monitoring gracefully.
        """
        if not self._monitor_task or self._monitor_task.done():
            logger.info("Connection monitor is not running, no action needed.")
            return

        logger.info("Stopping connection monitor task...")
        self._monitor_task.cancel()

        try:
            await self._monitor_task
        except asyncio.CancelledError:
            logger.info("Connection monitor task stopped successfully.")
        except Exception as e:
            logger.error(f"An error occurred while stopping the connection monitor: {e}", exc_info=True)
        finally:
            self._monitor_task = None
            logger.info("Connection monitor task cleaned up.")

    def _on_reconnect_from_kws(self, attempt_count: int, last_connect_time: Optional[float]) -> None:
        """
        Callback from KiteWebSocketClient when KiteTicker successfully reconnects.
        Triggers the data gap detection workflow.
        """
        logger.info(f"KiteTicker reconnected successfully after {attempt_count} attempts.")
        if last_connect_time:
            logger.info(f"Last connection time was at: {datetime.fromtimestamp(last_connect_time)}")
        logger.info("Triggering data gap detection and backfill process following reconnection.")
        self.on_data_gap_detected()

    def _on_noreconnect_from_kws(self) -> None:
        """
        Callback from KiteWebSocketClient when KiteTicker fails to reconnect after all attempts.
        This is a critical failure state.
        """
        logger.critical(
            "CRITICAL: KiteTicker has failed to reconnect after all attempts. The system can no longer receive live data."
        )
        logger.critical("Triggering the critical 'no-reconnect' workflow. Manual intervention is likely required.")
        self.on_noreconnect_critical()
