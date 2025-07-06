import asyncio
from typing import Any, Callable, Optional

from src.broker.websocket_client import KiteWebSocketClient
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger  # Centralized logger

# Load configuration
config = config_loader.get_config()


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
        self.websocket_client = websocket_client
        self.on_data_gap_detected = on_data_gap_detected
        self.on_noreconnect_critical = on_noreconnect_critical
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_interval = config.broker.connection_manager.monitor_interval

        # Load values from config
        self._monitor_interval = config.broker.connection_manager.monitor_interval

        # Assign KiteTicker callbacks to be handled by ConnectionManager
        self.websocket_client.on_reconnect_callback = self._on_reconnect_from_kws
        self.websocket_client.on_noreconnect_callback = self._on_noreconnect_from_kws

    async def _monitor_connection(self) -> None:
        """
        Monitors the WebSocket connection status.
        """
        while True:
            await asyncio.sleep(self._monitor_interval)
            if not self.websocket_client.is_connected():
                logger.critical(
                    "WebSocket is disconnected. KiteTicker failed to reconnect. Manual intervention may be required."
                )
                # Optionally, trigger a system-wide error state or alert here
                self.on_noreconnect_critical()

    def start_monitoring(self) -> None:
        """
        Starts the background task for connection monitoring.
        """
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_connection())
            logger.info("Connection monitoring started.")

    def stop_monitoring(self) -> None:
        """
        Stops the background task for connection monitoring.
        """
        if self._monitor_task:
            self._monitor_task.cancel()
            logger.info("Connection monitoring stopped.")

    def _on_reconnect_from_kws(self, attempt_count: int) -> None:
        """
        Callback from KiteWebSocketClient when KiteTicker successfully reconnects.
        """
        logger.info(f"KiteTicker reconnected successfully (attempt {attempt_count}).")
        self.on_data_gap_detected()  # Trigger data gap detection after reconnect

    def _on_noreconnect_from_kws(self) -> None:
        """
        Callback from KiteWebSocketClient when KiteTicker fails to reconnect.
        """
        logger.critical("KiteTicker could not reconnect. Triggering critical alert.")
        self.on_noreconnect_critical()
