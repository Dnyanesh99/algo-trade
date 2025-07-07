import asyncio
import json
from typing import Any, Callable, Optional

from kiteconnect import KiteTicker

from src.auth.token_manager import TokenManager
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger  # Centralized logger

# Load configuration
config = config_loader.get_config()


class KiteWebSocketClient:
    """
    Manages WebSocket connection to Zerodha KiteConnect for real-time data streaming.
    Handles connection, subscription, and incoming message parsing.
    """

    def __init__(
        self,
        on_ticks_callback: Callable[[list[dict[str, Any]]], None],
        on_reconnect_callback: Callable[[int], Any],
        on_noreconnect_callback: Callable[[], Any],
    ):
        self.api_key = config.broker.api_key if config.broker and config.broker.api_key else ""
        self.token_manager = TokenManager()
        self.kws: Optional[KiteTicker] = None
        self.on_ticks_callback = on_ticks_callback
        self.on_reconnect_callback = on_reconnect_callback
        self.on_noreconnect_callback = on_noreconnect_callback

    def _initialize_kws(self) -> None:
        """
        Initializes the KiteTicker (WebSocket) client.
        """
        access_token = self.token_manager.get_access_token()
        if not access_token:
            logger.error("Access token not available. Cannot initialize KiteTicker.")
            raise ValueError("Access token is required to initialize KiteTicker.")

        self.kws = KiteTicker(self.api_key, access_token)

        # Assign callbacks
        self.kws.on_connect = self._on_connect
        self.kws.on_close = self._on_close
        self.kws.on_error = self._on_error
        self.kws.on_reconnect = self._on_reconnect
        self.kws.on_noreconnect = self._on_noreconnect
        self.kws.on_ticks = self._on_ticks
        self.kws.on_order_update = self._on_order_update

        logger.info("KiteTicker client initialized.")

    def _on_connect(self, ws: Any, response: Any) -> None:
        """
        Callback for successful WebSocket connection.
        """
        logger.info("Kite WebSocket connected.")
        # self._connected = True # Managed by KiteTicker internally
        # You can subscribe to instruments here or manage subscriptions externally

    def _on_close(self, ws: Any, code: int, reason: str) -> None:
        """
        Callback for WebSocket disconnection.
        """
        logger.warning(f"Kite WebSocket closed. Code: {code}, Reason: {reason}")
        # self._connected = False # Managed by KiteTicker internally

    def _on_error(self, ws: Any, code: int, reason: str) -> None:
        """
        Callback for WebSocket errors.
        """
        logger.error(f"Kite WebSocket error. Code: {code}, Reason: {reason}")

    def _on_reconnect(self, ws: Any, attempt_count: int) -> None:
        """
        Callback for WebSocket reconnection attempts.
        """
        logger.info(f"Kite WebSocket reconnecting: Attempt {attempt_count}")
        if self.on_reconnect_callback is not None:
            self.on_reconnect_callback(attempt_count)

    def _on_noreconnect(self, ws: Any) -> None:
        """
        Callback when WebSocket reconnection attempts fail.
        """
        logger.critical("Kite WebSocket could not reconnect. Manual intervention may be required.")
        if self.on_noreconnect_callback is not None:
            self.on_noreconnect_callback()

    def _on_ticks(self, ws: Any, ticks: list[dict[str, Any]]) -> None:
        """
        Callback for incoming tick data.
        Passes ticks to the provided callback function.
        """
        # logger.debug(f"Received {len(ticks)} ticks.")
        self.on_ticks_callback(ticks)

    def _on_order_update(self, ws: Any, data: Any) -> None:
        """
        Callback for order update events.
        """
        logger.info(f"Order update received: {json.dumps(data)}")

    async def connect(self) -> None:
        """
        Establishes the WebSocket connection.
        """
        if not self.kws:
            self._initialize_kws()

        if self.kws and not self.kws.is_connected():
            logger.info("Connecting to Kite WebSocket...")
            # Run in a separate thread to not block the main event loop
            # KiteTicker's connect method is blocking
            await asyncio.to_thread(self.kws.connect)
            logger.info("Kite WebSocket connection initiated.")
        else:
            logger.info("Kite WebSocket already connected or KWS not initialized.")

    async def disconnect(self) -> None:
        """
        Closes the WebSocket connection.
        """
        if self.kws and self.kws.is_connected():
            logger.info("Disconnecting from Kite WebSocket...")
            self.kws.close()
            # self._connected = False # Managed by KiteTicker internally
            logger.info("Kite WebSocket disconnected.")
        else:
            logger.info("Kite WebSocket not connected.")

    def subscribe(self, instrument_tokens: list[int]) -> None:
        """
        Subscribes to a list of instrument tokens for tick data.
        """
        if self.kws and self.kws.is_connected():
            logger.info(f"Subscribing to {len(instrument_tokens)} instrument tokens.")
            self.kws.subscribe(instrument_tokens)
            mode_map = {
                "FULL": self.kws.MODE_FULL,
                "QUOTE": self.kws.MODE_QUOTE,
                "LTP": self.kws.MODE_LTP,
            }
            selected_mode = mode_map.get(config.broker.websocket_mode.upper() if config.broker and config.broker.websocket_mode else "FULL")
            self.kws.set_mode(selected_mode, instrument_tokens)
        else:
            logger.warning("Cannot subscribe: Kite WebSocket not connected.")

    def unsubscribe(self, instrument_tokens: list[int]) -> None:
        """
        Unsubscribes from a list of instrument tokens.
        """
        if self.kws and self.kws.is_connected():
            logger.info(f"Unsubscribing from {len(instrument_tokens)} instrument tokens.")
            self.kws.unsubscribe(instrument_tokens)
        else:
            logger.warning("Cannot unsubscribe: Kite WebSocket not connected.")

    def is_connected(self) -> bool:
        """
        Checks if the WebSocket is currently connected.
        """
        return self.kws.is_connected() if self.kws else False
