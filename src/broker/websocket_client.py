import asyncio
import json
from typing import Any, Callable, Optional

from kiteconnect import KiteTicker

from src.auth.token_manager import TokenManager
from src.utils.config_loader import BrokerConfig
from src.utils.logger import LOGGER as logger  # Centralized logger


class KiteWebSocketClient:
    """
    Manages WebSocket connection to Zerodha KiteConnect for real-time data streaming.
    Handles connection, subscription, and incoming message parsing with robust error handling.
    """

    def __init__(
        self,
        broker_config: BrokerConfig,
        on_ticks_callback: Callable[[list[dict[str, Any]]], None],
        on_reconnect_callback: Callable[[int, Optional[float]], Any],
        on_noreconnect_callback: Callable[[], Any],
        on_connect_callback: Optional[Callable[[], Any]] = None,
    ):
        if not all([isinstance(broker_config, BrokerConfig), broker_config.api_key]):
            logger.critical("CRITICAL: KiteWebSocketClient initialized with invalid or incomplete BrokerConfig.")
            raise ValueError("broker_config with a valid api_key is required.")
        if not all(callable(c) for c in [on_ticks_callback, on_reconnect_callback, on_noreconnect_callback]):
            logger.critical("CRITICAL: KiteWebSocketClient initialized with one or more invalid callbacks.")
            raise TypeError("All core callback handlers must be callable.")

        self.broker_config = broker_config
        self.api_key = broker_config.api_key
        self.token_manager = TokenManager()
        self.kws: Optional[KiteTicker] = None

        self.on_ticks_callback = on_ticks_callback
        self.on_reconnect_callback = on_reconnect_callback
        self.on_noreconnect_callback = on_noreconnect_callback
        self.on_connect_callback = on_connect_callback
        logger.info("KiteWebSocketClient instantiated.")

    def _initialize_kws(self) -> None:
        """
        Initializes the KiteTicker (WebSocket) client, ensuring a valid access token is present.

        Raises:
            ValueError: If the access token is missing, preventing initialization.
        """
        logger.info("Initializing KiteTicker client...")
        access_token = self.token_manager.get_access_token()
        if not access_token:
            logger.critical("CRITICAL: Access token not available. Cannot initialize KiteTicker.")
            raise ValueError("A valid access token is required to initialize the WebSocket client.")

        try:
            self.kws = KiteTicker(self.api_key, access_token)

            # Assign all callbacks for robust event handling
            self.kws.on_connect = self._on_connect
            self.kws.on_close = self._on_close
            self.kws.on_error = self._on_error
            self.kws.on_reconnect = self._on_reconnect
            self.kws.on_noreconnect = self._on_noreconnect
            self.kws.on_ticks = self._on_ticks
            self.kws.on_order_update = self._on_order_update

            logger.info("KiteTicker client initialized successfully and all callbacks are assigned.")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to instantiate KiteTicker: {e}", exc_info=True)
            raise RuntimeError("KiteTicker instantiation failed.") from e

    def _on_connect(self, ws: Any, response: Any) -> None:
        """
        Callback for successful WebSocket connection.
        """
        logger.info(f"Kite WebSocket connected successfully. Response: {response}")
        if self.on_connect_callback:
            try:
                self.on_connect_callback()
            except Exception as e:
                logger.error(f"Error executing on_connect_callback: {e}", exc_info=True)

    def _on_close(self, ws: Any, code: int, reason: str) -> None:
        """
        Callback for WebSocket disconnection. This is often a precursor to reconnection attempts.
        """
        logger.warning(f"Kite WebSocket connection closed. Code: {code}, Reason: {reason}")

    def _on_error(self, ws: Any, code: int, reason: str) -> None:
        """
        Callback for WebSocket errors. These are often critical.
        """
        logger.error(f"Kite WebSocket encountered an error. Code: {code}, Reason: {reason}")

    def _on_reconnect(self, ws: Any, attempt_count: int) -> None:
        """
        Callback for WebSocket reconnection attempts. Passes data to the ConnectionManager.
        """
        logger.info(f"Kite WebSocket attempting to reconnect: Attempt {attempt_count}")
        if self.on_reconnect_callback is not None:
            try:
                # The last_connect_time is not directly available here, so we pass None.
                # The ConnectionManager is responsible for tracking time if needed.
                self.on_reconnect_callback(attempt_count, None)
            except Exception as e:
                logger.error(f"Error executing on_reconnect_callback: {e}", exc_info=True)

    def _on_noreconnect(self, ws: Any) -> None:
        """
        Callback when WebSocket reconnection attempts fail permanently. This is a critical state.
        """
        logger.critical("CRITICAL: Kite WebSocket has failed all reconnection attempts.")
        if self.on_noreconnect_callback is not None:
            try:
                self.on_noreconnect_callback()
            except Exception as e:
                logger.error(f"Error executing on_noreconnect_callback: {e}", exc_info=True)

    def _on_ticks(self, ws: Any, ticks: list[dict[str, Any]]) -> None:
        """
        Callback for incoming tick data. Passes ticks to the designated handler.
        """
        logger.debug(f"Received {len(ticks)} ticks from WebSocket.")
        try:
            self.on_ticks_callback(ticks)
        except Exception as e:
            logger.error(f"An error occurred in the on_ticks_callback function: {e}", exc_info=True)

    def _on_order_update(self, ws: Any, data: Any) -> None:
        """
        Callback for order update events.
        """
        logger.info("Order update received from WebSocket.")
        try:
            logger.info(json.dumps(data, indent=2))
        except (TypeError, json.JSONDecodeError):
            logger.info(f"Raw order update data: {data}")

    async def connect(self) -> None:
        """
        Establishes the WebSocket connection in a non-blocking manner.

        Raises:
            RuntimeError: If the client fails to connect.
        """
        try:
            if not self.kws:
                self._initialize_kws()
        except (ValueError, RuntimeError) as e:
            logger.critical(f"Failed to connect: {e}")
            raise  # Re-raise the critical error

        if self.kws and not self.kws.is_connected():
            logger.info("Attempting to establish Kite WebSocket connection...")
            try:
                # The connect method is blocking, so it must be run in a separate thread.
                await asyncio.to_thread(self.kws.connect, threaded=True)
                logger.info("Kite WebSocket connection process has been initiated in a background thread.")
            except Exception as e:
                logger.critical(
                    f"CRITICAL: An exception occurred while trying to initiate the WebSocket connection: {e}",
                    exc_info=True,
                )
                raise RuntimeError("Failed to initiate WebSocket connection.") from e
        elif self.kws and self.kws.is_connected():
            logger.warning("Connect call made, but Kite WebSocket is already connected.")
        else:
            logger.critical("CRITICAL: Cannot connect because KiteTicker client (kws) is not initialized.")

    async def disconnect(self) -> None:
        """
        Closes the WebSocket connection gracefully.
        """
        if self.kws and self.kws.is_connected():
            logger.info("Disconnecting from Kite WebSocket...")
            try:
                # The close method can also be blocking
                await asyncio.to_thread(self.kws.close)
                logger.info("Kite WebSocket disconnected successfully.")
            except Exception as e:
                logger.error(f"An error occurred during WebSocket disconnection: {e}", exc_info=True)
        else:
            logger.info("Disconnect call made, but WebSocket is not currently connected.")

    def subscribe(self, instrument_tokens: list[int]) -> None:
        """
        Subscribes to a list of instrument tokens for tick data.
        """
        if not self.kws or not self.kws.is_connected():
            logger.error("Cannot subscribe: Kite WebSocket is not connected or initialized.")
            return
        if not instrument_tokens:
            logger.warning("Subscribe called with an empty list of instrument tokens.")
            return

        logger.info(f"Subscribing to {len(instrument_tokens)} instrument tokens.")
        try:
            self.kws.subscribe(instrument_tokens)
            self._set_subscription_mode(instrument_tokens)
        except Exception as e:
            logger.error(f"An error occurred during subscription: {e}", exc_info=True)

    def _set_subscription_mode(self, instrument_tokens: list[int]) -> None:
        """Sets the subscription mode for the given tokens."""
        if not self.kws:
            return

        mode_map = {
            "FULL": self.kws.MODE_FULL,
            "QUOTE": self.kws.MODE_QUOTE,
            "LTP": self.kws.MODE_LTP,
        }

        mode_string = self.broker_config.websocket_mode
        if not mode_string or mode_string.upper() not in mode_map:
            logger.error(
                f"CRITICAL: Invalid websocket_mode '{mode_string}' in config.yaml. "
                f"Must be one of {list(mode_map.keys())}. Defaulting to FULL."
            )
            mode_string = "FULL"

        selected_mode = mode_map[mode_string.upper()]
        logger.info(f"Setting subscription mode to '{mode_string.upper()}' for {len(instrument_tokens)} tokens.")
        try:
            self.kws.set_mode(selected_mode, instrument_tokens)
        except Exception as e:
            logger.error(f"An error occurred while setting subscription mode: {e}", exc_info=True)

    def unsubscribe(self, instrument_tokens: list[int]) -> None:
        """
        Unsubscribes from a list of instrument tokens.
        """
        if not self.kws or not self.kws.is_connected():
            logger.error("Cannot unsubscribe: Kite WebSocket is not connected or initialized.")
            return
        if not instrument_tokens:
            logger.warning("Unsubscribe called with an empty list of instrument tokens.")
            return

        logger.info(f"Unsubscribing from {len(instrument_tokens)} instrument tokens.")
        try:
            self.kws.unsubscribe(instrument_tokens)
        except Exception as e:
            logger.error(f"An error occurred during unsubscription: {e}", exc_info=True)

    def is_connected(self) -> bool:
        """
        Checks if the WebSocket is currently connected.
        """
        return self.kws.is_connected() if self.kws else False
