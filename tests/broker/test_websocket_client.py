import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.broker.websocket_client import KiteWebSocketClient
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

# Load configuration for tests
config = config_loader.get_config()


@pytest.fixture
def mock_token_manager():
    with patch("src.broker.websocket_client.TokenManager") as MockTokenManager:
        mock_instance = MockTokenManager.return_value
        yield mock_instance


@pytest.fixture
def mock_kite_ticker():
    with patch("src.broker.websocket_client.KiteTicker") as MockKiteTicker:
        mock_instance = MockKiteTicker.return_value
        yield mock_instance


@pytest.mark.asyncio
async def test_initialize_kws_with_token(mock_token_manager, mock_kite_ticker):
    logger.info("\n--- Starting test_initialize_kws_with_token ---")
    mock_token_manager.get_access_token.return_value = "test_access_token"
    ws_client = KiteWebSocketClient(lambda x: None, lambda x: None, lambda: None)
    ws_client._initialize_kws()
    mock_kite_ticker.assert_called_once_with(config.broker.api_key, "test_access_token")
    logger.info("test_initialize_kws_with_token completed successfully.")


@pytest.mark.asyncio
async def test_initialize_kws_without_token(mock_token_manager, mock_kite_ticker):
    logger.info("\n--- Starting test_initialize_kws_without_token ---")
    mock_token_manager.get_access_token.return_value = None
    ws_client = KiteWebSocketClient(lambda x: None, lambda x: None, lambda: None)
    with pytest.raises(ValueError, match="Access token is required to initialize KiteTicker."):
        ws_client._initialize_kws()
    mock_kite_ticker.assert_not_called()
    logger.info("test_initialize_kws_without_token completed successfully.")


@pytest.mark.asyncio
async def test_connect_disconnect(mock_token_manager, mock_kite_ticker):
    logger.info("\n--- Starting test_connect_disconnect ---")
    mock_token_manager.get_access_token.return_value = "test_access_token"
    ws_client = KiteWebSocketClient(lambda x: None, lambda x: None, lambda: None)

    # Mock the blocking connect call
    with patch("src.broker.websocket_client.asyncio.to_thread", new=AsyncMock()):
        await ws_client.connect()
        ws_client.kws.connect.assert_called_once()

        ws_client.kws.is_connected.return_value = True
        await ws_client.disconnect()
        ws_client.kws.close.assert_called_once()
    logger.info("test_connect_disconnect completed successfully.")


@pytest.mark.asyncio
async def test_subscribe_unsubscribe(mock_token_manager, mock_kite_ticker):
    logger.info("\n--- Starting test_subscribe_unsubscribe ---")
    mock_token_manager.get_access_token.return_value = "test_access_token"
    ws_client = KiteWebSocketClient(lambda x: None, lambda x: None, lambda: None)
    ws_client._initialize_kws()  # Manually initialize kws for testing subscribe
    ws_client.kws.is_connected.return_value = True

    instrument_tokens = [1, 2, 3]
    ws_client.subscribe(instrument_tokens)
    ws_client.kws.subscribe.assert_called_once_with(instrument_tokens)
    ws_client.kws.set_mode.assert_called_once_with(ws_client.kws.MODE_FULL, instrument_tokens)

    ws_client.unsubscribe(instrument_tokens)
    ws_client.kws.unsubscribe.assert_called_once_with(instrument_tokens)
    logger.info("test_subscribe_unsubscribe completed successfully.")


@pytest.mark.asyncio
async def test_on_ticks_callback():
    logger.info("\n--- Starting test_on_ticks_callback ---")
    mock_callback = MagicMock()
    ws_client = KiteWebSocketClient(mock_callback, lambda x: None, lambda: None)

    test_ticks = [{"instrument_token": 1, "last_traded_price": 100}]
    ws_client._on_ticks(None, test_ticks)  # ws argument is usually None for KiteTicker callbacks
    mock_callback.assert_called_once_with(test_ticks)
    logger.info("test_on_ticks_callback completed successfully.")


# Example of running tests directly (usually done via `pytest` command)
if __name__ == "__main__":
    logger.info("Running WebSocket client tests directly. Use `pytest` for proper test execution.")
    asyncio.run(test_initialize_kws_with_token(MagicMock(), MagicMock()))
    asyncio.run(test_initialize_kws_without_token(MagicMock(), MagicMock()))
    asyncio.run(test_connect_disconnect(MagicMock(), MagicMock()))
    asyncio.run(test_subscribe_unsubscribe(MagicMock(), MagicMock()))
    asyncio.run(test_on_ticks_callback())
