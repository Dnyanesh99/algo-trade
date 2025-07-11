import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from main import main_async

from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

# Load configuration for tests
config = config_loader.get_config()

@pytest.mark.asyncio
async def test_main_async_historical_mode():
    """
    Tests main_async function in HISTORICAL_MODE.
    Mocks dependencies to isolate main_async logic.
    """
    logger.info("\n--- Starting test_main_async_historical_mode ---")

    with (
        patch('main.KiteAuthenticator') as MockAuthenticator,
        patch('main.TokenManager') as MockTokenManager,
        patch('main.KiteRESTClient') as MockRESTClient,
        patch('main.InstrumentRepo') as MockInstrumentRepo,
        patch('main.InstrumentManager') as MockInstrumentManager,
        patch('main.config') as MockConfig,
    ):

        # Configure mocks
        mock_authenticator_instance = MockAuthenticator.return_value
        mock_token_manager_instance = MockTokenManager.return_value
        mock_rest_client_instance = MockRESTClient.return_value
        mock_instrument_repo_instance = MockInstrumentRepo.return_value
        mock_instrument_manager_instance = MockInstrumentManager.return_value

        mock_token_manager_instance.is_token_available.return_value = True
        mock_instrument_manager_instance.sync_instruments = AsyncMock()

        MockConfig.system.mode = "HISTORICAL_MODE"
        MockConfig.system.version = "test_version"
        MockConfig.system.timezone = "test_timezone"

        # Run the main_async function
        await main_async()

        # Assertions for HISTORICAL_MODE
        mock_token_manager_instance.is_token_available.assert_called_once()
        mock_instrument_manager_instance.sync_instruments.assert_called_once()
        logger.info("test_main_async_historical_mode completed successfully.")

@pytest.mark.asyncio
async def test_main_async_live_mode_success():
    """
    Tests main_async function in LIVE_MODE with successful connection and subscription.
    """
    logger.info("\n--- Starting test_main_async_live_mode_success ---")

    with patch('main.KiteAuthenticator') as MockAuthenticator, \
         patch('main.TokenManager') as MockTokenManager, \
         patch('main.KiteRESTClient') as MockRESTClient, \
         patch('main.InstrumentRepo') as MockInstrumentRepo, \
         patch('main.InstrumentManager') as MockInstrumentManager, \
         patch('main.SystemState') as MockSystemState, \
         patch('main.MarketCalendar') as MockMarketCalendar, \
         patch('main.TimeSynchronizer') as MockTimeSynchronizer, \
         patch('main.TickQueue') as MockTickQueue, \
         patch('main.TickValidator') as MockTickValidator, \
         patch('main.CandleBuffer') as MockCandleBuffer, \
         patch('main.StreamConsumer') as MockStreamConsumer, \
         patch('main.KiteWebSocketClient') as MockWebSocketClient, \
         patch('main.Scheduler') as MockScheduler, \
         patch('main.config') as MockConfig:

        # Configure mocks
        mock_authenticator_instance = MockAuthenticator.return_value
        mock_token_manager_instance = MockTokenManager.return_value
        mock_rest_client_instance = MockRESTClient.return_value
        mock_instrument_repo_instance = MockInstrumentRepo.return_value
        mock_instrument_manager_instance = MockInstrumentManager.return_value
        mock_system_state_instance = MockSystemState.return_value
        mock_market_calendar_instance = MockMarketCalendar.return_value
        mock_time_synchronizer_instance = MockTimeSynchronizer.return_value
        mock_tick_queue_instance = MockTickQueue.return_value
        mock_tick_validator_instance = MockTickValidator.return_value
        mock_candle_buffer_instance = MockCandleBuffer.return_value
        mock_stream_consumer_instance = MockStreamConsumer.return_value
        mock_websocket_client_instance = MockWebSocketClient.return_value
        mock_scheduler_instance = MockScheduler.return_value

        mock_token_manager_instance.is_token_available.return_value = True
        mock_instrument_manager_instance.sync_instruments = AsyncMock()
        mock_instrument_manager_instance.get_configured_instrument_tokens.return_value = [1, 2, 3]

        mock_websocket_client_instance.connect = AsyncMock()
        mock_websocket_client_instance.is_connected.return_value = True
        mock_websocket_client_instance.subscribe = MagicMock()
        mock_websocket_client_instance.disconnect = AsyncMock()

        mock_scheduler_instance.start = MagicMock()
        mock_scheduler_instance.stop = AsyncMock()

        MockConfig.system.mode = "LIVE_MODE"
        MockConfig.system.version = "test_version"
        MockConfig.system.timezone = "test_timezone"

        # Run the main_async function
        # We need to prevent the infinite loop in main_async for testing
        with patch('main.asyncio.sleep', new=AsyncMock(side_effect=[None, asyncio.CancelledError])):
            try:
                await main_async()
            except asyncio.CancelledError:
                pass # Expected to break the loop

        # Assertions for LIVE_MODE
        mock_token_manager_instance.is_token_available.assert_called_once()
        mock_instrument_manager_instance.sync_instruments.assert_called_once()
        mock_websocket_client_instance.connect.assert_called_once()
        mock_websocket_client_instance.is_connected.assert_called_once()
        mock_instrument_manager_instance.get_configured_instrument_tokens.assert_called_once()
        mock_websocket_client_instance.subscribe.assert_called_once_with([1, 2, 3])
        mock_scheduler_instance.start.assert_called_once()
        mock_scheduler_instance.stop.assert_called_once()
        mock_websocket_client_instance.disconnect.assert_called_once()
        logger.info("test_main_async_live_mode_success completed successfully.")

@pytest.mark.asyncio
async def test_main_async_live_mode_websocket_fail():
    """
    Tests main_async function in LIVE_MODE when WebSocket connection fails.
    """
    logger.info("\n--- Starting test_main_async_live_mode_websocket_fail ---")

    with patch('main.KiteAuthenticator') as MockAuthenticator, \
         patch('main.TokenManager') as MockTokenManager, \
         patch('main.KiteRESTClient') as MockRESTClient, \
         patch('main.InstrumentRepo') as MockInstrumentRepo, \
         patch('main.InstrumentManager') as MockInstrumentManager, \
         patch('main.SystemState') as MockSystemState, \
         patch('main.MarketCalendar') as MockMarketCalendar, \
         patch('main.TimeSynchronizer') as MockTimeSynchronizer, \
         patch('main.TickQueue') as MockTickQueue, \
         patch('main.TickValidator') as MockTickValidator, \
         patch('main.CandleBuffer') as MockCandleBuffer, \
         patch('main.StreamConsumer') as MockStreamConsumer, \
         patch('main.KiteWebSocketClient') as MockWebSocketClient, \
         patch('main.Scheduler') as MockScheduler, \
         patch('main.config') as MockConfig:

        # Configure mocks
        mock_token_manager_instance = MockTokenManager.return_value
        mock_instrument_manager_instance = MockInstrumentManager.return_value
        mock_websocket_client_instance = MockWebSocketClient.return_value
        mock_scheduler_instance = MockScheduler.return_value

        mock_token_manager_instance.is_token_available.return_value = True
        mock_instrument_manager_instance.sync_instruments = AsyncMock()

        mock_websocket_client_instance.connect = AsyncMock()
        mock_websocket_client_instance.is_connected.return_value = False # Simulate connection failure
        mock_websocket_client_instance.disconnect = AsyncMock()

        mock_scheduler_instance.stop = AsyncMock()

        MockConfig.system.mode = "LIVE_MODE"
        MockConfig.system.version = "test_version"
        MockConfig.system.timezone = "test_timezone"

        # Run the main_async function
        await main_async()

        # Assertions for LIVE_MODE with WebSocket failure
        mock_token_manager_instance.is_token_available.assert_called_once()
        mock_instrument_manager_instance.sync_instruments.assert_called_once()
        mock_websocket_client_instance.connect.assert_called_once()
        mock_websocket_client_instance.is_connected.assert_called_once()
        mock_websocket_client_instance.subscribe.assert_not_called() # Should not subscribe if not connected
        mock_scheduler_instance.start.assert_not_called() # Should not start scheduler
        mock_scheduler_instance.stop.assert_called_once() # Should still attempt to stop scheduler
        mock_websocket_client_instance.disconnect.assert_called_once()
        logger.info("test_main_async_live_mode_websocket_fail completed successfully.")


if __name__ == "__main__":
    # This block is for running tests directly, typically you'd use `pytest`
    # For demonstration, you can run specific tests here.
    asyncio.run(test_main_async_historical_mode())
    asyncio.run(test_main_async_live_mode_success())
    asyncio.run(test_main_async_live_mode_websocket_fail())
