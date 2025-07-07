import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.broker.rest_client import KiteRESTClient
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

# Load configuration for tests
config = config_loader.get_config()

@pytest.fixture
def mock_token_manager():
    with patch('src.broker.rest_client.TokenManager') as MockTokenManager:
        mock_instance = MockTokenManager.return_value
        yield mock_instance

@pytest.fixture
def mock_kite_connect():
    with patch('src.broker.rest_client.KiteConnect') as MockKiteConnect:
        mock_instance = MockKiteConnect.return_value
        yield mock_instance

@pytest.fixture
def mock_rate_limiter():
    with patch('src.broker.rest_client.RateLimiter') as MockRateLimiter:
        mock_instance = MockRateLimiter.return_value
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        yield mock_instance

@pytest.mark.asyncio
async def test_initialize_kite_client_with_token(mock_token_manager, mock_kite_connect):
    logger.info("\n--- Starting test_initialize_kite_client_with_token ---")
    mock_token_manager.get_access_token.return_value = "test_access_token"
    client = KiteRESTClient()
    mock_kite_connect.assert_called_once_with(api_key=config.broker.api_key, access_token="test_access_token")
    logger.info("test_initialize_kite_client_with_token completed successfully.")

@pytest.mark.asyncio
async def test_initialize_kite_client_without_token(mock_token_manager, mock_kite_connect):
    logger.info("\n--- Starting test_initialize_kite_client_without_token ---")
    mock_token_manager.get_access_token.return_value = None
    client = KiteRESTClient()
    mock_kite_connect.assert_called_once_with(api_key=config.broker.api_key)
    logger.info("test_initialize_kite_client_without_token completed successfully.")

@pytest.mark.asyncio
async def test_update_access_token(mock_token_manager, mock_kite_connect):
    logger.info("\n--- Starting test_update_access_token ---")
    client = KiteRESTClient()
    client.kite = MagicMock()
    client.kite.access_token = "old_token"
    mock_token_manager.get_access_token.return_value = "new_token"

    client._update_access_token()
    client.kite.set_access_token.assert_called_once_with("new_token")
    logger.info("test_update_access_token completed successfully.")

@pytest.mark.asyncio
async def test_get_historical_data(mock_token_manager, mock_kite_connect, mock_rate_limiter):
    logger.info("\n--- Starting test_get_historical_data ---")
    mock_token_manager.get_access_token.return_value = "test_access_token"
    client = KiteRESTClient()
    client.kite.historical_data.return_value = [{"date": "2023-01-01", "open": 100}]

    instrument_token = 256265
    from_date = "2023-01-01"
    to_date = "2023-01-02"
    interval = "minute"

    data = await client.get_historical_data(instrument_token, from_date, to_date, interval)

    assert data == [{"date": "2023-01-01", "open": 100}]
    mock_kite_connect.historical_data.assert_called_once_with(instrument_token, from_date, to_date, interval, continuous=False, ohlc=True)
    mock_rate_limiter.__aenter__.assert_called_once()
    logger.info("test_get_historical_data completed successfully.")

@pytest.mark.asyncio
async def test_get_instruments(mock_token_manager, mock_kite_connect, mock_rate_limiter):
    logger.info("\n--- Starting test_get_instruments ---")
    mock_token_manager.get_access_token.return_value = "test_access_token"
    client = KiteRESTClient()
    client.kite.instruments.return_value = [{"instrument_token": 1, "tradingsymbol": "NIFTY"}]

    instruments = await client.get_instruments(exchange="NSE")

    assert instruments == [{"instrument_token": 1, "tradingsymbol": "NIFTY"}]
    mock_kite_connect.instruments.assert_called_once_with(exchange="NSE")
    mock_rate_limiter.__aenter__.assert_called_once()
    logger.info("test_get_instruments completed successfully.")


# Example of running tests directly (usually done via `pytest` command)
if __name__ == "__main__":
    # This block is primarily for demonstrating the test structure.
    # You would typically run these tests using `pytest` from your terminal.
    logger.info("Running REST client tests directly. Use `pytest` for proper test execution.")
    asyncio.run(test_initialize_kite_client_with_token(MagicMock(), MagicMock()))
    asyncio.run(test_initialize_kite_client_without_token(MagicMock(), MagicMock()))
    asyncio.run(test_update_access_token(MagicMock(), MagicMock()))
    asyncio.run(test_get_historical_data(MagicMock(), MagicMock(), MagicMock()))
    asyncio.run(test_get_instruments(MagicMock(), MagicMock(), MagicMock()))
