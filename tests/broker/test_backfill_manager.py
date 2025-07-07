import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.broker.backfill_manager import BackfillManager
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

# Load configuration for tests
config = config_loader.get_config()

@pytest.fixture
def mock_rest_client():
    with patch('src.broker.backfill_manager.KiteRESTClient') as MockKiteRESTClient:
        mock_instance = MockKiteRESTClient.return_value
        yield mock_instance

@pytest.mark.asyncio
async def test_fetch_missing_data_success(mock_rest_client):
    logger.info("\n--- Starting test_fetch_missing_data_success ---")
    backfill_manager = BackfillManager(mock_rest_client)

    instrument_token = 12345
    last_received_timestamp = datetime.now() - timedelta(minutes=5)

    mock_rest_client.get_historical_data.return_value = [
        {"date": "2023-01-01 09:01:00", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 100},
        {"date": "2023-01-01 09:02:00", "open": 100.5, "high": 101.5, "low": 100, "close": 101, "volume": 120},
    ]

    missing_candles = await backfill_manager.fetch_missing_data(instrument_token, last_received_timestamp)

    assert missing_candles is not None
    assert len(missing_candles) == 2
    mock_rest_client.get_historical_data.assert_called_once()
    logger.info("test_fetch_missing_data_success completed successfully.")

@pytest.mark.asyncio
async def test_fetch_missing_data_no_candles(mock_rest_client):
    logger.info("\n--- Starting test_fetch_missing_data_no_candles ---")
    backfill_manager = BackfillManager(mock_rest_client)

    instrument_token = 12345
    last_received_timestamp = datetime.now() - timedelta(minutes=5)

    mock_rest_client.get_historical_data.return_value = []

    missing_candles = await backfill_manager.fetch_missing_data(instrument_token, last_received_timestamp)

    assert missing_candles is None
    mock_rest_client.get_historical_data.assert_called_once()
    logger.info("test_fetch_missing_data_no_candles completed successfully.")

@pytest.mark.asyncio
async def test_fetch_missing_data_error(mock_rest_client):
    logger.info("\n--- Starting test_fetch_missing_data_error ---")
    backfill_manager = BackfillManager(mock_rest_client)

    instrument_token = 12345
    last_received_timestamp = datetime.now() - timedelta(minutes=5)

    mock_rest_client.get_historical_data.side_effect = Exception("API Error")

    missing_candles = await backfill_manager.fetch_missing_data(instrument_token, last_received_timestamp)

    assert missing_candles is None
    mock_rest_client.get_historical_data.assert_called_once()
    logger.info("test_fetch_missing_data_error completed successfully.")

@pytest.mark.asyncio
async def test_merge_with_live_stream():
    logger.info("\n--- Starting test_merge_with_live_stream ---")
    # This test will remain a placeholder until actual merge logic is implemented
    backfill_manager = BackfillManager(MagicMock())

    missing_data = [{"date": "2023-01-01", "open": 100}]
    await backfill_manager.merge_with_live_stream(missing_data)
    # Assertions will be added once merge logic is concrete
    logger.info("test_merge_with_live_stream completed successfully (placeholder). ")


# Example of running tests directly (usually done via `pytest` command)
if __name__ == "__main__":
    logger.info("Running backfill manager tests directly. Use `pytest` for proper test execution.")
    asyncio.run(test_fetch_missing_data_success(MagicMock()))
    asyncio.run(test_fetch_missing_data_no_candles(MagicMock()))
    asyncio.run(test_fetch_missing_data_error(MagicMock()))
    asyncio.run(test_merge_with_live_stream())
