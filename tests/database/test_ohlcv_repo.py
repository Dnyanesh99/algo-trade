import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.db_utils import DatabaseManager
from src.database.ohlcv_repo import OHLCVRepository
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

# Load configuration for tests
config = config_loader.get_config()

@pytest.fixture
async def setup_ohlcv_repo():
    # Ensure a clean state for the singleton DatabaseManager
    DatabaseManager._instance = None
    DatabaseManager._pool = None
    db_manager_instance = DatabaseManager()

    with patch('src.database.db_utils.asyncpg.create_pool', new=AsyncMock()):
        await db_manager_instance.initialize()

        # Mock the db_manager for the repository
        with patch('src.database.ohlcv_repo.db_manager', new=db_manager_instance):
            repo = OHLCVRepository()
            yield repo

    await db_manager_instance.close()

@pytest.mark.asyncio
async def test_insert_ohlcv_data(setup_ohlcv_repo):
    logger.info("\n--- Starting test_insert_ohlcv_data ---")
    repo = setup_ohlcv_repo

    instrument_id = 1
    timeframe = "5min"
    sample_ohlcv = [
        {"date": datetime(2023, 1, 1, 9, 15), "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000, "oi": 500},
        {"date": datetime(2023, 1, 1, 9, 20), "open": 100.5, "high": 102, "low": 100, "close": 101.5, "volume": 1200, "oi": 550},
    ]

    repo.db_manager.transaction = AsyncMock()
    repo.db_manager.transaction.return_value.__aenter__.return_value.executemany = AsyncMock(return_value="INSERT 2")

    status = await repo.insert_ohlcv_data(instrument_id, timeframe, sample_ohlcv)
    assert status == "INSERT 2"
    repo.db_manager.transaction.return_value.__aenter__.return_value.executemany.assert_called_once()
    logger.info("test_insert_ohlcv_data completed successfully.")

@pytest.mark.asyncio
async def test_get_ohlcv_data(setup_ohlcv_repo):
    logger.info("\n--- Starting test_get_ohlcv_data ---")
    repo = setup_ohlcv_repo

    instrument_id = 1
    timeframe = "5min"
    start_time = datetime(2023, 1, 1, 9, 0)
    end_time = datetime(2023, 1, 1, 10, 0)

    mock_records = [MagicMock(ts=datetime(2023,1,1,9,15), open=100), MagicMock(ts=datetime(2023,1,1,9,20), open=100.5)]
    repo.db_manager.fetch_rows = AsyncMock(return_value=mock_records)

    rows = await repo.get_ohlcv_data(instrument_id, timeframe, start_time, end_time)
    assert len(rows) == 2
    repo.db_manager.fetch_rows.assert_called_once()
    logger.info("test_get_ohlcv_data completed successfully.")

@pytest.mark.asyncio
async def test_get_latest_candle(setup_ohlcv_repo):
    logger.info("\n--- Starting test_get_latest_candle ---")
    repo = setup_ohlcv_repo

    instrument_id = 1
    timeframe = "5min"

    mock_record = MagicMock(ts=datetime(2023,1,1,9,25), open=101.5)
    repo.db_manager.fetch_row = AsyncMock(return_value=mock_record)

    candle = await repo.get_latest_candle(instrument_id, timeframe)
    assert candle is not None
    assert candle.ts == datetime(2023,1,1,9,25)
    repo.db_manager.fetch_row.assert_called_once()
    logger.info("test_get_latest_candle completed successfully.")

@pytest.mark.asyncio
async def test_get_ohlcv_data_for_features(setup_ohlcv_repo):
    logger.info("\n--- Starting test_get_ohlcv_data_for_features ---")
    repo = setup_ohlcv_repo

    instrument_id = 1
    timeframe = "5min"
    limit = 2

    mock_records = [
        MagicMock(ts=datetime(2023,1,1,9,20), open=100.5, high=102, low=100, close=101.5, volume=1200),
        MagicMock(ts=datetime(2023,1,1,9,15), open=100, high=101, low=99, close=100.5, volume=1000),
    ]
    repo.db_manager.fetch_rows = AsyncMock(return_value=mock_records)

    rows = await repo.get_ohlcv_data_for_features(instrument_id, timeframe, limit)
    assert len(rows) == 2
    # Assert order is reversed (chronological)
    assert rows[0].ts == datetime(2023,1,1,9,15)
    assert rows[1].ts == datetime(2023,1,1,9,20)
    repo.db_manager.fetch_rows.assert_called_once()
    logger.info("test_get_ohlcv_data_for_features completed successfully.")


# Example of running tests directly (usually done via `pytest` command)
if __name__ == "__main__":
    logger.info("Running OHLCV repository tests directly. Use `pytest` for proper test execution.")
    # Note: Running fixtures directly like this is not how pytest is designed.
    # This is just for illustrative purposes if someone wanted to run without pytest setup.
    # Proper pytest setup involves `pytest.main()` or running `pytest` from CLI.
    asyncio.run(test_insert_ohlcv_data(MagicMock()))
    asyncio.run(test_get_ohlcv_data(MagicMock()))
    asyncio.run(test_get_latest_candle(MagicMock()))
    asyncio.run(test_get_ohlcv_data_for_features(MagicMock()))
