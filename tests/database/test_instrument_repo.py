import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import asyncpg

from src.database.db_utils import DatabaseManager
from src.database.instrument_repo import InstrumentRepository
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

# Load configuration for tests
config = config_loader.get_config()

@pytest.fixture
async def setup_instrument_repo():
    # Ensure a clean state for the singleton DatabaseManager
    DatabaseManager._instance = None
    DatabaseManager._pool = None
    db_manager_instance = DatabaseManager()
    
    with patch('src.database.db_utils.asyncpg.create_pool', new=AsyncMock()):
        await db_manager_instance.initialize()
        
        # Mock the db_manager for the repository
        with patch('src.database.instrument_repo.db_manager', new=db_manager_instance):
            repo = InstrumentRepository()
            yield repo

    await db_manager_instance.close()

@pytest.mark.asyncio
async def test_insert_instrument(setup_instrument_repo):
    logger.info("\n--- Starting test_insert_instrument ---")
    repo = setup_instrument_repo
    
    instrument_data = {
        "instrument_token": 12345,
        "exchange_token": "EXCH123",
        "tradingsymbol": "TESTSYM",
        "name": "Test Symbol",
        "last_price": 100.0,
        "expiry": None,
        "strike": None,
        "tick_size": 0.05,
        "lot_size": 1,
        "instrument_type": "EQ",
        "segment": "NSE",
        "exchange": "NSE",
    }

    repo.db_manager.get_connection.return_value.__aenter__.return_value.fetchval = AsyncMock(return_value=1)

    instrument_id = await repo.insert_instrument(instrument_data)
    assert instrument_id == 1
    repo.db_manager.get_connection.return_value.__aenter__.return_value.fetchval.assert_called_once()
    logger.info("test_insert_instrument completed successfully.")

@pytest.mark.asyncio
async def test_update_instrument(setup_instrument_repo):
    logger.info("\n--- Starting test_update_instrument ---")
    repo = setup_instrument_repo
    
    instrument_id = 1
    instrument_data = {
        "instrument_token": 12345,
        "exchange_token": "EXCH123",
        "tradingsymbol": "UPDATEDSYM",
        "name": "Updated Symbol",
        "last_price": 101.0,
        "expiry": None,
        "strike": None,
        "tick_size": 0.05,
        "lot_size": 1,
        "instrument_type": "EQ",
        "segment": "NSE",
        "exchange": "NSE",
    }

    repo.db_manager.get_connection.return_value.__aenter__.return_value.execute = AsyncMock(return_value="UPDATE 1")

    result = await repo.update_instrument(instrument_id, instrument_data)
    assert result is True
    repo.db_manager.get_connection.return_value.__aenter__.return_value.execute.assert_called_once()
    logger.info("test_update_instrument completed successfully.")

@pytest.mark.asyncio
async def test_get_instrument_by_tradingsymbol(setup_instrument_repo):
    logger.info("\n--- Starting test_get_instrument_by_tradingsymbol ---")
    repo = setup_instrument_repo
    
    tradingsymbol = "TESTSYM"
    exchange = "NSE"
    mock_record = MagicMock(instrument_token=12345, tradingsymbol="TESTSYM")
    repo.db_manager.get_connection.return_value.__aenter__.return_value.fetchrow = AsyncMock(return_value=mock_record)

    instrument = await repo.get_instrument_by_tradingsymbol(tradingsymbol, exchange)
    assert instrument["tradingsymbol"] == "TESTSYM"
    repo.db_manager.get_connection.return_value.__aenter__.return_value.fetchrow.assert_called_once()
    logger.info("test_get_instrument_by_tradingsymbol completed successfully.")

@pytest.mark.asyncio
async def test_get_instrument_by_token(setup_instrument_repo):
    logger.info("\n--- Starting test_get_instrument_by_token ---")
    repo = setup_instrument_repo
    
    instrument_token = 12345
    mock_record = MagicMock(instrument_token=12345, tradingsymbol="TESTSYM")
    repo.db_manager.get_connection.return_value.__aenter__.return_value.fetchrow = AsyncMock(return_value=mock_record)

    instrument = await repo.get_instrument_by_token(instrument_token)
    assert instrument["instrument_token"] == 12345
    repo.db_manager.get_connection.return_value.__aenter__.return_value.fetchrow.assert_called_once()
    logger.info("test_get_instrument_by_token completed successfully.")

@pytest.mark.asyncio
async def test_get_all_instruments(setup_instrument_repo):
    logger.info("\n--- Starting test_get_all_instruments ---")
    repo = setup_instrument_repo
    
    mock_records = [MagicMock(instrument_token=1, tradingsymbol="SYM1"), MagicMock(instrument_token=2, tradingsymbol="SYM2")]
    repo.db_manager.get_connection.return_value.__aenter__.return_value.fetch = AsyncMock(return_value=mock_records)

    instruments = await repo.get_all_instruments()
    assert len(instruments) == 2
    repo.db_manager.get_connection.return_value.__aenter__.return_value.fetch.assert_called_once()
    logger.info("test_get_all_instruments completed successfully.")

@pytest.mark.asyncio
async def test_get_instruments_by_type(setup_instrument_repo):
    logger.info("\n--- Starting test_get_instruments_by_type ---")
    repo = setup_instrument_repo
    
    instrument_type = "EQ"
    exchange = "NSE"
    mock_records = [MagicMock(instrument_token=1, tradingsymbol="SYM1")]
    repo.db_manager.get_connection.return_value.__aenter__.return_value.fetch = AsyncMock(return_value=mock_records)

    instruments = await repo.get_instruments_by_type(instrument_type, exchange)
    assert len(instruments) == 1
    repo.db_manager.get_connection.return_value.__aenter__.return_value.fetch.assert_called_once()
    logger.info("test_get_instruments_by_type completed successfully.")


# Example of running tests directly (usually done via `pytest` command)
if __name__ == "__main__":
    logger.info("Running instrument repository tests directly. Use `pytest` for proper test execution.")
    # Note: Running fixtures directly like this is not how pytest is designed.
    # This is just for illustrative purposes if someone wanted to run without pytest setup.
    # Proper pytest setup involves `pytest.main()` or running `pytest` from CLI.
    asyncio.run(test_insert_instrument(MagicMock()))
    asyncio.run(test_update_instrument(MagicMock()))
    asyncio.run(test_get_instrument_by_tradingsymbol(MagicMock()))
    asyncio.run(test_get_instrument_by_token(MagicMock()))
    asyncio.run(test_get_all_instruments(MagicMock()))
    asyncio.run(test_get_instruments_by_type(MagicMock()))
