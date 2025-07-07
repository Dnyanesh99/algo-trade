import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import asyncpg

from src.database.db_utils import DatabaseManager, db_manager
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

# Load configuration for tests
config = config_loader.get_config()

@pytest.fixture
async def setup_db_manager():
    # Ensure a clean state for the singleton
    DatabaseManager._instance = None
    DatabaseManager._pool = None
    manager = DatabaseManager()
    yield manager
    await manager.close()

@pytest.mark.asyncio
async def test_db_manager_singleton(setup_db_manager):
    logger.info("\n--- Starting test_db_manager_singleton ---")
    manager1 = DatabaseManager()
    manager2 = DatabaseManager()
    assert manager1 is manager2
    logger.info("test_db_manager_singleton completed successfully.")

@pytest.mark.asyncio
async def test_db_manager_initialize_close(setup_db_manager):
    logger.info("\n--- Starting test_db_manager_initialize_close ---")
    manager = setup_db_manager

    with (
        patch('src.database.db_utils.asyncpg.create_pool', new=AsyncMock()) as mock_create_pool,
        patch('src.database.db_utils.logger.info') as mock_logger_info,
    ):
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool

        await manager.initialize()
        mock_create_pool.assert_called_once()
        assert manager._pool is mock_pool
        mock_logger_info.assert_any_call("Database connection pool initialized successfully.")

        await manager.close()
        mock_pool.close.assert_called_once()
        assert manager._pool is None
        mock_logger_info.assert_any_call("Database connection pool closed.")
    logger.info("test_db_manager_initialize_close completed successfully.")

@pytest.mark.asyncio
async def test_db_manager_transaction(setup_db_manager):
    logger.info("\n--- Starting test_db_manager_transaction ---")
    manager = setup_db_manager

    mock_conn = AsyncMock()
    mock_conn.transaction.return_value = AsyncMock()
    manager.get_connection = AsyncMock(return_value=asyncio.AsyncContextManager(mock_conn))

    async with manager.transaction() as conn:
        assert conn is mock_conn
        conn.transaction.return_value.start.assert_called_once()
        # Simulate some operation
        await conn.execute("SELECT 1")

    conn.transaction.return_value.commit.assert_called_once()
    conn.transaction.return_value.rollback.assert_not_called()
    logger.info("test_db_manager_transaction completed successfully.")

@pytest.mark.asyncio
async def test_db_manager_transaction_rollback(setup_db_manager):
    logger.info("\n--- Starting test_db_manager_transaction_rollback ---")
    manager = setup_db_manager

    mock_conn = AsyncMock()
    mock_conn.transaction.return_value = AsyncMock()
    manager.get_connection = AsyncMock(return_value=asyncio.AsyncContextManager(mock_conn))

    with pytest.raises(ValueError):
        async with manager.transaction() as conn:
            await conn.execute("INSERT INTO test (col) VALUES (1)")
            raise ValueError("Simulated error")

    mock_conn.transaction.return_value.rollback.assert_called_once()
    mock_conn.transaction.return_value.commit.assert_not_called()
    logger.info("test_db_manager_transaction_rollback completed successfully.")

@pytest.mark.asyncio
async def test_db_manager_fetch_rows(setup_db_manager):
    logger.info("\n--- Starting test_db_manager_fetch_rows ---")
    manager = setup_db_manager

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = [MagicMock(id=1, name="test1"), MagicMock(id=2, name="test2")]

    manager.get_connection = AsyncMock(return_value=asyncio.AsyncContextManager(mock_conn))

    rows = await manager.fetch_rows("SELECT * FROM test_data")
    assert len(rows) == 2
    mock_conn.fetch.assert_called_once_with("SELECT * FROM test_data")
    logger.info("test_db_manager_fetch_rows completed successfully.")


# Example of running tests directly (usually done via `pytest` command)
if __name__ == "__main__":
    logger.info("Running db_utils tests directly. Use `pytest` for proper test execution.")
    asyncio.run(test_db_manager_singleton(DatabaseManager()))
    asyncio.run(test_db_manager_initialize_close(DatabaseManager()))
    asyncio.run(test_db_manager_transaction(DatabaseManager()))
    asyncio.run(test_db_manager_transaction_rollback(DatabaseManager()))
    asyncio.run(test_db_manager_fetch_rows(DatabaseManager()))
