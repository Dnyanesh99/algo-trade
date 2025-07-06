import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.broker.connection_manager import ConnectionManager
from src.broker.websocket_client import KiteWebSocketClient
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

# Load configuration for tests
config = config_loader.get_config()

@pytest.fixture
def mock_websocket_client():
    with patch('src.broker.connection_manager.KiteWebSocketClient') as MockKiteWebSocketClient:
        mock_instance = MockKiteWebSocketClient.return_value
        mock_instance.is_connected.return_value = True
        mock_instance.connect = AsyncMock()
        mock_instance.disconnect = AsyncMock()
        yield mock_instance

@pytest.mark.asyncio
async def test_connection_manager_init(mock_websocket_client):
    logger.info("\n--- Starting test_connection_manager_init ---")
    on_data_gap_detected = MagicMock()
    on_noreconnect_critical = MagicMock()
    manager = ConnectionManager(mock_websocket_client, on_data_gap_detected, on_noreconnect_critical)
    assert manager.websocket_client == mock_websocket_client
    assert manager.on_data_gap_detected == on_data_gap_detected
    assert manager.on_noreconnect_critical == on_noreconnect_critical
    mock_websocket_client.on_reconnect_callback = manager._on_reconnect_from_kws
    mock_websocket_client.on_noreconnect_callback = manager._on_noreconnect_from_kws
    logger.info("test_connection_manager_init completed successfully.")

@pytest.mark.asyncio
async def test_start_stop_monitoring(mock_websocket_client):
    logger.info("\n--- Starting test_start_stop_monitoring ---")
    on_data_gap_detected = MagicMock()
    on_noreconnect_critical = MagicMock()
    manager = ConnectionManager(mock_websocket_client, on_data_gap_detected, on_noreconnect_critical)

    manager.start_monitoring()
    assert manager._monitor_task is not None
    assert not manager._monitor_task.done()

    manager.stop_monitoring()
    await asyncio.sleep(0.1) # Give task a chance to cancel
    assert manager._monitor_task.done()
    logger.info("test_start_stop_monitoring completed successfully.")

@pytest.mark.asyncio
async def test_monitor_connection_disconnected(mock_websocket_client):
    logger.info("\n--- Starting test_monitor_connection_disconnected ---")
    on_data_gap_detected = MagicMock()
    on_noreconnect_critical = AsyncMock()
    manager = ConnectionManager(mock_websocket_client, on_data_gap_detected, on_noreconnect_critical)

    mock_websocket_client.is_connected.return_value = False

    # Run monitor once
    with patch('src.broker.connection_manager.asyncio.sleep', new=AsyncMock()):
        await manager._monitor_connection()

    on_noreconnect_critical.assert_called_once()
    logger.info("test_monitor_connection_disconnected completed successfully.")

@pytest.mark.asyncio
async def test_on_reconnect_from_kws(mock_websocket_client):
    logger.info("\n--- Starting test_on_reconnect_from_kws ---")
    on_data_gap_detected = MagicMock()
    on_noreconnect_critical = MagicMock()
    manager = ConnectionManager(mock_websocket_client, on_data_gap_detected, on_noreconnect_critical)

    manager._on_reconnect_from_kws(1)
    on_data_gap_detected.assert_called_once()
    logger.info("test_on_reconnect_from_kws completed successfully.")

@pytest.mark.asyncio
async def test_on_noreconnect_from_kws(mock_websocket_client):
    logger.info("\n--- Starting test_on_noreconnect_from_kws ---")
    on_data_gap_detected = MagicMock()
    on_noreconnect_critical = MagicMock()
    manager = ConnectionManager(mock_websocket_client, on_data_gap_detected, on_noreconnect_critical)

    manager._on_noreconnect_from_kws()
    on_noreconnect_critical.assert_called_once()
    logger.info("test_on_noreconnect_from_kws completed successfully.")


# Example of running tests directly (usually done via `pytest` command)
if __name__ == "__main__":
    logger.info("Running connection manager tests directly. Use `pytest` for proper test execution.")
    asyncio.run(test_connection_manager_init(MagicMock()))
    asyncio.run(test_start_stop_monitoring(MagicMock()))
    asyncio.run(test_monitor_connection_disconnected(MagicMock()))
    asyncio.run(test_on_reconnect_from_kws(MagicMock()))
    asyncio.run(test_on_noreconnect_from_kws(MagicMock()))
