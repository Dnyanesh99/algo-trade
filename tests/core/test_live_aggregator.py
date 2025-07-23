from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.live.candle_buffer import CandleBuffer
from src.live.live_aggregator import LiveAggregator
from src.live.tick_validator import TickValidator
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor


# Mock ConfigLoader for testing purposes
class MockConfigLoader:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {
            "live_aggregator": {
                "max_partial_candles": 10000,
                "partial_candle_cleanup_hours": 2,
                "health_check_success_rate_threshold": 0.95,
                "health_check_avg_processing_time_ms_threshold": 100,
                "health_check_validation_failures_threshold": 0.05,
                "required_tick_fields": ["instrument_token", "timestamp", "last_traded_price"],
                "required_candle_fields": [
                    "instrument_token",
                    "timeframe",
                    "start_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "trades",
                ],
            },
            "candle_buffer": {"timeframes": [1, 5, 15, 60]},
        }

    def load_config(self) -> dict[str, Any]:
        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        keys: list[str] = key.split(".")
        val: dict[str, Any] = self.config
        try:
            for k in keys:
                val = val[k]
            return val
        except KeyError:
            return default


@pytest.fixture
def mock_config_loader() -> MockConfigLoader:
    return MockConfigLoader()


@pytest.fixture
def live_aggregator(mock_config_loader: MockConfigLoader) -> Generator[LiveAggregator, None, None]:
    # Mock dependencies for LiveAggregator
    mock_candle_buffer: MagicMock = MagicMock(spec=CandleBuffer)
    mock_tick_validator: MagicMock = MagicMock(spec=TickValidator)
    mock_error_handler: MagicMock = MagicMock(spec=ErrorHandler)
    mock_health_monitor: MagicMock = MagicMock(spec=HealthMonitor)

    mock_tick_validator.validate_tick.return_value = (True, [])

    # Patch ConfigLoader to return our mock instance
    with patch("src.utils.config_loader.ConfigLoader") as MockCL:
        MockCL.return_value = mock_config_loader
        MockCL.return_value.get_config.return_value = mock_config_loader.config  # Simplified for testing
        yield LiveAggregator(
            mock_config_loader, mock_candle_buffer, mock_tick_validator, mock_error_handler, mock_health_monitor
        )


@pytest.fixture
def sample_tick_data() -> dict[str, Any]:
    return {
        "instrument_token": 12345,
        "timestamp": datetime.now(),
        "last_traded_price": 100.5,
        "last_traded_quantity": 10,
        "average_traded_price": 100.4,
        "volume": 1000,
        "buy_quantity": 500,
        "sell_quantity": 500,
        "ohlc": {"open": 100, "high": 101, "low": 99, "close": 100.5},
        "change": 0.5,
    }


def test_process_tick_valid(live_aggregator: LiveAggregator, sample_tick_data: dict[str, Any]) -> None:
    live_aggregator.process_tick(sample_tick_data)
    live_aggregator.tick_validator.validate_tick.assert_called_once_with(sample_tick_data)
    live_aggregator.candle_buffer.on_new_tick.assert_called_once_with(sample_tick_data)


def test_process_tick_invalid_tick(live_aggregator: LiveAggregator, sample_tick_data: dict[str, Any]) -> None:
    live_aggregator.tick_validator.validate_tick.return_value = (False, ["Missing field"])
    live_aggregator.process_tick(sample_tick_data)
    live_aggregator.tick_validator.validate_tick.assert_called_once_with(sample_tick_data)
    assert not live_aggregator.candle_buffer.on_new_tick.called  # Should not process invalid tick


def test_process_tick_exception_handling(live_aggregator: LiveAggregator, sample_tick_data: dict[str, Any]) -> None:
    live_aggregator.tick_validator.validate_tick.side_effect = Exception("Test Error")
    live_aggregator.process_tick(sample_tick_data)
    # Ensure no unhandled exceptions and logging (mocked)
    live_aggregator.tick_validator.validate_tick.assert_called_once_with(sample_tick_data)
    assert not live_aggregator.candle_buffer.on_new_tick.called


def test_get_health_metrics(live_aggregator: LiveAggregator) -> None:
    # Simulate some processing
    live_aggregator.processed_ticks = 100
    live_aggregator.validation_failures = 5
    live_aggregator.processing_times.append(50)
    live_aggregator.processing_times.append(60)

    metrics: dict[str, Any] = live_aggregator.get_health_metrics()

    assert "processed_ticks" in metrics
    assert metrics["processed_ticks"] == 100
    assert "validation_failures" in metrics
    assert metrics["validation_failures"] == 5
    assert "success_rate" in metrics
    assert metrics["success_rate"] == pytest.approx(0.95)
    assert "avg_processing_time_ms" in metrics
    assert metrics["avg_processing_time_ms"] == pytest.approx(55.0)


def test_check_health_ok(live_aggregator: LiveAggregator) -> None:
    live_aggregator.processed_ticks = 100
    live_aggregator.validation_failures = 2  # 2% failure rate
    live_aggregator.processing_times.extend([50] * 100)

    is_healthy: bool
    messages: list[str]
    is_healthy, messages = live_aggregator.check_health()
    assert is_healthy is True
    assert "Live Aggregator is healthy" in messages[0]


def test_check_health_low_success_rate(live_aggregator: LiveAggregator) -> None:
    live_aggregator.processed_ticks = 100
    live_aggregator.validation_failures = 10  # 10% failure rate
    live_aggregator.processing_times.extend([50] * 100)

    is_healthy: bool
    messages: list[str]
    is_healthy, messages = live_aggregator.check_health()
    assert is_healthy is False
    assert "Low tick processing success rate" in messages[0]


def test_check_health_high_processing_time(
    live_aggregator: LiveAggregator,
) -> None:
    live_aggregator.processed_ticks = 100
    live_aggregator.validation_failures = 0
    live_aggregator.processing_times.extend([150] * 100)  # High processing time

    is_healthy: bool
    messages: list[str]
    is_healthy, messages = live_aggregator.check_health()
    assert is_healthy is False
    assert "High average tick processing time" in messages[0]


def test_check_health_no_ticks_processed(live_aggregator: LiveAggregator) -> None:
    live_aggregator.processed_ticks = 0
    is_healthy: bool
    messages: list[str]
    is_healthy, messages = live_aggregator.check_health()
    assert is_healthy is False
    assert "No ticks processed yet" in messages[0]


def test_live_aggregator_processes_ticks_for_aggregation(
    live_aggregator: LiveAggregator, sample_tick_data: dict[str, Any]
) -> None:
    # Simulate multiple ticks arriving over time
    base_timestamp = datetime(2023, 1, 1, 9, 15, 0)

    # Simulate ticks for 15 minutes to ensure 1-min, 5-min, and 15-min candles can be formed
    for i in range(15 * 60 + 1):  # 15 minutes of 1-second ticks + 1 extra tick
        tick = sample_tick_data.copy()
        tick["timestamp"] = base_timestamp + timedelta(seconds=i)
        tick["last_traded_price"] = 100.5 + (i * 0.01)
        live_aggregator.process_tick(tick)

    # Verify that on_new_tick was called for each simulated tick
    assert live_aggregator.candle_buffer.on_new_tick.call_count == (15 * 60 + 1)

    # and assert on LiveAggregator's subsequent actions (e.g., calling feature calculator)
    # For now, we'll just assert that on_new_tick was called.

    # Simulate CandleBuffer completing a 60-minute candle (after more ticks)
    for i in range(15 * 60 + 1, 60 * 60 + 1):  # Continue for 60 minutes
        tick = sample_tick_data.copy()
        tick["timestamp"] = base_timestamp + timedelta(seconds=i)
        tick["last_traded_price"] = 100.5 + (i * 0.01)
        live_aggregator.process_tick(tick)

    assert live_aggregator.candle_buffer.on_new_tick.call_count == (60 * 60 + 1)

    # Further tests would involve mocking the feature calculator and signal generator
    # to ensure LiveAggregator passes the completed candles to them.
