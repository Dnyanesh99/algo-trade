from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.core.historical_aggregator import HistoricalAggregator
from src.database.ohlcv_repo import OHLCVRepository
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor


# Mock ConfigLoader for testing purposes
class MockConfigLoader:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {
            "trading": {
                "aggregation_timeframes": [5, 15, 60]  # minutes
            },
            "performance": {"processing": {"batch_size": 10000}, "max_retries": 3},
            "data_quality": {"validation": {"enabled": True, "quality_score_threshold": 80.0}},
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

    def get_config(self) -> MagicMock:
        mock_trading = MagicMock()
        mock_trading.aggregation_timeframes = self.config["trading"]["aggregation_timeframes"]

        mock_performance = MagicMock()
        mock_performance.processing = MagicMock(batch_size=self.config["performance"]["processing"]["batch_size"])
        mock_performance.max_retries = self.config["performance"]["max_retries"]

        mock_data_quality = MagicMock()
        mock_data_quality.validation = MagicMock(
            enabled=self.config["data_quality"]["validation"]["enabled"],
            quality_score_threshold=self.config["data_quality"]["validation"]["quality_score_threshold"],
        )

        mock_config = MagicMock()
        mock_config.trading = mock_trading
        mock_config.performance = mock_performance
        mock_config.data_quality = mock_data_quality
        return mock_config


@pytest.fixture
def mock_config_loader_instance() -> MockConfigLoader:
    return MockConfigLoader()


@pytest.fixture
async def historical_aggregator_instance(
    mock_config_loader_instance: MockConfigLoader,
) -> AsyncGenerator[HistoricalAggregator, None]:
    mock_ohlcv_repo = AsyncMock(spec=OHLCVRepository)
    mock_error_handler = AsyncMock(spec=ErrorHandler)
    mock_health_monitor = AsyncMock(spec=HealthMonitor)

    with patch("src.utils.config_loader.ConfigLoader") as MockCL:
        MockCL.return_value = mock_config_loader_instance
        MockCL.return_value.get_config.return_value = mock_config_loader_instance.get_config()
        with patch("src.core.candle_validator.CandleValidator") as MockCandleValidator:
            mock_candle_validator_instance = MockCandleValidator.return_value
            mock_candle_validator_instance.validate_candles = AsyncMock(
                return_value=MagicMock(
                    is_valid=True,
                    cleaned_data=pd.DataFrame(),
                    quality_report=MagicMock(quality_score=100.0, issues=[]),
                    validation_errors=[],
                    warnings=[],
                )
            )
            yield HistoricalAggregator(mock_ohlcv_repo, mock_error_handler, mock_health_monitor)


@pytest.fixture
def sample_ohlcv_1min_data() -> pd.DataFrame:
    start_time: datetime = datetime(2023, 1, 1, 9, 15)
    data: list[dict[str, Any]] = []
    for i in range(120):  # 2 hours of 1-min data
        ts: datetime = start_time + timedelta(minutes=i)
        open_price: float = 100 + i * 0.1
        close_price: float = open_price + 0.5
        high_price: float = max(open_price, close_price) + 0.2
        low_price: float = min(open_price, close_price) - 0.2
        volume: int = 100 + i * 10
        data.append(
            {"ts": ts, "open": open_price, "high": high_price, "low": low_price, "close": close_price, "volume": volume}
        )
    return pd.DataFrame(data)


@pytest.mark.asyncio
async def test_aggregate_and_store_5min(
    historical_aggregator_instance: HistoricalAggregator, sample_ohlcv_1min_data: pd.DataFrame
) -> None:
    df_1min: pd.DataFrame = sample_ohlcv_1min_data.copy()
    # Mock the insert_ohlcv_data to return success
    (historical_aggregator_instance.ohlcv_repo.insert_ohlcv_data).return_value = AsyncMock(return_value=None)

    results = await historical_aggregator_instance.aggregate_and_store(1, df_1min)

    assert len(results) == 3  # Expect results for 5min, 15min, 60min
    assert all(r.success for r in results)
    assert results[0].timeframe == "5min"
    assert results[0].stored_rows > 0


@pytest.mark.asyncio
async def test_aggregate_and_store_empty_data(
    historical_aggregator_instance: HistoricalAggregator,
) -> None:
    df_1min: pd.DataFrame = pd.DataFrame()
    results = await historical_aggregator_instance.aggregate_and_store(1, df_1min)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_aggregate_and_store_missing_columns(
    historical_aggregator_instance: HistoricalAggregator, sample_ohlcv_1min_data: pd.DataFrame
) -> None:
    df_1min: pd.DataFrame = sample_ohlcv_1min_data.copy().drop(columns=["volume"])
    results = await historical_aggregator_instance.aggregate_and_store(1, df_1min)
    assert len(results) == 0  # Should return empty list due to missing columns
    (historical_aggregator_instance.error_handler.handle_error).assert_called_once()


@pytest.mark.asyncio
async def test_aggregate_and_store_validation_failure(
    historical_aggregator_instance: HistoricalAggregator, sample_ohlcv_1min_data: pd.DataFrame
) -> None:
    df_1min: pd.DataFrame = sample_ohlcv_1min_data.copy()
    # Simulate validation failure
    (historical_aggregator_instance.candle_validator.validate_candles).return_value = AsyncMock(
        is_valid=False,
        cleaned_data=df_1min.head(10),  # Return some cleaned data
        quality_report=MagicMock(quality_score=50.0, issues=[]),
        validation_errors=["Data quality below threshold"],
        warnings=[],
    )

    results = await historical_aggregator_instance.aggregate_and_store(1, df_1min)
    assert len(results) == 3  # Still attempts all timeframes
    assert all(not r.success for r in results)  # All should fail due to initial validation
    assert any("Data quality below threshold" in r.validation_errors for r in results)


@pytest.mark.asyncio
async def test_aggregate_and_store_storage_failure(
    historical_aggregator_instance: HistoricalAggregator, sample_ohlcv_1min_data: pd.DataFrame
) -> None:
    df_1min: pd.DataFrame = sample_ohlcv_1min_data.copy()
    # Simulate storage failure after aggregation
    (historical_aggregator_instance.ohlcv_repo.insert_ohlcv_data).side_effect = AsyncMock(
        side_effect=Exception("DB Error")
    )

    results = await historical_aggregator_instance.aggregate_and_store(1, df_1min)
    assert len(results) == 3
    assert all(not r.success for r in results)
    assert any("DB Error" in r.validation_errors[0] for r in results)
    (historical_aggregator_instance.error_handler.handle_error).assert_called_once()


@pytest.mark.asyncio
async def test_aggregate_ohlcv_5min_logic(
    historical_aggregator_instance: HistoricalAggregator, sample_ohlcv_1min_data: pd.DataFrame
) -> None:
    df_1min: pd.DataFrame = sample_ohlcv_1min_data.copy()
    # Temporarily disable validation for this specific test to focus on aggregation logic
    with patch.object(historical_aggregator_instance, "validation_enabled", False):
        # Call the internal method directly for testing aggregation logic
        result = await historical_aggregator_instance._process_timeframe_aggregation(1, df_1min.set_index("ts"), 5)

        assert result.success
        assert result.timeframe == "5min"
        assert result.stored_rows > 0

        # Further detailed checks on the aggregated data can be added here if needed
        # For example, re-aggregate manually and compare with result.cleaned_data


@pytest.mark.asyncio
async def test_aggregate_ohlcv_15min_logic(
    historical_aggregator_instance: HistoricalAggregator, sample_ohlcv_1min_data: pd.DataFrame
) -> None:
    df_1min: pd.DataFrame = sample_ohlcv_1min_data.copy()
    with patch.object(historical_aggregator_instance, "validation_enabled", False):
        result = await historical_aggregator_instance._process_timeframe_aggregation(1, df_1min.set_index("ts"), 15)

        assert result.success
        assert result.timeframe == "15min"
        assert result.stored_rows > 0

        # or inspect the result.cleaned_data if _process_timeframe_aggregation returned it.
        # For now, we rely on stored_rows > 0 and success.


@pytest.mark.asyncio
async def test_aggregate_ohlcv_60min_logic(
    historical_aggregator_instance: HistoricalAggregator, sample_ohlcv_1min_data: pd.DataFrame
) -> None:
    df_1min: pd.DataFrame = sample_ohlcv_1min_data.copy()
    with patch.object(historical_aggregator_instance, "validation_enabled", False):
        result = await historical_aggregator_instance._process_timeframe_aggregation(1, df_1min.set_index("ts"), 60)

        assert result.success
        assert result.timeframe == "60min"
        assert result.stored_rows > 0


@pytest.mark.asyncio
async def test_get_aggregation_health(historical_aggregator_instance: HistoricalAggregator) -> None:
    health = await historical_aggregator_instance.get_aggregation_health()
    assert "component" in health
    assert health["component"] == "historical_aggregator"
    assert "status" in health
    assert "configuration" in health
    assert "timeframes" in health["configuration"]
