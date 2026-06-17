from collections.abc import Generator
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.feature_calculator import FeatureCalculator
from src.database.feature_repo import FeatureRepository
from src.database.ohlcv_repo import OHLCVRepository
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor


# Mock ConfigLoader for testing purposes
class MockConfigLoader:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {
            "trading": {
                "features": {
                    "lookback_period": 200,
                    "configurations": [
                        {
                            "name": "macd",
                            "function": "MACD",
                            "params": {
                                "default": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
                                "60m": {"fastperiod": 20, "slowperiod": 40, "signalperiod": 9},
                            },
                        },
                        {
                            "name": "rsi",
                            "function": "RSI",
                            "params": {"default": {"timeperiod": 14}, "60m": {"timeperiod": 21}},
                        },
                        {"name": "atr", "function": "ATR", "params": {"default": {"timeperiod": 14}}},
                        {
                            "name": "bbands",
                            "function": "BBANDS",
                            "params": {
                                "default": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
                                "60m": {"nbdevup": 2.5, "nbdevdn": 2.5},
                            },
                        },
                    ],
                    "name_mapping": {
                        "macdsignal": "macd_signal",
                        "macdhist": "macd_hist",
                        "upperband": "bb_upper",
                        "middleband": "bb_middle",
                        "lowerband": "bb_lower",
                        "aroonup": "aroon_up",
                        "aroondown": "aroon_down",
                    },
                }
            },
            "data_quality": {"validation": {"enabled": True}},
            "model_training": {"feature_engineering": {"enabled": False, "feature_selection": {"enabled": False}}},
            "system": {"mode": "HISTORICAL_MODE"},
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
        mock_trading.feature_timeframes = [5, 15, 60]
        mock_trading.features = MagicMock(
            lookback_period=self.config["trading"]["features"]["lookback_period"],
            configurations=self.config["trading"]["features"]["configurations"],
            name_mapping=self.config["trading"]["features"]["name_mapping"],
        )

        mock_data_quality = MagicMock()
        mock_data_quality.validation = MagicMock(enabled=self.config["data_quality"]["validation"]["enabled"])

        mock_model_training = MagicMock()
        mock_model_training.feature_engineering = MagicMock(
            enabled=self.config["model_training"]["feature_engineering"]["enabled"],
            feature_selection=MagicMock(
                enabled=self.config["model_training"]["feature_engineering"]["feature_selection"]["enabled"]
            ),
        )

        mock_system = MagicMock()
        mock_system.mode = self.config["system"]["mode"]

        mock_config = MagicMock()
        mock_config.trading = mock_trading
        mock_config.data_quality = mock_data_quality
        mock_config.model_training = mock_model_training
        mock_config.system = mock_system
        return mock_config


@pytest.fixture
def mock_config_loader_instance() -> MockConfigLoader:
    return MockConfigLoader()


@pytest.fixture
def feature_calculator_instance(
    mock_config_loader_instance: MockConfigLoader,
) -> Generator[FeatureCalculator, None, None]:
    mock_ohlcv_repo = AsyncMock(spec=OHLCVRepository)
    mock_feature_repo = AsyncMock(spec=FeatureRepository)
    mock_error_handler = AsyncMock(spec=ErrorHandler)
    mock_health_monitor = AsyncMock(spec=HealthMonitor)

    # Mock the get_ohlcv_data_for_features method to return a list of OHLCVData objects
    mock_ohlcv_repo.get_ohlcv_data_for_features.return_value = AsyncMock(
        return_value=[
            MagicMock(ts=datetime.now(), open=100, high=105, low=95, close=102, volume=1000) for _ in range(500)
        ]
    )

    with patch("src.utils.config_loader.ConfigLoader") as MockCL:
        MockCL.return_value = mock_config_loader_instance
        MockCL.return_value.get_config.return_value = mock_config_loader_instance.get_config()
        with patch("src.core.feature_validator.FeatureValidator") as MockFeatureValidator:
            MockFeatureValidator.return_value.validate_features.return_value = AsyncMock(
                return_value=(
                    True,  # is_valid
                    pd.DataFrame(),  # cleaned_df (will be replaced by actual data in tests)
                    MagicMock(quality_score=100.0, issues=[]),  # DataQualityReport
                )
            )
            yield FeatureCalculator(mock_ohlcv_repo, mock_feature_repo, mock_error_handler, mock_health_monitor)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    # Generate enough data for all indicators, including lookback periods
    data_size: int = 500
    np.random.seed(42)
    dates: pd.DatetimeIndex = pd.date_range(start="2023-01-01", periods=data_size, freq="min")
    open_prices: np.ndarray = np.random.uniform(100, 110, data_size).cumsum() + 1000
    close_prices: np.ndarray = open_prices + np.random.uniform(-2, 2, data_size)
    high_prices: np.ndarray = np.maximum(open_prices, close_prices) + np.random.uniform(0, 1, data_size)
    low_prices: np.ndarray = np.minimum(open_prices, close_prices) - np.random.uniform(0, 1, data_size)
    volume: np.ndarray = np.random.randint(1000, 5000, data_size)

    df = pd.DataFrame(
        {
            "ts": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        }
    )
    return df.set_index("ts")


@pytest.mark.asyncio
async def test_calculate_and_store_features_macd_default(
    feature_calculator_instance: FeatureCalculator, sample_ohlcv_data: pd.DataFrame
) -> None:
    # Mock the internal _calculate_all_features to return a predictable DataFrame
    with patch.object(feature_calculator_instance, "_calculate_all_features") as mock_calc_all_features:
        mock_calc_all_features.return_value = AsyncMock(
            return_value=pd.DataFrame(
                {
                    "macd": np.random.rand(len(sample_ohlcv_data)),
                    "macd_signal": np.random.rand(len(sample_ohlcv_data)),
                    "macd_hist": np.random.rand(len(sample_ohlcv_data)),
                },
                index=sample_ohlcv_data.index,
            )
        )
        mock_calc_all_features.return_value.return_value["timestamp"] = (
            mock_calc_all_features.return_value.return_value.index
        )

        results = await feature_calculator_instance.calculate_and_store_features(1, datetime.now())

        assert len(results) > 0
        assert results[0].success
        assert results[0].features_calculated > 0
        feature_calculator_instance.feature_repo.insert_features.assert_called()


@pytest.mark.asyncio
async def test_calculate_and_store_features_insufficient_data(
    feature_calculator_instance: FeatureCalculator,
) -> None:
    # Mock get_ohlcv_data_for_features to return insufficient data
    feature_calculator_instance.ohlcv_repo.get_ohlcv_data_for_features.return_value = AsyncMock(
        return_value=[
            MagicMock(ts=datetime.now(), open=100, high=105, low=95, close=102, volume=1000) for _ in range(10)
        ]
    )

    results = await feature_calculator_instance.calculate_and_store_features(1, datetime.now())

    assert len(results) > 0
    assert not results[0].success
    assert "Insufficient data" in results[0].validation_errors[0]


@pytest.mark.asyncio
async def test_calculate_and_store_features_no_features_calculated(
    feature_calculator_instance: FeatureCalculator, sample_ohlcv_data: pd.DataFrame
) -> None:
    with patch.object(feature_calculator_instance, "_calculate_all_features") as mock_calc_all_features:
        mock_calc_all_features.return_value = AsyncMock(return_value=pd.DataFrame())  # Simulate no features calculated
        mock_calc_all_features.return_value.return_value[
            "timestamp"
        ] = []  # Ensure timestamp column is present but empty

        results = await feature_calculator_instance.calculate_and_store_features(1, datetime.now())

        assert len(results) > 0
        assert not results[0].success
        assert "No features were calculated." in results[0].validation_errors[0]


@pytest.mark.asyncio
async def test_calculate_and_store_features_validation_failure(
    feature_calculator_instance: FeatureCalculator, sample_ohlcv_data: pd.DataFrame
) -> None:
    with patch.object(feature_calculator_instance, "_calculate_all_features") as mock_calc_all_features:
        mock_calc_all_features.return_value = AsyncMock(
            return_value=pd.DataFrame(
                {
                    "rsi": np.random.rand(len(sample_ohlcv_data)),
                },
                index=sample_ohlcv_data.index,
            )
        )
        mock_calc_all_features.return_value.return_value["timestamp"] = (
            mock_calc_all_features.return_value.return_value.index
        )

        feature_calculator_instance.feature_validator.validate_features.return_value = AsyncMock(
            return_value=(
                False,  # is_valid
                pd.DataFrame(),  # cleaned_df (empty to simulate severe validation failure)
                MagicMock(quality_score=0.0, issues=[]),
            )
        )

        results = await feature_calculator_instance.calculate_and_store_features(1, datetime.now())

        assert len(results) > 0
        assert not results[0].success
        assert "Validation failed" in results[0].validation_errors[0]


@pytest.mark.asyncio
async def test_calculate_and_store_features_storage_failure(
    feature_calculator_instance: FeatureCalculator, sample_ohlcv_data: pd.DataFrame
) -> None:
    with patch.object(feature_calculator_instance, "_calculate_all_features") as mock_calc_all_features:
        mock_calc_all_features.return_value = AsyncMock(
            return_value=pd.DataFrame(
                {
                    "macd": np.random.rand(len(sample_ohlcv_data)),
                },
                index=sample_ohlcv_data.index,
            )
        )
        mock_calc_all_features.return_value.return_value["timestamp"] = (
            mock_calc_all_features.return_value.return_value.index
        )

        feature_calculator_instance.feature_repo.insert_features.side_effect = AsyncMock(
            side_effect=Exception("DB Error")
        )

        results = await feature_calculator_instance.calculate_and_store_features(1, datetime.now())

        assert len(results) > 0
        assert not results[0].success
        assert "DB Error" in results[0].validation_errors[0]
        feature_calculator_instance.error_handler.handle_error.assert_called_once()


@pytest.mark.asyncio
async def test_calculate_all_features_macd_default(
    feature_calculator_instance: FeatureCalculator, sample_ohlcv_data: pd.DataFrame
) -> None:
    # This tests the internal _calculate_all_features method directly
    features_df = await feature_calculator_instance._calculate_all_features(sample_ohlcv_data, 5)

    assert "macd" in features_df.columns
    assert "macd_signal" in features_df.columns
    assert "macd_hist" in features_df.columns
    assert not features_df["macd"].isnull().all()  # Ensure some values are calculated
    assert not features_df["macd_signal"].isnull().all()
    assert not features_df["macd_hist"].isnull().all()


@pytest.mark.asyncio
async def test_calculate_all_features_rsi_default(
    feature_calculator_instance: FeatureCalculator, sample_ohlcv_data: pd.DataFrame
) -> None:
    features_df = await feature_calculator_instance._calculate_all_features(sample_ohlcv_data, 5)

    assert "rsi" in features_df.columns
    assert not features_df["rsi"].isnull().all()
    assert (features_df["rsi"].dropna() >= 0).all() and (features_df["rsi"].dropna() <= 100).all()


@pytest.mark.asyncio
async def test_calculate_all_features_insufficient_data(
    feature_calculator_instance: FeatureCalculator,
) -> None:
    # Create a DataFrame with very few rows, less than any lookback period
    data: dict[str, list[Any]] = {
        "ts": pd.to_datetime(["2023-01-01 09:15", "2023-01-01 09:16", "2023-01-01 09:17"]),
        "open": [100, 101, 102],
        "high": [101, 102, 103],
        "low": [99, 100, 101],
        "close": [100, 101, 102],
        "volume": [100, 110, 120],
    }
    df: pd.DataFrame = pd.DataFrame(data).set_index("ts")

    features_df = await feature_calculator_instance._calculate_all_features(df, 5)

    # All feature columns should be present but filled with NaNs due to insufficient data
    assert "macd" in features_df.columns
    assert features_df["macd"].isnull().all()
    assert "rsi" in features_df.columns
    assert features_df["rsi"].isnull().all()
    assert "atr" in features_df.columns
    assert features_df["atr"].isnull().all()


@pytest.mark.asyncio
async def test_feature_name_mapping(
    feature_calculator_instance: FeatureCalculator, sample_ohlcv_data: pd.DataFrame
) -> None:
    features_df = await feature_calculator_instance._calculate_all_features(sample_ohlcv_data, 5)

    # Check if original TA-Lib names are mapped to desired names
    assert "macdsignal" not in features_df.columns  # Original name should not exist
    assert "macd_signal" in features_df.columns  # Mapped name should exist
    assert "macdhist" not in features_df.columns
    assert "macd_hist" in features_df.columns
    assert "upperband" not in features_df.columns
    assert "bb_upper" in features_df.columns
    assert "middleband" not in features_df.columns
    assert "bb_middle" in features_df.columns
    assert "lowerband" not in features_df.columns
    assert "bb_lower" in features_df.columns
