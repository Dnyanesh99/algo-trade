import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import pytz

from src.core.feature_engineering_pipeline import (
    FeatureEngineeringPipeline,
    FeatureScore,
)
from src.core.feature_validator import FeatureValidator
from src.database.feature_repo import FeatureRepository
from src.database.ohlcv_repo import OHLCVRepository
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor


# Mock ConfigLoader for testing purposes
class MockConfigLoader:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {
            "system": {"timezone": "UTC"},
            "model_training": {
                "feature_engineering": {
                    "enabled": True,
                    "feature_generation": {
                        "enabled": True,
                        "patterns": [
                            "momentum_ratios",
                            "volatility_adjustments",
                            "trend_confirmations",
                            "mean_reversion_signals",
                            "breakout_indicators",
                        ],
                        "lookback_periods": [5, 10, 20],
                        "min_quality_score": 0.3,
                    },
                    "feature_selection": {
                        "enabled": True,
                        "target_feature_count": 50,
                        "importance_history_length": 5,
                        "correlation_threshold": 0.8,
                        "importance_weight": 0.5,
                        "stability_weight": 0.3,
                        "consistency_weight": 0.2,
                        "correlation_data_lookback_multiplier": 2,
                    },
                    "cross_asset": {
                        "enabled": True,
                        "max_related_instruments": 3,
                        "minimum_correlation_threshold": 0.5,
                        "correlation_lookback_days": 60,
                        "cache_duration_minutes": 15,
                        "instruments": {101: "NIFTY", 202: "BANKNIFTY"},
                    },
                }
            },
            "trading": {
                "features": {
                    "configurations": [
                        {"name": "rsi_14", "function": "RSI", "params": {"timeperiod": 14}},
                        {"name": "macd", "function": "MACD"},
                    ]
                },
                "feature_timeframes": ["5min", "15min", "60min"],
            },
        }

    def get_config(self) -> MagicMock:
        mock_config = MagicMock()
        mock_config.system.timezone = self.config["system"]["timezone"]
        mock_config.model_training.feature_engineering = MagicMock()
        fe_config = self.config["model_training"]["feature_engineering"]
        mock_config.model_training.feature_engineering.feature_generation.enabled = fe_config["feature_generation"][
            "enabled"
        ]
        mock_config.model_training.feature_engineering.feature_generation.patterns = fe_config["feature_generation"][
            "patterns"
        ]
        mock_config.model_training.feature_engineering.feature_generation.lookback_periods = fe_config[
            "feature_generation"
        ]["lookback_periods"]
        mock_config.model_training.feature_engineering.feature_generation.min_quality_score = fe_config[
            "feature_generation"
        ]["min_quality_score"]
        mock_config.model_training.feature_engineering.feature_selection.enabled = fe_config["feature_selection"][
            "enabled"
        ]
        mock_config.model_training.feature_engineering.feature_selection.target_feature_count = fe_config[
            "feature_selection"
        ]["target_feature_count"]
        mock_config.model_training.feature_engineering.feature_selection.importance_history_length = fe_config[
            "feature_selection"
        ]["importance_history_length"]
        mock_config.model_training.feature_engineering.feature_selection.correlation_threshold = fe_config[
            "feature_selection"
        ]["correlation_threshold"]
        mock_config.model_training.feature_engineering.feature_selection.importance_weight = fe_config[
            "feature_selection"
        ]["importance_weight"]
        mock_config.model_training.feature_engineering.feature_selection.stability_weight = fe_config[
            "feature_selection"
        ]["stability_weight"]
        mock_config.model_training.feature_engineering.feature_selection.consistency_weight = fe_config[
            "feature_selection"
        ]["consistency_weight"]
        mock_config.model_training.feature_engineering.feature_selection.correlation_data_lookback_multiplier = (
            fe_config["feature_selection"]["correlation_data_lookback_multiplier"]
        )
        mock_config.model_training.feature_engineering.cross_asset.enabled = fe_config["cross_asset"]["enabled"]
        mock_config.model_training.feature_engineering.cross_asset.max_related_instruments = fe_config["cross_asset"][
            "max_related_instruments"
        ]
        mock_config.model_training.feature_engineering.cross_asset.minimum_correlation_threshold = fe_config[
            "cross_asset"
        ]["minimum_correlation_threshold"]
        mock_config.model_training.feature_engineering.cross_asset.correlation_lookback_days = fe_config["cross_asset"][
            "correlation_lookback_days"
        ]
        mock_config.model_training.feature_engineering.cross_asset.cache_duration_minutes = fe_config["cross_asset"][
            "cache_duration_minutes"
        ]
        mock_config.model_training.feature_engineering.cross_asset.instruments = fe_config["cross_asset"]["instruments"]
        mock_config.trading.features.configurations = self.config["trading"]["features"]["configurations"]
        mock_config.trading.feature_timeframes = self.config["trading"]["feature_timeframes"]
        return mock_config


@pytest.fixture
def mock_config_loader_instance() -> MockConfigLoader:
    return MockConfigLoader()


@pytest.fixture
async def feature_engineering_pipeline(
    mock_config_loader_instance: MockConfigLoader,
) -> AsyncGenerator[FeatureEngineeringPipeline, None]:
    mock_ohlcv_repo = AsyncMock(spec=OHLCVRepository)
    mock_feature_repo = AsyncMock(spec=FeatureRepository)
    mock_feature_validator = AsyncMock(spec=FeatureValidator)
    mock_error_handler = AsyncMock(spec=ErrorHandler)
    mock_health_monitor = AsyncMock(spec=HealthMonitor)

    with patch("src.utils.config_loader.ConfigLoader") as MockCL:
        MockCL.return_value = mock_config_loader_instance
        MockCL.return_value.get_config.return_value = mock_config_loader_instance.get_config()
        yield FeatureEngineeringPipeline(
            mock_ohlcv_repo, mock_feature_repo, mock_feature_validator, mock_error_handler, mock_health_monitor
        )


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    data_size = 100
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=data_size, freq="min")
    prices = np.random.uniform(100, 110, data_size).cumsum() + 1000
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": prices + np.random.uniform(0, 1, data_size),
            "low": prices - np.random.uniform(0, 1, data_size),
            "close": prices + np.random.uniform(-0.5, 0.5, data_size),
            "volume": np.random.randint(1000, 5000, data_size),
        }
    )
    return df.set_index("timestamp")


@pytest.fixture
def base_features() -> dict[str, float]:
    return {
        "rsi_14": 60.0,
        "macd": 1.5,
        "macd_signal": 1.2,
        "atr_14": 2.5,
        "bb_upper": 110.0,
        "bb_lower": 100.0,
        "adx": 30.0,
        "sar": 99.0,
        "aroon_up": 80.0,
        "aroon_down": 20.0,
        "willr": -25.0,
        "obv": 10000.0,
    }


@pytest.mark.asyncio
async def test_generate_engineered_features_success(
    feature_engineering_pipeline: FeatureEngineeringPipeline,
    sample_ohlcv_data: pd.DataFrame,
    base_features: dict[str, float],
):
    # Mock the helper to return a valid DataFrame
    with patch.object(
        feature_engineering_pipeline, "_get_recent_ohlcv_data", return_value=asyncio.Future()
    ) as mock_get_data:
        mock_get_data.return_value.set_result(sample_ohlcv_data)

        instrument_id = 1
        timeframe = "5min"
        timestamp = datetime.now(pytz.UTC)

        engineered_features = await feature_engineering_pipeline.generate_engineered_features(
            instrument_id, timeframe, timestamp, base_features
        )

        assert isinstance(engineered_features, dict)
        assert len(engineered_features) > 0
        assert "momentum_ratio_5d" in engineered_features
        assert "rsi_momentum_normalized" in engineered_features
        assert "volatility_adjusted_return" in engineered_features
        feature_engineering_pipeline.feature_repo.insert_engineered_features.assert_called_once()


@pytest.mark.asyncio
async def test_generate_engineered_features_disabled(feature_engineering_pipeline: FeatureEngineeringPipeline):
    feature_engineering_pipeline.config.feature_generation.enabled = False
    engineered_features = await feature_engineering_pipeline.generate_engineered_features(1, "5min", datetime.now(), {})
    assert engineered_features == {}


@pytest.mark.asyncio
async def test_generate_engineered_features_no_data(
    feature_engineering_pipeline: FeatureEngineeringPipeline, base_features: dict[str, float]
):
    with patch.object(
        feature_engineering_pipeline, "_get_recent_ohlcv_data", return_value=asyncio.Future()
    ) as mock_get_data:
        mock_get_data.return_value.set_result(pd.DataFrame())  # Empty DataFrame

        engineered_features = await feature_engineering_pipeline.generate_engineered_features(
            1, "5min", datetime.now(), base_features
        )
        assert engineered_features == {}


@pytest.mark.asyncio
async def test_update_feature_selection_success(feature_engineering_pipeline: FeatureEngineeringPipeline):
    instrument_id = 1
    timeframe = "60min"
    model_metadata = {
        "features": {"importance": {"feature1": 0.8, "feature2": 0.2, "feature3": 0.5}},
        "metrics": {"accuracy": 0.95},
        "version_id": "v1.0",
    }

    # Mock dependencies
    feature_engineering_pipeline.feature_repo.get_features_for_correlation.return_value = asyncio.Future()
    feature_engineering_pipeline.feature_repo.get_features_for_correlation.return_value.set_result(
        pd.DataFrame({"feature1": [1, 2], "feature2": [2, 3], "feature3": [3, 4]})
    )
    feature_engineering_pipeline.feature_repo.insert_feature_selection_history.return_value = asyncio.Future()
    feature_engineering_pipeline.feature_repo.insert_feature_selection_history.return_value.set_result(None)

    # Populate importance history to allow selection calculation
    feature_engineering_pipeline.importance_history[f"{instrument_id}_{timeframe}"] = [
        {"timestamp": datetime.now(), "importance": {"feature1": 0.7, "feature2": 0.3}, "accuracy": 0.9}
    ]

    selected_features = await feature_engineering_pipeline.update_feature_selection(
        instrument_id, timeframe, model_metadata
    )

    assert selected_features is not None
    assert isinstance(selected_features, list)
    assert len(selected_features) > 0
    feature_engineering_pipeline.feature_repo.insert_feature_scores.assert_called_once()
    feature_engineering_pipeline.feature_repo.insert_feature_selection_history.assert_called_once()


@pytest.mark.asyncio
async def test_get_selected_features_cache_hit(feature_engineering_pipeline: FeatureEngineeringPipeline):
    model_key = "1_5min"
    cached_features = {"feature1", "feature2"}
    feature_engineering_pipeline.selected_features_cache[model_key] = cached_features

    features = await feature_engineering_pipeline.get_selected_features(1, "5min")
    assert features == cached_features
    feature_engineering_pipeline.feature_repo.get_latest_feature_selection.assert_not_called()


@pytest.mark.asyncio
async def test_get_selected_features_db_load(feature_engineering_pipeline: FeatureEngineeringPipeline):
    model_key = "1_5min"
    db_features = ["featureA", "featureB"]
    feature_engineering_pipeline.feature_repo.get_latest_feature_selection.return_value = asyncio.Future()
    feature_engineering_pipeline.feature_repo.get_latest_feature_selection.return_value.set_result(db_features)

    features = await feature_engineering_pipeline.get_selected_features(1, "5min")
    assert features == set(db_features)
    assert feature_engineering_pipeline.selected_features_cache[model_key] == set(db_features)
    feature_engineering_pipeline.feature_repo.get_latest_feature_selection.assert_called_once_with(1, "5min")


@pytest.mark.asyncio
async def test_calculate_cross_asset_features_success(feature_engineering_pipeline: FeatureEngineeringPipeline):
    primary_instrument_id = 1
    timeframe = "15min"
    timestamp = datetime.now(pytz.UTC)

    # Mock dependencies
    with (
        patch.object(
            feature_engineering_pipeline, "_get_related_instruments", return_value=asyncio.Future()
        ) as mock_get_related,
        patch.object(
            feature_engineering_pipeline, "_calculate_instrument_correlation", return_value=asyncio.Future()
        ) as mock_calc_corr,
    ):
        mock_get_related.return_value.set_result([(101, "NIFTY"), (202, "BANKNIFTY")])
        mock_calc_corr.return_value.set_result(0.85)  # High correlation

        cross_features = await feature_engineering_pipeline.calculate_cross_asset_features(
            primary_instrument_id, timeframe, timestamp
        )

        assert "correlation_nifty" in cross_features
        assert "correlation_banknifty" in cross_features
        assert cross_features["correlation_nifty"] == 0.85
        assert len(feature_engineering_pipeline.correlation_cache) > 0


@pytest.mark.asyncio
async def test_remove_correlated_features(feature_engineering_pipeline: FeatureEngineeringPipeline):
    feature_scores = {
        "featA": FeatureScore(
            feature_name="featA", importance_score=0.9, stability_score=0.9, consistency_score=1.0, composite_score=0.9
        ),
        "featB": FeatureScore(
            feature_name="featB", importance_score=0.8, stability_score=0.9, consistency_score=1.0, composite_score=0.8
        ),
        "featC": FeatureScore(
            feature_name="featC", importance_score=0.7, stability_score=0.9, consistency_score=1.0, composite_score=0.7
        ),
    }
    # featA and featB are highly correlated, featC is not. featA has higher score.
    correlation_df = pd.DataFrame(
        {"featA": [1, 2, 3], "featB": [1.1, 2.2, 3.3], "featC": [5, -2, 1]}  # featA and featB correlated
    )
    feature_engineering_pipeline.feature_repo.get_features_for_correlation.return_value = asyncio.Future()
    feature_engineering_pipeline.feature_repo.get_features_for_correlation.return_value.set_result(correlation_df)

    reduced_features = await feature_engineering_pipeline._remove_correlated_features(feature_scores, 1, "5min")

    assert "featA" in reduced_features
    assert "featB" not in reduced_features  # Should be removed
    assert "featC" in reduced_features
    assert len(reduced_features) == 2
