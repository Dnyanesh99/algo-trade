from collections.abc import Generator
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.feature_validator import FeatureValidator

if TYPE_CHECKING:
    from src.utils.data_quality import DataQualityReport


# Mock ConfigLoader for testing purposes
class MockConfigLoader:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {
            "model_training": {
                "feature_ranges": {
                    "rsi": [0, 100],
                    "macd": [-5, 5],
                    "bb_upper": [0, np.inf],  # Assuming upper band can be any positive value
                    "bb_lower": [-np.inf, np.inf],  # Assuming lower band can be any value
                    "bb_middle": [-np.inf, np.inf],
                }
            },
            "data_quality": {
                "penalties": {"out_of_range_penalty": 0.1, "inconsistency_penalty": 0.05},
                "validation": {
                    "min_valid_rows": 5,
                    "quality_score_threshold": 80.0,
                },
                "outlier_detection": {"iqr_multiplier": 1.5, "handling_strategy": "clip"},
            },
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
        mock_data_quality = MagicMock()
        mock_data_quality.validation = MagicMock(
            min_valid_rows=self.config["data_quality"]["validation"]["min_valid_rows"],
            quality_score_threshold=self.config["data_quality"]["validation"]["quality_score_threshold"],
        )
        mock_data_quality.outlier_detection = MagicMock(
            iqr_multiplier=self.config["data_quality"]["outlier_detection"]["iqr_multiplier"],
            handling_strategy=self.config["data_quality"]["outlier_detection"]["handling_strategy"]
        )
        mock_data_quality.penalties = MagicMock(
            outlier_of_range_penalty=self.config["data_quality"]["penalties"]["out_of_range_penalty"],
            inconsistency_penalty=self.config["data_quality"]["penalties"]["inconsistency_penalty"],
        )
        mock_config = MagicMock()
        mock_config.data_quality = mock_data_quality
        return mock_config


@pytest.fixture
def mock_config_loader_instance() -> MockConfigLoader:
    return MockConfigLoader()


@pytest.fixture
def feature_validator_instance(
    mock_config_loader_instance: MockConfigLoader,
) -> Generator[FeatureValidator, None, None]:
    with patch("src.utils.config_loader.ConfigLoader") as MockCL:
        MockCL.return_value = mock_config_loader_instance
        MockCL.return_value.get_config.return_value = mock_config_loader_instance.get_config()
        with patch("src.utils.data_quality.DataValidator.calculate_quality_score") as mock_calculate_quality_score:
            mock_calculate_quality_score.return_value = 100.0  # Default to 100 for valid cases
            yield FeatureValidator()


@pytest.fixture
def sample_features_data() -> pd.DataFrame:
    data: dict[str, list[Any]] = {
        "ts": pd.to_datetime(
            ["2023-01-01 09:15", "2023-01-01 09:16", "2023-01-01 09:17", "2023-01-01 09:18", "2023-01-01 09:19"]
        ),
        "rsi": [30, 50, 70, 20, 80],
        "macd": [0.1, 0.5, -0.2, 1.0, -0.5],
        "bb_upper": [105, 106, 107, 108, 109],
        "bb_middle": [100, 101, 102, 103, 104],
        "bb_lower": [95, 96, 97, 98, 99],
    }
    return pd.DataFrame(data).set_index("ts").rename_axis("timestamp")


def test_validate_features_overall_valid(
    feature_validator_instance: FeatureValidator, sample_features_data: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_features_data.copy()
    is_valid: bool
    cleaned_df: pd.DataFrame
    report: DataQualityReport
    is_valid, cleaned_df, report = feature_validator_instance.validate_features(df)
    assert is_valid is True
    assert report.quality_score == 100.0
    assert not report.issues


def test_validate_features_overall_with_issues(
    feature_validator_instance: FeatureValidator, sample_features_data: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_features_data.copy()
    df.loc[df.index[0], "rsi"] = -10  # Out of range
    df.loc[df.index[1], "macd"] = 10  # Out of range

    is_valid: bool
    cleaned_df: pd.DataFrame
    report: DataQualityReport
    is_valid, cleaned_df, report = feature_validator_instance.validate_features(df)
    assert is_valid is False
    assert report.quality_score < 100.0
    assert any("Clipped" in issue for issue in report.issues)
    assert len(report.issues) > 0


def test_validate_features_empty_df(feature_validator_instance: FeatureValidator) -> None:
    df: pd.DataFrame = pd.DataFrame()
    is_valid: bool
    cleaned_df: pd.DataFrame
    report: DataQualityReport
    is_valid, cleaned_df, report = feature_validator_instance.validate_features(df)
    assert is_valid is False
    assert "No feature data to validate" in report.issues


def test_validate_features_nan_inf_values(
    feature_validator_instance: FeatureValidator, sample_features_data: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_features_data.copy()
    df.loc[0, "rsi"] = np.nan
    df.loc[1, "macd"] = np.inf
    is_valid: bool
    cleaned_df: pd.DataFrame
    report: DataQualityReport
    is_valid, cleaned_df, report = feature_validator_instance.validate_features(df)
    assert is_valid is False  # Should be False because min_valid_rows will be violated
    assert "Removed 2 rows with NaN/Inf values." in report.issues
    assert len(cleaned_df) == len(sample_features_data) - 2


def test_validate_features_outliers_clipping(
    feature_validator_instance: FeatureValidator, sample_features_data: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_features_data.copy()
    # Introduce an outlier that should be clipped
    df.loc[0, "rsi"] = 500  # Far outside [0, 100]
    is_valid: bool
    cleaned_df: pd.DataFrame
    report: DataQualityReport
    is_valid, cleaned_df, report = feature_validator_instance.validate_features(df)
    assert is_valid is False  # Likely invalid due to quality score drop
    assert "Clipped 1 outliers in feature 'rsi'" in report.issues
    # Check if the value was actually clipped (it should be within the bounds after clipping)
    assert cleaned_df.loc[cleaned_df.index[0], "rsi"] <= 100
    assert cleaned_df.loc[cleaned_df.index[0], "rsi"] >= 0
