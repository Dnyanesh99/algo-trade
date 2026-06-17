from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.candle_validator import CandleValidator, ValidationResult


# Mock ConfigLoader for testing purposes
class MockConfigLoader:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {
            "data_quality": {
                "time_series": {"gap_multiplier": 2.0},
                "outlier_detection": {"iqr_multiplier": 1.5, "handling_strategy": "clip"},
                "penalties": {
                    "outlier_penalty": 0.1,
                    "gap_penalty": 0.2,
                    "ohlc_violation_penalty": 0.2,
                    "duplicate_penalty": 0.1,
                },
                "validation": {
                    "enabled": True,
                    "min_valid_rows": 50,
                    "quality_score_threshold": 80.0,
                    "required_columns": ["ts", "open", "high", "low", "close", "volume"],
                    "expected_columns": {
                        "date": "datetime64[ns]",
                        "open": "float64",
                        "high": "float64",
                        "low": "float64",
                        "close": "float64",
                        "volume": "int64",
                    },
                },
            }
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
            required_columns=self.config["data_quality"]["validation"]["required_columns"],
        )
        mock_data_quality.outlier_detection = MagicMock(
            iqr_multiplier=self.config["data_quality"]["outlier_detection"]["iqr_multiplier"],
            handling_strategy=self.config["data_quality"]["outlier_detection"]["handling_strategy"]
        )
        mock_data_quality.time_series = MagicMock(
            gap_multiplier=self.config["data_quality"]["time_series"]["gap_multiplier"]
        )
        mock_data_quality.penalties = MagicMock(
            outlier_penalty=self.config["data_quality"]["penalties"]["outlier_penalty"],
            gap_penalty=self.config["data_quality"]["penalties"]["gap_penalty"],
            ohlc_violation_penalty=self.config["data_quality"]["penalties"]["ohlc_violation_penalty"],
            duplicate_penalty=self.config["data_quality"]["penalties"]["duplicate_penalty"],
        )
        mock_config = MagicMock()
        mock_config.data_quality = mock_data_quality
        return mock_config


@pytest.fixture
def mock_config_loader_instance() -> MockConfigLoader:
    return MockConfigLoader()


@pytest.fixture
def candle_validator_instance(mock_config_loader_instance: MockConfigLoader) -> Generator[CandleValidator, None, None]:
    with patch("src.utils.config_loader.ConfigLoader") as MockCL:
        MockCL.return_value = mock_config_loader_instance
        MockCL.return_value.get_config.return_value = mock_config_loader_instance.get_config()
        # Mock DataValidator.calculate_quality_score as it's a static method
        with patch("src.utils.data_quality.DataValidator.calculate_quality_score") as mock_calculate_quality_score:
            mock_calculate_quality_score.return_value = 100.0  # Default to 100 for valid cases
            yield CandleValidator()


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    data: dict[str, list[Any]] = {
        "ts": [datetime(2023, 1, 1, 9, 15) + timedelta(minutes=i) for i in range(10)],
        "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
    }
    return pd.DataFrame(data)


@pytest.mark.asyncio
async def test_overall_validate_candles_valid(
    candle_validator_instance: CandleValidator, sample_ohlcv_data: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_ohlcv_data.copy()
    validation_result: ValidationResult = await candle_validator_instance.validate_candles(df, 1, "1min")
    final_score: float = validation_result.quality_report.quality_score
    all_issues: list[str] = validation_result.quality_report.issues
    assert final_score == 100.0
    assert not all_issues
    assert validation_result.is_valid is True


@pytest.mark.asyncio
async def test_overall_validate_candles_with_multiple_issues(
    candle_validator_instance: CandleValidator, sample_ohlcv_data: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_ohlcv_data.copy()
    df.loc[0, "open"] = 102  # OHLC violation
    df.loc[1, "volume"] = -50  # Negative volume
    df = df.drop(index=[5]).reset_index(drop=True)  # Gap
    df.loc[2, "close"] = 5000  # Outlier

    validation_result: ValidationResult = await candle_validator_instance.validate_candles(df, 1, "1min")
    final_score: float = validation_result.quality_report.quality_score
    all_issues: list[str] = validation_result.quality_report.issues
    assert final_score < 100.0
    assert any("OHLC_VIOLATION" in issue for issue in all_issues)
    assert any("NEGATIVE_VOLUME" in issue for issue in all_issues)
    assert any("TIME_GAP" in issue for issue in all_issues)
    assert any("OUTLIER" in issue for issue in all_issues)
    assert len(all_issues) > 0
    assert validation_result.is_valid is False


@pytest.mark.asyncio
async def test_overall_validate_candles_empty_df(
    candle_validator_instance: CandleValidator,
) -> None:
    df: pd.DataFrame = pd.DataFrame()
    validation_result: ValidationResult = await candle_validator_instance.validate_candles(df, 1, "1min")
    assert validation_result.is_valid is False
    assert "Empty dataset provided" in validation_result.quality_report.issues
    assert "Empty dataset" in validation_result.validation_errors


@pytest.mark.asyncio
async def test_overall_validate_candles_missing_required_columns(
    candle_validator_instance: CandleValidator, sample_ohlcv_data: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_ohlcv_data.copy().drop(columns=["volume"])
    validation_result: ValidationResult = await candle_validator_instance.validate_candles(df, 1, "1min")
    assert validation_result.is_valid is False
    assert "Missing required columns: ['volume']" in validation_result.quality_report.issues
    assert "Missing required columns: ['volume']" in validation_result.validation_errors


@pytest.mark.asyncio
async def test_overall_validate_candles_below_min_rows(
    candle_validator_instance: CandleValidator, sample_ohlcv_data: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_ohlcv_data.copy().head(5)  # Less than min_valid_rows (50)
    validation_result: ValidationResult = await candle_validator_instance.validate_candles(df, 1, "1min")
    assert validation_result.is_valid is False
    assert "Data quality score" in validation_result.validation_errors[0]


@pytest.mark.asyncio
async def test_overall_validate_candles_with_nan_values(
    candle_validator_instance: CandleValidator, sample_ohlcv_data: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_ohlcv_data.copy()
    df.loc[0, "open"] = None  # Introduce NaN
    validation_result: ValidationResult = await candle_validator_instance.validate_candles(df, 1, "1min")
    assert validation_result.is_valid is False  # Should be invalid due to NaN removal affecting min_valid_rows
    assert "Removed 1 rows with NaN values in critical columns." in validation_result.quality_report.issues


@pytest.mark.asyncio
async def test_overall_validate_candles_duplicate_timestamps(
    candle_validator_instance: CandleValidator, sample_ohlcv_data: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_ohlcv_data.copy()
    df = pd.concat([df, df.iloc[[0]]])  # Add a duplicate row
    validation_result: ValidationResult = await candle_validator_instance.validate_candles(df, 1, "1min")
    assert validation_result.is_valid is False  # Should be invalid due to duplicate removal affecting min_valid_rows
    assert "Removed 1 duplicate timestamps, keeping the first entry." in validation_result.quality_report.issues


@pytest.mark.asyncio
async def test_overall_validate_candles_ohlc_correction(
    candle_validator_instance: CandleValidator, sample_ohlcv_data: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_ohlcv_data.copy()
    df.loc[0, "high"] = 90  # high < low
    df.loc[1, "low"] = 120  # low > high
    df.loc[2, "open"] = 150  # open > high
    df.loc[3, "close"] = 50  # close < low

    validation_result: ValidationResult = await candle_validator_instance.validate_candles(df, 1, "1min")
    assert validation_result.is_valid is False  # Should be invalid due to corrections
    assert "Detected and corrected" in validation_result.quality_report.issues[0]
    assert "Corrected" in validation_result.warnings[0]


@pytest.mark.asyncio
async def test_overall_validate_candles_time_gaps(
    candle_validator_instance: CandleValidator, sample_ohlcv_data: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_ohlcv_data.copy()
    # Create a large gap
    df.loc[5:, "ts"] = df.loc[5:, "ts"] + timedelta(minutes=100)
    validation_result: ValidationResult = await candle_validator_instance.validate_candles(df, 1, "1min")
    assert validation_result.is_valid is False
    assert "Detected 1 significant time gaps in the data series." in validation_result.quality_report.issues


@pytest.mark.asyncio
async def test_overall_validate_candles_outliers(
    candle_validator_instance: CandleValidator, sample_ohlcv_data: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_ohlcv_data.copy()
    df.loc[0, "close"] = 100000  # Introduce an extreme outlier
    validation_result: ValidationResult = await candle_validator_instance.validate_candles(df, 1, "1min")
    assert validation_result.is_valid is False
    assert "Detected 1 statistical outliers across columns." in validation_result.quality_report.issues
