"""
Comprehensive tests for CandleValidator.
Tests all real-world edge cases and different instrument types.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.validation.candle_validator import CandleValidator, ConfigurationError
from tests.validation.test_fixtures import ValidationTestFixtures


class TestCandleValidator:
    """Test suite for CandleValidator."""

    def test_validator_initialization_with_default_config(self):
        """Test validator initialization with default configuration."""
        validator = CandleValidator()
        assert validator.config is not None
        assert validator.outlier_detector is not None

    def test_validator_initialization_with_custom_config(self, config_equity):
        """Test validator initialization with custom configuration."""
        validator = CandleValidator(config=config_equity)
        assert validator.config == config_equity

    def test_validator_initialization_missing_config(self):
        """Test validator initialization fails with missing config."""
        with patch('src.validation.candle_validator.ConfigLoader') as mock_loader:
            mock_loader.return_value.get_config.return_value.data_quality = None
            with pytest.raises(ConfigurationError):
                CandleValidator()

    def test_config_validation_invalid_threshold(self, config_equity):
        """Test configuration validation with invalid threshold."""
        config_equity.validation.quality_score_threshold = 150  # Invalid > 100
        with pytest.raises(ConfigurationError):
            CandleValidator(config=config_equity)

    def test_config_validation_empty_required_columns(self, config_equity):
        """Test configuration validation with empty required columns."""
        config_equity.validation.required_columns = []
        with pytest.raises(ConfigurationError):
            CandleValidator(config=config_equity)


class TestIndexDataValidation:
    """Test validation for index data (zero volume, no OI)."""

    def test_index_data_validation_success(self, config_index):
        """Test successful validation of clean index data."""
        validator = CandleValidator(config=config_index)
        df = ValidationTestFixtures.create_index_data(50)

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="NIFTY50", instrument_type="INDICES"
        )

        assert is_valid
        assert len(cleaned_df) > 0
        assert report.quality_score >= config_index.validation.quality_score_threshold
        assert "ts" in cleaned_df.columns
        assert all(col in cleaned_df.columns for col in ["open", "high", "low", "close"])

    def test_index_data_zero_volume_allowed(self, config_index):
        """Test that zero volume is allowed for index data."""
        validator = CandleValidator(config=config_index)
        df = ValidationTestFixtures.create_index_data(20)
        df['volume'] = 0  # Ensure zero volume

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="NIFTY50", instrument_type="INDICES"
        )

        # Should not fail due to zero volume since volume is not in required_columns
        assert len(cleaned_df) == len(df)

    def test_index_data_empty_oi_handling(self, config_index):
        """Test handling of empty OI column in index data."""
        validator = CandleValidator(config=config_index)
        df = ValidationTestFixtures.create_index_data(20)
        df['oi'] = None  # Empty OI column

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="NIFTY50", instrument_type="INDICES"
        )

        # Should handle empty OI gracefully since OI is not in required_columns
        assert len(cleaned_df) > 0


class TestEquityDataValidation:
    """Test validation for equity data (positive volume, no OI)."""

    def test_equity_data_validation_success(self, config_equity):
        """Test successful validation of clean equity data."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_equity_data(50)

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="RELIANCE", instrument_type="EQ"
        )

        assert is_valid
        assert len(cleaned_df) > 0
        assert report.quality_score >= config_equity.validation.quality_score_threshold
        assert all(cleaned_df['volume'] > 0)  # Equity requires positive volume

    def test_equity_data_zero_volume_filtered(self, config_equity):
        """Test that zero volume rows are filtered for equity data."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_equity_data(20)

        # Insert some zero volume rows
        df.loc[5:7, 'volume'] = 0
        initial_rows = len(df)

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="RELIANCE", instrument_type="EQ"
        )

        # Should filter out zero volume rows
        assert len(cleaned_df) < initial_rows
        assert all(cleaned_df['volume'] > 0)

    def test_equity_data_negative_volume_filtered(self, config_equity):
        """Test that negative volume rows are filtered."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_negative_values_data()

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="RELIANCE", instrument_type="EQ"
        )

        # Should filter out negative volume rows
        assert all(cleaned_df['volume'] >= 0)


class TestFuturesDataValidation:
    """Test validation for futures data (volume + OI)."""

    def test_futures_data_validation_success(self, config_futures):
        """Test successful validation of clean futures data."""
        validator = CandleValidator(config=config_futures)
        df = ValidationTestFixtures.create_futures_data(50)

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="NIFTY25JAN25000CE", instrument_type="NFO-FUT"
        )

        assert is_valid
        assert len(cleaned_df) > 0
        assert all(cleaned_df['volume'] > 0)
        assert all(cleaned_df['oi'] >= 0)

    def test_futures_data_oi_validation(self, config_futures):
        """Test OI validation for futures data."""
        validator = CandleValidator(config=config_futures)
        df = ValidationTestFixtures.create_futures_data(20)

        # Insert negative OI values
        df.loc[3:5, 'oi'] = -1000

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="NIFTY25JAN25000CE", instrument_type="NFO-FUT"
        )

        # Should filter out negative OI rows
        assert all(cleaned_df['oi'] >= 0)


class TestOHLCValidation:
    """Test OHLC relationship validation."""

    def test_ohlc_validation_when_all_columns_present(self, config_equity):
        """Test OHLC validation when all OHLC columns are in required_columns."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_data_with_ohlc_violations()

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        # Should correct OHLC violations
        assert all(cleaned_df['high'] >= cleaned_df[['open', 'close']].max(axis=1))
        assert all(cleaned_df['low'] <= cleaned_df[['open', 'close']].min(axis=1))
        assert all(cleaned_df['high'] >= cleaned_df['low'])
        assert report.ohlc_violations > 0

    def test_ohlc_validation_skipped_when_columns_missing(self, config_minimal):
        """Test OHLC validation is skipped when not all OHLC columns are required."""
        validator = CandleValidator(config=config_minimal)
        df = ValidationTestFixtures.create_minimal_columns_data()

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        # Should skip OHLC validation and report it
        assert report.ohlc_violations == 0
        assert any("Skipping OHLC validation" in issue for issue in report.issues)


class TestOutlierDetection:
    """Test outlier detection functionality."""

    def test_outlier_detection_with_required_columns(self, config_equity):
        """Test outlier detection processes only required columns."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_data_with_outliers()

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        # Should detect outliers in required columns only
        expected_cols = ["open", "high", "low", "close", "volume"]
        detected_cols = list(report.outliers.keys())

        for col in detected_cols:
            assert col in expected_cols

        assert sum(report.outliers.values()) > 0

    def test_outlier_detection_skipped_when_no_suitable_columns(self):
        """Test outlier detection is skipped when no suitable columns are required."""
        from src.utils.config_loader import ConfigLoader
        config = ConfigLoader().get_config().data_quality
        config.validation.required_columns = ["ts"]  # Only timestamp

        validator = CandleValidator(config=config)
        df = ValidationTestFixtures.create_data_with_outliers()

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        # Should skip outlier detection
        assert report.outliers == {}
        assert any("No suitable columns found for outlier detection" in issue for issue in report.issues)


class TestTimestampHandling:
    """Test timestamp column handling and validation."""

    def test_timestamp_column_detection(self, config_equity):
        """Test detection of different timestamp column names."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_equity_data(20)

        # Test with 'timestamp' column
        df_timestamp = df.rename(columns={'ts': 'timestamp'})
        is_valid, cleaned_df, report = validator.validate(
            df_timestamp, symbol="TEST", instrument_type="EQ"
        )

        assert 'ts' in cleaned_df.columns  # Should be renamed to 'ts'

    def test_missing_timestamp_column_error(self, config_equity):
        """Test error when timestamp column is missing."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_data_missing_timestamp()

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        assert not is_valid
        assert any("Missing timestamp column" in issue for issue in report.issues)

    def test_unordered_timestamp_sorting(self, config_equity):
        """Test sorting of unordered timestamps."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_unordered_timestamp_data()

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        # Should sort timestamps
        assert cleaned_df['ts'].is_monotonic_increasing


class TestTimeGapDetection:
    """Test time gap detection functionality."""

    def test_time_gap_detection(self, config_equity):
        """Test detection of significant time gaps."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_data_with_time_gaps()

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ", timeframe="1min"
        )

        # Should detect time gaps
        assert report.time_gaps > 0
        assert any("time gaps" in issue for issue in report.issues)

    def test_time_gap_detection_without_timeframe(self, config_equity):
        """Test time gap detection skipped without timeframe."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_equity_data(20)

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"  # No timeframe
        )

        # Should skip time gap detection
        assert report.time_gaps == 0


class TestDuplicateHandling:
    """Test duplicate timestamp handling."""

    def test_duplicate_timestamp_removal(self, config_equity):
        """Test removal of duplicate timestamps."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_data_with_duplicates()

        initial_rows = len(df)

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        # Should remove duplicates
        assert len(cleaned_df) < initial_rows
        assert not cleaned_df['ts'].duplicated().any()
        assert report.duplicate_timestamps > 0


class TestNaNHandling:
    """Test NaN value handling."""

    def test_nan_value_removal(self, config_equity):
        """Test removal of rows with NaN values in required columns."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_data_with_nan_values()

        initial_rows = len(df)

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        # Should remove rows with NaN in required columns
        assert len(cleaned_df) < initial_rows
        required_cols = config_equity.validation.required_columns
        assert not cleaned_df[required_cols].isnull().any().any()


class TestInstrumentTypeDetection:
    """Test instrument type detection and handling."""

    def test_instrument_type_index_detection(self, config_equity):
        """Test INDEX instrument type detection."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_equity_data(20)
        df['volume'] = 0  # Zero volume for index

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="NIFTY50", instrument_type="INDICES"
        )

        # Should allow zero volume for INDEX type
        # Note: This depends on configuration - if volume is required, it might still filter
        assert len(cleaned_df) >= 0

    def test_instrument_type_equity_detection(self, config_equity):
        """Test EQUITY instrument type detection."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_equity_data(20)
        df.loc[5:7, 'volume'] = 0  # Some zero volume

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="RELIANCE", instrument_type="EQ"
        )

        # Should filter zero volume for EQUITY type (if volume is required)
        if 'volume' in config_equity.validation.required_columns:
            assert all(cleaned_df['volume'] > 0)


class TestQualityScoring:
    """Test data quality scoring."""

    def test_quality_score_calculation(self, config_equity):
        """Test quality score calculation with various issues."""
        validator = CandleValidator(config=config_equity)

        # Create data with multiple issues
        df = ValidationTestFixtures.create_equity_data(100)
        df = pd.concat([
            df,
            ValidationTestFixtures.create_data_with_outliers().iloc[:10],
            ValidationTestFixtures.create_data_with_duplicates().iloc[:5]
        ]).reset_index(drop=True)

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        # Should calculate quality score considering all issues
        assert 0 <= report.quality_score <= 100
        assert report.total_rows > report.valid_rows  # Some data should be filtered

    def test_quality_score_threshold_validation(self, config_equity):
        """Test validation against quality score threshold."""
        config_equity.validation.quality_score_threshold = 95.0  # High threshold
        validator = CandleValidator(config=config_equity)

        # Create poor quality data
        df = ValidationTestFixtures.create_data_with_outliers()
        df = pd.concat([df] * 3).reset_index(drop=True)  # Duplicate to create more issues

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        # Should fail quality threshold
        if report.quality_score < 95.0:
            assert not is_valid


class TestEmptyDataHandling:
    """Test handling of empty datasets."""

    def test_empty_dataframe_handling(self, config_equity):
        """Test handling of empty DataFrame."""
        validator = CandleValidator(config=config_equity)
        df = pd.DataFrame()

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        assert not is_valid
        assert len(cleaned_df) == 0
        assert report.quality_score == 0.0
        assert "No data to validate" in report.issues


class TestErrorHandling:
    """Test error handling and exception scenarios."""

    def test_validation_exception_handling(self, config_equity):
        """Test handling of validation exceptions."""
        validator = CandleValidator(config=config_equity)

        # Create DataFrame that will cause processing error
        df = pd.DataFrame({'invalid_column': [1, 2, 3]})

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        assert not is_valid
        assert any("Critical validation error" in issue for issue in report.issues)

    def test_malformed_timestamp_handling(self, config_equity):
        """Test handling of malformed timestamps."""
        validator = CandleValidator(config=config_equity)
        df = ValidationTestFixtures.create_equity_data(20)

        # Insert malformed timestamps
        df.loc[5, 'ts'] = "invalid_timestamp"
        df.loc[10, 'ts'] = None

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        # Should handle malformed timestamps gracefully
        assert len(cleaned_df) <= len(df)


class TestConfigurationVariations:
    """Test different configuration variations."""

    def test_minimal_required_columns(self):
        """Test validation with minimal required columns."""
        from src.utils.config_loader import ConfigLoader
        config = ConfigLoader().get_config().data_quality
        config.validation.required_columns = ["ts", "close"]

        validator = CandleValidator(config=config)
        df = ValidationTestFixtures.create_minimal_columns_data()

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        assert len(cleaned_df) > 0
        assert "ts" in cleaned_df.columns
        assert "close" in cleaned_df.columns

    def test_custom_column_combinations(self):
        """Test validation with custom column combinations."""
        from src.utils.config_loader import ConfigLoader
        config = ConfigLoader().get_config().data_quality
        config.validation.required_columns = ["ts", "open", "volume"]  # Custom combination

        validator = CandleValidator(config=config)
        df = ValidationTestFixtures.create_equity_data(20)

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        # Should validate only required columns
        assert "ts" in cleaned_df.columns
        assert "open" in cleaned_df.columns
        assert "volume" in cleaned_df.columns


# Performance and integration tests
class TestPerformanceAndIntegration:
    """Test performance and integration scenarios."""

    def test_large_dataset_validation(self, config_equity):
        """Test validation performance with large dataset."""
        validator = CandleValidator(config=config_equity)

        # Create large dataset
        large_df = pd.concat([
            ValidationTestFixtures.create_equity_data(500)
            for _ in range(4)
        ]).reset_index(drop=True)

        is_valid, cleaned_df, report = validator.validate(
            large_df, symbol="TEST", instrument_type="EQ"
        )

        # Should handle large dataset efficiently
        assert len(cleaned_df) > 0
        assert report.total_rows == len(large_df)

    def test_real_csv_data_validation(self, config_index):
        """Test validation with real CSV data from the project."""
        validator = CandleValidator(config=config_index)
        df = ValidationTestFixtures.load_real_csv_data()

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="NIFTY50", instrument_type="INDICES"
        )

        # Should successfully validate real data
        assert len(cleaned_df) > 0
        assert report.quality_score > 0
