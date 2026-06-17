"""
Test module for outlier handling strategies in data quality validation.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.utils.data_quality import DataValidator, DataQualityReport
from src.utils.config_loader import DataQualityConfig


class TestOutlierHandlingStrategies:
    """Test class for outlier handling strategies."""
    
    @pytest.fixture
    def mock_config_clip(self):
        """Mock configuration with 'clip' outlier handling strategy."""
        config = MagicMock()
        config.outlier_detection = MagicMock()
        config.outlier_detection.iqr_multiplier = 1.5
        config.outlier_detection.handling_strategy = "clip"
        config.validation = MagicMock()
        config.validation.min_valid_rows = 10
        config.validation.quality_score_threshold = 70.0
        config.validation.required_columns = ["ts", "open", "high", "low", "close", "volume"]
        config.penalties = MagicMock()
        config.penalties.outlier_penalty = 0.1
        config.penalties.gap_penalty = 0.2
        config.penalties.ohlc_violation_penalty = 0.2
        config.penalties.duplicate_penalty = 0.1
        config.time_series = MagicMock()
        config.time_series.gap_multiplier = 2.0
        return config

    @pytest.fixture
    def mock_config_remove(self):
        """Mock configuration with 'remove' outlier handling strategy."""
        config = MagicMock()
        config.outlier_detection = MagicMock()
        config.outlier_detection.iqr_multiplier = 1.5
        config.outlier_detection.handling_strategy = "remove"
        config.validation = MagicMock()
        config.validation.min_valid_rows = 10
        config.validation.quality_score_threshold = 70.0
        config.validation.required_columns = ["ts", "open", "high", "low", "close", "volume"]
        config.penalties = MagicMock()
        config.penalties.outlier_penalty = 0.1
        config.penalties.gap_penalty = 0.2
        config.penalties.ohlc_violation_penalty = 0.2
        config.penalties.duplicate_penalty = 0.1
        config.time_series = MagicMock()
        config.time_series.gap_multiplier = 2.0
        return config

    @pytest.fixture
    def mock_config_flag(self):
        """Mock configuration with 'flag' outlier handling strategy."""
        config = MagicMock()
        config.outlier_detection = MagicMock()
        config.outlier_detection.iqr_multiplier = 1.5
        config.outlier_detection.handling_strategy = "flag"
        config.validation = MagicMock()
        config.validation.min_valid_rows = 10
        config.validation.quality_score_threshold = 70.0
        config.validation.required_columns = ["ts", "open", "high", "low", "close", "volume"]
        config.penalties = MagicMock()
        config.penalties.outlier_penalty = 0.1
        config.penalties.gap_penalty = 0.2
        config.penalties.ohlc_violation_penalty = 0.2
        config.penalties.duplicate_penalty = 0.1
        config.time_series = MagicMock()
        config.time_series.gap_multiplier = 2.0
        return config

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Sample OHLCV data with outliers."""
        return pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })

    @pytest.fixture
    def sample_ohlcv_data_with_outliers(self):
        """Sample OHLCV data with known outliers."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })
        # Add outliers
        df.loc[10, 'close'] = 1000  # Extreme high outlier
        df.loc[20, 'close'] = 10    # Extreme low outlier
        df.loc[30, 'volume'] = 50000  # Volume outlier
        return df

    def test_clip_outlier_strategy(self, mock_config_clip, sample_ohlcv_data_with_outliers):
        """Test that 'clip' strategy clips outliers to bounds."""
        is_valid, cleaned_df, report = DataValidator.validate_ohlcv(
            sample_ohlcv_data_with_outliers, mock_config_clip
        )
        
        # Check that outliers were clipped
        assert cleaned_df['close'].max() < 1000, "Extreme high outlier should be clipped"
        assert cleaned_df['close'].min() > 10, "Extreme low outlier should be clipped"
        assert cleaned_df['volume'].max() < 50000, "Volume outlier should be clipped"
        
        # Check that clipping was reported
        clip_issues = [issue for issue in report.issues if "Clipped" in issue]
        assert len(clip_issues) > 0, "Should report clipped outliers"
        
        # Check that row count is preserved
        assert len(cleaned_df) == len(sample_ohlcv_data_with_outliers), "Row count should be preserved with clip strategy"

    def test_remove_outlier_strategy(self, mock_config_remove, sample_ohlcv_data_with_outliers):
        """Test that 'remove' strategy removes rows with outliers."""
        is_valid, cleaned_df, report = DataValidator.validate_ohlcv(
            sample_ohlcv_data_with_outliers, mock_config_remove
        )
        
        # Check that rows with outliers were removed
        assert len(cleaned_df) < len(sample_ohlcv_data_with_outliers), "Should remove rows with outliers"
        
        # Check that removal was reported
        remove_issues = [issue for issue in report.issues if "Removed" in issue and "outliers" in issue]
        assert len(remove_issues) > 0, "Should report removed outliers"
        
        # Check that no extreme values remain
        original_close_range = sample_ohlcv_data_with_outliers['close'].quantile([0.25, 0.75])
        cleaned_close_range = cleaned_df['close'].quantile([0.25, 0.75])
        assert cleaned_close_range[0.25] >= original_close_range[0.25], "Should remove extreme low values"
        assert cleaned_close_range[0.75] <= original_close_range[0.75] or abs(cleaned_close_range[0.75] - original_close_range[0.75]) < 10, "Should remove extreme high values"

    def test_flag_outlier_strategy(self, mock_config_flag, sample_ohlcv_data_with_outliers):
        """Test that 'flag' strategy adds flag columns for outliers."""
        is_valid, cleaned_df, report = DataValidator.validate_ohlcv(
            sample_ohlcv_data_with_outliers, mock_config_flag
        )
        
        # Check that original values are preserved
        assert len(cleaned_df) == len(sample_ohlcv_data_with_outliers), "Row count should be preserved with flag strategy"
        
        # Check that flag columns were added
        flag_columns = [col for col in cleaned_df.columns if col.endswith('_outlier_flag')]
        assert len(flag_columns) > 0, "Should add outlier flag columns"
        
        # Check that flagging was reported
        flag_issues = [issue for issue in report.issues if "Flagged" in issue]
        assert len(flag_issues) > 0, "Should report flagged outliers"
        
        # Check that outliers are properly flagged
        if 'close_outlier_flag' in cleaned_df.columns:
            flagged_outliers = cleaned_df[cleaned_df['close_outlier_flag'] == True]
            assert len(flagged_outliers) > 0, "Should flag outliers in close column"

    def test_feature_outlier_handling_clip(self, mock_config_clip):
        """Test outlier handling for feature validation with clip strategy."""
        # Create feature data with outliers
        feature_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'rsi': [50 + i * 0.1 for i in range(100)],
            'atr': [2 + i * 0.01 for i in range(100)],
            'macd': [0.5 + i * 0.001 for i in range(100)]
        })
        
        # Add outliers
        feature_data.loc[10, 'rsi'] = 200  # Extreme RSI outlier
        feature_data.loc[20, 'atr'] = 100  # Extreme ATR outlier
        
        is_valid, cleaned_df, report = DataValidator.validate_features(
            feature_data, mock_config_clip
        )
        
        # Check that outliers were clipped
        assert cleaned_df['rsi'].max() < 200, "RSI outlier should be clipped"
        assert cleaned_df['atr'].max() < 100, "ATR outlier should be clipped"
        
        # Check that clipping was reported
        clip_issues = [issue for issue in report.issues if "Clipped" in issue and "outliers" in issue]
        assert len(clip_issues) > 0, "Should report clipped feature outliers"

    def test_feature_outlier_handling_remove(self, mock_config_remove):
        """Test outlier handling for feature validation with remove strategy."""
        # Create feature data with outliers
        feature_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'rsi': [50 + i * 0.1 for i in range(100)],
            'atr': [2 + i * 0.01 for i in range(100)],
            'macd': [0.5 + i * 0.001 for i in range(100)]
        })
        
        # Add outliers
        feature_data.loc[10, 'rsi'] = 200  # Extreme RSI outlier
        feature_data.loc[20, 'atr'] = 100  # Extreme ATR outlier
        
        is_valid, cleaned_df, report = DataValidator.validate_features(
            feature_data, mock_config_remove
        )
        
        # Check that rows with outliers were removed
        assert len(cleaned_df) < len(feature_data), "Should remove rows with feature outliers"
        
        # Check that removal was reported
        remove_issues = [issue for issue in report.issues if "Removed" in issue and "outliers" in issue]
        assert len(remove_issues) > 0, "Should report removed feature outliers"

    def test_feature_outlier_handling_flag(self, mock_config_flag):
        """Test outlier handling for feature validation with flag strategy."""
        # Create feature data with outliers
        feature_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'rsi': [50 + i * 0.1 for i in range(100)],
            'atr': [2 + i * 0.01 for i in range(100)],
            'macd': [0.5 + i * 0.001 for i in range(100)]
        })
        
        # Add outliers
        feature_data.loc[10, 'rsi'] = 200  # Extreme RSI outlier
        feature_data.loc[20, 'atr'] = 100  # Extreme ATR outlier
        
        is_valid, cleaned_df, report = DataValidator.validate_features(
            feature_data, mock_config_flag
        )
        
        # Check that original values are preserved
        assert len(cleaned_df) == len(feature_data), "Row count should be preserved with flag strategy"
        
        # Check that flag columns were added
        flag_columns = [col for col in cleaned_df.columns if col.endswith('_outlier_flag')]
        assert len(flag_columns) > 0, "Should add outlier flag columns for features"
        
        # Check that flagging was reported
        flag_issues = [issue for issue in report.issues if "Flagged" in issue]
        assert len(flag_issues) > 0, "Should report flagged feature outliers"

    def test_no_outliers_detected(self, mock_config_clip, sample_ohlcv_data):
        """Test behavior when no outliers are detected."""
        is_valid, cleaned_df, report = DataValidator.validate_ohlcv(
            sample_ohlcv_data, mock_config_clip
        )
        
        # Should not report any outlier handling
        outlier_issues = [issue for issue in report.issues if "outlier" in issue.lower()]
        # Only the general detection message should be present if any outliers are found
        assert len(outlier_issues) <= 1, "Should not report outlier handling when no outliers"
        
        # Data should be unchanged
        assert len(cleaned_df) == len(sample_ohlcv_data), "Data should be unchanged when no outliers"

    def test_invalid_strategy_fallback(self, mock_config_clip, sample_ohlcv_data_with_outliers):
        """Test behavior with invalid strategy configuration."""
        # Set invalid strategy
        mock_config_clip.outlier_detection.handling_strategy = "invalid_strategy"
        
        # Should not crash, but behavior is undefined
        # The test ensures the system is robust to configuration errors
        try:
            is_valid, cleaned_df, report = DataValidator.validate_ohlcv(
                sample_ohlcv_data_with_outliers, mock_config_clip
            )
            # If it doesn't crash, that's good
            assert True, "Should handle invalid strategy gracefully"
        except Exception as e:
            # If it does crash, it should be with a clear error message
            assert "strategy" in str(e).lower(), "Error should mention strategy"

    def test_empty_dataframe_handling(self, mock_config_clip):
        """Test outlier handling with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        is_valid, cleaned_df, report = DataValidator.validate_ohlcv(
            empty_df, mock_config_clip
        )
        
        assert not is_valid, "Empty DataFrame should not be valid"
        assert len(cleaned_df) == 0, "Empty DataFrame should remain empty"
        assert "No data to validate" in report.issues, "Should report no data"

    def test_quality_score_impact(self, mock_config_clip, sample_ohlcv_data_with_outliers):
        """Test that outlier handling affects quality score appropriately."""
        # Test with clip strategy
        is_valid_clip, cleaned_df_clip, report_clip = DataValidator.validate_ohlcv(
            sample_ohlcv_data_with_outliers, mock_config_clip
        )
        
        # Test with remove strategy
        mock_config_clip.outlier_detection.handling_strategy = "remove"
        is_valid_remove, cleaned_df_remove, report_remove = DataValidator.validate_ohlcv(
            sample_ohlcv_data_with_outliers, mock_config_clip
        )
        
        # Both should result in valid data (assuming thresholds are reasonable)
        # The quality scores might differ based on the strategy
        assert isinstance(report_clip.quality_score, (int, float)), "Quality score should be numeric"
        assert isinstance(report_remove.quality_score, (int, float)), "Quality score should be numeric"
        assert 0 <= report_clip.quality_score <= 100, "Quality score should be in valid range"
        assert 0 <= report_remove.quality_score <= 100, "Quality score should be in valid range"