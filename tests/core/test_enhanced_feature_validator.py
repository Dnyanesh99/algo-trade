"""
Test suite for Enhanced FeatureValidator - Production-grade feature validation system.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.feature_validator import FeatureValidator
from src.utils.data_quality import DataQualityReport


class TestEnhancedFeatureValidator:
    """Test suite for Enhanced FeatureValidator."""

    @pytest.fixture
    def mock_config(self):
        """Create a comprehensive mock configuration."""
        config = MagicMock()
        
        # Mock data_quality configuration - explicitly create nested structure
        config.data_quality = MagicMock()
        config.data_quality.outlier_detection = MagicMock()
        config.data_quality.outlier_detection.handling_strategy = "clip"
        config.data_quality.outlier_detection.iqr_multiplier = 1.5
        config.data_quality.validation = MagicMock()
        config.data_quality.validation.min_valid_rows = 10
        config.data_quality.validation.quality_score_threshold = 80.0
        config.data_quality.penalties = MagicMock()
        config.data_quality.penalties.outlier_penalty = 0.1
        config.data_quality.penalties.gap_penalty = 0.2
        config.data_quality.penalties.ohlc_violation_penalty = 0.2
        config.data_quality.penalties.duplicate_penalty = 0.1
        
        # Mock model_training feature ranges (ensure it passes hasattr checks)
        config.model_training = MagicMock()
        config.model_training.feature_ranges = {
            "rsi_14": [0, 100],
            "stoch_k": [0, 100],
            "stoch_d": [0, 100],
            "willr": [-100, 0],
            "bb_position": [-3, 3],
        }
        
        # Mock trading features configurations (ensure it passes hasattr checks)
        config.trading = MagicMock()
        config.trading.features = MagicMock()
        config.trading.features.configurations = [
            {"name": "rsi", "function": "RSI"},
            {"name": "stoch", "function": "STOCH"},
            {"name": "willr", "function": "WILLR"},
            {"name": "macd", "function": "MACD"},
            {"name": "atr", "function": "ATR"},
        ]
        
        return config

    @pytest.fixture
    def sample_feature_data(self):
        """Create sample feature data for testing."""
        return pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="1min"),
            "rsi_14": np.random.uniform(20, 80, 100),
            "stoch_k": np.random.uniform(10, 90, 100),
            "stoch_d": np.random.uniform(10, 90, 100),
            "willr": np.random.uniform(-80, -20, 100),
            "macd": np.random.uniform(-1, 1, 100),
            "macd_signal": np.random.uniform(-1, 1, 100),
            "macd_hist": np.random.uniform(-0.5, 0.5, 100),
            "atr": np.random.uniform(0.1, 2.0, 100),
            "bb_upper": np.random.uniform(100, 120, 100),
            "bb_middle": np.random.uniform(95, 105, 100),
            "bb_lower": np.random.uniform(90, 100, 100),
        })

    def test_initialization(self, mock_config):
        """Test validator initialization."""
        validator = FeatureValidator(mock_config)
        
        assert validator.config == mock_config
        assert validator.validation_registry is not None
        assert validator.data_validator is not None

    def test_validate_features_basic(self, mock_config, sample_feature_data):
        """Test basic feature validation."""
        validator = FeatureValidator(mock_config)
        
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        assert isinstance(is_valid, bool)
        assert isinstance(cleaned_df, pd.DataFrame)
        assert isinstance(report, DataQualityReport)
        assert len(cleaned_df) <= len(sample_feature_data)

    def test_validate_features_with_violations(self, mock_config, sample_feature_data):
        """Test feature validation with range violations."""
        validator = FeatureValidator(mock_config)
        
        # Add violations
        sample_feature_data.loc[0, "rsi_14"] = 150  # Above max
        sample_feature_data.loc[1, "rsi_14"] = -50  # Below min
        sample_feature_data.loc[2, "stoch_k"] = 120  # Above max
        sample_feature_data.loc[3, "willr"] = 50   # Above max (should be negative)
        
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        # Check that violations were handled
        assert cleaned_df["rsi_14"].max() <= 100
        assert cleaned_df["rsi_14"].min() >= 0
        assert cleaned_df["stoch_k"].max() <= 100
        assert cleaned_df["willr"].max() <= 0
        
        # Check that issues were reported
        assert len(report.issues) > 0
        assert any("Clipped" in issue for issue in report.issues)

    def test_validate_features_remove_strategy(self, mock_config, sample_feature_data):
        """Test feature validation with remove strategy."""
        mock_config.data_quality.outlier_detection.handling_strategy = "remove"
        validator = FeatureValidator(mock_config)
        
        # Add violations
        sample_feature_data.loc[0, "rsi_14"] = 150
        sample_feature_data.loc[1, "stoch_k"] = 120
        
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        # Check that rows were removed
        assert len(cleaned_df) < len(sample_feature_data)
        assert any("Removed" in issue for issue in report.issues)

    def test_validate_features_flag_strategy(self, mock_config, sample_feature_data):
        """Test feature validation with flag strategy."""
        mock_config.data_quality.outlier_detection.handling_strategy = "flag"
        validator = FeatureValidator(mock_config)
        
        # Add violations
        sample_feature_data.loc[0, "rsi_14"] = 150
        sample_feature_data.loc[1, "stoch_k"] = 120
        
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        # Check that flag columns were added
        flag_columns = [col for col in cleaned_df.columns if col.endswith("_range_violation_flag")]
        assert len(flag_columns) > 0
        assert any("Flagged" in issue for issue in report.issues)

    def test_cross_feature_validation_stochastic(self, mock_config, sample_feature_data):
        """Test cross-feature validation for stochastic indicators."""
        validator = FeatureValidator(mock_config)
        
        # Make %D more volatile than %K (violation)
        sample_feature_data["stoch_k"] = [50] * 100  # Constant K
        sample_feature_data["stoch_d"] = np.random.uniform(10, 90, 100)  # Volatile D
        
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        # Should detect volatility inconsistency
        volatility_issues = [issue for issue in report.issues if "volatile" in issue]
        assert len(volatility_issues) > 0

    def test_cross_feature_validation_bollinger_bands(self, mock_config, sample_feature_data):
        """Test cross-feature validation for Bollinger Bands."""
        validator = FeatureValidator(mock_config)
        
        # Create ordering violation (upper < lower)
        sample_feature_data.loc[0, "bb_upper"] = 90
        sample_feature_data.loc[0, "bb_middle"] = 100
        sample_feature_data.loc[0, "bb_lower"] = 110
        
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        # Should detect ordering violation
        bollinger_issues = [issue for issue in report.issues if "Bollinger" in issue]
        assert len(bollinger_issues) > 0

    def test_cross_feature_validation_macd(self, mock_config, sample_feature_data):
        """Test cross-feature validation for MACD."""
        validator = FeatureValidator(mock_config)
        
        # Create MACD consistency violation
        sample_feature_data["macd"] = [1.0] * 100
        sample_feature_data["macd_signal"] = [0.5] * 100
        sample_feature_data["macd_hist"] = [0.4] * 100  # Should be 0.5, not 0.4
        
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        # Should detect MACD inconsistency
        macd_issues = [issue for issue in report.issues if "MACD" in issue]
        assert len(macd_issues) > 0

    def test_fallback_to_generic_outlier_detection(self, mock_config, sample_feature_data):
        """Test fallback to generic outlier detection for unknown features."""
        validator = FeatureValidator(mock_config)
        
        # Add unknown feature with outliers
        unknown_data = [1, 2, 3, 1000, 5] + [2] * 95  # 1000 is outlier, pad to 100 rows
        sample_feature_data["unknown_feature"] = unknown_data
        
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        # Should apply generic outlier detection
        outlier_issues = [issue for issue in report.issues if "outlier" in issue.lower()]
        assert len(outlier_issues) > 0

    def test_empty_dataframe(self, mock_config):
        """Test validation with empty DataFrame."""
        validator = FeatureValidator(mock_config)
        
        empty_df = pd.DataFrame()
        is_valid, cleaned_df, report = validator.validate_features(empty_df)
        
        assert not is_valid
        assert len(cleaned_df) == 0
        assert "No feature data to validate" in report.issues

    def test_nan_handling(self, mock_config, sample_feature_data):
        """Test handling of NaN values."""
        validator = FeatureValidator(mock_config)
        
        # Add NaN values
        sample_feature_data.loc[0:5, "rsi_14"] = np.nan
        sample_feature_data.loc[10:15, "stoch_k"] = np.nan
        
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        # Should report NaN values
        nan_issues = [issue for issue in report.issues if "NaN" in issue]
        assert len(nan_issues) > 0

    def test_quality_score_calculation(self, mock_config, sample_feature_data):
        """Test quality score calculation."""
        validator = FeatureValidator(mock_config)
        
        # Add various issues
        sample_feature_data.loc[0, "rsi_14"] = 150  # Violation
        sample_feature_data.loc[1, "stoch_k"] = 120  # Violation
        sample_feature_data.loc[2:5, "willr"] = np.nan  # NaN values
        
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        # Quality score should be less than 100
        assert 0 <= report.quality_score <= 100
        assert report.quality_score < 100  # Should be penalized for issues

    def test_validation_summary(self, mock_config):
        """Test getting validation summary."""
        validator = FeatureValidator(mock_config)
        
        summary = validator.get_validation_summary()
        
        assert isinstance(summary, dict)
        assert "total_rules" in summary
        assert "by_validator_type" in summary
        assert "by_priority" in summary
        assert "rules" in summary

    def test_feature_ranges_from_config(self, mock_config, sample_feature_data):
        """Test that feature ranges from config are properly used."""
        validator = FeatureValidator(mock_config)
        
        # Test with feature that has config range
        bb_position_data = [-5, 5, 2, -2, 1] + [0] * 95  # Violations at -5 and 5, pad to 100 rows
        sample_feature_data["bb_position"] = bb_position_data
        
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        # Should clip to [-3, 3] range
        assert cleaned_df["bb_position"].min() >= -3
        assert cleaned_df["bb_position"].max() <= 3

    def test_auto_discovery_indicators(self, mock_config, sample_feature_data):
        """Test auto-discovery of indicator validation rules."""
        validator = FeatureValidator(mock_config)
        
        # Test with indicator that should be auto-discovered
        sample_feature_data["test_rsi"] = [150, -50, 50, 75, 25]  # Violations at 150 and -50
        
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        # Should apply RSI rules (0-100 range)
        assert cleaned_df["test_rsi"].min() >= 0
        assert cleaned_df["test_rsi"].max() <= 100

    def test_instrument_id_logging(self, mock_config, sample_feature_data):
        """Test that instrument ID is properly logged."""
        validator = FeatureValidator(mock_config)
        
        with patch('src.core.feature_validator.logger') as mock_logger:
            validator.validate_features(sample_feature_data, instrument_id=123)
            
            # Check that instrument ID was logged
            mock_logger.info.assert_called()
            log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
            assert any("instrument 123" in call for call in log_calls)

    def test_outlier_counts_aggregation(self, mock_config, sample_feature_data):
        """Test aggregation of outlier counts in report."""
        validator = FeatureValidator(mock_config)
        
        # Add violations in multiple features
        sample_feature_data.loc[0, "rsi_14"] = 150
        sample_feature_data.loc[1, "stoch_k"] = 120
        sample_feature_data.loc[2, "willr"] = 50
        
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        # Should have outlier counts for each feature
        assert isinstance(report.outliers, dict)
        assert len(report.outliers) > 0
        assert sum(report.outliers.values()) > 0

    def test_error_handling_graceful_degradation(self, mock_config, sample_feature_data):
        """Test graceful error handling."""
        validator = FeatureValidator(mock_config)
        
        # Corrupt some data to trigger errors
        sample_feature_data["problematic_feature"] = [1, "invalid", 3, 4, 5]
        
        # Should handle gracefully without crashing
        is_valid, cleaned_df, report = validator.validate_features(sample_feature_data)
        
        # Should still return valid results
        assert isinstance(is_valid, bool)
        assert isinstance(cleaned_df, pd.DataFrame)
        assert isinstance(report, DataQualityReport)