"""
Comprehensive tests for OutlierDetector.
Tests all outlier detection strategies and edge cases.
"""

import numpy as np
import pandas as pd

from src.utils.config_loader import ConfigLoader
from src.validation.outlier_detector import OutlierDetector
from tests.validation.test_fixtures import ValidationTestFixtures


class TestOutlierDetectorBasics:
    """Test basic outlier detector functionality."""

    def test_outlier_detector_initialization(self):
        """Test OutlierDetector can be instantiated."""
        detector = OutlierDetector()
        assert detector is not None

    def test_detect_and_handle_method_exists(self):
        """Test detect_and_handle method exists and is callable."""
        detector = OutlierDetector()
        assert hasattr(detector, 'detect_and_handle')
        assert callable(detector.detect_and_handle)


class TestIQROutlierDetection:
    """Test IQR-based outlier detection algorithm."""

    def test_iqr_outlier_detection_basic(self):
        """Test basic IQR outlier detection."""
        # Create data with known outliers
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'value': data
        })

        config = ConfigLoader().get_config().data_quality
        issues_log = []

        # Test outlier detection
        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['value'], config, issues_log
        )

        # Should detect the outlier
        assert outliers['value'] > 0
        assert len(issues_log) > 0

    def test_iqr_calculation_correctness(self):
        """Test IQR calculation correctness."""
        # Create data with known quartiles
        data = list(range(1, 101))  # 1 to 100
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'value': data
        })

        # Q1 = 25.5, Q3 = 75.5, IQR = 50
        # With multiplier 1.5: lower = 25.5 - 75 = -49.5, upper = 75.5 + 75 = 150.5
        # No outliers expected in range 1-100

        config = ConfigLoader().get_config().data_quality
        config.outlier_detection.iqr_multiplier = 1.5
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['value'], config, issues_log
        )

        # Should detect no outliers
        assert outliers['value'] == 0

    def test_iqr_with_extreme_outliers(self):
        """Test IQR detection with extreme outliers."""
        # Create data with extreme outliers
        normal_data = np.random.normal(100, 10, 95)
        outlier_data = [1000, -1000, 5000, -5000, 10000]
        data = list(normal_data) + outlier_data

        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'value': data
        })

        config = ConfigLoader().get_config().data_quality
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['value'], config, issues_log
        )

        # Should detect multiple outliers
        assert outliers['value'] >= 5


class TestOutlierHandlingStrategies:
    """Test different outlier handling strategies."""

    def test_clip_strategy(self):
        """Test clip strategy for outlier handling."""
        # Create data with outliers
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'price': [10, 11, 12, 13, 1000, 15, 16, 17, 18, 19]  # 1000 is outlier
        })

        config = ConfigLoader().get_config().data_quality
        config.outlier_detection.handling_strategy = "clip"
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['price'], config, issues_log
        )

        # Should clip the outlier, not remove rows
        assert len(processed_df) == len(df)
        assert outliers['price'] > 0
        assert max(processed_df['price']) < 1000  # Outlier should be clipped
        assert any("Clipped" in issue for issue in issues_log)

    def test_remove_strategy(self):
        """Test remove strategy for outlier handling."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'price': [10, 11, 12, 13, 1000, 15, 16, 17, 18, 19],
            'volume': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        })

        config = ConfigLoader().get_config().data_quality
        config.outlier_detection.handling_strategy = "remove"
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['price'], config, issues_log
        )

        # Should remove rows with outliers
        assert len(processed_df) < len(df)
        assert 1000 not in processed_df['price'].values
        assert any("Removed" in issue for issue in issues_log)

    def test_flag_strategy(self):
        """Test flag strategy for outlier handling."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'price': [10, 11, 12, 13, 1000, 15, 16, 17, 18, 19]
        })

        config = ConfigLoader().get_config().data_quality
        config.outlier_detection.handling_strategy = "flag"
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['price'], config, issues_log
        )

        # Should add flag column and keep all rows
        assert len(processed_df) == len(df)
        assert 'price_outlier_flag' in processed_df.columns
        assert processed_df['price_outlier_flag'].sum() > 0  # At least one flagged
        assert any("Flagged" in issue for issue in issues_log)


class TestMultipleColumnsOutlierDetection:
    """Test outlier detection across multiple columns."""

    def test_multiple_columns_detection(self):
        """Test outlier detection across multiple columns."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'high': [11, 12, 13, 14, 15, 16, 17, 18, 19, 1000],  # Outlier in high
            'low': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            'close': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5],
            'volume': [100, 110, 120, 130, 140, 10000, 160, 170, 180, 190]  # Outlier in volume
        })

        config = ConfigLoader().get_config().data_quality
        config.outlier_detection.handling_strategy = "flag"
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['open', 'high', 'low', 'close', 'volume'], config, issues_log
        )

        # Should detect outliers in multiple columns
        assert outliers['high'] > 0
        assert outliers['volume'] > 0
        assert outliers['open'] == 0  # No outliers in open
        assert outliers['low'] == 0   # No outliers in low
        assert outliers['close'] == 0 # No outliers in close

    def test_ohlcv_columns_realistic_data(self):
        """Test outlier detection on realistic OHLCV data."""
        df = ValidationTestFixtures.create_data_with_outliers()

        config = ConfigLoader().get_config().data_quality
        config.outlier_detection.handling_strategy = "clip"
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['open', 'high', 'low', 'close', 'volume'], config, issues_log
        )

        # Should process all OHLCV columns
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            assert col in outliers

        # Should detect some outliers (we inserted them in test data)
        total_outliers = sum(outliers.values())
        assert total_outliers > 0


class TestEdgeCases:
    """Test edge cases for outlier detection."""

    def test_empty_dataframe(self):
        """Test outlier detection on empty DataFrame."""
        df = pd.DataFrame()

        config = ConfigLoader().get_config().data_quality
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['price'], config, issues_log
        )

        # Should handle empty DataFrame gracefully
        assert len(processed_df) == 0
        assert outliers['price'] == 0

    def test_single_row_dataframe(self):
        """Test outlier detection on single row DataFrame."""
        df = pd.DataFrame({
            'ts': [pd.Timestamp('2023-01-01')],
            'price': [100]
        })

        config = ConfigLoader().get_config().data_quality
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['price'], config, issues_log
        )

        # Should handle single row gracefully (no outliers possible)
        assert len(processed_df) == 1
        assert outliers['price'] == 0

    def test_all_same_values(self):
        """Test outlier detection when all values are the same."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'price': [100] * 10  # All same values, IQR = 0
        })

        config = ConfigLoader().get_config().data_quality
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['price'], config, issues_log
        )

        # Should handle zero IQR gracefully
        assert len(processed_df) == 10
        assert outliers['price'] == 0

    def test_all_nan_values(self):
        """Test outlier detection with all NaN values."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'price': [np.nan] * 10
        })

        config = ConfigLoader().get_config().data_quality
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['price'], config, issues_log
        )

        # Should handle all NaN values gracefully
        assert len(processed_df) == 10
        assert outliers['price'] == 0

    def test_mixed_nan_and_valid_values(self):
        """Test outlier detection with mixed NaN and valid values."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'price': [10, np.nan, 12, 13, np.nan, 15, 16, 1000, 18, np.nan]
        })

        config = ConfigLoader().get_config().data_quality
        config.outlier_detection.handling_strategy = "flag"
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['price'], config, issues_log
        )

        # Should detect outliers only in valid values
        assert outliers['price'] > 0  # Should detect the 1000 outlier
        assert len(processed_df) == 10  # Should preserve all rows

    def test_insufficient_data_points(self):
        """Test outlier detection with insufficient data points."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=3, freq='1min'),
            'price': [10, 11, 12]
        })

        config = ConfigLoader().get_config().data_quality
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['price'], config, issues_log
        )

        # Should handle insufficient data points (< 4 required for meaningful IQR)
        assert len(processed_df) == 3
        assert outliers['price'] == 0


class TestColumnTypeHandling:
    """Test handling of different column types."""

    def test_datetime_column_skipping(self):
        """Test that datetime columns are skipped."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'price': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        })

        config = ConfigLoader().get_config().data_quality
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['ts', 'price'], config, issues_log
        )

        # Should skip datetime column and process price column
        assert outliers['ts'] == 0  # Datetime column should be skipped
        assert 'price' in outliers

    def test_missing_column_handling(self):
        """Test handling of columns that don't exist in DataFrame."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'price': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        })

        config = ConfigLoader().get_config().data_quality
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['price', 'nonexistent_column'], config, issues_log
        )

        # Should handle missing column gracefully
        assert outliers['nonexistent_column'] == 0
        assert 'price' in outliers

    def test_string_column_handling(self):
        """Test handling of string/categorical columns."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'symbol': ['AAPL'] * 10,
            'price': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        })

        config = ConfigLoader().get_config().data_quality
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['symbol', 'price'], config, issues_log
        )

        # Should handle string column appropriately
        assert 'price' in outliers
        # String column behavior depends on implementation


class TestConfigurationOptions:
    """Test different configuration options."""

    def test_different_iqr_multipliers(self):
        """Test outlier detection with different IQR multipliers."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'price': [10, 11, 12, 13, 14, 15, 16, 17, 18, 100]  # 100 is outlier
        })

        config = ConfigLoader().get_config().data_quality
        issues_log = []

        # Test with strict multiplier (1.0)
        config.outlier_detection.iqr_multiplier = 1.0
        processed_df1, outliers1 = OutlierDetector.detect_and_handle(
            df, ['price'], config, issues_log
        )

        # Test with lenient multiplier (3.0)
        config.outlier_detection.iqr_multiplier = 3.0
        processed_df2, outliers2 = OutlierDetector.detect_and_handle(
            df, ['price'], config, issues_log
        )

        # Strict multiplier should detect more outliers
        assert outliers1['price'] >= outliers2['price']

    def test_strategy_configuration_consistency(self):
        """Test that configuration strategy is applied consistently."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'price1': [10, 11, 12, 13, 1000, 15, 16, 17, 18, 19],
            'price2': [20, 21, 22, 23, 2000, 25, 26, 27, 28, 29]
        })

        config = ConfigLoader().get_config().data_quality
        config.outlier_detection.handling_strategy = "remove"
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['price1', 'price2'], config, issues_log
        )

        # Both columns should be processed with same strategy
        assert len(processed_df) < len(df)  # Should remove outlier rows
        assert 1000 not in processed_df['price1'].values
        assert 2000 not in processed_df['price2'].values


class TestPerformanceWithLargeData:
    """Test performance with large datasets."""

    def test_large_dataset_performance(self):
        """Test outlier detection performance with large dataset."""
        # Create large dataset
        np.random.seed(42)
        n_rows = 10000
        normal_data = np.random.normal(100, 10, n_rows - 10)
        outlier_data = np.random.normal(1000, 100, 10)  # 10 outliers
        price_data = np.concatenate([normal_data, outlier_data])
        np.random.shuffle(price_data)

        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=n_rows, freq='1min'),
            'price': price_data
        })

        config = ConfigLoader().get_config().data_quality
        config.outlier_detection.handling_strategy = "clip"
        issues_log = []

        # Should handle large dataset efficiently
        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['price'], config, issues_log
        )

        assert len(processed_df) == n_rows
        assert outliers['price'] > 0  # Should detect some outliers

    def test_many_columns_performance(self):
        """Test outlier detection performance with many columns."""
        n_rows = 1000
        n_cols = 20

        # Create data with many columns
        data = {}
        data['ts'] = pd.date_range('2023-01-01', periods=n_rows, freq='1min')

        np.random.seed(42)
        for i in range(n_cols):
            normal_data = np.random.normal(100, 10, n_rows - 5)
            outlier_data = np.random.normal(1000, 100, 5)
            col_data = np.concatenate([normal_data, outlier_data])
            np.random.shuffle(col_data)
            data[f'price_{i}'] = col_data

        df = pd.DataFrame(data)
        columns_to_check = [f'price_{i}' for i in range(n_cols)]

        config = ConfigLoader().get_config().data_quality
        config.outlier_detection.handling_strategy = "flag"
        issues_log = []

        # Should handle many columns efficiently
        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, columns_to_check, config, issues_log
        )

        assert len(processed_df) == n_rows
        assert len(outliers) == n_cols
        assert sum(outliers.values()) > 0  # Should detect some outliers


class TestIntegrationWithRealData:
    """Test integration with real-world data scenarios."""

    def test_real_csv_data_outlier_detection(self):
        """Test outlier detection on real CSV data."""
        df = ValidationTestFixtures.load_real_csv_data()

        # Add some artificial outliers to test detection
        df_with_outliers = df.copy()
        if len(df_with_outliers) > 10:
            df_with_outliers.loc[5, 'close'] = df_with_outliers['close'].mean() * 10
            df_with_outliers.loc[10, 'high'] = df_with_outliers['high'].mean() * 5

        config = ConfigLoader().get_config().data_quality
        config.outlier_detection.handling_strategy = "clip"
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df_with_outliers, ['open', 'high', 'low', 'close'], config, issues_log
        )

        # Should detect the artificial outliers we added
        assert len(processed_df) == len(df_with_outliers)
        total_outliers = sum(outliers.values())
        assert total_outliers > 0

    def test_index_data_outlier_characteristics(self):
        """Test outlier detection characteristics on index data."""
        df = ValidationTestFixtures.create_index_data(100)

        # Add some realistic index outliers (gap ups/downs)
        df.loc[20, 'open'] = df.loc[20, 'close'] * 1.05  # 5% gap up
        df.loc[50, 'low'] = df.loc[50, 'close'] * 0.95   # 5% gap down

        config = ConfigLoader().get_config().data_quality
        config.outlier_detection.handling_strategy = "flag"
        issues_log = []

        processed_df, outliers = OutlierDetector.detect_and_handle(
            df, ['open', 'high', 'low', 'close'], config, issues_log
        )

        # Should handle index data appropriately
        assert len(processed_df) == len(df)
        # May or may not detect outliers depending on volatility

    def test_equity_vs_futures_outlier_patterns(self):
        """Test different outlier patterns in equity vs futures data."""
        equity_df = ValidationTestFixtures.create_equity_data(100)
        futures_df = ValidationTestFixtures.create_futures_data(100)

        config = ConfigLoader().get_config().data_quality
        config.outlier_detection.handling_strategy = "flag"
        issues_log_equity = []
        issues_log_futures = []

        # Test equity outlier detection
        processed_equity, outliers_equity = OutlierDetector.detect_and_handle(
            equity_df, ['open', 'high', 'low', 'close', 'volume'], config, issues_log_equity
        )

        # Test futures outlier detection (includes OI)
        processed_futures, outliers_futures = OutlierDetector.detect_and_handle(
            futures_df, ['open', 'high', 'low', 'close', 'volume', 'oi'], config, issues_log_futures
        )

        # Both should process successfully
        assert len(processed_equity) == len(equity_df)
        assert len(processed_futures) == len(futures_df)

        # Futures should have OI outlier data
        assert 'oi' in outliers_futures
        assert 'oi' not in outliers_equity
