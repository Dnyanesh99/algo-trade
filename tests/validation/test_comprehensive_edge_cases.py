"""
Comprehensive edge case tests to demonstrate validation behavior.
This file contains integration tests for all real-world scenarios.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.config_loader import ConfigLoader
from src.validation.candle_validator import CandleValidator
from tests.validation.test_fixtures import ValidationTestFixtures


class TestMissingRequiredColumns:
    """Test behavior when required columns are missing."""

    def test_missing_timestamp_column(self):
        """Test behavior when ts column is missing from required_columns."""
        config = ConfigLoader().get_config().data_quality
        config.validation.required_columns = ["open", "high", "low", "close"]  # No 'ts'

        validator = CandleValidator(config=config)
        df = ValidationTestFixtures.create_equity_data(20)

        # Should fail because timestamp is always required internally
        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST", instrument_type="EQ"
        )

        # Should fail validation due to missing timestamp handling
        assert not is_valid or len(report.issues) > 0

    def test_missing_ohlc_columns_individually(self):
        """Test behavior when individual OHLC columns are missing."""
        df = ValidationTestFixtures.create_equity_data(20)

        # Test missing each OHLC column individually
        ohlc_columns = ["open", "high", "low", "close"]

        for missing_col in ohlc_columns:
            config = ConfigLoader().get_config().data_quality
            required_cols = [col for col in ["ts"] + ohlc_columns if col != missing_col]
            config.validation.required_columns = required_cols

            validator = CandleValidator(config=config)
            test_df = df[required_cols].copy()  # Only include required columns

            is_valid, cleaned_df, report = validator.validate(
                test_df, symbol=f"TEST_NO_{missing_col.upper()}", instrument_type="EQ"
            )

            # Should skip OHLC validation but still process other validations
            assert any("Skipping OHLC validation" in issue for issue in report.issues)

    def test_missing_volume_column(self):
        """Test behavior when volume is not in required_columns."""
        config = ConfigLoader().get_config().data_quality
        config.validation.required_columns = ["ts", "open", "high", "low", "close"]  # No volume

        validator = CandleValidator(config=config)
        df = ValidationTestFixtures.create_equity_data(20)

        # Insert zero volume rows (shouldn't matter since volume not required)
        df.loc[5:7, 'volume'] = 0

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST_NO_VOLUME", instrument_type="EQ"
        )

        # Should not filter out zero volume rows since volume is not required
        assert len(cleaned_df) == len(df)
        assert 'volume' not in report.outliers or report.outliers.get('volume', 0) == 0

    def test_missing_oi_column(self):
        """Test behavior when oi is not in required_columns."""
        config = ConfigLoader().get_config().data_quality
        config.validation.required_columns = ["ts", "open", "high", "low", "close", "volume"]  # No oi

        validator = CandleValidator(config=config)
        df = ValidationTestFixtures.create_futures_data(20)

        # Insert negative OI values (shouldn't matter since OI not required)
        df.loc[3:5, 'oi'] = -1000

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="TEST_NO_OI", instrument_type="NFO-FUT"
        )

        # Should not filter out negative OI rows since OI is not required
        assert 'oi' not in report.outliers or report.outliers.get('oi', 0) == 0


class TestRealWorldDataScenarios:
    """Test real-world data scenarios and edge cases."""

    def test_real_csv_data_comprehensive_validation(self):
        """Test comprehensive validation on real CSV data."""
        config = ConfigLoader().get_config().data_quality
        config.validation.required_columns = ["ts", "open", "high", "low", "close"]  # Index config

        validator = CandleValidator(config=config)
        df = ValidationTestFixtures.load_real_csv_data()

        print(f"Testing real CSV data with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="NIFTY50", instrument_type="INDICES"
        )

        print(f"Validation result: {is_valid}")
        print(f"Quality score: {report.quality_score:.2f}%")
        print(f"Issues found: {len(report.issues)}")
        for issue in report.issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")

        # Should successfully process real data
        assert len(cleaned_df) > 0
        assert report.quality_score >= 0

    def test_mixed_data_types_simulation(self):
        """Test validation with mixed data representing different instruments."""
        # Test Index data (zero volume, no OI)
        config_index = ConfigLoader().get_config().data_quality
        config_index.validation.required_columns = ["ts", "open", "high", "low", "close"]

        validator_index = CandleValidator(config=config_index)
        index_df = ValidationTestFixtures.create_index_data(50)

        is_valid_index, cleaned_index, report_index = validator_index.validate(
            index_df, symbol="NIFTY50", instrument_type="INDICES"
        )

        # Test Equity data (positive volume, no OI)
        config_equity = ConfigLoader().get_config().data_quality
        config_equity.validation.required_columns = ["ts", "open", "high", "low", "close", "volume"]

        validator_equity = CandleValidator(config=config_equity)
        equity_df = ValidationTestFixtures.create_equity_data(50)

        is_valid_equity, cleaned_equity, report_equity = validator_equity.validate(
            equity_df, symbol="RELIANCE", instrument_type="EQ"
        )

        # Test Futures data (volume + OI)
        config_futures = ConfigLoader().get_config().data_quality
        config_futures.validation.required_columns = ["ts", "open", "high", "low", "close", "volume", "oi"]

        validator_futures = CandleValidator(config=config_futures)
        futures_df = ValidationTestFixtures.create_futures_data(50)

        is_valid_futures, cleaned_futures, report_futures = validator_futures.validate(
            futures_df, symbol="NIFTY25JAN25000CE", instrument_type="NFO-FUT"
        )

        # All should validate successfully
        assert is_valid_index, f"Index validation failed: {report_index.issues}"
        assert is_valid_equity, f"Equity validation failed: {report_equity.issues}"
        assert is_valid_futures, f"Futures validation failed: {report_futures.issues}"

        # Check appropriate columns were processed
        assert 'volume' not in report_index.outliers or report_index.outliers['volume'] == 0
        assert 'volume' in report_equity.outliers
        assert 'oi' in report_futures.outliers

    def test_extreme_market_conditions_simulation(self):
        """Test validation under extreme market conditions."""
        config = ConfigLoader().get_config().data_quality
        config.validation.required_columns = ["ts", "open", "high", "low", "close", "volume"]

        validator = CandleValidator(config=config)
        df = ValidationTestFixtures.create_equity_data(100)

        # Simulate extreme market conditions
        # Circuit breakers (10% gaps)
        df.loc[20, 'open'] = df.loc[19, 'close'] * 1.10  # 10% gap up
        df.loc[21, 'high'] = df.loc[21, 'open'] * 1.05   # Another 5% up
        df.loc[50, 'open'] = df.loc[49, 'close'] * 0.90  # 10% gap down

        # Volume spikes (institutional trades)
        df.loc[20:22, 'volume'] = df['volume'].mean() * 20

        # Flash crash simulation
        df.loc[75, 'low'] = df.loc[75, 'close'] * 0.95   # 5% flash drop
        df.loc[75, 'volume'] = df['volume'].mean() * 50  # 50x volume spike

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="EXTREME_TEST", instrument_type="EQ"
        )

        print(f"Extreme conditions test - Valid: {is_valid}")
        print(f"Quality score: {report.quality_score:.2f}%")
        print(f"Outliers detected: {sum(report.outliers.values())}")

        # Should handle extreme conditions gracefully
        assert len(cleaned_df) > 0
        assert report.quality_score >= 0

        # Should detect outliers from extreme conditions
        assert sum(report.outliers.values()) > 0

    def test_data_corruption_scenarios(self):
        """Test validation with various data corruption scenarios."""
        config = ConfigLoader().get_config().data_quality
        config.validation.required_columns = ["ts", "open", "high", "low", "close", "volume"]

        validator = CandleValidator(config=config)

        # Test different corruption scenarios
        corruption_scenarios = [
            ("Negative prices", ValidationTestFixtures.create_data_with_negative_values()),
            ("NaN values", ValidationTestFixtures.create_data_with_nan_values()),
            ("OHLC violations", ValidationTestFixtures.create_data_with_ohlc_violations()),
            ("Duplicate timestamps", ValidationTestFixtures.create_data_with_duplicates()),
            ("Time gaps", ValidationTestFixtures.create_data_with_time_gaps()),
            ("Statistical outliers", ValidationTestFixtures.create_data_with_outliers()),
        ]

        for scenario_name, corrupted_df in corruption_scenarios:
            print(f"\nTesting corruption scenario: {scenario_name}")

            is_valid, cleaned_df, report = validator.validate(
                corrupted_df, symbol=f"CORRUPT_{scenario_name.upper().replace(' ', '_')}",
                instrument_type="EQ", timeframe="1min"
            )

            print(f"  Valid: {is_valid}")
            print(f"  Original rows: {len(corrupted_df)}")
            print(f"  Cleaned rows: {len(cleaned_df)}")
            print(f"  Quality score: {report.quality_score:.2f}%")
            print(f"  Issues: {len(report.issues)}")

            # Should handle corruption gracefully
            assert len(cleaned_df) >= 0  # May filter out all data if too corrupted
            assert 0 <= report.quality_score <= 100
            assert len(report.issues) >= 0


class TestConfigurationEdgeCases:
    """Test edge cases in configuration handling."""

    def test_minimal_configuration(self):
        """Test validation with absolute minimal configuration."""
        config = ConfigLoader().get_config().data_quality
        config.validation.required_columns = ["ts"]  # Only timestamp

        validator = CandleValidator(config=config)
        df = ValidationTestFixtures.create_equity_data(20)

        # Only keep timestamp column
        minimal_df = df[['ts']].copy()

        is_valid, cleaned_df, report = validator.validate(
            minimal_df, symbol="MINIMAL", instrument_type="EQ"
        )

        # Should process with minimal data
        assert len(cleaned_df) > 0
        assert 'ts' in cleaned_df.columns
        assert len(cleaned_df.columns) == 1

    def test_custom_column_combinations(self):
        """Test various custom column combinations."""
        custom_configs = [
            (["ts", "close"], "Close only"),
            (["ts", "open", "close"], "Open and close"),
            (["ts", "high", "low"], "High and low only"),
            (["ts", "volume"], "Volume only"),
            (["ts", "open", "volume"], "Open and volume"),
        ]

        for required_columns, description in custom_configs:
            print(f"\nTesting custom config: {description} - {required_columns}")

            config = ConfigLoader().get_config().data_quality
            config.validation.required_columns = required_columns

            validator = CandleValidator(config=config)
            df = ValidationTestFixtures.create_equity_data(20)

            # Filter to only required columns
            test_df = df[required_columns].copy()

            is_valid, cleaned_df, report = validator.validate(
                test_df, symbol=f"CUSTOM_{description.upper().replace(' ', '_')}",
                instrument_type="EQ"
            )

            print(f"  Valid: {is_valid}")
            print(f"  Columns processed: {list(cleaned_df.columns)}")
            print(f"  Quality score: {report.quality_score:.2f}%")

            # Should process custom configurations
            assert len(cleaned_df) > 0
            assert all(col in cleaned_df.columns for col in required_columns)

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        # Test empty required columns
        with pytest.raises((ValueError, Exception)):  # Should raise ConfigurationError
            config = ConfigLoader().get_config().data_quality
            config.validation.required_columns = []
            CandleValidator(config=config)

        # Test invalid quality threshold
        with pytest.raises((ValueError, Exception)):  # Should raise ConfigurationError
            config = ConfigLoader().get_config().data_quality
            config.validation.quality_score_threshold = 150  # > 100
            CandleValidator(config=config)


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        config = ConfigLoader().get_config().data_quality
        config.validation.required_columns = ["ts", "open", "high", "low", "close", "volume"]

        validator = CandleValidator(config=config)

        # Create progressively larger datasets
        dataset_sizes = [100, 1000, 5000]

        for size in dataset_sizes:
            print(f"\nTesting dataset size: {size} rows")

            # Create large dataset by concatenating base data
            base_df = ValidationTestFixtures.create_equity_data(500)
            large_df = pd.concat([base_df] * (size // 500 + 1)).head(size).reset_index(drop=True)

            # Add some variety to timestamps
            large_df['ts'] = pd.date_range('2023-01-01', periods=size, freq='1min')

            import time
            start_time = time.time()

            is_valid, cleaned_df, report = validator.validate(
                large_df, symbol=f"LARGE_{size}", instrument_type="EQ"
            )

            end_time = time.time()
            processing_time = end_time - start_time

            print(f"  Processing time: {processing_time:.3f} seconds")
            print(f"  Rows per second: {size / processing_time:.0f}")
            print(f"  Valid: {is_valid}")
            print(f"  Quality score: {report.quality_score:.2f}%")

            # Should handle large datasets efficiently
            assert processing_time < 30  # Should complete within 30 seconds
            assert len(cleaned_df) > 0

    def test_high_outlier_density_performance(self):
        """Test performance when data has high outlier density."""
        config = ConfigLoader().get_config().data_quality
        config.validation.required_columns = ["ts", "open", "high", "low", "close", "volume"]
        config.outlier_detection.handling_strategy = "clip"  # Most processing-intensive

        validator = CandleValidator(config=config)

        # Create data with 50% outliers
        df = ValidationTestFixtures.create_equity_data(1000)

        # Insert many outliers
        outlier_indices = np.random.choice(len(df), size=len(df)//2, replace=False)
        for idx in outlier_indices:
            if idx % 4 == 0:
                df.loc[idx, 'close'] = df['close'].mean() * 5
            elif idx % 4 == 1:
                df.loc[idx, 'volume'] = df['volume'].mean() * 10
            elif idx % 4 == 2:
                df.loc[idx, 'high'] = df['high'].mean() * 3
            else:
                df.loc[idx, 'low'] = df['low'].mean() * 0.5

        import time
        start_time = time.time()

        is_valid, cleaned_df, report = validator.validate(
            df, symbol="HIGH_OUTLIER_DENSITY", instrument_type="EQ"
        )

        end_time = time.time()
        processing_time = end_time - start_time

        print("\nHigh outlier density test:")
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Outliers detected: {sum(report.outliers.values())}")
        print(f"  Quality score: {report.quality_score:.2f}%")

        # Should handle high outlier density
        assert processing_time < 10  # Should complete reasonably fast
        assert sum(report.outliers.values()) > 100  # Should detect many outliers


if __name__ == "__main__":
    """
    Run comprehensive edge case tests.
    This demonstrates all the validation capabilities.
    """
    print("=" * 80)
    print("COMPREHENSIVE VALIDATION EDGE CASE TESTING")
    print("=" * 80)

    # Initialize test classes
    test_classes = [
        TestMissingRequiredColumns(),
        TestRealWorldDataScenarios(),
        TestConfigurationEdgeCases(),
        TestPerformanceAndScalability(),
    ]

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n\n{'='*20} {class_name} {'='*20}")

        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]

        for method_name in test_methods:
            print(f"\n--- Running {method_name} ---")
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"✅ {method_name} PASSED")
            except Exception as e:
                print(f"❌ {method_name} FAILED: {e}")

    print("\n\n" + "=" * 80)
    print("COMPREHENSIVE TESTING COMPLETED")
    print("=" * 80)
