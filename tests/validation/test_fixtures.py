"""
Test fixtures for validation tests.
Contains sample data for different instrument types and edge cases.
"""

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.utils.config_loader import DataQualityConfig
from tests.validation.comprehensive_config_mock import create_comprehensive_mock_config


class ValidationTestFixtures:
    """Test fixtures for validation testing."""

    @staticmethod
    def load_real_csv_data() -> pd.DataFrame:
        """Load the real CSV data from the project."""
        csv_path = "/home/dnyanesh/ai-trading/algo-trade/src/validation/ohlcv_1min_202507191130.csv"
        # Read CSV and handle empty trailing columns
        df = pd.read_csv(csv_path)
        
        # Handle empty OI column (trailing commas)
        if 'oi' in df.columns and df['oi'].isna().all():
            df['oi'] = np.nan
            
        # Convert timestamp to datetime - handle timezone format
        try:
            df['ts'] = pd.to_datetime(df['ts'], format='%Y-%m-%d %H:%M:%S.%f %z')
        except ValueError:
            # Fallback to flexible parsing
            df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
        
        # Ensure OHLCV columns are numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df

    @staticmethod
    def create_index_data(num_rows: int = 100) -> pd.DataFrame:
        """Create index data with zero volume and empty OI (like NIFTY)."""
        base_df = ValidationTestFixtures.load_real_csv_data().head(num_rows).copy()
        # Ensure index characteristics
        base_df['volume'] = 0
        base_df['oi'] = None  # Empty OI for indices
        return base_df

    @staticmethod
    def create_equity_data(num_rows: int = 100) -> pd.DataFrame:
        """Create equity data with realistic volume and no OI."""
        base_df = ValidationTestFixtures.load_real_csv_data().head(num_rows).copy()

        # Simulate realistic equity volume (1000-50000 shares)
        np.random.seed(42)
        base_df['volume'] = np.random.randint(1000, 50000, size=len(base_df))
        base_df['oi'] = None  # No OI for equity

        return base_df

    @staticmethod
    def create_futures_data(num_rows: int = 100) -> pd.DataFrame:
        """Create futures data with volume and OI."""
        base_df = ValidationTestFixtures.load_real_csv_data().head(num_rows).copy()

        # Simulate realistic futures volume and OI
        np.random.seed(42)
        base_df['volume'] = np.random.randint(500, 10000, size=len(base_df))
        base_df['oi'] = np.random.randint(1000, 100000, size=len(base_df))

        return base_df

    @staticmethod
    def create_data_with_ohlc_violations() -> pd.DataFrame:
        """Create data with intentional OHLC violations."""
        df = ValidationTestFixtures.create_equity_data(10)

        # Create violations
        df.loc[0, 'high'] = df.loc[0, 'low'] - 1  # high < low
        df.loc[1, 'high'] = df.loc[1, 'open'] - 1  # high < open
        df.loc[2, 'low'] = df.loc[2, 'close'] + 1  # low > close

        return df

    @staticmethod
    def create_data_with_outliers() -> pd.DataFrame:
        """Create data with statistical outliers."""
        df = ValidationTestFixtures.create_equity_data(100)

        # Insert extreme outliers
        df.loc[10, 'close'] = df['close'].mean() + 10 * df['close'].std()
        df.loc[20, 'volume'] = df['volume'].mean() + 5 * df['volume'].std()
        df.loc[30, 'high'] = df['high'].mean() + 8 * df['high'].std()

        return df

    @staticmethod
    def create_data_with_time_gaps() -> pd.DataFrame:
        """Create data with significant time gaps."""
        df = ValidationTestFixtures.create_equity_data(20)

        # Create time gaps
        df.loc[5, 'ts'] = df.loc[4, 'ts'] + timedelta(hours=2)  # 2-hour gap
        df.loc[10, 'ts'] = df.loc[9, 'ts'] + timedelta(minutes=30)  # 30-min gap

        # Sort by timestamp
        return df.sort_values('ts').reset_index(drop=True)

    @staticmethod
    def create_data_with_duplicates() -> pd.DataFrame:
        """Create data with duplicate timestamps."""
        df = ValidationTestFixtures.create_equity_data(20)

        # Create duplicates
        df.loc[5, 'ts'] = df.loc[4, 'ts']
        df.loc[10, 'ts'] = df.loc[9, 'ts']

        return df

    @staticmethod
    def create_data_with_nan_values() -> pd.DataFrame:
        """Create data with NaN values in critical columns."""
        df = ValidationTestFixtures.create_equity_data(20)

        # Insert NaN values
        df.loc[3, 'open'] = np.nan
        df.loc[7, 'close'] = np.nan
        df.loc[12, 'volume'] = np.nan

        return df

    @staticmethod
    def create_data_with_negative_values() -> pd.DataFrame:
        """Create data with negative prices/volume."""
        df = ValidationTestFixtures.create_equity_data(20)

        # Insert negative values
        df.loc[2, 'open'] = -100.5
        df.loc[5, 'volume'] = -1000
        df.loc[8, 'close'] = -50.25

        return df

    @staticmethod
    def create_negative_values_data() -> pd.DataFrame:
        """Alias for create_data_with_negative_values for backward compatibility."""
        return ValidationTestFixtures.create_data_with_negative_values()

    @staticmethod
    def create_minimal_columns_data() -> pd.DataFrame:
        """Create data with only minimal required columns."""
        df = ValidationTestFixtures.create_equity_data(20)
        return df[['ts', 'close']].copy()

    @staticmethod
    def create_data_missing_timestamp() -> pd.DataFrame:
        """Create data missing timestamp column."""
        df = ValidationTestFixtures.create_equity_data(20)
        return df.drop(columns=['ts'])

    @staticmethod
    def create_unordered_timestamp_data() -> pd.DataFrame:
        """Create data with unordered timestamps."""
        df = ValidationTestFixtures.create_equity_data(20)

        # Shuffle timestamps to create disorder
        timestamps = df['ts'].sample(frac=1).reset_index(drop=True)
        df['ts'] = timestamps

        return df


@pytest.fixture
def config_index():
    """Configuration for index validation."""
    return create_comprehensive_mock_config(["ts", "open", "high", "low", "close"])


@pytest.fixture
def config_equity():
    """Configuration for equity validation."""
    return create_comprehensive_mock_config(["ts", "open", "high", "low", "close", "volume"])


@pytest.fixture
def config_futures():
    """Configuration for futures validation."""
    return create_comprehensive_mock_config(["ts", "open", "high", "low", "close", "volume", "oi"])


@pytest.fixture
def config_minimal():
    """Configuration with minimal required columns."""
    return create_comprehensive_mock_config(["ts", "close"])


@pytest.fixture
def sample_index_data() -> pd.DataFrame:
    """Sample index data fixture."""
    return ValidationTestFixtures.create_index_data()


@pytest.fixture
def sample_equity_data() -> pd.DataFrame:
    """Sample equity data fixture."""
    return ValidationTestFixtures.create_equity_data()


@pytest.fixture
def sample_futures_data() -> pd.DataFrame:
    """Sample futures data fixture."""
    return ValidationTestFixtures.create_futures_data()


@pytest.fixture
def data_with_violations() -> pd.DataFrame:
    """Data with OHLC violations fixture."""
    return ValidationTestFixtures.create_data_with_ohlc_violations()


@pytest.fixture
def data_with_outliers() -> pd.DataFrame:
    """Data with outliers fixture."""
    return ValidationTestFixtures.create_data_with_outliers()


@pytest.fixture
def data_with_time_gaps() -> pd.DataFrame:
    """Data with time gaps fixture."""
    return ValidationTestFixtures.create_data_with_time_gaps()


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Empty DataFrame fixture."""
    return pd.DataFrame()
