"""
Configuration for validation tests.
"""
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import config mocks
from tests.validation.comprehensive_config_mock import create_full_app_config
from tests.validation.config_loader_mock import mock_config_loader

# Import fixtures
from tests.validation.test_fixtures import (
    config_index,
    config_equity,
    config_futures,
    config_minimal,
    sample_index_data,
    sample_equity_data,
    sample_futures_data,
    data_with_violations,
    data_with_outliers,
    data_with_time_gaps,
    empty_dataframe,
)

@pytest.fixture(autouse=True)
def patch_config_loader():
    """Automatically patch ConfigLoader for all tests in validation module."""
    # Create the mock instance
    mock_instance = mock_config_loader()
    
    # Patch both the class and its instantiation
    with patch('src.validation.candle_validator.ConfigLoader', return_value=mock_instance) as mock_class:
        yield mock_class

# Make fixtures available
__all__ = [
    "config_index",
    "config_equity", 
    "config_futures",
    "config_minimal",
    "sample_index_data",
    "sample_equity_data",
    "sample_futures_data",
    "data_with_violations",
    "data_with_outliers",
    "data_with_time_gaps",
    "empty_dataframe",
    "patch_config_loader",
]