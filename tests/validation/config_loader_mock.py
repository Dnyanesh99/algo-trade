"""
Mock ConfigLoader for tests that try to use default configuration.
"""
import pytest
from unittest.mock import patch, MagicMock
from tests.validation.comprehensive_config_mock import create_full_app_config


def mock_config_loader():
    """Create a mock ConfigLoader that returns our comprehensive mock config."""
    mock_loader = MagicMock()
    mock_config = create_full_app_config()
    mock_loader.get_config.return_value = mock_config
    return mock_loader


@pytest.fixture(autouse=True)
def patch_config_loader():
    """Automatically patch ConfigLoader for all tests in validation module."""
    with patch('src.utils.config_loader.ConfigLoader', return_value=mock_config_loader()):
        yield