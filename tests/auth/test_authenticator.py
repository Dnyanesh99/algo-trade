import asyncio
from unittest.mock import patch

import pytest
from kiteconnect.exceptions import TokenException

from src.auth.authenticator import KiteAuthenticator
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

# Load configuration for tests
config = config_loader.get_config()

@pytest.fixture
def mock_kite_connect():
    """
    Fixture to mock KiteConnect for authenticator tests.
    """
    with patch('src.auth.authenticator.KiteConnect') as MockKiteConnect:
        mock_instance = MockKiteConnect.return_value
        yield mock_instance

@pytest.fixture
def mock_token_manager():
    """
    Fixture to mock TokenManager for authenticator tests.
    """
    with patch('src.auth.authenticator.TokenManager') as MockTokenManager:
        mock_instance = MockTokenManager.return_value
        yield mock_instance

@pytest.mark.asyncio
async def test_generate_login_url(mock_kite_connect):
    """
    Tests that generate_login_url calls KiteConnect.login_url.
    """
    logger.info("\n--- Starting test_generate_login_url ---")
    authenticator = KiteAuthenticator()
    mock_kite_connect.login_url.return_value = "http://mock.login.url"

    login_url = authenticator.generate_login_url()

    assert login_url == "http://mock.login.url"
    mock_kite_connect.login_url.assert_called_once()
    logger.info("test_generate_login_url completed successfully.")

@pytest.mark.asyncio
async def test_generate_session_success(mock_kite_connect, mock_token_manager):
    """
    Tests successful session generation.
    """
    logger.info("\n--- Starting test_generate_session_success ---")
    authenticator = KiteAuthenticator()
    mock_kite_connect.generate_session.return_value = {"access_token": "mock_access_token"}

    access_token = authenticator.generate_session("mock_request_token")

    assert access_token == "mock_access_token"
    mock_kite_connect.generate_session.assert_called_once_with("mock_request_token", api_secret=config.broker.api_secret)
    logger.info("test_generate_session_success completed successfully.")

@pytest.mark.asyncio
async def test_authenticate_success(mock_kite_connect, mock_token_manager):
    """
    Tests successful authentication flow.
    """
    logger.info("\n--- Starting test_authenticate_success ---")
    authenticator = KiteAuthenticator()
    mock_kite_connect.generate_session.return_value = {"access_token": "mock_access_token"}

    access_token = authenticator.authenticate("mock_request_token")

    assert access_token == "mock_access_token"
    mock_token_manager.set_access_token.assert_called_once_with("mock_access_token")
    logger.info("test_authenticate_success completed successfully.")

@pytest.mark.asyncio
async def test_authenticate_token_exception(mock_kite_connect, mock_token_manager):
    """
    Tests authentication failure due to TokenException.
    """
    logger.info("\n--- Starting test_authenticate_token_exception ---")
    authenticator = KiteAuthenticator()
    mock_kite_connect.generate_session.side_effect = TokenException("Invalid token")

    with pytest.raises(TokenException):
        authenticator.authenticate("mock_request_token")
    mock_token_manager.set_access_token.assert_not_called()
    logger.info("test_authenticate_token_exception completed successfully.")

@pytest.mark.asyncio
async def test_authenticate_value_error():
    """
    Tests authentication with empty request token.
    """
    logger.info("\n--- Starting test_authenticate_value_error ---")
    authenticator = KiteAuthenticator()

    with pytest.raises(ValueError, match="Request token cannot be empty."):
        authenticator.authenticate("")
    logger.info("test_authenticate_value_error completed successfully.")


# Example of running tests directly (usually done via `pytest` command)
if __name__ == "__main__":
    # To run these tests, you would typically use `pytest` from your terminal.
    # For direct execution, you might need to configure pytest or run individual async functions.
    # This block is primarily for demonstrating the test structure.
    logger.info("Running authenticator tests directly. Use `pytest` for proper test execution.")
    asyncio.run(test_generate_login_url(mock_kite_connect()))
    # Note: Running fixtures directly like this is not how pytest is designed.
    # This is just for illustrative purposes if someone wanted to run without pytest setup.
    # Proper pytest setup involves `pytest.main()` or running `pytest` from CLI.
