import asyncio
import pytest

from src.auth.token_manager import TokenManager
from src.utils.logger import LOGGER as logger

@pytest.mark.asyncio
async def test_token_manager_functionality():
    """
    Tests the functionality of the TokenManager.
    """
    logger.info("\n--- Starting TokenManager Functionality Test ---")

    token_manager = TokenManager()

    logger.info(f"Is token available initially? {token_manager.is_token_available()}")
    assert not token_manager.is_token_available()

    try:
        token_manager.set_access_token("test_access_token_123")
        logger.info(f"Is token available after setting? {token_manager.is_token_available()}")
        assert token_manager.is_token_available()
        logger.info(f"Retrieved token: {token_manager.get_access_token()}")
        assert token_manager.get_access_token() == "test_access_token_123"

        token_manager.clear_access_token()
        logger.info(f"Is token available after clearing? {token_manager.is_token_available()}")
        assert not token_manager.is_token_available()

        # Test with invalid token
        try:
            token_manager.set_access_token("")
            pytest.fail("ValueError was not raised for empty token.")
        except ValueError as e:
            logger.error(f"Caught expected error: {e}")
            assert "non-empty string" in str(e)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        pytest.fail(f"An unexpected error occurred: {e}")

    logger.info("--- TokenManager Functionality Test Completed ---")


if __name__ == "__main__":
    asyncio.run(test_token_manager_functionality())
