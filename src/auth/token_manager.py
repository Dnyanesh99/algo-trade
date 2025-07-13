import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from kiteconnect.exceptions import TokenException

from src.utils.logger import LOGGER as logger


class TokenManager:
    """
    Manages the in-memory storage and retrieval of the Zerodha KiteConnect access token.
    Implemented as a singleton to ensure a single source of truth for the token.
    Additionally, handles persistence of the token to a file.
    """

    _instance: Optional["TokenManager"] = None
    _lock: threading.Lock = threading.Lock()
    _access_token: Optional[str] = None
    _token_file_path: Path
    _last_validation_time: float = 0
    _validation_cache_duration: float = 300  # 5 minutes
    _is_token_valid: bool = False
    _auth_failure_count: int = 0
    _max_auth_failures: int = 3

    def __new__(cls) -> "TokenManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._access_token = None
                cls._instance._last_validation_time = 0
                cls._instance._is_token_valid = False
                cls._instance._auth_failure_count = 0

                # Get token file path from environment variable with default
                token_file_path = os.getenv("ACCESS_TOKEN_FILE_PATH", "config/access_token.json")
                cls._instance._token_file_path = Path(token_file_path)
                cls._instance._load_token_from_file()
                logger.info("TokenManager initialized.")
        return cls._instance

    def __init__(self) -> None:
        pass

    def _save_token_to_file(self, token: str) -> None:
        """
        Saves the access token to a file with secure permissions.
        """
        try:
            self._token_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._token_file_path, "w") as f:
                json.dump({"access_token": token}, f)
            # Set file permissions to 600 (read/write for owner only)
            os.chmod(self._token_file_path, 0o600)
            logger.info(f"Access token saved to {self._token_file_path} with secure permissions.")
        except OSError as e:
            logger.error(f"Failed to save access token to file {self._token_file_path}: {e}")

    def _load_token_from_file(self) -> None:
        """
        Loads the access token from a file.
        """
        if self._token_file_path.exists():
            try:
                with open(self._token_file_path) as f:
                    data = json.load(f)
                    token = data.get("access_token")
                    if token and token.strip():
                        self._access_token = token.strip()
                        logger.info(f"Access token loaded from {self._token_file_path}")
                    else:
                        self._access_token = None
                        logger.warning(f"Access token file {self._token_file_path} is empty or malformed.")
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to load access token from file {self._token_file_path}: {e}")
        else:
            logger.info(f"Access token file {self._token_file_path} does not exist. No token loaded.")

    def set_access_token(self, token: str) -> None:
        """
        Sets the access token and persists it to a file.
        """
        if not isinstance(token, str) or not token:
            raise ValueError("Access token must be a non-empty string.")
        with self._lock:
            self._access_token = token
            self._save_token_to_file(token)
            # Reset validation state when new token is set
            self._last_validation_time = 0
            self._is_token_valid = False
            self._auth_failure_count = 0
        logger.info("Access token set successfully.")

    def get_access_token(self) -> Optional[str]:
        """
        Retrieves the stored access token.
        """
        with self._lock:
            token = self._access_token
        if token is None:
            logger.warning("Attempted to retrieve access token, but no token is set.")
        return token

    def clear_access_token(self) -> None:
        """
        Clears the stored access token (e.g., on logout or token expiry).
        Also deletes the token file.
        """
        with self._lock:
            self._access_token = None
            self._last_validation_time = 0
            self._is_token_valid = False
            if self._token_file_path.exists():
                try:
                    self._token_file_path.unlink()
                    logger.info(f"Access token file {self._token_file_path} deleted.")
                except OSError as e:
                    logger.error(f"Failed to delete access token file {self._token_file_path}: {e}")
        logger.info("Access token cleared.")

    def is_token_available(self) -> bool:
        """
        Checks if an access token is currently available and valid (non-empty).
        """
        with self._lock:
            status = self._access_token is not None and self._access_token.strip() != ""
        logger.debug(f"Token availability check: {status}")
        return status

    async def validate_token_with_api(self, api_key: str, force: bool = False) -> bool:
        """
        Validates the current token by making a simple API call to Kite.
        Uses caching to avoid excessive API calls.

        Args:
            api_key: Kite API key
            force: Force validation even if recently validated

        Returns True if token is valid, False if invalid/expired.
        """
        if not self.is_token_available():
            logger.debug("No token available for validation")
            return False

        current_time = time.time()

        # Check cache unless forced
        if not force and (current_time - self._last_validation_time) < self._validation_cache_duration:
            logger.debug(f"Using cached token validation result: {self._is_token_valid}")
            return self._is_token_valid

        try:
            from kiteconnect import KiteConnect

            kite = KiteConnect(api_key=api_key, access_token=self.get_access_token())

            # Make a simple API call to validate the token
            import asyncio

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: kite.margins())

            # Update cache
            with self._lock:
                self._last_validation_time = current_time
                self._is_token_valid = True
                self._auth_failure_count = 0

            logger.info("Token validation successful")
            return True

        except TokenException as e:
            with self._lock:
                self._last_validation_time = current_time
                self._is_token_valid = False
                self._auth_failure_count += 1

            logger.warning(f"Token validation failed - token is invalid/expired: {e}")
            return False
        except Exception as e:
            logger.error(f"Token validation failed due to unexpected error: {e}")
            # Don't update cache on network/other errors
            return False

    def record_auth_failure(self) -> bool:
        """
        Record an authentication failure and return True if max failures reached.
        """
        with self._lock:
            self._auth_failure_count += 1
            self._is_token_valid = False
            failure_count = self._auth_failure_count

        logger.warning(f"Authentication failure recorded. Count: {failure_count}/{self._max_auth_failures}")
        return failure_count >= self._max_auth_failures

    def reset_auth_failures(self) -> None:
        """Reset authentication failure counter."""
        with self._lock:
            self._auth_failure_count = 0
        logger.info("Authentication failure counter reset")

    def should_trigger_reauth(self) -> bool:
        """Check if re-authentication should be triggered based on failure count."""
        with self._lock:
            return self._auth_failure_count >= self._max_auth_failures
