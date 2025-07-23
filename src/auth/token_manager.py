import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from kiteconnect.exceptions import NetworkException, TokenException

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
    _refresh_token: Optional[str] = None
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
                cls._instance._refresh_token = None
                cls._instance._last_validation_time = 0
                cls._instance._is_token_valid = False
                cls._instance._auth_failure_count = 0

                token_file_path_str = os.getenv("ACCESS_TOKEN_FILE_PATH")
                if not token_file_path_str:
                    logger.error("ACCESS_TOKEN_FILE_PATH environment variable not set.")
                    raise ValueError("ACCESS_TOKEN_FILE_PATH is not set in the environment.")

                cls._instance._token_file_path = Path(token_file_path_str)
                cls._instance._load_tokens_from_file()
                logger.info(f"TokenManager initialized. Token file path: '{cls._instance._token_file_path}'")
        return cls._instance

    def __init__(self) -> None:
        pass

    def _save_tokens_to_file(self, access_token: str, refresh_token: str) -> None:
        """Saves access and refresh tokens to a file."""
        if not access_token:
            logger.error("Attempted to save empty access token. Aborting file save.")
            return
        try:
            self._token_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._token_file_path, "w") as f:
                json.dump({"access_token": access_token, "refresh_token": refresh_token}, f)
            os.chmod(self._token_file_path, 0o600)
            logger.info(f"Tokens securely saved to {self._token_file_path}")
        except OSError as e:
            logger.critical(f"CRITICAL: Failed to save tokens to file '{self._token_file_path}': {e}", exc_info=True)
            raise RuntimeError(f"Could not persist tokens to {self._token_file_path}") from e

    def _load_tokens_from_file(self) -> None:
        """Loads access and refresh tokens from a file."""
        if self._token_file_path.exists():
            try:
                logger.debug(f"Attempting to load tokens from '{self._token_file_path}'")
                with open(self._token_file_path) as f:
                    data = json.load(f)
                    self._access_token = data.get("access_token")
                    self._refresh_token = data.get("refresh_token")
                    if self._access_token:
                        if self._refresh_token:
                            logger.info(f"Access and refresh tokens successfully loaded from {self._token_file_path}")
                        else:
                            logger.info(f"Access token successfully loaded from {self._token_file_path} (no refresh token)")
                    else:
                        logger.warning(
                            f"Token file '{self._token_file_path}' is missing 'access_token'."
                        )
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to read or parse token file '{self._token_file_path}': {e}", exc_info=True)
                # Corrupt or unreadable file, treat as if no token is available
                self._access_token = None
                self._refresh_token = None
        else:
            logger.info(f"Token file '{self._token_file_path}' does not exist. No tokens loaded.")

    def set_tokens(self, access_token: str, refresh_token: str) -> None:
        """Sets the access and refresh tokens, ensuring access token is not empty."""
        if not access_token:
            logger.error("Attempted to set empty access token.")
            raise ValueError("Access token cannot be empty.")
        
        if not refresh_token:
            logger.warning("Refresh token is empty. Only access token will be stored.")

        with self._lock:
            self._access_token = access_token
            self._refresh_token = refresh_token
            logger.info("In-memory tokens updated. Persisting to file.")
            self._save_tokens_to_file(access_token, refresh_token)
            self._last_validation_time = 0  # Invalidate cache
            self._is_token_valid = False
            self.reset_auth_failures()  # Reset failure count on new tokens
        if refresh_token:
            logger.info("Access and refresh tokens set successfully.")
        else:
            logger.info("Access token set successfully (no refresh token provided).")

    def get_access_token(self) -> Optional[str]:
        """Retrieves the stored access token."""
        with self._lock:
            return self._access_token

    def get_refresh_token(self) -> Optional[str]:
        """Retrieves the stored refresh token."""
        with self._lock:
            return self._refresh_token

    def clear_tokens(self) -> None:
        """Clears tokens and deletes the token file."""
        with self._lock:
            self._access_token = None
            self._refresh_token = None
            self._last_validation_time = 0
            self._is_token_valid = False
            if self._token_file_path.exists():
                try:
                    self._token_file_path.unlink()
                    logger.info(f"Token file {self._token_file_path} deleted.")
                except OSError as e:
                    logger.error(f"Failed to delete token file: {e}")
        logger.info("All tokens cleared.")

    async def refresh_access_token(self, api_key: str, api_secret: str) -> bool:
        """Refreshes the access token using the refresh token."""
        refresh_token = self.get_refresh_token()
        if not refresh_token:
            logger.error("No refresh token available to refresh access token.")
            return False

        logger.info("Attempting to refresh access token...")
        try:
            import asyncio

            from kiteconnect import KiteConnect

            kite = KiteConnect(api_key=api_key)
            # The generate_session method is used for both initial generation and refreshing
            data = await asyncio.to_thread(kite.generate_session, refresh_token, api_secret)
            new_access_token = data["access_token"]
            new_refresh_token = data.get(
                "refresh_token", refresh_token
            )  # Use old refresh token if new one isn't provided
            self.set_tokens(new_access_token, new_refresh_token)
            logger.info("Access token refreshed successfully.")
            return True
        except (TokenException, NetworkException) as e:
            logger.error(f"Failed to refresh access token: {e}. The session may be expired.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during token refresh: {e}")
            return False

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
