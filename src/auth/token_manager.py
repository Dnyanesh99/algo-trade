import threading
from typing import Optional

from src.utils.logger import LOGGER as logger  # Centralized logger


class TokenManager:
    """
    Manages the in-memory storage and retrieval of the Zerodha KiteConnect access token.
    Implemented as a singleton to ensure a single source of truth for the token.
    """

    _instance: Optional["TokenManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "TokenManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                logger.info("TokenManager initialized.")
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_access_token"):
            self._access_token: Optional[str] = None

    def set_access_token(self, token: str) -> None:
        """
        Sets the access token.
        """
        if not isinstance(token, str) or not token:
            raise ValueError("Access token must be a non-empty string.")
        with self._lock:
            self._access_token = token
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
        """
        with self._lock:
            self._access_token = None
        logger.info("Access token cleared.")

    def is_token_available(self) -> bool:
        """
        Checks if an access token is currently available.
        """
        with self._lock:
            status = self._access_token is not None
        logger.debug(f"Token availability check: {status}")
        return status
