import json
import threading
from pathlib import Path
from typing import Optional

from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

config = config_loader.get_config()


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

    def __new__(cls) -> "TokenManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._access_token = None
                cls._instance._token_file_path = Path(config.broker.token_file_path)
                cls._instance._load_token_from_file()
                logger.info("TokenManager initialized.")
        return cls._instance

    def __init__(self) -> None:
        pass

    def _save_token_to_file(self, token: str) -> None:
        """
        Saves the access token to a file.
        """
        try:
            self._token_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._token_file_path, "w") as f:
                json.dump({"access_token": token}, f)
            logger.info(f"Access token saved to {self._token_file_path}")
        except IOError as e:
            logger.error(f"Failed to save access token to file {self._token_file_path}: {e}")

    def _load_token_from_file(self) -> None:
        """
        Loads the access token from a file.
        """
        if self._token_file_path.exists():
            try:
                with open(self._token_file_path, "r") as f:
                    data = json.load(f)
                    self._access_token = data.get("access_token")
                if self._access_token:
                    logger.info(f"Access token loaded from {self._token_file_path}")
                else:
                    logger.warning(f"Access token file {self._token_file_path} is empty or malformed.")
            except (IOError, json.JSONDecodeError) as e:
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
            if self._token_file_path.exists():
                try:
                    self._token_file_path.unlink()
                    logger.info(f"Access token file {self._token_file_path} deleted.")
                except IOError as e:
                    logger.error(f"Failed to delete access token file {self._token_file_path}: {e}")
        logger.info("Access token cleared.")

    def is_token_available(self) -> bool:
        """
        Checks if an access token is currently available.
        """
        with self._lock:
            status = self._access_token is not None
        logger.debug(f"Token availability check: {status}")
        return status
