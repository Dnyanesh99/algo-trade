import time

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException, NetworkException, TokenException

from src.auth.token_manager import TokenManager
from src.metrics import metrics_registry
from src.utils.config_loader import BrokerConfig
from src.utils.logger import LOGGER as logger


class KiteAuthenticator:
    """
    Manages the authentication flow with Zerodha KiteConnect API.
    Handles initial login, request token exchange for access token.
    """

    def __init__(self, broker_config: BrokerConfig) -> None:
        self.broker_config = broker_config
        self.redirect_url = broker_config.redirect_url
        self.kite = KiteConnect(api_key=broker_config.api_key)
        self.token_manager = TokenManager()

    def set_redirect_url(self, new_url: str) -> None:
        """
        Sets a new redirect URL for the KiteConnect instance.
        This is useful when the auth server needs to dynamically assign a port.
        """
        self.redirect_url = new_url
        self.kite.redirect_url = new_url
        logger.info(f"KiteConnect redirect URL updated to: {new_url}")

    def generate_login_url(self) -> str:
        """
        Generates the login URL for the KiteConnect authentication.
        The user needs to visit this URL to grant access.
        """
        try:
            login_url: str = self.kite.login_url()
            logger.info(f"Generated KiteConnect login URL: {login_url}")
            return login_url
        except Exception as e:
            logger.error(f"Error generating login URL: {e}")
            raise

    def generate_session(self, request_token: str) -> str:
        """
        Exchanges the request token for an access token.
        """
        start_time = time.monotonic()
        try:
            data = self.kite.generate_session(request_token, self.broker_config.api_secret)
            access_token: str = data["access_token"]

            # Record successful authentication metrics
            duration = time.monotonic() - start_time
            metrics_registry.record_auth_request("generate_session", True, duration)

            logger.info("Successfully generated KiteConnect session and retrieved access token.")
            return access_token
        except (TokenException, NetworkException) as e:
            # Record failed authentication metrics
            duration = time.monotonic() - start_time
            metrics_registry.record_auth_request("generate_session", False, duration)

            logger.error(f"Error generating session: {e}")
            raise
        except KiteException as e:
            logger.error(f"KiteConnect error generating session: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating session: {e}")
            raise

    def authenticate(self, request_token: str) -> str:
        """
        Main authentication method. Takes a request token and returns the access token.
        """
        if not request_token:
            raise ValueError("Request token cannot be empty.")

        logger.info(f"Attempting to authenticate with request token: {request_token}")
        try:
            access_token = self.generate_session(request_token)
            self.token_manager.set_access_token(access_token)
            logger.info("Authentication successful and token stored in TokenManager.")
            return access_token
        except (TokenException, NetworkException, KiteException) as e:
            logger.critical(f"Authentication failed with KiteConnect error: {e}")
            raise
        except Exception as e:
            logger.critical(f"Authentication failed with unexpected error: {e}")
            raise
