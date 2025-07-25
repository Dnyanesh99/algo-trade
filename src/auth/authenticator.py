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

    Note: Zerodha Kite Connect deliberately does not provide refresh tokens.
    Access tokens are valid only for the trading day and expire around 07:00-07:30 AM IST.
    """

    def __init__(self, broker_config: BrokerConfig) -> None:
        if not broker_config.api_key or not broker_config.api_secret:
            logger.critical("CRITICAL: Broker API key or API secret is not configured.")
            raise ValueError("BrokerConfig is missing required 'api_key' or 'api_secret'. System cannot start.")

        self.broker_config = broker_config
        self.redirect_url = broker_config.redirect_url

        try:
            self.kite = KiteConnect(api_key=broker_config.api_key)
            logger.info("KiteConnect client initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize KiteConnect client: {e}", exc_info=True)
            raise RuntimeError("Could not initialize KiteConnect client.") from e

        self.token_manager = TokenManager()
        logger.info("KiteAuthenticator initialized.")

    def set_redirect_url(self, new_url: str) -> None:
        """
        Sets a new redirect URL for the KiteConnect instance.
        This is useful when the auth server needs to dynamically assign a port.
        """
        if not new_url or not new_url.startswith("http"):
            logger.error(f"Attempted to set an invalid redirect URL: '{new_url}'")
            raise ValueError("A valid, absolute redirect URL must be provided.")

        self.redirect_url = new_url
        self.kite.redirect_url = new_url
        logger.info(f"KiteConnect redirect URL successfully updated to: {new_url}")

    def generate_login_url(self) -> str:
        """
        Generates the login URL for the KiteConnect authentication.
        The user needs to visit this URL to grant access.
        """
        logger.info("Generating KiteConnect login URL.")
        try:
            login_url: str = self.kite.login_url()
            logger.info(f"Successfully generated KiteConnect login URL: {login_url}")
            return login_url
        except Exception as e:
            logger.critical(f"Failed to generate login URL: {e}", exc_info=True)
            raise RuntimeError("Could not generate KiteConnect login URL.") from e

    def authenticate(self, request_token: str) -> str:
        """
        Main authentication method. Exchanges a request token for an access token.
        This is a critical step in the OAuth flow.

        Args:
            request_token: The token received from the broker after user login.

        Returns:
            The access token.

        Raises:
            ValueError: If the request_token is empty or invalid.
            RuntimeError: If the authentication process fails for any reason.
        """
        if not request_token or not isinstance(request_token, str):
            logger.error("Authenticate called with invalid or empty request_token.")
            raise ValueError("Request token must be a non-empty string.")

        logger.info(
            f"Attempting to authenticate and generate session with request token ending in '...{request_token[-4:]}'"
        )
        start_time = time.monotonic()

        try:
            data = self.kite.generate_session(request_token, self.broker_config.api_secret)
            duration = time.monotonic() - start_time

            access_token = data.get("access_token")

            if not access_token:
                logger.critical(f"Authentication response from broker is missing access_token. Response: {data}")
                metrics_registry.record_auth_request("generate_session", False, duration)
                raise RuntimeError("Invalid token data received from broker.")

            # Note: Zerodha deliberately returns an empty string for refresh_token
            # We don't need to check for it or handle it

            metrics_registry.record_auth_request("generate_session", True, duration)
            logger.info(f"Successfully generated session in {duration:.2f} seconds. Received access token.")
            return str(access_token)

        except (TokenException, NetworkException, KiteException) as e:
            duration = time.monotonic() - start_time
            metrics_registry.record_auth_request("generate_session", False, duration)
            logger.critical(f"Authentication failed due to a KiteConnect API error: {e}", exc_info=True)
            raise RuntimeError("Failed to generate session due to broker API error.") from e

        except Exception as e:
            duration = time.monotonic() - start_time
            metrics_registry.record_auth_request("generate_session", False, duration)
            logger.critical(f"An unexpected critical error occurred during authentication: {e}", exc_info=True)
            raise RuntimeError("An unexpected error occurred during session generation.") from e
