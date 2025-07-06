from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException, NetworkException, TokenException

from src.auth.token_manager import TokenManager
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

# Load configuration
config = config_loader.get_config()

# No need for direct logging configuration here, it's handled by src.utils.logger


class KiteAuthenticator:
    """
    Manages the authentication flow with Zerodha KiteConnect API.
    Handles initial login, request token exchange for access token.
    """

    def __init__(self) -> None:
        self.redirect_url = config.broker.redirect_url
        self.kite = KiteConnect(api_key=config.broker.api_key)

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
        try:
            data = self.kite.generate_session(request_token, api_secret=config.broker.api_secret)
            access_token: str = data["access_token"]
            logger.info("Successfully generated KiteConnect session and retrieved access token.")
            return access_token
        except TokenException as e:
            logger.error(f"Token error generating session: {e}")
            raise
        except NetworkException as e:
            logger.error(f"Network error generating session: {e}")
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
            TokenManager().set_access_token(access_token)
            logger.info("Authentication successful and token stored in TokenManager.")
            return access_token
        except (TokenException, NetworkException, KiteException) as e:
            logger.critical(f"Authentication failed with KiteConnect error: {e}")
            raise
        except Exception as e:
            logger.critical(f"Authentication failed with unexpected error: {e}")
            raise
