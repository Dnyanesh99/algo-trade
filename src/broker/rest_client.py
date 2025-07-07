import asyncio
from typing import Any, Optional

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException, NetworkException, TokenException

from src.auth.token_manager import TokenManager
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger  # Centralized logger
from src.utils.rate_limiter import RateLimiter

# Load configuration
config = config_loader.get_config()


class KiteRESTClient:
    """
    Handles all REST-like HTTP API calls to the Zerodha KiteConnect API.
    Ensures requests are signed and rate-limited.
    """

    def __init__(self) -> None:
        self.api_key = config.broker.api_key if config.broker and config.broker.api_key else ""
        self.token_manager = TokenManager()
        self.kite = self._initialize_kite_client()
        self.historical_data_rate_limiter = RateLimiter("historical_data")
        self.general_api_rate_limiter = RateLimiter("general_api")

    def _initialize_kite_client(self) -> KiteConnect:
        """
        Initializes the KiteConnect client with the API key and access token.
        The access token is fetched from the TokenManager.
        """
        access_token = self.token_manager.get_access_token()
        if not access_token:
            logger.warning("Access token not available. KiteConnect client initialized without token.")
            logger.warning("API calls requiring authentication will fail until token is set.")
            return KiteConnect(api_key=self.api_key)
        logger.info("KiteConnect client initialized with access token.")
        return KiteConnect(api_key=self.api_key, access_token=access_token)

    def _update_access_token(self) -> None:
        """
        Updates the access token for the KiteConnect client if it has changed.
        This is crucial for long-running applications where the token might be refreshed.
        """
        current_token = self.token_manager.get_access_token()
        if current_token and self.kite.access_token != current_token:
            self.kite.set_access_token(current_token)
            logger.info("KiteConnect client access token updated.")

    async def get_historical_data(
        self,
        instrument_token: int,
        from_date: str,
        to_date: str,
        interval: str,
        continuous: bool = False,
    ) -> Optional[list[dict[str, Any]]]:
        """
        Fetches historical OHLCV data for a given instrument.
        Applies rate limiting before making the API call.

        Args:
            instrument_token (int): Instrument token of the instrument.
            from_date (str): Start date (yyyy-mm-dd).
            to_date (str): End date (yyyy-mm-dd).
            interval (str): Candle interval (e.g., "minute", "5minute", "15minute", "day").
            continuous (bool, optional): Whether to fetch continuous data for futures. Defaults to False.

        Returns:
            Optional[list[Dict[str, Any]]]: List of historical data candles, or None if an error occurs.
        """
        self._update_access_token()
        try:
            async with self.historical_data_rate_limiter:
                logger.info(
                    f"Fetching historical data for {instrument_token} from {from_date} to {to_date} ({interval} interval)"
                )
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.kite.historical_data(
                        instrument_token,
                        from_date,
                        to_date,
                        interval,
                        continuous=continuous,
                    ),
                )
        except TokenException as e:
            logger.error(f"Authentication token error fetching historical data for {instrument_token}: {e}")
            # Consider triggering re-authentication flow or critical alert
            return None
        except NetworkException as e:
            logger.error(f"Network error fetching historical data for {instrument_token}: {e}")
            return None
        except KiteException as e:
            logger.error(f"KiteConnect API error fetching historical data for {instrument_token}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching historical data for {instrument_token}: {e}")
            return None

    async def get_instruments(self, exchange: Optional[str] = None) -> Optional[list[dict[str, Any]]]:
        """
        Fetches the list of instruments.
        Applies rate limiting before making the API call.

        Args:
            exchange (str, optional): Exchange name (e.g., "NSE", "BSE"). Defaults to None (all instruments).

        Returns:
            Optional[list[Dict[str, Any]]]: List of instrument details, or None if an error occurs.
        """
        self._update_access_token()
        try:
            async with self.general_api_rate_limiter:  # Using general API limiter for instruments
                logger.info(f"Fetching instruments for exchange: {exchange if exchange else 'All'}")
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: self.kite.instruments(exchange=exchange))
        except TokenException as e:
            logger.error(f"Authentication token error fetching instruments: {e}")
            # Consider triggering re-authentication flow or critical alert
            return None
        except NetworkException as e:
            logger.error(f"Network error fetching instruments: {e}")
            return None
        except KiteException as e:
            logger.error(f"KiteConnect API error fetching instruments: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching instruments: {e}")
            return None

    # Add other REST API methods as needed (e.g., get_quotes, place_order, etc.)
    # Remember to apply appropriate rate limiting for each endpoint.
