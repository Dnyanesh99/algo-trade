import asyncio
import time
from typing import Any, Optional

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException, NetworkException, TokenException

from src.auth.token_manager import TokenManager
from src.metrics import metrics_registry
from src.utils.config_loader import AppConfig
from src.utils.logger import LOGGER as logger
from src.utils.rate_limiter import RateLimiter, get_rate_limiter


class KiteRESTClient:
    """
    Handles all REST-like HTTP API calls to the Zerodha KiteConnect API.
    Ensures requests are signed and rate-limited.
    """

    def __init__(self, token_manager: TokenManager, api_key: str) -> None:
        self.api_key = api_key
        self.token_manager = token_manager
        self.kite: Optional[KiteConnect] = None
        self.historical_data_rate_limiter: Optional[RateLimiter] = None
        self.general_api_rate_limiter: Optional[RateLimiter] = None

    async def initialize(self, config: Optional[AppConfig] = None) -> None:
        """
        Initializes the KiteRESTClient, including the KiteConnect client and rate limiters.
        """
        self.kite = self._initialize_kite_client()
        self.historical_data_rate_limiter = await get_rate_limiter("historical_data", config)
        self.general_api_rate_limiter = await get_rate_limiter("general_api", config)

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
        if not self.kite:
            raise RuntimeError("KiteRESTClient not initialized. Call initialize() first.")
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
        """
        if not self.kite or not self.historical_data_rate_limiter:
            raise RuntimeError("KiteRESTClient not initialized. Call initialize() first.")

        self._update_access_token()
        start_time = time.monotonic()
        try:
            async with self.historical_data_rate_limiter:
                logger.info(
                    f"Fetching historical data for {instrument_token} from {from_date} to {to_date} ({interval} interval)"
                )
                loop = asyncio.get_running_loop()
                if self.kite is None:
                    raise RuntimeError("Kite client is not initialized.")
                kite = self.kite  # Assign to local variable for lambda
                result: Optional[list[dict[str, Any]]] = await loop.run_in_executor(
                    None,
                    lambda: kite.historical_data(
                        instrument_token,
                        from_date,
                        to_date,
                        interval,
                        continuous=continuous,
                    ),
                )

                duration = time.monotonic() - start_time
                metrics_registry.record_broker_api_request("historical_data", True, duration)

                return result

        except (TokenException, NetworkException, KiteException) as e:
            duration = time.monotonic() - start_time
            metrics_registry.record_broker_api_request("historical_data", False, duration)
            logger.error(f"Error fetching historical data for {instrument_token}: {e}")
            return None
        except Exception as e:
            duration = time.monotonic() - start_time
            metrics_registry.record_broker_api_request("historical_data", False, duration)
            logger.error(f"Unexpected error fetching historical data for {instrument_token}: {e}")
            return None

    async def get_instruments(self, exchange: Optional[str] = None) -> Optional[list[dict[str, Any]]]:
        """
        Fetches the list of instruments.
        Applies rate limiting before making the API call.
        """
        if not self.kite or not self.general_api_rate_limiter:
            raise RuntimeError("KiteRESTClient not initialized. Call initialize() first.")

        self._update_access_token()
        try:
            async with self.general_api_rate_limiter:
                logger.info(f"Fetching instruments for exchange: {exchange if exchange else 'All'}")
                loop = asyncio.get_running_loop()
                if self.kite is None:
                    raise RuntimeError("Kite client is not initialized.")
                kite = self.kite  # Assign to local variable for lambda
                result: Optional[list[dict[str, Any]]] = await loop.run_in_executor(
                    None, lambda: kite.instruments(exchange=exchange)
                )
                return result
        except TokenException as e:
            logger.error(f"Authentication token error fetching instruments: {e}")
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
