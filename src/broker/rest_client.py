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
    Ensures requests are signed, rate-limited, and robustly handled.
    """

    def __init__(self, token_manager: TokenManager, api_key: str):
        # Type hints ensure token_manager is a TokenManager instance
        if not api_key or not isinstance(api_key, str):
            logger.critical("CRITICAL: KiteRESTClient initialized with an invalid or empty API key.")
            raise ValueError("api_key must be a non-empty string.")

        self.api_key = api_key
        self.token_manager = token_manager
        self.kite: Optional[KiteConnect] = None
        self.historical_data_rate_limiter: Optional[RateLimiter] = None
        self.general_api_rate_limiter: Optional[RateLimiter] = None
        logger.info("KiteRESTClient instantiated.")

    async def initialize(self, config: AppConfig) -> None:
        """
        Initializes the KiteRESTClient, including the KiteConnect client and rate limiters.
        This method is essential for preparing the client for making API calls.
        """
        logger.info("Initializing KiteRESTClient...")
        if not config:
            logger.critical("CRITICAL: Application configuration is required to initialize KiteRESTClient.")
            raise ValueError("Configuration (AppConfig) must be provided.")

        try:
            self.kite = self._initialize_kite_client()
            self.historical_data_rate_limiter = await get_rate_limiter("historical_data", config)
            self.general_api_rate_limiter = await get_rate_limiter("general_api", config)
            logger.info("KiteRESTClient initialized successfully with rate limiters.")
        except (ValueError, TypeError) as e:
            logger.critical(f"Failed to initialize KiteRESTClient due to invalid configuration: {e}", exc_info=True)
            raise RuntimeError("Could not initialize KiteRESTClient due to configuration errors.") from e
        except Exception as e:
            logger.critical(f"An unexpected error occurred during KiteRESTClient initialization: {e}", exc_info=True)
            raise RuntimeError("An unexpected error prevented KiteRESTClient initialization.") from e

    def _initialize_kite_client(self) -> KiteConnect:
        """
        Initializes the KiteConnect client with the API key and access token.
        The access token is fetched from the TokenManager.
        """
        logger.info("Initializing KiteConnect client...")
        access_token = self.token_manager.get_access_token()
        if not access_token:
            logger.warning("Access token is not yet available. Initializing KiteConnect client without it.")
            logger.warning("Any API calls requiring authentication will fail until the token is set and updated.")
            # Allow initialization without a token, but it will be unusable for most APIs.
            return KiteConnect(api_key=self.api_key, disable_ssl=True)

        logger.info("KiteConnect client initialized with a valid access token.")
        return KiteConnect(api_key=self.api_key, access_token=access_token, disable_ssl=True)

    def _update_access_token(self) -> None:
        """
        Updates the access token for the KiteConnect client if it has changed.
        This is crucial for long-running applications where the token might be refreshed.
        """
        if not self.kite:
            logger.critical("CRITICAL: Attempted to update access token on a non-initialized KiteRESTClient.")
            raise RuntimeError("KiteRESTClient must be initialized before updating the token.")

        current_token = self.token_manager.get_access_token()
        if not current_token:
            logger.warning("No access token available in TokenManager. Unable to update KiteConnect client.")
            # Ensure the client also has no token if it's been cleared.
            if self.kite.access_token:
                self.kite.set_access_token(None)
                logger.info("Cleared expired/invalid token from KiteConnect client.")
            return

        if self.kite.access_token != current_token:
            self.kite.set_access_token(current_token)
            logger.info("KiteConnect client's access token has been successfully updated.")
        else:
            logger.debug("Access token is already up-to-date.")

    async def get_historical_data(
        self,
        instrument_token: int,
        from_date: str,
        to_date: str,
        interval: str,
        continuous: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Fetches historical OHLCV data for a given instrument, applying rate limiting.

        Returns:
            A list of historical data points. Returns an empty list on recoverable errors.

        Raises:
            RuntimeError: On critical, non-recoverable errors like uninitialization or auth failure.
        """
        if not self.kite or not self.historical_data_rate_limiter:
            logger.critical("CRITICAL: get_historical_data called before KiteRESTClient was initialized.")
            raise RuntimeError("KiteRESTClient must be initialized before fetching data.")

        self._update_access_token()
        if not self.kite.access_token:
            logger.error("Cannot fetch historical data: No valid access token is available.")
            raise RuntimeError("Authentication token is missing, cannot fetch historical data.")

        start_time = time.monotonic()
        api_name = "historical_data"
        try:
            async with self.historical_data_rate_limiter:
                logger.info(
                    f"Fetching historical data for token {instrument_token} from {from_date} to {to_date} ({interval})."
                )
                loop = asyncio.get_running_loop()
                # Assign to a local variable for thread safety in the executor.
                kite_client = self.kite
                result: list[dict[str, Any]] = await loop.run_in_executor(
                    None,
                    lambda: kite_client.historical_data(
                        instrument_token, from_date, to_date, interval, continuous=continuous
                    ),
                )
                duration = time.monotonic() - start_time
                metrics_registry.record_broker_api_request(api_name, True, duration)
                logger.debug(f"Successfully fetched {len(result)} records for {instrument_token} in {duration:.2f}s.")
                return result

        except TokenException as e:
            duration = time.monotonic() - start_time
            metrics_registry.record_broker_api_request(api_name, False, duration)
            logger.critical(
                f"CRITICAL: TokenException for {instrument_token}. Re-authentication required. Error: {e}",
                exc_info=True,
            )
            raise RuntimeError("Authentication token is invalid or expired.") from e
        except (NetworkException, KiteException) as e:
            duration = time.monotonic() - start_time
            metrics_registry.record_broker_api_request(api_name, False, duration)
            logger.error(f"Broker API error for {instrument_token}: {e}", exc_info=True)
            return []  # Return empty list for recoverable API/network errors
        except Exception as e:
            duration = time.monotonic() - start_time
            metrics_registry.record_broker_api_request(api_name, False, duration)
            logger.error(f"Unexpected error fetching historical data for {instrument_token}: {e}", exc_info=True)
            return []

    async def get_instruments_csv(self) -> Optional[list[dict[str, Any]]]:
        """
        Fetches the complete instrument list from Zerodha in CSV format.
        This is the optimized approach recommended by Zerodha documentation.
        Downloads a gzipped CSV dump of all instruments across all exchanges.
        """
        if not self.general_api_rate_limiter:
            logger.critical("CRITICAL: get_instruments_csv called before KiteRESTClient was initialized.")
            raise RuntimeError("KiteRESTClient must be initialized before fetching instruments.")

        self._update_access_token()
        access_token = self.token_manager.get_access_token()
        if not access_token:
            logger.error("Cannot fetch instruments CSV: No valid access token is available.")
            return None

        logger.info("Attempting to fetch complete instrument list using the optimized CSV method.")
        try:
            async with self.general_api_rate_limiter:
                import csv
                import gzip
                import io

                import httpx

                url = "https://api.kite.trade/instruments"
                headers = {
                    "X-Kite-Version": "3",
                    "Authorization": f"token {self.api_key}:{access_token}",
                }

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()  # Will raise for 4xx/5xx responses

                logger.info("Successfully downloaded instruments CSV. Parsing...")
                
                # Handle both gzipped and already-decompressed responses
                try:
                    # Try to decompress first (in case it's still gzipped)
                    decompressed_data = gzip.decompress(response.content)
                    csv_content = decompressed_data.decode("utf-8")
                    logger.debug("Response was gzipped, successfully decompressed")
                except gzip.BadGzipFile:
                    # Already decompressed by HTTP client
                    csv_content = response.content.decode("utf-8")
                    logger.debug("Response was already decompressed by HTTP client")
                csv_reader = csv.DictReader(io.StringIO(csv_content))

                instruments = []
                for row in csv_reader:
                    try:
                        instrument = {
                            "instrument_token": int(row["instrument_token"]),
                            "exchange_token": row.get("exchange_token"),
                            "tradingsymbol": row["tradingsymbol"],
                            "name": row.get("name"),
                            "last_price": float(row["last_price"]) if row.get("last_price") else None,
                            "expiry": row.get("expiry"),
                            "strike": float(row["strike"]) if row.get("strike") else None,
                            "tick_size": float(row["tick_size"]) if row.get("tick_size") else None,
                            "lot_size": int(row["lot_size"]) if row.get("lot_size") else None,
                            "instrument_type": row["instrument_type"],
                            "segment": row["segment"],
                            "exchange": row["exchange"],
                        }
                        instruments.append(instrument)
                    except (KeyError, TypeError, ValueError) as e:
                        logger.warning(f"Skipping a row from instruments CSV due to parsing error: {e}. Row: {row}")
                        continue

                logger.info(f"Successfully parsed {len(instruments)} instruments from CSV.")
                return instruments

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.critical(
                    f"CRITICAL: Authentication failed (403 Forbidden) while fetching instruments CSV. Token may be invalid. Error: {e}"
                )
            else:
                logger.error(
                    f"HTTP error fetching instruments CSV: {e.response.status_code} - {e.response.text}", exc_info=True
                )
            return None
        except httpx.RequestError as e:
            logger.error(f"A network error occurred while fetching the instruments CSV: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while fetching or parsing the instruments CSV: {e}", exc_info=True
            )
            return None

    async def get_instruments(self, exchange: Optional[str] = None) -> Optional[list[dict[str, Any]]]:
        """
        Fetches the list of instruments, preferring the optimized CSV approach.
        Falls back to the legacy API if the CSV method fails.

        Args:
            exchange: Optional exchange to filter instruments by (e.g., 'NFO', 'NSE').

        Returns:
            A list of instruments, or None if both methods fail.
        """
        if not self.kite or not self.general_api_rate_limiter:
            logger.critical("CRITICAL: get_instruments called before KiteRESTClient was initialized.")
            raise RuntimeError("KiteRESTClient must be initialized before fetching instruments.")

        logger.info(f"Fetching instruments for exchange: {exchange or 'ALL'}.")

        # Primary method: Optimized CSV download
        instruments = await self.get_instruments_csv()
        if instruments is not None:
            if exchange:
                logger.debug(f"Filtering {len(instruments)} total instruments for exchange '{exchange}'.")
                filtered_instruments = [inst for inst in instruments if inst.get("exchange") == exchange]
                logger.info(f"Found {len(filtered_instruments)} instruments for exchange '{exchange}' via CSV method.")
                return filtered_instruments
            return instruments

        # Fallback method: Legacy API call
        logger.warning("CSV instrument fetch failed. Falling back to legacy API method.")
        self._update_access_token()
        if not self.kite.access_token:
            logger.error("Cannot fetch instruments via legacy API: No valid access token.")
            return None

        try:
            async with self.general_api_rate_limiter:
                logger.info(f"Executing legacy instrument fetch for exchange: {exchange or 'All'}")
                loop = asyncio.get_running_loop()
                kite_client = self.kite
                result: list[dict[str, Any]] = await loop.run_in_executor(
                    None, lambda: kite_client.instruments(exchange=exchange)
                )
                logger.info(f"Successfully fetched {len(result)} instruments via legacy API.")
                return result
        except TokenException as e:
            logger.critical(
                f"CRITICAL: TokenException on legacy instrument fetch. Re-authentication required. Error: {e}",
                exc_info=True,
            )
            return None
        except (NetworkException, KiteException) as e:
            logger.error(f"Broker API error on legacy instrument fetch: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error on legacy instrument fetch: {e}", exc_info=True)
            return None
