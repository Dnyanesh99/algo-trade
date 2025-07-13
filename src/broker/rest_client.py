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

    async def get_instruments_csv(self) -> Optional[list[dict[str, Any]]]:
        """
        Fetches the complete instrument list from Zerodha in CSV format.
        This is the optimized approach recommended by Zerodha documentation.
        Downloads a gzipped CSV dump of all instruments across all exchanges.
        """
        if not self.general_api_rate_limiter:
            raise RuntimeError("KiteRESTClient not initialized. Call initialize() first.")

        self._update_access_token()

        try:
            async with self.general_api_rate_limiter:
                logger.info("Fetching complete instrument list in CSV format...")

                import csv
                import gzip
                import io

                import httpx

                # Zerodha instruments CSV URL
                url = "https://api.kite.trade/instruments"
                headers = {
                    "X-Kite-Version": "3",
                    "Authorization": f"token {self.api_key}:{self.token_manager.get_access_token()}",
                }

                # Download the gzipped CSV
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()

                    # Decompress gzipped content
                    decompressed_data = gzip.decompress(response.content)
                    csv_content = decompressed_data.decode("utf-8")

                    # Parse CSV content
                    csv_reader = csv.DictReader(io.StringIO(csv_content))
                    instruments = []

                    for row in csv_reader:
                        # Convert types for consistency with KiteConnect API
                        instrument = {
                            "instrument_token": int(row["instrument_token"]) if row["instrument_token"] else 0,
                            "exchange_token": row["exchange_token"] if row["exchange_token"] else None,
                            "tradingsymbol": row["tradingsymbol"],
                            "name": row["name"] if row["name"] else None,
                            "last_price": float(row["last_price"]) if row["last_price"] else None,
                            "expiry": row["expiry"] if row["expiry"] else None,
                            "strike": float(row["strike"]) if row["strike"] else None,
                            "tick_size": float(row["tick_size"]) if row["tick_size"] else None,
                            "lot_size": int(row["lot_size"]) if row["lot_size"] else None,
                            "instrument_type": row["instrument_type"],
                            "segment": row["segment"],
                            "exchange": row["exchange"],
                        }
                        instruments.append(instrument)

                    logger.info(f"Successfully parsed {len(instruments)} instruments from CSV")
                    return instruments

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.error("Authentication failed - invalid access token for instruments CSV")
            else:
                logger.error(f"HTTP error fetching instruments CSV: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Network error fetching instruments CSV: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching instruments CSV: {e}")
            return None

    async def get_instruments(self, exchange: Optional[str] = None) -> Optional[list[dict[str, Any]]]:
        """
        Fetches the list of instruments.
        Uses optimized CSV approach for better performance.
        Falls back to legacy API if CSV approach fails.
        """
        if not self.kite or not self.general_api_rate_limiter:
            raise RuntimeError("KiteRESTClient not initialized. Call initialize() first.")

        # Try optimized CSV approach first
        instruments = await self.get_instruments_csv()
        if instruments:
            # Filter by exchange if specified
            if exchange:
                instruments = [inst for inst in instruments if inst["exchange"] == exchange]
                logger.info(f"Filtered to {len(instruments)} instruments for exchange: {exchange}")
            return instruments

        # Fallback to legacy API approach
        logger.warning("CSV approach failed, falling back to legacy instruments API")
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
