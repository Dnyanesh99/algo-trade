import asyncio
import time
from collections.abc import Awaitable
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from src.broker.rest_client import KiteRESTClient
from src.metrics import metrics_registry
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

_F = TypeVar("_F", bound=Callable[..., Awaitable[Any]])


def measure_fetch_operation(func: _F) -> _F:
    """
    Decorator to measure the execution time of fetch operations and record metrics.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.monotonic()
        success = True
        endpoint = func.__name__

        try:
            return await func(*args, **kwargs)
        except Exception:
            success = False
            raise
        finally:
            duration = time.monotonic() - start_time

            # Record broker API metrics
            metrics_registry.record_broker_api_request(api_name=endpoint, success=success, duration=duration)

            logger.debug(f"Fetch operation {endpoint} completed in {duration:.3f}s, success: {success}")

    return cast("_F", wrapper)


class HistoricalFetcher:
    """
    Fetches 1 year of 1-minute historical data from the Zerodha KiteConnect API.
    Handles the 60-day limit per request by performing date range chunking.
    """

    def __init__(self, rest_client: KiteRESTClient):
        config = ConfigLoader().get_config()
        if config.data_pipeline is None:
            raise ValueError("Data pipeline configuration is required")
        if config.broker is None:
            raise ValueError("Broker configuration is required")
        if config.performance is None:
            raise ValueError("Performance configuration is required")
        self.rest_client = rest_client
        self.max_days_per_request = config.data_pipeline.historical_data_max_days_per_request
        self.historical_interval = config.broker.historical_interval
        self.max_retries = config.performance.max_retries

    @measure_fetch_operation
    async def fetch_historical_data(
        self, instrument_token: int, from_date: datetime, to_date: datetime
    ) -> list[dict[str, Any]]:
        """
        Fetches historical data for a given instrument and date range.
        Automatically chunks requests.

        Args:
            instrument_token (int): The instrument token.
            from_date (datetime): The start date for data fetching.
            to_date (datetime): The end date for data fetching.

        Returns:
            List[Dict[str, Any]]: A list of historical data candles.
        """
        all_candles: list[dict[str, Any]] = []
        current_from_date = from_date

        logger.info(f"Starting historical data fetch for {instrument_token} from {from_date} to {to_date}")

        while current_from_date <= to_date:
            chunk_to_date = current_from_date + timedelta(days=self.max_days_per_request - 1)
            if chunk_to_date > to_date:
                chunk_to_date = to_date

            from_date_str = current_from_date.strftime("%Y-%m-%d %H:%M:%S")
            to_date_str = chunk_to_date.strftime("%Y-%m-%d %H:%M:%S")

            retries = 0
            success = False
            while retries < self.max_retries and not success:
                try:
                    logger.info(
                        f"Fetching chunk for {instrument_token}: {from_date_str} to {to_date_str} (Attempt {retries + 1}/{self.max_retries})"
                    )
                    candles = await self.rest_client.get_historical_data(
                        instrument_token, from_date_str, to_date_str, self.historical_interval
                    )
                    success = True

                    if candles:
                        all_candles.extend(candles)
                        logger.info(f"Fetched {len(candles)} candles in this chunk. Total: {len(all_candles)}")
                    else:
                        logger.warning(f"No candles returned for chunk {from_date_str} to {to_date_str}.")

                except Exception as e:
                    retries += 1
                    logger.error(
                        f"Error fetching chunk for {instrument_token} from {from_date_str} to {to_date_str}: {e}. "
                        f"Retrying in {2**retries} seconds... (Attempt {retries}/{self.max_retries})",
                        exc_info=True,
                    )
                    await asyncio.sleep(2**retries)  # Exponential backoff

            if not success:
                logger.critical(
                    f"Failed to fetch historical data for {instrument_token} from {from_date_str} to {to_date_str} "
                    f"after {self.max_retries} attempts. Data gap will exist."
                )

            current_from_date = chunk_to_date + timedelta(days=1)

        logger.info(f"Finished historical data fetch for {instrument_token}. Total candles: {len(all_candles)}")
        return all_candles
