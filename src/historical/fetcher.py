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
        if not isinstance(rest_client, KiteRESTClient):
            raise TypeError("rest_client must be a valid KiteRESTClient instance.")

        config = ConfigLoader().get_config()
        if not all([config.data_pipeline, config.broker, config.model_training, config.performance]):
            raise ValueError("Data pipeline, broker, model_training, and performance configurations are required.")

        self.rest_client = rest_client
        self.max_days_per_request = config.data_pipeline.historical_data_max_days_per_request
        self.historical_interval = config.broker.historical_interval
        self.historical_data_lookback_days = config.model_training.historical_data_lookback_days
        self.max_retries = config.performance.max_retries
        logger.info("HistoricalFetcher initialized with production configuration.")

    @measure_fetch_operation
    async def fetch_historical_data(
        self, instrument_token: int, from_date: datetime, to_date: datetime
    ) -> list[dict[str, Any]]:
        """
        Fetches historical data for a given instrument and date range.
        Automatically chunks requests to respect broker API limits.
        """
        if from_date >= to_date:
            logger.error(
                f"Invalid date range for historical fetch: from_date ({from_date}) must be before to_date ({to_date})."
            )
            return []

        total_days = (to_date - from_date).days
        logger.info(
            f"Starting historical data fetch for instrument {instrument_token} from {from_date.date()} to {to_date.date()} ({total_days} days)."
        )

        all_candles: list[dict[str, Any]] = []
        current_from_date = from_date

        while current_from_date <= to_date:
            chunk_to_date = min(to_date, current_from_date + timedelta(days=self.max_days_per_request - 1))
            from_date_str = current_from_date.strftime("%Y-%m-%d %H:%M:%S")
            to_date_str = chunk_to_date.strftime("%Y-%m-%d %H:%M:%S")

            success = False
            for attempt in range(self.max_retries):
                try:
                    logger.info(
                        f"Fetching chunk for {instrument_token}: {from_date_str} to {to_date_str} (Attempt {attempt + 1}/{self.max_retries})"
                    )
                    candles = await self.rest_client.get_historical_data(
                        instrument_token, from_date_str, to_date_str, self.historical_interval
                    )
                    if candles:
                        all_candles.extend(candles)
                        logger.info(f"Fetched {len(candles)} candles in this chunk. Total so far: {len(all_candles)}.")
                    else:
                        logger.warning(
                            f"No candles returned for chunk {from_date_str} to {to_date_str}. This may be expected."
                        )

                    success = True
                    break  # Exit retry loop on success

                except Exception as e:
                    logger.error(
                        f"Error fetching chunk for {instrument_token}: {e}. "
                        f"Retrying in {2**attempt} seconds... (Attempt {attempt + 1}/{self.max_retries})",
                        exc_info=True,
                    )
                    await asyncio.sleep(2**attempt)

            if not success:
                logger.critical(
                    f"Failed to fetch historical data for {instrument_token} from {from_date_str} to {to_date_str} "
                    f"after {self.max_retries} attempts. This will result in a data gap."
                )
                # Depending on system requirements, you might want to raise an exception here
                # to halt processing if a full dataset is critical.

            current_from_date = chunk_to_date + timedelta(days=1)

        logger.info(
            f"Finished historical data fetch for {instrument_token}. Total candles fetched: {len(all_candles)}."
        )
        return all_candles
