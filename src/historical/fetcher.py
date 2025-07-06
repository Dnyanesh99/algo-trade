import asyncio
from datetime import datetime, timedelta
from typing import Any

from src.broker.rest_client import KiteRESTClient
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger
from src.utils.rate_limiter import RateLimiter

config = config_loader.get_config()


class HistoricalFetcher:
    """
    Fetches 1 year of 1-minute historical data from the Zerodha KiteConnect API.
    Handles the 60-day limit per request by performing date range chunking.
    Uses a rate limiter to avoid API throttling.
    """

    def __init__(self, rest_client: KiteRESTClient):
        self.rest_client = rest_client
        self.max_days_per_request = config.data_pipeline.historical_data_max_days_per_request
        self.rate_limiter = RateLimiter("historical_data")
        self.historical_interval = config.broker.historical_interval
        self.max_retries = config.performance.max_retries

    async def fetch_historical_data(
        self, instrument_token: int, from_date: datetime, to_date: datetime
    ) -> list[dict[str, Any]]:
        """
        Fetches historical data for a given instrument and date range.
        Automatically chunks requests and applies rate limiting.

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
                    async with self.rate_limiter:
                        logger.info(f"Fetching chunk for {instrument_token}: {from_date_str} to {to_date_str} (Attempt {retries + 1}/{self.max_retries})")
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
                        f"Retrying in {2 ** retries} seconds... (Attempt {retries}/{self.max_retries})",
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



