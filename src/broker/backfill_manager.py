from datetime import datetime, timedelta
from typing import Any

from kiteconnect.exceptions import KiteException, NetworkException, TokenException

from src.broker.rest_client import KiteRESTClient
from src.utils.logger import LOGGER as logger


class BackfillManager:
    """
    Manages the backfilling of missing historical data, typically triggered after a WebSocket
    reconnection or detected data gaps. Fetches missing 1-minute candles using the REST client.
    """

    def __init__(self, rest_client: KiteRESTClient):
        # Type hints ensure rest_client is a KiteRESTClient instance
        self.rest_client = rest_client
        logger.info("BackfillManager initialized successfully.")

    async def fetch_missing_data(
        self, instrument_token: int, last_received_timestamp: datetime
    ) -> list[dict[str, Any]]:
        """
        Fetches missing 1-minute historical data from the last received timestamp up to now.

        Args:
            instrument_token: The instrument token for which to fetch data.
            last_received_timestamp: The timestamp of the last successfully received data point.

        Returns:
            A list of missing 1-minute candles. Can be empty if no new data is available.

        Raises:
            ValueError: If input parameters are invalid.
            RuntimeError: If a critical, non-recoverable error occurs during the fetch.
        """
        if not isinstance(instrument_token, int) or instrument_token <= 0:
            logger.error(f"Invalid instrument_token provided for backfill: {instrument_token}")
            raise ValueError("instrument_token must be a positive integer.")
        # Type hints ensure last_received_timestamp is a datetime object

        # Add a small buffer to `from_date` to ensure no overlap and fetch the next candle
        from_date = last_received_timestamp + timedelta(minutes=1)
        to_date = datetime.now()

        if from_date >= to_date:
            logger.warning(
                f"Skipping backfill for {instrument_token}. "
                f"Last received data at {last_received_timestamp.strftime('%Y-%m-%d %H:%M:%S')} is too recent.",
            )
            return []

        from_date_str = from_date.strftime("%Y-%m-%d %H:%M:%S")
        to_date_str = to_date.strftime("%Y-%m-%d %H:%M:%S")

        logger.info(
            f"Initiating backfill for instrument {instrument_token} "
            f"from {from_date_str} to {to_date_str} (1-minute interval)."
        )

        try:
            missing_candles = await self.rest_client.get_historical_data(
                instrument_token,
                from_date_str,
                to_date_str,
                "minute",
            )

            if missing_candles:
                logger.info(f"Successfully fetched {len(missing_candles)} missing candles for {instrument_token}.")
            else:
                logger.warning(
                    f"No missing candles returned from REST client for {instrument_token}. Check previous logs."
                )
            return missing_candles
        except TokenException as e:
            logger.critical(
                f"CRITICAL: Authentication token error during backfill for {instrument_token}: {e}", exc_info=True
            )
            raise RuntimeError("Authentication failed during data backfill. System may need re-authentication.") from e
        except (NetworkException, KiteException) as e:
            logger.error(f"Broker API or network error during backfill for {instrument_token}: {e}", exc_info=True)
            # Depending on strategy, this could be a soft failure (return empty list) or hard (raise)
            # For now, we return empty and let the caller decide.
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during backfill for {instrument_token}: {e}", exc_info=True)
            # Unexpected errors are often more serious, but we still return empty for now.
            return []

    async def merge_with_live_stream(self, missing_data: list[dict[str, Any]]) -> None:
        """
        Placeholder for merging fetched missing data with the live data stream.
        In a real implementation, this would involve inserting data into a buffer or database
        and ensuring data integrity (no duplicates, correct order).

        Args:
            missing_data: The list of missing candles to merge.
        """
        if not missing_data:
            logger.info("No missing data to merge.")
            return

        logger.info(f"[PLACEHOLDER] Merging {len(missing_data)} missing data points with live stream.")
        logger.warning("This is a placeholder implementation. In a production system, this method must:")
        logger.warning("1. Connect to the appropriate data buffer or database repository.")
        logger.warning("2. Insert the historical candles, handling potential duplicates and ordering.")
        logger.warning("3. Trigger downstream processes to re-calculate features if necessary.")
        # Example of what real logic might look like:
        # await self.data_repository.insert_candles(missing_data)
        # await self.feature_calculator.recompute_for_instrument(instrument_token)
