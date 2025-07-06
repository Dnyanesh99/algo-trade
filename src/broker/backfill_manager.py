from datetime import datetime, timedelta
from typing import Any, Optional

from kiteconnect.exceptions import KiteException, NetworkException, TokenException

from src.broker.rest_client import KiteRESTClient
from src.utils.logger import LOGGER as logger  # Centralized logger


class BackfillManager:
    """
    Manages the backfilling of missing historical data, typically triggered after a WebSocket
    reconnection or detected data gaps. Fetches missing 1-minute candles using the REST client.
    """

    def __init__(self, rest_client: KiteRESTClient):
        self.rest_client = rest_client

    async def fetch_missing_data(
        self, instrument_token: int, last_received_timestamp: datetime
    ) -> Optional[list[dict[str, Any]]]:
        """
        Fetches missing 1-minute historical data from the last received timestamp up to now.

        Args:
            instrument_token (int): The instrument token for which to fetch data.
            last_received_timestamp (datetime): The timestamp of the last successfully received data point.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of missing 1-minute candles, or None if an error occurs.
        """
        # Add a small buffer to `from_date` to ensure no overlap and fetch the next candle
        from_date = last_received_timestamp + timedelta(minutes=1)
        to_date = datetime.now()  # Fetch up to the current time

        # KiteConnect historical data API expects string dates in 'YYYY-MM-DD HH:MM:SS' format
        from_date_str = from_date.strftime("%Y-%m-%d %H:%M:%S")
        to_date_str = to_date.strftime("%Y-%m-%d %H:%M:%S")

        logger.info(
            f"Initiating backfill for {instrument_token} from {from_date_str} to {to_date_str} (1-minute interval)"
        )

        try:
            # Fetch 1-minute candles
            missing_candles = await self.rest_client.get_historical_data(
                instrument_token,
                from_date_str,
                to_date_str,
                "minute",  # Always fetch 1-minute candles for backfill
            )

            if missing_candles:
                logger.info(f"Successfully fetched {len(missing_candles)} missing candles for {instrument_token}.")
                return missing_candles
            logger.warning(f"No missing candles found or error occurred during backfill for {instrument_token}.")
            return None
        except TokenException as e:
            logger.error(f"Authentication token error during backfill for {instrument_token}: {e}")
            # Consider triggering re-authentication flow or critical alert
            return None
        except NetworkException as e:
            logger.error(f"Network error during backfill for {instrument_token}: {e}")
            return None
        except KiteException as e:
            logger.error(f"KiteConnect API error during backfill for {instrument_token}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during backfill for {instrument_token}: {e}")
            return None

    async def merge_with_live_stream(self, missing_data: list[dict[str, Any]]) -> None:
        """
        Placeholder for merging fetched missing data with the live data stream.
        In a real implementation, this would involve inserting data into a buffer or database.

        Args:
            missing_data (List[Dict[str, Any]]): The list of missing candles to merge.
        """
        if missing_data:
            logger.info(f"Merging {len(missing_data)} missing data points with live stream (placeholder).")
            # TODO: Implement actual merging logic, e.g., insert into tick queue or candle buffer
        else:
            logger.info("No missing data to merge.")
