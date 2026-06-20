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

    def __init__(self, rest_client: KiteRESTClient, ohlcv_repo: Any = None):
        # Type hints ensure rest_client is a KiteRESTClient instance
        self.rest_client = rest_client
        self.ohlcv_repo = ohlcv_repo
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

    async def merge_with_live_stream(self, instrument_id: int, missing_data: list[dict[str, Any]]) -> None:
        """
        Merges fetched missing data by inserting it into the database.

        Args:
            instrument_id: The database ID of the instrument.
            missing_data: The list of missing candles to merge.
        """
        if not missing_data:
            logger.info("No missing data to merge.")
            return

        if not self.ohlcv_repo:
            logger.error("OHLCVRepository not provided to BackfillManager. Cannot merge data.")
            return

        from src.database.models import OHLCVData

        logger.info(f"Merging {len(missing_data)} missing data points for instrument_id {instrument_id}.")
        
        # 1. Parse and insert historical 1-minute candles
        try:
            ohlcv_records = []
            for row in missing_data:
                # The Kite API returns timestamp under 'date'
                ts = row.get("date")
                if not ts:
                    continue
                    
                ohlcv_records.append(
                    OHLCVData(
                        ts=ts,
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                        oi=row.get("oi", None)
                    )
                )
                
            if ohlcv_records:
                await self.ohlcv_repo.insert_ohlcv_data(instrument_id, "1min", ohlcv_records)
                logger.info(f"Successfully backfilled {len(ohlcv_records)} 1-minute candles.")
            else:
                logger.warning("No valid records to backfill after parsing.")
                
        except Exception as e:
            logger.error(f"Error merging backfill data for instrument {instrument_id}: {e}", exc_info=True)
