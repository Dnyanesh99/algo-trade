from datetime import datetime
from typing import Optional

import pytz

from src.database.db_utils import db_manager
from src.database.models import OHLCVData
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

config = ConfigLoader().get_config()
queries = ConfigLoader().get_queries()


class OHLCVRepository:
    """
    Repository for interacting with OHLCV data in the TimescaleDB.
    Provides methods for inserting, fetching, and managing OHLCV data across different timeframes.
    """

    def __init__(self) -> None:
        self.db_manager = db_manager
        if not config.trading or not config.trading.aggregation_timeframes:
            logger.critical("CRITICAL: trading.aggregation_timeframes not found in config.yaml.")
            raise ValueError("OHLCVRepository requires aggregation_timeframes to be configured.")

        self._table_map = {f"{tf}min": f"ohlcv_{tf}min" for tf in config.trading.aggregation_timeframes}
        self._table_map["1min"] = "ohlcv_1min"  # Ensure base timeframe is always present
        # Set up timezone for consistent datetime handling
        self.timezone = pytz.timezone(config.system.timezone)
        logger.info(f"OHLCVRepository initialized with table map: {self._table_map}")

    def _normalize_datetime(self, dt: datetime) -> datetime:
        """
        Normalize datetime to be timezone-aware using system timezone.

        Args:
            dt: DateTime object (timezone-aware or naive)

        Returns:
            Timezone-aware datetime object
        """
        if dt.tzinfo is None:
            # If timezone-naive, assume it's in system timezone
            return self.timezone.localize(dt)
        # If already timezone-aware, convert to system timezone
        return dt.astimezone(self.timezone)

    def _get_table_name(self, timeframe: str) -> str:
        """
        Validates the timeframe and returns the corresponding table name.
        This prevents SQL injection and ensures only valid tables are queried.
        """
        table_name = self._table_map.get(timeframe)
        if not table_name:
            logger.error(
                f"Invalid or unconfigured timeframe '{timeframe}' requested. Must be one of {list(self._table_map.keys())}"
            )
            raise ValueError(f"Invalid or unconfigured timeframe: {timeframe}")
        logger.debug(f"Resolved timeframe '{timeframe}' to table '{table_name}'.")
        return table_name

    async def insert_ohlcv_data(self, instrument_id: int, timeframe: str, ohlcv_data: list[OHLCVData]) -> None:
        """
        Inserts OHLCV data into the specified timeframe table.

        Raises:
            ValueError: If input arguments are invalid.
            RuntimeError: If the database insertion fails.
        """
        if not ohlcv_data:
            logger.warning(f"insert_ohlcv_data called with no data for instrument {instrument_id} ({timeframe}).")
            return

        table_name = self._get_table_name(timeframe)
        query = queries.ohlcv_repo["insert_ohlcv_data"].format(table_name=table_name)
        records = [(d.ts, instrument_id, d.open, d.high, d.low, d.close, d.volume, d.oi) for d in ohlcv_data]

        try:
            async with self.db_manager.transaction() as conn:
                status = await conn.executemany(query, records)
            logger.info(
                f"Successfully inserted/updated {len(records)} OHLCV records for instrument {instrument_id} ({timeframe}). Status: {status}"
            )
        except Exception as e:
            logger.error(
                f"Database error inserting OHLCV data for instrument {instrument_id} ({timeframe}): {e}", exc_info=True
            )
            raise RuntimeError("Failed to insert OHLCV data.") from e

    async def get_ohlcv_data(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> list[OHLCVData]:
        """
        Fetches OHLCV data for a given instrument and timeframe within a time range.

        Raises:
            RuntimeError: If the database query fails.
        """
        table_name = self._get_table_name(timeframe)
        query = queries.ohlcv_repo["get_ohlcv_data"].format(table_name=table_name)
        logger.debug(
            f"Fetching OHLCV data for instrument {instrument_id} ({timeframe}) from {start_time} to {end_time}."
        )
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, start_time, end_time)
            logger.info(f"Fetched {len(rows)} OHLCV records for instrument {instrument_id} ({timeframe}).")
            return [OHLCVData.model_validate(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Database error fetching OHLCV data for {instrument_id} ({timeframe}): {e}", exc_info=True)
            raise RuntimeError("Failed to fetch OHLCV data.") from e

    async def get_latest_candle(self, instrument_id: int, timeframe: str) -> Optional[OHLCVData]:
        """
        Fetches the latest OHLCV candle for a given instrument and timeframe.

        Raises:
            RuntimeError: If the database query fails.
        """
        table_name = self._get_table_name(timeframe)
        query = queries.ohlcv_repo["get_latest_candle"].format(table_name=table_name)
        logger.debug(f"Fetching latest {timeframe} candle for instrument {instrument_id}.")
        try:
            row = await self.db_manager.fetch_row(query, instrument_id)
            if row:
                logger.info(f"Latest {timeframe} candle for instrument {instrument_id} is at {row['ts']}.")
                return OHLCVData.model_validate(dict(row))
            logger.info(f"No latest {timeframe} candle found for instrument {instrument_id}.")
            return None
        except Exception as e:
            logger.error(f"Database error fetching latest candle for {instrument_id} ({timeframe}): {e}", exc_info=True)
            raise RuntimeError("Failed to fetch latest candle.") from e

    async def get_ohlcv_data_for_features(self, instrument_id: int, timeframe: int, limit: int) -> list[OHLCVData]:
        """
        Fetches the most recent OHLCV data points up to a specified limit,
        ordered by timestamp ascending, suitable for feature calculation.

        Raises:
            RuntimeError: If the database query fails.
        """
        table_name = self._get_table_name(f"{timeframe}min")
        query = queries.ohlcv_repo["get_ohlcv_data_for_features"].format(table_name=table_name)
        logger.debug(
            f"Fetching {limit} recent {timeframe}min candles for instrument {instrument_id} for feature calculation."
        )
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, limit)
            logger.info(f"Fetched {len(rows)} candles for feature calculation for instrument {instrument_id}.")
            return [OHLCVData.model_validate(dict(row)) for row in rows]
        except Exception as e:
            logger.error(
                f"Database error fetching OHLCV data for features for {instrument_id} ({timeframe}min): {e}",
                exc_info=True,
            )
            raise RuntimeError("Failed to fetch OHLCV data for features.") from e

    async def check_data_availability(self, timeframe: str, start_time: datetime, end_time: datetime) -> int:
        """
        Check data availability for a given timeframe and time range.
        Returns the count of candles available in the specified period.

        Raises:
            RuntimeError: If the database query fails.
        """
        table_name = self._get_table_name(timeframe)
        query = queries.ohlcv_repo["check_data_availability"].format(table_name=table_name)
        logger.debug(f"Checking data availability for {timeframe} between {start_time} and {end_time}.")
        try:
            candle_count = await self.db_manager.fetchval(query, start_time, end_time)
            count = candle_count if candle_count is not None else 0
            logger.info(f"Found {count} candles for {timeframe} in the specified range.")
            return count
        except Exception as e:
            logger.error(f"Database error checking data availability for {timeframe}: {e}", exc_info=True)
            raise RuntimeError("Failed to check data availability.") from e

    async def get_data_range(self, instrument_id: int, timeframe: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the earliest and latest timestamps for an instrument/timeframe.

        Returns:
            Tuple of (earliest_timestamp, latest_timestamp) or (None, None) if no data.

        Raises:
            RuntimeError: If the database query fails.
        """
        table_name = self._get_table_name(timeframe)
        query = queries.ohlcv_repo["get_data_range_for_instrument"].format(table_name=table_name)
        logger.debug(f"Getting data range for instrument {instrument_id} ({timeframe}).")
        try:
            row = await self.db_manager.fetch_row(query, instrument_id)
            if row and row["earliest_ts"] and row["latest_ts"]:
                logger.info(
                    f"Data range for {instrument_id} ({timeframe}): {row['earliest_ts']} to {row['latest_ts']}."
                )
                return (row["earliest_ts"], row["latest_ts"])
            logger.info(f"No data range found for instrument {instrument_id} ({timeframe}).")
            return (None, None)
        except Exception as e:
            logger.error(f"Database error getting data range for {instrument_id} ({timeframe}): {e}", exc_info=True)
            raise RuntimeError("Failed to get data range.") from e

    async def get_last_successful_fetch(self, instrument_id: int) -> Optional[datetime]:
        """
        Get timestamp of last successful data fetch for an instrument.
        This checks the latest data in the base 1-minute timeframe.

        Raises:
            RuntimeError: If the database query fails.
        """
        logger.debug(f"Getting last successful fetch time for instrument {instrument_id}.")
        try:
            latest_1min = await self.get_latest_candle(instrument_id, "1min")
            if latest_1min:
                logger.info(f"Last successful fetch for instrument {instrument_id} was at {latest_1min.ts}.")
                return latest_1min.ts
            logger.info(f"No successful fetch found for instrument {instrument_id}.")
            return None
        except Exception as e:
            logger.error(f"Database error getting last successful fetch for {instrument_id}: {e}", exc_info=True)
            raise RuntimeError("Failed to get last successful fetch time.") from e

    async def has_sufficient_data(
        self, instrument_id: int, timeframe: str, required_candles: int, lookback_hours: int = 24
    ) -> bool:
        """
        Check if there's sufficient data for processing.

        Returns:
            True if sufficient data exists, False otherwise.

        Raises:
            RuntimeError: If the database query fails.
        """
        from datetime import timedelta

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)

        # Normalize datetimes to ensure timezone consistency
        end_time = self._normalize_datetime(end_time)
        start_time = self._normalize_datetime(start_time)
        logger.debug(
            f"Checking for {required_candles} candles for instrument {instrument_id} ({timeframe}) in the last {lookback_hours} hours."
        )

        try:
            candle_count = await self.check_data_availability(timeframe, start_time, end_time)
            has_sufficient = candle_count >= required_candles

            logger.info(
                f"Data sufficiency check for {instrument_id} ({timeframe}): "
                f"Found {candle_count}/{required_candles} required candles. Sufficient: {has_sufficient}"
            )
            return has_sufficient

        except Exception as e:
            logger.error(
                f"Database error checking data sufficiency for {instrument_id} ({timeframe}): {e}", exc_info=True
            )
            raise RuntimeError("Failed to check for sufficient data.") from e
