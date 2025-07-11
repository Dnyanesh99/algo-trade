from datetime import datetime
from typing import Optional

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
        # --- REFACTOR: Create a validated mapping of timeframes to table names ---
        self._table_map = (
            {f"{tf}min": f"ohlcv_{tf}min" for tf in config.trading.aggregation_timeframes} if config.trading else {}
        )
        self._table_map["1min"] = "ohlcv_1min"  # Add base timeframe
        logger.info(f"OHLCVRepository initialized with table map: {self._table_map}")

    def _get_table_name(self, timeframe: str) -> str:
        """
        Validates the timeframe and returns the corresponding table name.
        This prevents SQL injection and ensures only valid tables are queried.
        """
        table_name = self._table_map.get(timeframe)
        if not table_name:
            logger.error(f"Invalid timeframe '{timeframe}' requested. Must be one of {list(self._table_map.keys())}")
            raise ValueError(f"Invalid timeframe: {timeframe}")
        return table_name

    async def insert_ohlcv_data(self, instrument_id: int, timeframe: str, ohlcv_data: list[OHLCVData]) -> str:
        """
        Inserts OHLCV data into the specified timeframe table.

        Args:
            instrument_id (int): The ID of the instrument.
            timeframe (str): The timeframe (e.g., '5min', '15min', '60min').
            ohlcv_data (List[OHLCVData]): List of OHLCVData objects.

        Returns:
            str: Command status from the database.
        """
        table_name = self._get_table_name(timeframe)
        query = queries.ohlcv_repo["insert_ohlcv_data"].format(table_name=table_name)
        records = [
            (
                d.ts,
                instrument_id,
                d.open,
                d.high,
                d.low,
                d.close,
                d.volume,
                d.oi,
            )
            for d in ohlcv_data
        ]

        try:
            async with self.db_manager.transaction() as conn:
                # Use executemany for proper upsert handling
                status = await conn.executemany(query, records)
                logger.info(
                    f"Inserted/Updated {len(ohlcv_data)} OHLCV records for {instrument_id} ({timeframe}). Status: {status}"
                )
                return str(status)
        except Exception as e:
            logger.error(f"Error inserting OHLCV data for {instrument_id} ({timeframe}): {e}")
            raise

    async def get_ohlcv_data(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> list[OHLCVData]:
        """
        Fetches OHLCV data for a given instrument and timeframe within a time range.
        """
        table_name = self._get_table_name(timeframe)
        query = queries.ohlcv_repo["get_ohlcv_data"].format(table_name=table_name)
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, start_time, end_time)
            logger.debug(
                f"Fetched {len(rows)} OHLCV records for {instrument_id} ({timeframe}) from {start_time} to {end_time}."
            )
            return [OHLCVData.model_validate(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {instrument_id} ({timeframe}): {e}")
            raise

    async def get_latest_candle(self, instrument_id: int, timeframe: str) -> Optional[OHLCVData]:
        """
        Fetches the latest OHLCV candle for a given instrument and timeframe.
        """
        table_name = self._get_table_name(timeframe)
        query = queries.ohlcv_repo["get_latest_candle"].format(table_name=table_name)
        try:
            row = await self.db_manager.fetch_row(query, instrument_id)
            if row:
                logger.debug(f"Fetched latest {timeframe} candle for {instrument_id}: {row['ts']}")
                return OHLCVData.model_validate(dict(row))
            logger.debug(f"No latest {timeframe} candle found for {instrument_id}.")
            return None
        except Exception as e:
            logger.error(f"Error fetching latest candle for {instrument_id} ({timeframe}): {e}")
            raise

    async def get_ohlcv_data_for_features(self, instrument_id: int, timeframe: int, limit: int) -> list[OHLCVData]:
        """
        Fetches the most recent OHLCV data points up to a specified limit,
        ordered by timestamp ascending, suitable for feature calculation.
        """
        table_name = self._get_table_name(f"{timeframe}min")
        query = queries.ohlcv_repo["get_ohlcv_data_for_features"].format(table_name=table_name)
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, limit)
            logger.debug(f"Fetched {len(rows)} recent {timeframe} candles for {instrument_id} for feature calculation.")
            return [OHLCVData.model_validate(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for features for {instrument_id} ({timeframe}): {e}")
            raise

    async def check_data_availability(self, timeframe: str, start_time: datetime, end_time: datetime) -> int:
        """
        Check data availability for a given timeframe and time range.
        Returns the count of candles available in the specified period.
        """
        table_name = self._get_table_name(timeframe)
        query = queries.ohlcv_repo["check_data_availability"].format(table_name=table_name)
        try:
            row = await self.db_manager.fetch_row(query, start_time, end_time)
            candle_count = row["candle_count"] if row else 0
            logger.debug(
                f"Data availability check for {timeframe}: {candle_count} candles between {start_time} and {end_time}"
            )
            return candle_count
        except Exception as e:
            logger.error(f"Error checking data availability for {timeframe}: {e}")
            raise
