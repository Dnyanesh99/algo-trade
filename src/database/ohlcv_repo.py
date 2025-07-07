from datetime import datetime
from typing import Optional

from src.database.db_utils import db_manager
from src.database.models import OHLCVData
from src.utils.logger import LOGGER as logger


class OHLCVRepository:
    """
    Repository for interacting with OHLCV data in the TimescaleDB.
    Provides methods for inserting, fetching, and managing OHLCV data across different timeframes.
    """

    def __init__(self) -> None:
        self.db_manager = db_manager

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
        table_name = f"ohlcv_{timeframe}"
        query = f"""
            INSERT INTO {table_name} (ts, instrument_id, open, high, low, close, volume, oi)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (instrument_id, ts) DO UPDATE
            SET open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                oi = EXCLUDED.oi
        """
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
        table_name = f"ohlcv_{timeframe}"
        query = f"SELECT * FROM {table_name} WHERE instrument_id = $1 AND ts BETWEEN $2 AND $3 ORDER BY ts ASC"
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
        table_name = f"ohlcv_{timeframe}"
        query = f"SELECT * FROM {table_name} WHERE instrument_id = $1 ORDER BY ts DESC LIMIT 1"
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
        table_name = f"ohlcv_{timeframe}min"
        query = f"SELECT ts, open, high, low, close, volume FROM {table_name} WHERE instrument_id = $1 ORDER BY ts DESC LIMIT $2"
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, limit)
            logger.debug(f"Fetched {len(rows)} recent {timeframe} candles for {instrument_id} for feature calculation.")
            return [OHLCVData.model_validate(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for features for {instrument_id} ({timeframe}): {e}")
            raise
