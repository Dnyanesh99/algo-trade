import asyncio
from datetime import datetime
from typing import Any, Optional

import asyncpg

from src.database.db_utils import db_manager
from src.database.models import FeatureData
from src.utils.logger import LOGGER as logger


class FeatureRepository:
    """
    Repository for interacting with calculated features in the TimescaleDB.
    Provides methods for inserting and fetching feature data.
    """

    def __init__(self):
        self.db_manager = db_manager

    async def insert_features(self, instrument_id: int, timeframe: str, features_data: list[FeatureData]) -> str:
        """
        Inserts calculated features data into the features table.

        Args:
            instrument_id (int): The ID of the instrument.
            timeframe (str): The timeframe (e.g., '5min', '15min', '60min').
            features_data (List[FeatureData]): List of FeatureData objects.

        Returns:
            str: Command status from the database.
        """
        query = """
            INSERT INTO features (ts, instrument_id, timeframe, feature_name, feature_value)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (instrument_id, timeframe, feature_name, ts) DO UPDATE
            SET feature_value = EXCLUDED.feature_value
        """
        records = [
            (d.ts, instrument_id, timeframe, d.feature_name, d.feature_value)
            for d in features_data
        ]

        try:
            async with self.db_manager.transaction() as conn:
                # Using executemany for batch insert/update
                await conn.executemany(query, records)
                logger.info(f"Inserted/Updated {len(features_data)} feature records for {instrument_id} ({timeframe}).")
                return "INSERT OK"
        except Exception as e:
            logger.error(f"Error inserting features data for {instrument_id} ({timeframe}): {e}")
            raise

    async def get_features(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> list[asyncpg.Record]:
        """
        Fetches features data for a given instrument and timeframe within a time range.
        """
        query = """
            SELECT ts, feature_name, feature_value
            FROM features
            WHERE instrument_id = $1 AND timeframe = $2 AND ts BETWEEN $3 AND $4
            ORDER BY ts ASC, feature_name ASC
        """
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe, start_time, end_time)
            logger.debug(
                f"Fetched {len(rows)} feature records for {instrument_id} ({timeframe}) from {start_time} to {end_time}."
            )
            return rows
        except Exception as e:
            logger.error(f"Error fetching features data for {instrument_id} ({timeframe}): {e}")
            raise

    async def get_latest_features(
        self, instrument_id: int, timeframe: str, num_features: Optional[int] = None
    ) -> list[asyncpg.Record]:
        """
        Fetches the latest features for a given instrument and timeframe.
        If num_features is specified, returns that many latest feature sets.
        Otherwise, returns all features for the latest timestamp.
        """
        if num_features:
            # This query fetches the latest 'num_features' entries, assuming each entry is a feature value
            # If you need a complete set of features for the *latest timestamp*, a different query is needed.
            query = """
                SELECT ts, feature_name, feature_value
                FROM features
                WHERE instrument_id = $1 AND timeframe = $2
                ORDER BY ts DESC
                LIMIT $3
            """
            try:
                rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe, num_features)
                rows.reverse()  # Return in chronological order
                logger.debug(f"Fetched {len(rows)} latest feature records for {instrument_id} ({timeframe}).")
                return rows
            except Exception as e:
                logger.error(f"Error fetching latest features for {instrument_id} ({timeframe}): {e}")
                raise
        else:
            # Fetch all features for the single latest timestamp
            query = """
                SELECT ts, feature_name, feature_value
                FROM features
                WHERE instrument_id = $1 AND timeframe = $2 AND ts = (
                    SELECT MAX(ts) FROM features WHERE instrument_id = $1 AND timeframe = $2
                )
                ORDER BY feature_name ASC
            """
            try:
                rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe)
                logger.debug(f"Fetched all features for the latest timestamp for {instrument_id} ({timeframe}).")
                return rows
            except Exception as e:
                logger.error(f"Error fetching latest features for {instrument_id} ({timeframe}): {e}")
                raise