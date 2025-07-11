from datetime import datetime
from typing import Any

import pytz

from src.database.db_utils import db_manager
from src.database.models import LabelData
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

config = ConfigLoader().get_config()
queries = ConfigLoader().get_queries()

# Get the timezone from the config and create a timezone object
app_timezone = pytz.timezone(config.system.timezone) if config.system else pytz.UTC


class LabelRepository:
    """
    Repository for interacting with labels data in the TimescaleDB.
    Provides methods for inserting and fetching label data.
    """

    def __init__(self) -> None:
        self.db_manager = db_manager

    async def insert_labels(self, instrument_id: int, labels_data: list[LabelData]) -> str:
        """
        Inserts labels data into the labels table.

        Args:
            instrument_id (int): The ID of the instrument.
            labels_data (List[LabelData]): List of LabelData objects.

        Returns:
            str: Command status from the database.
        """
        query = queries.label_repo["insert_labels"]
        records = [
            (
                app_timezone.localize(d.ts) if d.ts.tzinfo is None else d.ts,
                instrument_id,
                d.timeframe,
                d.label,
                d.tp_price,
                d.sl_price,
                d.exit_price,
                d.exit_reason,
                d.exit_bar_offset,
                d.barrier_return,
                d.max_favorable_excursion,
                d.max_adverse_excursion,
                d.risk_reward_ratio,
                d.volatility_at_entry,
            )
            for d in labels_data
        ]

        try:
            async with self.db_manager.transaction() as conn:
                await conn.executemany(query, records)
                logger.info(f"Inserted/Updated {len(labels_data)} label records for {instrument_id}.")
                return "INSERT OK"
        except Exception as e:
            logger.error(f"Error inserting labels data for {instrument_id}: {e}")
            raise

    async def get_labels(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> list[LabelData]:
        """
        Fetches labels data for a given instrument and timeframe within a time range.
        """
        query = queries.label_repo["get_labels"]
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe, start_time, end_time)
            logger.debug(
                f"Fetched {len(rows)} label records for {instrument_id} ({timeframe}) from {start_time} to {end_time}."
            )
            return [LabelData.model_validate(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching labels data for {instrument_id} ({timeframe}): {e}")
            raise

    async def get_label_statistics(self, instrument_id: int, timeframe: str) -> dict[str, Any]:
        """
        Calculates and returns statistics about labels for a given instrument and timeframe.
        """
        query = queries.label_repo["get_label_statistics"]
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe)
            stats: dict[str, Any] = {"total_labels": 0}
            for row in rows:
                label_name = "BUY" if row["label"] == 1 else ("SELL" if row["label"] == -1 else "NEUTRAL")
                stats[label_name] = {
                    "count": row["count"],
                    "avg_return": row["avg_return"],
                    "std_return": row["std_return"],
                }
                stats["total_labels"] += row["count"]
            logger.debug(f"Fetched label statistics for {instrument_id} ({timeframe}): {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error fetching label statistics for {instrument_id} ({timeframe}): {e}")
            raise
