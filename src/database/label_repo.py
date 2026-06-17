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
        if not config.system or not config.system.timezone:
            logger.critical("CRITICAL: System timezone is not configured in config.yaml.")
            raise ValueError("System timezone must be configured for the LabelRepository.")
        self.app_timezone = pytz.timezone(config.system.timezone)
        logger.info(f"LabelRepository initialized with timezone '{self.app_timezone}'.")

    async def insert_labels(self, instrument_id: int, labels_data: list[LabelData]) -> None:
        """
        Inserts labels data into the labels table.

        Args:
            instrument_id: The ID of the instrument.
            labels_data: A list of LabelData objects.

        Raises:
            ValueError: If input arguments are invalid.
            RuntimeError: If the database insertion fails.
        """
        if not labels_data:
            logger.warning(f"insert_labels called with no data for instrument {instrument_id}. Nothing to do.")
            return
        if not isinstance(instrument_id, int):
            raise ValueError("Invalid instrument_id type.")

        query = queries.label_repo["insert_labels"]
        records = [
            (
                self.app_timezone.localize(d.ts) if d.ts.tzinfo is None else d.ts,
                instrument_id,
                d.timeframe,
                d.label,
                d.tp_price,
                d.sl_price,
                d.exit_price,
                d.exit_reason,
                d.exit_bar_offset,
                d.barrier_return,
                d.path_adjusted_return,
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
            logger.info(f"Successfully inserted/updated {len(records)} label records for instrument {instrument_id}.")
        except Exception as e:
            logger.error(f"Database error inserting labels for instrument {instrument_id}: {e}", exc_info=True)
            raise RuntimeError("Failed to insert label data.") from e

    async def get_labels(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> list[LabelData]:
        """
        Fetches labels data for a given instrument and timeframe within a time range.

        Raises:
            RuntimeError: If the database query fails.
        """
        query = queries.label_repo["get_labels"]
        logger.debug(
            f"Fetching labels for instrument {instrument_id} ({timeframe}) between {start_time} and {end_time}."
        )
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe, start_time, end_time)
            logger.info(f"Fetched {len(rows)} label records for instrument {instrument_id} ({timeframe}).")
            return [LabelData.model_validate(dict(row)) for row in rows]
        except Exception as e:
            logger.error(
                f"Database error fetching labels for instrument {instrument_id} ({timeframe}): {e}", exc_info=True
            )
            raise RuntimeError("Failed to fetch label data.") from e

    async def get_label_statistics(self, instrument_id: int, timeframe: str) -> dict[str, Any]:
        """
        Calculates and returns statistics about labels for a given instrument and timeframe.

        Raises:
            RuntimeError: If the database query fails.
        """
        query = queries.label_repo["get_label_statistics"]
        logger.debug(f"Fetching label statistics for instrument {instrument_id} ({timeframe}).")
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe)
            stats: dict[str, Any] = {"total_labels": 0, "BUY": {}, "SELL": {}, "NEUTRAL": {}}
            total_labels = 0
            for row in rows:
                label_val = row.get("label")
                count = row.get("count", 0)
                label_name = "BUY" if label_val == 1 else ("SELL" if label_val == -1 else "NEUTRAL")
                stats[label_name] = {
                    "count": count,
                    "avg_return": row.get("avg_return"),
                    "std_return": row.get("std_return"),
                }
                total_labels += count
            stats["total_labels"] = total_labels

            logger.info(f"Successfully fetched label statistics for instrument {instrument_id} ({timeframe}).")
            return stats
        except Exception as e:
            logger.error(
                f"Database error fetching label statistics for {instrument_id} ({timeframe}): {e}", exc_info=True
            )
            raise RuntimeError("Failed to fetch label statistics.") from e
