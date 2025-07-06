from datetime import datetime
from typing import Any

from src.database.db_utils import db_manager
from src.database.models import LabelData
from src.utils.logger import LOGGER as logger


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
        query = """
            INSERT INTO labels (ts, instrument_id, timeframe, label, tp_price, sl_price, exit_price, exit_reason, exit_bar_offset, barrier_return, max_favorable_excursion, max_adverse_excursion, risk_reward_ratio, volatility_at_entry)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (instrument_id, timeframe, ts) DO UPDATE
            SET label = EXCLUDED.label,
                tp_price = EXCLUDED.tp_price,
                sl_price = EXCLUDED.sl_price,
                exit_price = EXCLUDED.exit_price,
                exit_reason = EXCLUDED.exit_reason,
                exit_bar_offset = EXCLUDED.exit_bar_offset,
                barrier_return = EXCLUDED.barrier_return,
                max_favorable_excursion = EXCLUDED.max_favorable_excursion,
                max_adverse_excursion = EXCLUDED.max_adverse_excursion,
                risk_reward_ratio = EXCLUDED.risk_reward_ratio,
                volatility_at_entry = EXCLUDED.volatility_at_entry
        """
        records = [
            (
                d.ts,
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
        query = """
            SELECT ts, timeframe, label, tp_price, sl_price, exit_price, exit_reason, exit_bar_offset, barrier_return, max_favorable_excursion, max_adverse_excursion, risk_reward_ratio, volatility_at_entry
            FROM labels
            WHERE instrument_id = $1 AND timeframe = $2 AND ts BETWEEN $3 AND $4
            ORDER BY ts ASC
        """
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe, start_time, end_time)
            logger.debug(
                f"Fetched {len(rows)} label records for {instrument_id} ({timeframe}) from {start_time} to {end_time}."
            )
            return [LabelData.model_validate(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching labels data for {instrument_id} ({timeframe}): {e}")
            raise

    async def get_label_statistics(self, instrument_id: int, timeframe: str) -> dict[str, Any]:
        """
        Calculates and returns statistics about labels for a given instrument and timeframe.
        """
        query = """
            SELECT
                label,
                COUNT(*) as count,
                AVG(barrier_return) as avg_return,
                STDDEV(barrier_return) as std_return
            FROM labels
            WHERE instrument_id = $1 AND timeframe = $2
            GROUP BY label
        """
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
