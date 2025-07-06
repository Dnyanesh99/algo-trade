from datetime import datetime
from typing import Optional

from src.database.db_utils import db_manager
from src.database.models import SignalData
from src.utils.logger import LOGGER as logger


class SignalRepository:
    """
    Repository for interacting with trading signals data in the TimescaleDB.
    Provides methods for inserting and fetching signal data.
    """

    def __init__(self) -> None:
        self.db_manager = db_manager

    async def insert_signal(self, instrument_id: int, signal_data: SignalData) -> str:
        """
        Inserts a single trading signal into the signals table.

        Args:
            instrument_id (int): The ID of the instrument.
            signal_data (SignalData): Pydantic model containing signal details.

        Returns:
            str: Command status from the database.
        """
        query = """
            INSERT INTO signals (ts, instrument_id, signal_type, direction, confidence_score, source_feature_name, price_at_signal, source_feature_value, details)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (instrument_id, signal_type, ts) DO UPDATE
            SET direction = EXCLUDED.direction,
                confidence_score = EXCLUDED.confidence_score,
                source_feature_name = EXCLUDED.source_feature_name,
                price_at_signal = EXCLUDED.price_at_signal,
                source_feature_value = EXCLUDED.source_feature_value,
                details = EXCLUDED.details
        """
        try:
            async with self.db_manager.transaction() as conn:
                status = await conn.execute(
                    query,
                    signal_data.ts,
                    instrument_id,
                    signal_data.signal_type,
                    signal_data.direction,
                    signal_data.confidence_score,
                    signal_data.source_feature_name,
                    signal_data.price_at_signal,
                    signal_data.source_feature_value,
                    signal_data.details,
                )
                logger.info(
                    f"Inserted/Updated signal for {instrument_id} ({signal_data.signal_type}). Status: {status}"
                )
                return str(status)
        except Exception as e:
            logger.error(f"Error inserting signal data for {instrument_id}: {e}")
            raise

    async def get_signals(
        self, instrument_id: int, start_time: datetime, end_time: datetime, signal_type: Optional[str] = None
    ) -> list[SignalData]:
        """
        Fetches signals data for a given instrument within a time range, optionally filtered by signal type.
        """
        query = """
            SELECT ts, signal_type, direction, confidence_score, source_feature_name, price_at_signal, source_feature_value, details
            FROM signals
            WHERE instrument_id = $1 AND ts BETWEEN $2 AND $3
        """
        params = [instrument_id, start_time, end_time]

        if signal_type:
            query += " AND signal_type = $4"
            params.append(signal_type)

        query += " ORDER BY ts ASC"

        try:
            rows = await self.db_manager.fetch_rows(query, *params)
            logger.debug(f"Fetched {len(rows)} signal records for {instrument_id} from {start_time} to {end_time}.")
            return [SignalData.model_validate(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching signals data for {instrument_id}: {e}")
            raise

    async def get_signal_history(self, instrument_id: int, limit: int = 100) -> list[SignalData]:
        """
        Fetches the most recent signal history for a given instrument.
        """
        query = """
            SELECT ts, signal_type, direction, confidence_score, price_at_signal
            FROM signals
            WHERE instrument_id = $1
            ORDER BY ts DESC
            LIMIT $2
        """
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, limit)
            logger.debug(f"Fetched {len(rows)} latest signal records for {instrument_id}.")
            return [SignalData.model_validate(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching signal history for {instrument_id}: {e}")
            raise
