from datetime import datetime
from typing import Optional

from src.database.db_utils import db_manager
from src.database.models import SignalData
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

queries = ConfigLoader().get_queries()


class SignalRepository:
    """
    Repository for interacting with trading signals data in the TimescaleDB.
    Provides methods for inserting and fetching signal data.
    """

    def __init__(self) -> None:
        self.db_manager = db_manager
        logger.info("SignalRepository initialized.")

    async def insert_signal(self, instrument_id: int, signal_data: SignalData) -> None:
        """
        Inserts a single trading signal into the signals table.

        Raises:
            ValueError: If input arguments are invalid.
            RuntimeError: If the database insertion fails.
        """
        if not isinstance(instrument_id, int) or not isinstance(signal_data, SignalData):
            raise ValueError("Invalid instrument_id or SignalData object provided.")

        query = queries.signal_repo["insert_signal"]
        logger.debug(
            f"Inserting signal for instrument {instrument_id}: {signal_data.direction} {signal_data.signal_type}"
        )
        try:
            await self.db_manager.execute(
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
            logger.info(f"Successfully inserted signal for instrument {instrument_id} at {signal_data.ts}.")
        except Exception as e:
            logger.error(f"Database error inserting signal for instrument {instrument_id}: {e}", exc_info=True)
            raise RuntimeError("Failed to insert signal data.") from e

    async def get_signals(
        self, instrument_id: int, start_time: datetime, end_time: datetime, signal_type: Optional[str] = None
    ) -> list[SignalData]:
        """
        Fetches signals data for a given instrument within a time range, optionally filtered by signal type.

        Raises:
            RuntimeError: If the database query fails.
        """
        query = queries.signal_repo["get_signals"]
        params = [instrument_id, start_time, end_time]

        if signal_type:
            query += " AND signal_type = $4"
            params.append(signal_type)

        query += " ORDER BY ts ASC"
        logger.debug(f"Fetching signals for instrument {instrument_id} with params: {params}")

        try:
            rows = await self.db_manager.fetch_rows(query, *params)
            logger.info(f"Fetched {len(rows)} signal records for instrument {instrument_id}.")
            return [SignalData.model_validate(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Database error fetching signals for instrument {instrument_id}: {e}", exc_info=True)
            raise RuntimeError("Failed to fetch signal data.") from e

    async def get_signal_history(self, instrument_id: int, limit: int = 100) -> list[SignalData]:
        """
        Fetches the most recent signal history for a given instrument.

        Raises:
            RuntimeError: If the database query fails.
        """
        if limit <= 0:
            logger.warning(f"get_signal_history called with non-positive limit: {limit}. Returning empty list.")
            return []

        query = queries.signal_repo["get_signal_history"]
        logger.debug(f"Fetching latest {limit} signals for instrument {instrument_id}.")
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, limit)
            logger.info(f"Fetched {len(rows)} latest signal records for instrument {instrument_id}.")
            return [SignalData.model_validate(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Database error fetching signal history for {instrument_id}: {e}", exc_info=True)
            raise RuntimeError("Failed to fetch signal history.") from e
