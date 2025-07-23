"""
Production-grade repository for managing processing state persistence.
Handles CRUD operations for processing state tracking with proper error handling.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from src.database.db_utils import db_manager
from src.utils.config_loader import ConfigLoader

config = ConfigLoader().get_config()
queries = ConfigLoader().get_queries()


class ProcessingState(BaseModel):
    """Model for processing state data."""

    instrument_id: int
    processing_type: str
    completed_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    status: str = "completed"


class DataRange(BaseModel):
    """Model for data range information."""

    instrument_id: int
    timeframe: str
    earliest_ts: datetime
    latest_ts: datetime


class ProcessingStateRepository:
    """
    Repository for managing processing state and data range persistence.

    Responsibilities:
    - Persist processing completion states to database
    - Track data ranges for gap detection
    - Provide efficient queries for state checks
    - Handle state recovery on system restart
    """

    def __init__(self) -> None:
        """Initialize the repository with database manager."""
        self.db_manager = db_manager
        logger.info("ProcessingStateRepository initialized")

    async def mark_processing_complete(
        self,
        instrument_id: int,
        process_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "completed",
    ) -> None:
        """
        Mark a processing step as complete in the database.

        Args:
            instrument_id: The ID of the instrument being processed
            process_type: The type of processing being completed
            metadata: Optional metadata about the processing
            status: The status of the processing (completed, failed, in_progress)

        Raises:
            Exception: If the database operation fails
        """
        query = queries.processing_state_repo["insert_processing_state"]
        logger.debug(f"Marking process '{process_type}' as {status} for instrument {instrument_id}.")
        try:
            # Convert metadata dict to JSON string for JSONB column
            json_metadata = json.dumps(metadata) if metadata is not None else "{}"

            await self.db_manager.execute(query, instrument_id, process_type, datetime.now(), json_metadata, status)
            logger.info(f"Successfully marked '{process_type}' as {status} for instrument {instrument_id}.")
        except Exception as e:
            logger.error(
                f"Database error marking '{process_type}' as {status} for instrument {instrument_id}: {str(e)}",
                exc_info=True,
            )
            raise

    async def mark_processing_failed(
        self, instrument_id: int, process_type: str, error_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark a processing step as failed in the database.

        Args:
            instrument_id: The ID of the instrument being processed
            process_type: The type of processing that failed
            error_metadata: Optional metadata about the failure
        """
        await self.mark_processing_complete(
            instrument_id=instrument_id, process_type=process_type, metadata=error_metadata, status="failed"
        )

    async def mark_processing_in_progress(
        self, instrument_id: int, process_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark a processing step as in progress in the database.

        Args:
            instrument_id: The ID of the instrument being processed
            process_type: The type of processing being started
            metadata: Optional metadata about the processing
        """
        await self.mark_processing_complete(
            instrument_id=instrument_id, process_type=process_type, metadata=metadata, status="in_progress"
        )

    async def is_processing_complete(self, instrument_id: int, process_type: str) -> bool:
        """
        Check if a processing step is complete for an instrument.

        Args:
            instrument_id: The ID of the instrument to check
            process_type: The type of processing to check

        Returns:
            bool: True if processing is complete, False otherwise
        """
        query = queries.processing_state_repo["check_processing_complete"]
        logger.debug(f"Checking completion status of '{process_type}' for instrument {instrument_id}.")
        try:
            result = await self.db_manager.fetchval(query, instrument_id, process_type)
            is_complete = bool(result)
            logger.debug(
                f"Processing status for '{process_type}' on instrument {instrument_id}: "
                f"{'Complete' if is_complete else 'Incomplete'}."
            )
            return is_complete
        except Exception as e:
            logger.error(f"Error checking processing status: {e}", exc_info=True)
            return False

    async def get_last_processing_timestamp(self, instrument_id: int, process_type: str) -> Optional[datetime]:
        """
        Get the timestamp of last successful processing for an instrument.

        Args:
            instrument_id: The ID of the instrument to check
            process_type: The type of processing to check

        Returns:
            Optional[datetime]: Timestamp of last processing or None if never processed
        """
        query = queries.processing_state_repo["get_last_processing_timestamp"]
        try:
            result = await self.db_manager.fetchval(query, instrument_id, process_type)
            return result if result else None
        except Exception as e:
            logger.error(f"Error fetching last processing timestamp: {e}", exc_info=True)
            return None

    async def set_data_range(self, instrument_id: int, timeframe: str, start_ts: datetime, end_ts: datetime) -> None:
        """
        Track data ranges for gap detection.

        Args:
            instrument_id: The ID of the instrument
            timeframe: The timeframe of the data
            start_ts: The start timestamp of the data range
            end_ts: The end timestamp of the data range
        """
        query = queries.processing_state_repo["upsert_data_range"]
        try:
            await self.db_manager.execute(query, instrument_id, timeframe, start_ts, end_ts, datetime.now())
            logger.info(f"Updated data range for {instrument_id} ({timeframe}): {start_ts} -> {end_ts}")
        except Exception as e:
            logger.error(f"Error setting data range: {e}", exc_info=True)
            raise

    async def get_data_range(self, instrument_id: int, timeframe: str) -> Optional[DataRange]:
        """
        Get known data range for an instrument/timeframe.

        Args:
            instrument_id: The ID of the instrument
            timeframe: The timeframe to check

        Returns:
            Optional[DataRange]: The data range information or None if not found
        """
        query = queries.processing_state_repo["get_data_range"]
        try:
            result = await self.db_manager.fetch_one(query, instrument_id, timeframe)
            if result:
                return DataRange(
                    instrument_id=instrument_id,
                    timeframe=timeframe,
                    earliest_ts=result["earliest_ts"],
                    latest_ts=result["latest_ts"],
                )
            return None
        except Exception as e:
            logger.error(f"Error fetching data range: {e}", exc_info=True)
            return None

    async def get_all_processing_states(self) -> List[ProcessingState]:
        """
        Get all processing states for system state hydration.

        Returns:
            List[ProcessingState]: List of all processing states
        """
        query = queries.processing_state_repo["get_all_processing_states"]
        try:
            rows = await self.db_manager.fetch_all(query)
            logger.info(f"Fetched {len(rows)} processing state records.")
            return [
                ProcessingState(
                    instrument_id=row["instrument_id"],
                    processing_type=row["processing_type"],
                    completed_at=row["completed_at"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                    status=row.get("status", "completed"),
                )
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error fetching all processing states: {e}", exc_info=True)
            return []

    async def get_all_data_ranges(self) -> List[DataRange]:
        """
        Get all data ranges for system state hydration.

        Returns:
            List[DataRange]: List of all data ranges
        """
        query = queries.processing_state_repo["get_all_data_ranges"]
        try:
            rows = await self.db_manager.fetch_all(query)
            logger.info(f"Fetched {len(rows)} data range records.")
            return [
                DataRange(
                    instrument_id=row["instrument_id"],
                    timeframe=row["timeframe"],
                    earliest_ts=row["earliest_ts"],
                    latest_ts=row["latest_ts"],
                )
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error fetching all data ranges: {e}", exc_info=True)
            return []

    async def has_actual_data_for_processing(self, instrument_id: int, process_type: str, config: Any = None) -> bool:
        """
        Check if actual data exists for a specific processing type.

        Args:
            instrument_id: The ID of the instrument to check
            process_type: The type of processing to check
            config: Optional configuration object for thresholds

        Returns:
            bool: True if sufficient data exists, False otherwise
        """
        try:
            query_name = f"has_actual_data_{process_type}"
            if query_name not in queries.processing_state_repo:
                logger.error(f"No query defined for data check: {query_name}")
                return False

            query = queries.processing_state_repo[query_name]

            # Handle timeframe-specific queries
            if "{timeframe}" in query:
                if not config or not hasattr(config.trading, f"{process_type}_timeframes"):
                    logger.error(f"No timeframes configured for {process_type}")
                    return False

                timeframes = getattr(config.trading, f"{process_type}_timeframes")
                for tf in timeframes:
                    formatted_query = query.format(timeframe=tf)
                    count = await self.db_manager.fetchval(formatted_query, instrument_id)
                    min_count = getattr(config.processing_control, f"min_candles_for_{process_type}", 100)
                    if not count or count < min_count:
                        return False
                return True
            count = await self.db_manager.fetchval(query, instrument_id)
            min_count = getattr(config.processing_control, f"min_candles_for_{process_type}", 100)
            return bool(count and count >= min_count)

        except Exception as e:
            logger.error(f"Error checking actual data availability: {e}", exc_info=True)
            return False

    async def reset_processing_state(
        self, instrument_id: Optional[int] = None, process_type: Optional[str] = None
    ) -> None:
        """
        Reset processing state for debugging or reprocessing.

        Args:
            instrument_id: Optional instrument ID to reset
            process_type: Optional process type to reset
        """
        try:
            if instrument_id and process_type:
                query = queries.processing_state_repo["delete_specific_processing_state"]
                await self.db_manager.execute(query, instrument_id, process_type)
            elif instrument_id:
                query = queries.processing_state_repo["delete_processing_state_by_instrument"]
                await self.db_manager.execute(query, instrument_id)
            else:
                query = queries.processing_state_repo["delete_all_processing_state"]
                await self.db_manager.execute(query)

            logger.info("Successfully reset processing state")
        except Exception as e:
            logger.error(f"Error resetting processing state: {e}", exc_info=True)
            raise
