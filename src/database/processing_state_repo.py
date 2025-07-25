"""
Production-grade repository for managing processing state persistence.
Handles CRUD operations for processing state tracking with proper error handling.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Optional

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
    metadata: Optional[dict[str, Any]] = None
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
    - Validate processing dependencies
    """

    # Processing dependency mapping (simple and minimal)
    _PROCESSING_DEPENDENCIES = {
        "historical_fetch": [],  # No dependencies
        "aggregation": ["historical_fetch"],
        "features": ["aggregation"],
        "labeling": ["features"],
        "training": ["labeling"],
    }

    def __init__(self) -> None:
        """Initialize the repository with database manager."""
        self.db_manager = db_manager
        # Whitelist of allowed timeframes to prevent SQL injection
        self._allowed_timeframes = {"1", "3", "5", "10", "15", "30", "60", "240", "1440"}
        logger.info("ProcessingStateRepository initialized")

    def _validate_timeframe_safe(self, timeframe: str) -> bool:
        """
        Safely validate timeframe parameter against a whitelist to prevent SQL injection.

        Args:
            timeframe: The timeframe string to validate

        Returns:
            True if timeframe is valid, False otherwise
        """
        return timeframe in self._allowed_timeframes

    def _create_safe_timeframe_query(self, template_query: str, table_name: str, format_params: dict[str, Any]) -> str:
        """
        Create a safe query by replacing timeframe placeholders with validated table names.

        Args:
            template_query: Query template with {timeframe} placeholder
            table_name: Validated table name (e.g., "ohlcv_15min")
            format_params: Other format parameters

        Returns:
            Safe SQL query string
        """
        # Replace {timeframe} with the validated table name in the template
        safe_query = template_query.replace("{timeframe}", table_name.split("_")[1].replace("min", ""))

        # Apply other safe format parameters
        safe_format_params = {k: v for k, v in format_params.items() if k != "timeframe"}

        return safe_query.format(**safe_format_params)

    async def mark_processing_complete(
        self,
        instrument_id: int,
        process_type: str,
        metadata: Optional[dict[str, Any]] = None,
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
        self, instrument_id: int, process_type: str, error_metadata: Optional[dict[str, Any]] = None
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
        self, instrument_id: int, process_type: str, metadata: Optional[dict[str, Any]] = None
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

    async def is_processing_failed(self, instrument_id: int, process_type: str) -> bool:
        """
        Check if a processing step has failed for an instrument.

        Args:
            instrument_id: The ID of the instrument to check
            process_type: The type of processing to check

        Returns:
            bool: True if processing has failed, False otherwise
        """
        query = queries.processing_state_repo["check_processing_failed"]
        logger.debug(f"Checking failure status of '{process_type}' for instrument {instrument_id}.")
        try:
            result = await self.db_manager.fetchval(query, instrument_id, process_type)
            is_failed = bool(result)
            logger.debug(
                f"Processing failure status for '{process_type}' on instrument {instrument_id}: "
                f"{'Failed' if is_failed else 'Not Failed'}."
            )
            return is_failed
        except Exception as e:
            logger.error(f"Error checking processing failure status: {e}", exc_info=True)
            return False

    async def get_processing_status(self, instrument_id: int, process_type: str) -> Optional[dict[str, Any]]:
        """
        Get detailed processing status for an instrument and process type.

        Args:
            instrument_id: The ID of the instrument to check
            process_type: The type of processing to check

        Returns:
            Optional[dict]: Processing status details or None if not found
        """
        query = queries.processing_state_repo["get_processing_status"]
        logger.debug(f"Getting detailed status of '{process_type}' for instrument {instrument_id}.")
        try:
            result = await self.db_manager.fetch_row(query, instrument_id, process_type)
            if result:
                status_info = {
                    "status": result["status"],
                    "completed_at": result["completed_at"],
                    "metadata": json.loads(result["metadata"]) if result["metadata"] else None,
                }
                logger.debug(
                    f"Processing status details for '{process_type}' on instrument {instrument_id}: {status_info['status']}"
                )
                return status_info
            return None
        except Exception as e:
            logger.error(f"Error getting processing status details: {e}", exc_info=True)
            return None

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
            result = await self.db_manager.fetch_row(query, instrument_id, timeframe)
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

    async def get_all_processing_states(self) -> list[ProcessingState]:
        """
        Get all processing states for system state hydration.

        Returns:
            List[ProcessingState]: List of all processing states
        """
        query = queries.processing_state_repo["get_all_processing_states"]
        try:
            rows = await self.db_manager.fetch_rows(query)
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

    async def get_all_data_ranges(self) -> list[DataRange]:
        """
        Get all data ranges for system state hydration.

        Returns:
            List[DataRange]: List of all data ranges
        """
        query = queries.processing_state_repo["get_all_data_ranges"]
        try:
            rows = await self.db_manager.fetch_rows(query)
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

            # Prepare format parameters from config
            format_params = {}
            if config and hasattr(config, "processing_control"):
                processing_control = config.processing_control
                # Add time interval configurations
                format_params.update(
                    {
                        "historical_data_availability_days": getattr(
                            processing_control, "historical_data_availability_days", 7
                        ),
                        "aggregation_data_availability_days": getattr(
                            processing_control, "aggregation_data_availability_days", 7
                        ),
                        "features_data_availability_days": getattr(
                            processing_control, "features_data_availability_days", 3
                        ),
                        "labeling_data_availability_days": getattr(
                            processing_control, "labeling_data_availability_days", 5
                        ),
                        "existing_features_availability_days": getattr(
                            processing_control, "existing_features_availability_days", 2
                        ),
                        "existing_labels_availability_days": getattr(
                            processing_control, "existing_labels_availability_days", 3
                        ),
                    }
                )

            # Handle timeframe-specific queries
            if "{timeframe}" in query:
                # Map process types to their corresponding configuration attribute names
                timeframe_config_map = {
                    "features": "feature_timeframes",
                    "labeling": "labeling_timeframes",
                    "aggregation": "aggregation_timeframes",
                }

                config_attr = timeframe_config_map.get(process_type, f"{process_type}_timeframes")

                if not config or not hasattr(config.trading, config_attr):
                    logger.error(f"No timeframes configured for {process_type} (looking for {config_attr})")
                    return False

                timeframes = getattr(config.trading, config_attr)
                for tf in timeframes:
                    # Validate timeframe to prevent SQL injection
                    if not self._validate_timeframe_safe(str(tf)):
                        logger.error(f"Invalid timeframe value: {tf}. Skipping.")
                        continue

                    # Use safe table name lookup instead of string formatting
                    table_name = f"ohlcv_{tf}min"

                    # Create safe query with validated table name
                    safe_query = self._create_safe_timeframe_query(query, table_name, format_params)
                    count = await self.db_manager.fetchval(safe_query, instrument_id)
                    min_count = getattr(config.processing_control, f"min_candles_for_{process_type}", 100)
                    if not count or count < min_count:
                        return False
                return True

            # Format query with config parameters (non-timeframe queries are safe)
            formatted_query = query.format(**format_params)
            count = await self.db_manager.fetchval(formatted_query, instrument_id)
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
                logger.info(f"Reset processing state for instrument {instrument_id}, process {process_type}")
            elif instrument_id:
                query = queries.processing_state_repo["delete_processing_state_by_instrument"]
                await self.db_manager.execute(query, instrument_id)
                logger.info(f"Reset all processing states for instrument {instrument_id}")
            else:
                query = queries.processing_state_repo["delete_all_processing_state"]
                await self.db_manager.execute(query)
                logger.info("Reset all processing states")

        except Exception as e:
            logger.error(f"Error resetting processing state: {e}", exc_info=True)
            raise

    async def reset_failed_processing_state(self, instrument_id: int, process_type: str) -> bool:
        """
        Reset only failed processing state to allow retry.

        Args:
            instrument_id: The ID of the instrument
            process_type: The type of processing to reset

        Returns:
            bool: True if reset was successful, False otherwise
        """
        try:
            # Check if the process is actually in failed state
            is_failed = await self.is_processing_failed(instrument_id, process_type)
            if not is_failed:
                logger.warning(
                    f"Process {process_type} for instrument {instrument_id} is not in failed state. No reset needed."
                )
                return False

            # Get failure details for logging
            status_info = await self.get_processing_status(instrument_id, process_type)

            # Reset the failed state
            query = queries.processing_state_repo["delete_specific_processing_state"]
            await self.db_manager.execute(query, instrument_id, process_type)

            logger.info(
                f"✅ Reset failed processing state for instrument {instrument_id}, process {process_type}. "
                f"Previous failure: {status_info.get('metadata', {}).get('error', 'Unknown') if status_info else 'Unknown'}"
            )
            return True

        except Exception as e:
            logger.error(f"Error resetting failed processing state: {e}", exc_info=True)
            return False

    async def is_system_operation_complete(self, operation_type: str) -> bool:
        """
        Check if a system-level operation is complete.

        Args:
            operation_type: The type of system operation to check

        Returns:
            bool: True if operation is complete, False otherwise
        """
        query = queries.processing_state_repo.get("check_system_operation_complete")
        if not query:
            logger.error("Query 'check_system_operation_complete' not found in queries.yaml")
            return False

        logger.debug(f"Checking completion status of system operation '{operation_type}'.")
        try:
            result = await self.db_manager.fetchval(query, operation_type)
            is_complete = bool(result)
            logger.debug(
                f"System operation status for '{operation_type}': {'Complete' if is_complete else 'Incomplete'}."
            )
            return is_complete
        except Exception as e:
            logger.error(f"Error checking system operation status: {e}", exc_info=True)
            return False

    async def mark_system_operation_complete(
        self,
        operation_type: str,
        metadata: Optional[dict[str, Any]] = None,
        status: str = "completed",
    ) -> None:
        """
        Mark a system-level operation as complete.

        Args:
            operation_type: The type of system operation
            metadata: Optional metadata about the operation
            status: The status of the operation (completed, failed, in_progress)

        Raises:
            Exception: If the database operation fails
        """
        query = queries.processing_state_repo.get("upsert_system_operation")
        if not query:
            logger.error("Query 'upsert_system_operation' not found in queries.yaml")
            raise RuntimeError("System operation query not configured")

        logger.debug(f"Marking system operation '{operation_type}' as {status}.")
        try:
            # Convert metadata dict to JSON string for JSONB column
            json_metadata = json.dumps(metadata) if metadata is not None else "{}"

            await self.db_manager.execute(query, operation_type, datetime.now(), json_metadata, status, datetime.now())
            logger.info(f"Successfully marked system operation '{operation_type}' as {status}.")
        except Exception as e:
            logger.error(
                f"Database error marking system operation '{operation_type}' as {status}: {str(e)}",
                exc_info=True,
            )
            raise

    async def validate_processing_dependencies(self, instrument_id: int, process_type: str) -> bool:
        """
        Validate that all required dependencies for a process type are satisfied.

        This method enforces the linear processing flow while allowing intelligent
        skipping when dependencies are disabled but sufficient data exists.

        Args:
            instrument_id: The ID of the instrument to check
            process_type: The process type to validate dependencies for

        Returns:
            bool: True if all dependencies are satisfied, False otherwise
        """
        if process_type not in self._PROCESSING_DEPENDENCIES:
            logger.error(f"Unknown process type for dependency validation: {process_type}")
            return False

        dependencies = self._PROCESSING_DEPENDENCIES[process_type]

        # No dependencies means it can always run
        if not dependencies:
            return True

        import os

        def _get_bool_env(env_var_name: str, default: bool = True) -> bool:
            """Parse boolean environment variables with consistent logic."""
            env_value = os.getenv(env_var_name)
            if env_value is None:
                return default
            return env_value.lower() in ("true", "1", "yes", "on")

        # Map dependency process types to their corresponding environment variables
        enabled_env_map = {
            "historical_fetch": "HISTORICAL_PROCESSING_ENABLED",
            "aggregation": "AGGREGATION_PROCESSING_ENABLED",
            "features": "FEATURE_PROCESSING_ENABLED",
            "labeling": "LABELING_PROCESSING_ENABLED",
            "training": "TRAINING_PROCESSING_ENABLED",
        }

        # Check each dependency
        for dep_process in dependencies:
            try:
                # Check if dependency processing is enabled
                enabled_env_var = enabled_env_map.get(dep_process)
                is_dependency_enabled = _get_bool_env(enabled_env_var, True) if enabled_env_var else True

                if is_dependency_enabled:
                    # Dependency is enabled - check if processing completed successfully
                    is_dep_complete = await self.is_processing_complete(instrument_id, dep_process)
                    if is_dep_complete:
                        logger.debug(f"✅ Dependency {dep_process} completed successfully for {process_type}")
                        continue

                    # Check if dependency failed
                    is_dep_failed = await self.is_processing_failed(instrument_id, dep_process)
                    if is_dep_failed:
                        status_info = await self.get_processing_status(instrument_id, dep_process)
                        error_details = ""
                        if status_info and status_info.get("metadata"):
                            error_details = f" Error: {status_info['metadata'].get('error', 'Unknown error')}"

                        logger.error(
                            f"❌ Cannot process {process_type} for instrument {instrument_id}: "
                            f"dependency {dep_process} FAILED.{error_details}"
                        )
                        return False

                    # Dependency is enabled but not completed yet
                    logger.warning(
                        f"⏳ Cannot process {process_type} for instrument {instrument_id}: "
                        f"dependency {dep_process} has not been completed yet"
                    )
                    return False

                # Dependency is disabled - check if sufficient data exists to proceed
                logger.info(
                    f"🔍 Dependency {dep_process} disabled via {enabled_env_var}=false. "
                    f"Checking if sufficient data exists for {process_type} to proceed..."
                )

                # Check if we have sufficient data for the current process type to run
                # This handles smart processing + force reprocess scenarios
                has_sufficient_data = await self.has_actual_data_for_processing(instrument_id, process_type, config)

                if has_sufficient_data:
                    logger.info(
                        f"✅ Dependency {dep_process} satisfied for {process_type} - "
                        f"disabled but sufficient data exists"
                    )
                    continue

                logger.error(
                    f"❌ Cannot process {process_type} for instrument {instrument_id}: "
                    f"dependency {dep_process} is disabled but insufficient data exists to proceed"
                )
                return False

            except Exception as e:
                logger.error(f"Error checking dependency {dep_process} for {process_type}: {e}")
                logger.error(
                    f"🚫 Blocking {process_type} for instrument {instrument_id} due to dependency check failure"
                )
                return False

        logger.info(f"✅ All dependencies satisfied for {process_type} on instrument {instrument_id}")
        return True

    async def get_instrument_processing_summary(self, instrument_id: int) -> dict[str, Any]:
        """
        Get a comprehensive summary of processing status for an instrument.

        Args:
            instrument_id: The ID of the instrument to check

        Returns:
            dict: Summary of all processing stages and their status
        """
        summary: dict[str, Any] = {
            "instrument_id": instrument_id,
            "processing_stages": {},
            "dependency_chain_status": "unknown",
            "can_proceed_to": [],
            "blocked_processes": [],
        }

        try:
            # Check status of all known process types
            for process_type in self._PROCESSING_DEPENDENCIES:
                status_info = await self.get_processing_status(instrument_id, process_type)
                if status_info:
                    summary["processing_stages"][process_type] = {
                        "status": status_info["status"],
                        "completed_at": status_info["completed_at"].isoformat()
                        if status_info["completed_at"]
                        else None,
                        "error": status_info.get("metadata", {}).get("error") if status_info.get("metadata") else None,
                    }
                else:
                    summary["processing_stages"][process_type] = {
                        "status": "not_started",
                        "completed_at": None,
                        "error": None,
                    }

            # Determine what processes can proceed
            for process_type in self._PROCESSING_DEPENDENCIES:
                can_proceed = await self.validate_processing_dependencies(instrument_id, process_type)
                if can_proceed:
                    # Check if it's already completed or failed
                    current_status = summary["processing_stages"][process_type]["status"]
                    if current_status not in ["completed", "failed"]:
                        summary["can_proceed_to"].append(process_type)
                else:
                    summary["blocked_processes"].append(process_type)

            # Determine overall dependency chain status
            all_completed = all(stage["status"] == "completed" for stage in summary["processing_stages"].values())
            any_failed = any(stage["status"] == "failed" for stage in summary["processing_stages"].values())

            if all_completed:
                summary["dependency_chain_status"] = "all_completed"
            elif any_failed:
                summary["dependency_chain_status"] = "has_failures"
            else:
                summary["dependency_chain_status"] = "in_progress"

            return summary

        except Exception as e:
            logger.error(f"Error getting processing summary for instrument {instrument_id}: {e}")
            summary["error"] = str(e)
            return summary

    async def log_processing_summary(self, instrument_id: int) -> None:
        """
        Log a detailed processing summary for debugging.

        Args:
            instrument_id: The ID of the instrument to check
        """
        summary = await self.get_instrument_processing_summary(instrument_id)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"PROCESSING SUMMARY for Instrument {instrument_id}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Overall Status: {summary['dependency_chain_status']}")

        logger.info("\nProcessing Stages:")
        for process_type, stage_info in summary["processing_stages"].items():
            status_emoji = {"completed": "✅", "failed": "❌", "in_progress": "🔄", "not_started": "⏳"}.get(
                stage_info["status"], "❓"
            )

            error_info = f" - Error: {stage_info['error']}" if stage_info["error"] else ""
            completed_info = f" - Completed: {stage_info['completed_at']}" if stage_info["completed_at"] else ""

            logger.info(f"  {status_emoji} {process_type}: {stage_info['status']}{completed_info}{error_info}")

        if summary["can_proceed_to"]:
            logger.info(f"\nCan proceed to: {', '.join(summary['can_proceed_to'])}")

        if summary["blocked_processes"]:
            logger.info(f"Blocked processes: {', '.join(summary['blocked_processes'])}")

        logger.info(f"{'=' * 60}\n")

    async def check_and_perform_truncation_atomic(self, session_id: str) -> tuple[bool, bool]:
        """
        Atomically check if truncation should be performed and execute it if needed.
        Uses PostgreSQL advisory locks to prevent race conditions across multiple processes.

        Args:
            session_id: Unique identifier for this processing session

        Returns:
            tuple[bool, bool]: (should_truncate_was_requested, truncation_was_performed)
        """
        # Use a fixed advisory lock ID for truncation coordination
        TRUNCATION_LOCK_ID = 12345  # Fixed ID for truncation operations

        try:
            # Try to acquire advisory lock (non-blocking)
            lock_acquired = await db_manager.fetchval("SELECT pg_try_advisory_lock($1)", TRUNCATION_LOCK_ID)

            if not lock_acquired:
                # Another process is handling truncation, wait and check status
                logger.info("Another process is handling truncation. Waiting for completion...")
                await asyncio.sleep(1)  # Brief wait

                # Check if truncation was completed by other process
                truncation_completed = await self.is_system_operation_complete("global_truncation")
                return True, truncation_completed

            try:
                # We have the lock, check if truncation already completed
                truncation_already_done = await self.is_system_operation_complete("global_truncation")
                if truncation_already_done:
                    logger.info("Global truncation already completed by previous session. Skipping.")
                    return True, True

                # Perform the truncation
                logger.critical("🚨 PERFORMING ATOMIC GLOBAL TRUNCATION - ONE-TIME OPERATION...")
                truncation_success = await self._execute_truncation_with_state_tracking(session_id)

                if truncation_success:
                    # Mark truncation as globally complete
                    await self.mark_system_operation_complete(
                        "global_truncation",
                        {
                            "performed_by_session": session_id,
                            "truncated_at": datetime.now().isoformat(),
                            "method": "atomic_advisory_lock",
                        },
                    )
                    logger.warning("🔄 Atomic global truncation completed successfully.")
                    return True, True
                logger.error("💥 Atomic truncation failed")
                return True, False

            finally:
                # Always release the advisory lock
                await db_manager.fetchval("SELECT pg_advisory_unlock($1)", TRUNCATION_LOCK_ID)

        except Exception as e:
            logger.error(f"Error in atomic truncation check: {e}", exc_info=True)
            return False, False

    async def _execute_truncation_with_state_tracking(self, session_id: str) -> bool:
        """
        Execute the actual truncation with proper state tracking.
        This is the internal method that performs the database operations.

        Args:
            session_id: Session ID for tracking

        Returns:
            bool: True if successful, False otherwise
        """
        return await self.truncate_all_data_tables()

    async def truncate_all_data_tables(self) -> bool:
        """
        DANGEROUS: Truncate configured data tables for a complete fresh start.

        NOTE: This method should only be called through check_and_perform_truncation_atomic()
        to ensure thread safety and prevent race conditions.

        Uses the data_reset configuration from config.yaml to determine which tables to truncate.
        This will permanently delete ALL data from selected tables!

        Configuration Controls:
        - data_reset.truncate_*_tables: Enable/disable categories of tables
        - data_reset.tables.{table_name}: Override specific table inclusion

        Returns:
            bool: True if successful, False otherwise
        """
        logger.warning("⚠️  TRUNCATE_ALL_DATA: Starting configurable data truncation - DATA WILL BE PERMANENTLY LOST!")

        try:
            # Get data reset configuration (with defaults if not configured)
            data_reset_config = getattr(config, "data_reset", None)
            if not data_reset_config:
                logger.error("💥 No data_reset configuration found in config.yaml")
                return False

            # Category to table mapping for automatic categorization
            table_categories = {
                "ohlcv_1min": "truncate_ohlcv_tables",
                "ohlcv_5min": "truncate_ohlcv_tables",
                "ohlcv_15min": "truncate_ohlcv_tables",
                "ohlcv_60min": "truncate_ohlcv_tables",
                "features": "truncate_feature_tables",
                "engineered_features": "truncate_feature_tables",
                "feature_scores": "truncate_feature_tables",
                "feature_selection_history": "truncate_feature_tables",
                "feature_lineage": "truncate_feature_tables",
                "feature_performance_metrics": "truncate_statistics_tables",
                "current_selected_features": "truncate_feature_tables",
                "cross_asset_features": "truncate_feature_tables",
                "labels": "truncate_ml_tables",
                "signals": "truncate_ml_tables",
                "model_registry": "truncate_ml_tables",
                "labeling_statistics": "truncate_ml_tables",
                "processing_state": "truncate_state_tables",
                "data_ranges": "truncate_state_tables",
                "system_operations_state": "truncate_state_tables",
                "daily_ohlcv_agg": "truncate_aggregation_tables",
                "hourly_feature_agg": "truncate_aggregation_tables",
                "daily_signal_performance": "truncate_statistics_tables",
            }

            # Get all existing tables from database
            get_tables_query = queries.data_reset_repo["get_all_tables"]
            existing_tables = await self.db_manager.fetch_rows(get_tables_query)
            existing_table_names = {row["table_name"] for row in existing_tables}

            logger.info(f"🔍 Found {len(existing_table_names)} tables in database")

            tables_to_truncate = []

            # Determine which tables to truncate based on configuration
            for table_name in existing_table_names:
                should_truncate = False

                # Check if table is explicitly configured in data_reset.tables
                if hasattr(data_reset_config, "tables") and table_name in data_reset_config.tables:
                    should_truncate = data_reset_config.tables[table_name]
                    logger.debug(f"📋 Table {table_name}: explicit config = {should_truncate}")

                # Otherwise, check category-based configuration
                elif table_name in table_categories:
                    category_setting = table_categories[table_name]
                    should_truncate = getattr(data_reset_config, category_setting, False)
                    logger.debug(f"📂 Table {table_name}: category {category_setting} = {should_truncate}")

                # Unknown tables are not truncated by default (safety)
                else:
                    should_truncate = False
                    logger.debug(f"❓ Table {table_name}: unknown table, skipping for safety")

                if should_truncate:
                    tables_to_truncate.append(table_name)

            if not tables_to_truncate:
                logger.warning("🚫 No tables configured for truncation. Check data_reset configuration.")
                return True

            logger.warning(
                f"🗑️  Will truncate {len(tables_to_truncate)} tables: {', '.join(sorted(tables_to_truncate))}"
            )

            # Confirm all tables exist before starting truncation
            check_table_query = queries.data_reset_repo["check_table_exists"]
            verified_tables = []

            for table_name in tables_to_truncate:
                table_exists = await self.db_manager.fetchval(check_table_query, table_name)
                if table_exists:
                    verified_tables.append(table_name)
                else:
                    logger.warning(f"⚠️  Table {table_name} configured for truncation but does not exist")

            logger.info(f"✅ Verified {len(verified_tables)} tables exist and will be truncated")

            # Perform truncation
            truncated_count = 0
            failed_tables = []

            for table_name in verified_tables:
                try:
                    # Get row count before truncation for logging
                    count_query = queries.data_reset_repo["get_table_row_count"].format(table_name=table_name)
                    row_count = await self.db_manager.fetchval(count_query)

                    # Truncate the table
                    truncate_query = queries.data_reset_repo["truncate_table"].format(table_name=table_name)
                    await self.db_manager.execute(truncate_query)

                    truncated_count += 1
                    logger.info(f"🗑️  Truncated {table_name}: {row_count:,} rows deleted")

                except Exception as e:
                    logger.error(f"❌ Failed to truncate table {table_name}: {e}")
                    failed_tables.append(table_name)
                    continue

            # Summary
            if failed_tables:
                logger.error(
                    f"💥 TRUNCATE_ALL_DATA: {truncated_count}/{len(verified_tables)} tables truncated successfully. Failed: {failed_tables}"
                )
                return False
            logger.warning(f"🎉 TRUNCATE_ALL_DATA: Successfully truncated all {truncated_count} configured tables")
            logger.warning("🔄 System is now in fresh state - all configured data has been removed!")
            return True

        except Exception as e:
            logger.error(f"💥 TRUNCATE_ALL_DATA: Critical error during data truncation: {e}", exc_info=True)
            return False
