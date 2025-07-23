import threading
from datetime import datetime
from typing import Any, Optional

from src.utils.logger import LOGGER as logger  # Centralized logger

# No need for direct logging configuration here, it's handled by src.utils.logger


class SystemState:
    """
    A central state machine that tracks the overall system status.
    It holds flags like `60_min_data_available`.
    """

    _instance: Optional["SystemState"] = None
    _lock: threading.Lock = threading.Lock()
    _60_min_data_available: bool
    _system_status: str
    _component_health: dict[str, Any]
    _processing_state: dict[str, dict[str, Any]]
    _data_ranges: dict[tuple[int, str], tuple[datetime, datetime]]
    start_time: Optional[datetime] = None

    def __new__(cls) -> "SystemState":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._60_min_data_available = False
                cls._instance._system_status = "INITIALIZING"
                cls._instance._component_health = {}
                cls._instance._processing_state = {}
                cls._instance._data_ranges = {}
                cls._instance.start_time = datetime.now()
                logger.info("SystemState initialized with processing control capabilities.")
            else:
                logger.warning("Attempted to re-initialize SystemState. Returning existing instance.")
        return cls._instance

    def is_60_min_data_available(self) -> bool:
        """
        Checks if 60-minute data is available for prediction.
        """
        status = self._60_min_data_available
        logger.debug(f"60-min data availability check: {status}")
        return status

    def set_60_min_data_available(self, status: bool) -> None:
        """
        Sets the status of 60-minute data availability.
        """
        self._60_min_data_available = status
        logger.info(f"60-min data availability set to: {status}")

    def get_system_status(self) -> str:
        """
        Returns the overall system status.
        """
        status = self._system_status
        logger.debug(f"Current system status: {status}")
        return status

    def set_system_status(self, status: str) -> None:
        """
        Sets the overall system status.
        """
        self._system_status = status
        logger.info(f"System status updated to: {status}")

    def update_component_health(self, component_name: str, health_info: Any) -> None:
        """
        Updates the health information for a specific component.
        """
        self._component_health[component_name] = health_info
        logger.debug(f"Component {component_name} health updated: {health_info}")

    def get_component_health(self, component_name: Optional[str] = None) -> Any:
        """
        Returns health information for a specific component or all components.
        """
        if component_name:
            health = self._component_health.get(component_name)
            logger.debug(f"Retrieved health for {component_name}: {health}")
            return health
        logger.debug(f"Retrieved all component health: {self._component_health}")
        return self._component_health

    def mark_processing_complete(
        self, instrument_id: int, process_type: str, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Mark a processing step as complete for an instrument.

        Args:
            instrument_id: The instrument ID
            process_type: Type of processing (historical_fetch, aggregation, features, labeling)
            metadata: Optional metadata about the processing
        """
        with self._lock:
            key = f"{instrument_id}_{process_type}"
            self._processing_state[key] = {
                "completed_at": datetime.now(),
                "status": "completed",
                "metadata": metadata or {},
                "instrument_id": instrument_id,
                "process_type": process_type,
            }
            logger.info(f"Marked {process_type} complete for instrument {instrument_id}")

    def is_processing_complete(self, instrument_id: int, process_type: str) -> bool:
        """
        Check if a processing step is complete for an instrument.

        Args:
            instrument_id: The instrument ID
            process_type: Type of processing to check

        Returns:
            True if processing is complete, False otherwise
        """
        with self._lock:
            key = f"{instrument_id}_{process_type}"
            is_complete = key in self._processing_state and self._processing_state[key].get("status") == "completed"
            logger.debug(f"Processing check for {process_type} on instrument {instrument_id}: {is_complete}")
            return is_complete

    def get_last_processing_timestamp(self, instrument_id: int, process_type: str) -> Optional[datetime]:
        """
        Get the timestamp of last successful processing for an instrument.

        Args:
            instrument_id: The instrument ID
            process_type: Type of processing

        Returns:
            Timestamp of last processing or None if never processed
        """
        with self._lock:
            key = f"{instrument_id}_{process_type}"
            state = self._processing_state.get(key)
            if state:
                timestamp = state.get("completed_at")
                if isinstance(timestamp, datetime):
                    logger.debug(
                        f"Last processing timestamp for {process_type} on instrument {instrument_id}: {timestamp}"
                    )
                    return timestamp
            logger.debug(f"No processing timestamp found for {process_type} on instrument {instrument_id}")
            return None

    def set_data_range(self, instrument_id: int, timeframe: str, start_ts: datetime, end_ts: datetime) -> None:
        """
        Track data ranges for gap detection.

        Args:
            instrument_id: The instrument ID
            timeframe: The timeframe (e.g., '1min', '5min')
            start_ts: Start timestamp of data range
            end_ts: End timestamp of data range
        """
        with self._lock:
            key = (instrument_id, timeframe)
            self._data_ranges[key] = (start_ts, end_ts)
            logger.debug(f"Data range set for instrument {instrument_id} {timeframe}: {start_ts} to {end_ts}")

    def get_data_range(self, instrument_id: int, timeframe: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """
        Get known data range for an instrument/timeframe.

        Args:
            instrument_id: The instrument ID
            timeframe: The timeframe

        Returns:
            Tuple of (start_timestamp, end_timestamp) or (None, None) if not tracked
        """
        with self._lock:
            key = (instrument_id, timeframe)
            range_data = self._data_ranges.get(key, (None, None))
            logger.debug(f"Data range for instrument {instrument_id} {timeframe}: {range_data}")
            return range_data

    def reset_processing_state(self, instrument_id: Optional[int] = None, process_type: Optional[str] = None) -> None:
        """
        Reset processing state for debugging/reprocessing.

        Args:
            instrument_id: If provided, reset only for this instrument
            process_type: If provided (with instrument_id), reset only this process type
        """
        with self._lock:
            if instrument_id and process_type:
                # Reset specific process for specific instrument
                key = f"{instrument_id}_{process_type}"
                if key in self._processing_state:
                    del self._processing_state[key]
                    logger.info(f"Reset processing state for {process_type} on instrument {instrument_id}")

                # Also reset data range if resetting historical_fetch
                if process_type == "historical_fetch":
                    for data_range_key in list(self._data_ranges.keys()):
                        if data_range_key[0] == instrument_id:
                            del self._data_ranges[data_range_key]
                    logger.info(f"Reset data ranges for instrument {instrument_id}")

            elif instrument_id:
                # Reset all processing for this instrument
                for key in list(self._processing_state.keys()):
                    if key.startswith(f"{instrument_id}_"):
                        del self._processing_state[key]

                # Reset data ranges for this instrument
                for data_range_key in list(self._data_ranges.keys()):
                    if data_range_key[0] == instrument_id:
                        del self._data_ranges[data_range_key]

                logger.info(f"Reset all processing state for instrument {instrument_id}")
            else:
                # Reset all processing state
                self._processing_state.clear()
                self._data_ranges.clear()
                logger.warning("Reset ALL processing state and data ranges")

    def get_processing_summary(self) -> dict[str, Any]:
        """
        Get a summary of all processing states.

        Returns:
            Dictionary with processing state summary
        """
        with self._lock:
            summary: dict[str, Any] = {
                "total_processing_entries": len(self._processing_state),
                "total_data_ranges": len(self._data_ranges),
                "processing_by_type": {},
                "instruments_processed": set(),
                "latest_processing": None,
            }

            latest_time: Optional[datetime] = None
            for _key, state in self._processing_state.items():
                process_type = state.get("process_type")
                instrument_id = state.get("instrument_id")
                completed_at = state.get("completed_at")

                if process_type and isinstance(process_type, str):
                    if process_type not in summary["processing_by_type"]:
                        summary["processing_by_type"][process_type] = 0
                    summary["processing_by_type"][process_type] += 1

                if instrument_id:
                    summary["instruments_processed"].add(instrument_id)

                if (
                    completed_at
                    and isinstance(completed_at, datetime)
                    and (latest_time is None or completed_at > latest_time)
                ):
                    latest_time = completed_at
                    summary["latest_processing"] = {
                        "instrument_id": instrument_id,
                        "process_type": process_type,
                        "completed_at": completed_at.isoformat(),
                    }

            summary["instruments_processed"] = list(summary["instruments_processed"])

            logger.debug(f"Processing summary generated: {summary}")
            return summary
