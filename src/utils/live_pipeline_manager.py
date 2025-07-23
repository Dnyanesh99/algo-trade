"""Manages the state and health of the live, event-driven pipeline."""

from datetime import datetime
from typing import Any, Optional

from src.database.processing_state_repo import ProcessingStateRepository
from src.state.system_state import SystemState
from src.utils.logger import LOGGER as logger


class LivePipelineManager:
    """Monitors and manages the state of the live trading pipeline."""

    def __init__(self, system_state: SystemState, processing_state_repo: ProcessingStateRepository):
        self.system_state = system_state
        self.processing_state_repo = processing_state_repo
        logger.info("LivePipelineManager initialized.")

    async def update_websocket_status(self, status: str, details: Optional[dict[str, Any]] = None) -> None:
        """Update the status of the WebSocket connection."""
        self.system_state.update_component_health("live_websocket", {"status": status, "details": details or {}})
        try:
            await self.processing_state_repo.mark_processing_complete(
                instrument_id=0,  # System-wide state
                process_type="live_websocket_connection",
                metadata={
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                    "details": details or {},
                },
            )
        except Exception as e:
            logger.error(f"Error updating WebSocket status in processing state repo: {e}", exc_info=True)

    async def update_tick_stream_status(self, status: str, last_tick_time: Optional[datetime] = None) -> None:
        """Update the status of the incoming tick stream."""
        self.system_state.update_component_health(
            "live_tick_stream", {"status": status, "last_tick_time": last_tick_time}
        )
        try:
            await self.processing_state_repo.mark_processing_complete(
                instrument_id=0,  # System-wide state
                process_type="live_tick_stream",
                metadata={
                    "status": status,
                    "last_tick_time": last_tick_time.isoformat() if last_tick_time else None,
                },
            )
        except Exception as e:
            logger.error(f"Error updating tick stream status in processing state repo: {e}", exc_info=True)

    async def record_candle_completion(self, timeframe: str, candle_timestamp: datetime) -> None:
        """Record the completion of a live candle."""
        process_type = f"live_candle_aggregation_{timeframe}"
        try:
            await self.processing_state_repo.mark_processing_complete(
                instrument_id=0,  # System-wide for now, can be instrument-specific
                process_type=process_type,
                metadata={"last_candle_timestamp": candle_timestamp.isoformat()},
            )
        except Exception as e:
            logger.error(
                f"Error recording candle completion in processing state repo for {timeframe} at {candle_timestamp}: {e}",
                exc_info=True,
            )
