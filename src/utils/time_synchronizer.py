"""
Production-grade time synchronization for precise candle boundary alignment.
Handles timezone conversions, network latency compensation, and drift correction.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import pytz

from src.utils.config_loader import SystemConfig, TimeSynchronizerConfig
from src.utils.logger import LOGGER as logger
from src.utils.time_helper import get_next_candle_boundary


class TimeSynchronizer:
    """
    Ensures precise candle boundary synchronization with network latency compensation.

    Features:
    - Configurable candle intervals and latency buffers
    - Timezone-aware time handling
    - Drift detection and correction
    - Production-grade error handling
    """

    def __init__(self, time_synchronizer_config: TimeSynchronizerConfig, system_config: SystemConfig) -> None:
        if not time_synchronizer_config:
            raise ValueError("Time synchronizer configuration is required")
        if not system_config:
            raise ValueError("System configuration is required")

        self.candle_interval_minutes = time_synchronizer_config.candle_interval_minutes
        self.latency_buffer_seconds = time_synchronizer_config.latency_buffer_seconds
        self.max_sync_attempts = time_synchronizer_config.max_sync_attempts
        self.sync_tolerance_seconds = time_synchronizer_config.sync_tolerance_seconds
        self.timezone = system_config.timezone

        # Validate configuration
        if self.candle_interval_minutes <= 0:
            raise ValueError(f"Invalid candle interval: {self.candle_interval_minutes}")
        if self.latency_buffer_seconds < 0:
            raise ValueError(f"Invalid latency buffer: {self.latency_buffer_seconds}")

        # Initialize timezone
        try:
            self.tz = pytz.timezone(self.timezone)
        except pytz.UnknownTimeZoneError:
            logger.error(f"Unknown timezone: {self.timezone}. Falling back to Asia/Kolkata.")
            self.tz = pytz.timezone("Asia/Kolkata")

        logger.info(
            f"TimeSynchronizer initialized: {self.candle_interval_minutes}min intervals, "
            f"{self.latency_buffer_seconds}s buffer, timezone: {self.tz.zone}"
        )

    def get_current_time(self) -> datetime:
        """Get current time in configured timezone."""
        return datetime.now(self.tz)

    def get_sync_target_time(self, boundary_time: datetime) -> datetime:
        """
        Calculate the target synchronization time before the boundary.

        Args:
            boundary_time: The exact candle boundary time

        Returns:
            Target sync time (boundary - 1 second - latency buffer)
        """
        return boundary_time - timedelta(seconds=1 + self.latency_buffer_seconds)

    async def wait_until_next_candle_boundary(self) -> datetime:
        """
        Wait until the next precise candle boundary with latency compensation.

        Returns:
            The synchronized datetime when wait concluded

        Raises:
            RuntimeError: If synchronization fails after maximum attempts
        """
        for attempt in range(self.max_sync_attempts):
            try:
                now = self.get_current_time()
                next_boundary = get_next_candle_boundary(now, self.candle_interval_minutes, self.tz)
                target_sync_time = self.get_sync_target_time(next_boundary)

                time_to_wait = (target_sync_time - now).total_seconds()

                if time_to_wait > 0:
                    logger.info(
                        f"Waiting {time_to_wait:.2f}s until candle boundary "
                        f"(target: {target_sync_time.strftime('%H:%M:%S')}, "
                        f"boundary: {next_boundary.strftime('%H:%M:%S')})"
                    )
                    await asyncio.sleep(time_to_wait)
                else:
                    logger.warning(
                        f"Target sync time already passed by {abs(time_to_wait):.2f}s. Proceeding immediately."
                    )

                # Verify synchronization
                synchronized_time = self.get_current_time()
                sync_error = abs((synchronized_time - target_sync_time).total_seconds())

                if sync_error <= self.sync_tolerance_seconds:
                    logger.info(
                        f"Successfully synchronized to candle boundary "
                        f"(error: {sync_error:.3f}s, time: {synchronized_time.strftime('%H:%M:%S.%f')})"
                    )
                    return synchronized_time
                logger.warning(
                    f"Synchronization drift detected: {sync_error:.3f}s "
                    f"(attempt {attempt + 1}/{self.max_sync_attempts})"
                )
                if attempt == self.max_sync_attempts - 1:
                    logger.error(
                        f"Failed to synchronize after {self.max_sync_attempts} attempts. Final error: {sync_error:.3f}s"
                    )
                    raise RuntimeError(
                        f"Time synchronization failed after {self.max_sync_attempts} attempts. "
                        f"Final synchronization error: {sync_error:.3f}s (tolerance: {self.sync_tolerance_seconds:.3f}s)"
                    )

                # Short wait before retry
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in synchronization attempt {attempt + 1}: {e}")
                if attempt == self.max_sync_attempts - 1:
                    raise RuntimeError(f"Time synchronization failed after {self.max_sync_attempts} attempts") from e
                await asyncio.sleep(0.1)

        # This should never be reached due to the logic above, but added for completeness
        raise RuntimeError("Synchronization failed - unexpected code path")

    def is_at_candle_boundary(self, check_time: Optional[datetime] = None, tolerance_seconds: float = 1.0) -> bool:
        """
        Check if the given time is at a candle boundary within tolerance.

        Args:
            check_time: Time to check (if None, uses current time)
            tolerance_seconds: Acceptable deviation from exact boundary

        Returns:
            True if at candle boundary within tolerance
        """
        if check_time is None:
            check_time = self.get_current_time()

        # Find the nearest boundary
        boundary = get_next_candle_boundary(check_time, self.candle_interval_minutes, self.tz)
        prev_boundary = boundary - timedelta(minutes=self.candle_interval_minutes)

        # Check distance to both boundaries
        dist_to_next = abs((check_time - boundary).total_seconds())
        dist_to_prev = abs((check_time - prev_boundary).total_seconds())

        return min(dist_to_next, dist_to_prev) <= tolerance_seconds
