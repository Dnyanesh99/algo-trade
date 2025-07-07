"""
Production-grade market-aware scheduler for trading signal generation.
Handles market hours, readiness checks, and precise timing for predictions.
"""

import asyncio
import threading
from datetime import datetime, time, timedelta
from typing import Any, Callable, Optional

from src.state.system_state import SystemState
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger
from src.utils.market_calendar import MarketCalendar

# Load configuration
config = config_loader.get_config()


class Scheduler:
    """
    Production-grade market-aware scheduler for trading signal generation.

    Features:
    - Market calendar integration for trading hours
    - Configurable readiness checks and intervals
    - Precise timing with candle boundary alignment
    - Thread-safe operation with proper error handling
    """

    def __init__(
        self,
        market_calendar: MarketCalendar,
        system_state: SystemState,
        prediction_pipeline_callback: Callable,
    ):
        self.market_calendar = market_calendar
        self.system_state = system_state
        self.prediction_pipeline_callback = prediction_pipeline_callback

        # Configuration-driven parameters
        if config.scheduler:
            self.candle_interval_minutes = config.scheduler.prediction_interval_minutes
            self.readiness_check_time = datetime.strptime(config.scheduler.readiness_check_time, "%H:%M").time()
        else:
            logger.warning("Scheduler configuration not found. Using default values.")
            self.candle_interval_minutes = 15  # Default
            self.readiness_check_time = time(9, 30) # Default

        # State management
        self._scheduler_task: Optional[asyncio.Task] = None
        self._last_triggered_candle = -1
        self._is_running = False
        self._lock = threading.Lock()

        # Validation
        if self.candle_interval_minutes <= 0:
            raise ValueError(f"Invalid candle interval: {self.candle_interval_minutes}")

        logger.info(
            f"Scheduler initialized: {self.candle_interval_minutes}min intervals, "
            f"readiness check at {self.readiness_check_time.strftime('%H:%M')}"
        )

    async def _run_scheduler(self) -> None:
        """
        Main scheduler loop with improved market awareness and error handling.
        """
        logger.info("Scheduler started. Monitoring market hours...")

        try:
            while self._is_running:
                now = datetime.now(self.market_calendar.tz)
                time_to_sleep = 1.0  # Minimum sleep to prevent busy-waiting

                if not self.market_calendar.is_market_open(now):
                    # Market closed - reset state and wait until next market open
                    if self._last_triggered_candle != -1:
                        logger.info(f"Market closed at {now.strftime('%H:%M:%S')}. Resetting state.")
                        self._last_triggered_candle = -1

                    next_market_open = self.market_calendar.get_next_trading_day(now)
                    time_to_sleep = (next_market_open - now).total_seconds()
                    logger.debug(f"Market closed. Sleeping for {time_to_sleep:.2f}s until next market open.")

                else:
                    # Market is open - check readiness
                    if not self._check_system_readiness(now):
                        # If not ready, sleep until readiness check time or next minute
                        readiness_dt = now.replace(
                            hour=self.readiness_check_time.hour,
                            minute=self.readiness_check_time.minute,
                            second=0,
                            microsecond=0,
                        )
                        if now < readiness_dt:
                            time_to_sleep = (
                                readiness_dt - now
                            ).total_seconds() + 1  # Add 1 second to ensure we pass the time
                            logger.debug(f"Not ready. Sleeping for {time_to_sleep:.2f}s until readiness check time.")
                        else:
                            # Already past readiness time, but not ready (e.g., 60-min data not available)
                            # Sleep for a short period to re-check
                            time_to_sleep = 5.0  # Re-check every 5 seconds
                            logger.debug(f"Past readiness time but not ready. Sleeping for {time_to_sleep:.2f}s.")

                    else:
                        # System is ready, calculate sleep until next candle boundary
                        next_boundary = self._get_next_candle_boundary_time(now)
                        time_to_sleep = (next_boundary - now).total_seconds()

                        if time_to_sleep <= 0:
                            # Already past boundary, trigger immediately and sleep for next interval
                            logger.warning(
                                f"Scheduler running behind. Triggering immediately. Time behind: {abs(time_to_sleep):.2f}s"
                            )
                            time_to_sleep = (
                                self.candle_interval_minutes * 60
                            )  # Sleep for full interval after triggering

                        logger.debug(f"Ready. Sleeping for {time_to_sleep:.2f}s until next candle boundary.")

                        # Check if we should trigger prediction pipeline
                        if self._should_trigger_prediction(now):
                            current_candle = self._get_current_candle_id(now)

                            if current_candle != self._last_triggered_candle:
                                logger.info(
                                    f"Triggering {self.candle_interval_minutes}-min prediction pipeline "
                                    f"at {now.strftime('%H:%M:%S')} (candle #{current_candle})"
                                )

                                try:
                                    await self.prediction_pipeline_callback()
                                    self._last_triggered_candle = current_candle
                                    logger.debug("Pipeline execution completed successfully")
                                except Exception as e:
                                    logger.error(f"Pipeline execution failed: {e}")
                                    # Continue running despite pipeline failure

                # Ensure sleep is not negative or excessively small
                await asyncio.sleep(max(1.0, time_to_sleep))

        except asyncio.CancelledError:
            logger.info("Scheduler cancelled")
            raise
        except Exception as e:
            logger.error(f"Fatal error in scheduler: {e}")
            raise
        finally:
            logger.info("Scheduler loop ended")

    def _check_system_readiness(self, now: datetime) -> bool:
        """
        Check if system is ready for predictions.
        """
        # Time-based readiness check
        if now.time() < self.readiness_check_time:
            return False

        # 60-min data availability check
        if not self.system_state.is_60_min_data_available():
            logger.info("Performing 60-min data readiness check...")
            # TODO: Implement actual database query or state check for 60-min data availability.
            # This is a critical component for production readiness.
            # For now, raising an error to ensure it's addressed.
            # self.system_state.set_60_min_data_available(True) # REMOVE THIS HARDCODED LINE
            # raise NotImplementedError("60-min data availability check not implemented. This must query the database.")
            return False  # Temporarily return False until implemented

        return self.system_state.is_60_min_data_available()

    def _should_trigger_prediction(self, now: datetime) -> bool:
        """
        Determine if prediction should be triggered based on timing.
        Triggers at the last second of the candle interval (e.g., XX:14:59 for 15-min).
        """
        # Calculate the minute within the current candle interval
        minute_in_interval = now.minute % self.candle_interval_minutes

        # The target minute is the last minute of the interval (e.g., 14 for a 15-min candle)
        target_minute = self.candle_interval_minutes - 1

        # Trigger if we are in the target minute and at least 59 seconds into it
        return minute_in_interval == target_minute and now.second >= 59

    def _get_current_candle_id(self, now: datetime) -> int:
        """
        Get unique identifier for current candle interval.
        This ID changes only when a new candle interval begins.
        """
        minutes_since_midnight = now.hour * 60 + now.minute
        return minutes_since_midnight // self.candle_interval_minutes

    def _get_next_candle_boundary_time(self, now: datetime) -> datetime:
        """
        Calculates the exact timestamp of the next candle boundary.
        For a 15-min candle, if now is 10:05:30, next boundary is 10:15:00.
        """
        current_minute = now.minute
        # Calculate minutes to the next multiple of candle_interval_minutes
        minutes_to_add = self.candle_interval_minutes - (current_minute % self.candle_interval_minutes)

        # If current_minute is already a multiple of candle_interval_minutes (e.g., 10:15 for 15-min)
        # and we are exactly at 0 seconds, the next boundary is current time + interval.
        # Otherwise, if we are past 0 seconds, the next boundary is the current time + interval.
        if minutes_to_add == self.candle_interval_minutes:
            minutes_to_add = 0  # This means we are exactly on a boundary, so next boundary is current time + interval

        next_boundary = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)

        # If the calculated next_boundary is in the past or exactly now (and we want the *next* one)
        # and we are not exactly at the start of the current boundary (i.e., seconds > 0 or microseconds > 0)
        if next_boundary <= now and (now.second > 0 or now.microsecond > 0):
            next_boundary += timedelta(minutes=self.candle_interval_minutes)

        return next_boundary

    def start(self) -> None:
        """
        Start the scheduler in a non-blocking way.
        """
        with self._lock:
            if self._is_running:
                logger.warning("Scheduler is already running")
                return

            if self._scheduler_task is not None and not self._scheduler_task.done():
                logger.warning("Scheduler task already exists")
                return

            self._is_running = True
            self._scheduler_task = asyncio.create_task(self._run_scheduler())
            logger.info("Scheduler started successfully")

    async def stop(self) -> None:
        """
        Stop the scheduler gracefully.
        """
        with self._lock:
            if not self._is_running:
                logger.warning("Scheduler is not running")
                return

            self._is_running = False

        if self._scheduler_task:
            logger.info("Stopping scheduler...")
            self._scheduler_task.cancel()

            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                logger.info("Scheduler stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")
            finally:
                self._scheduler_task = None

    def is_running(self) -> bool:
        """
        Check if scheduler is currently running.
        """
        return self._is_running

    def get_status(self) -> dict[str, Any]:
        """
        Get current scheduler status and statistics.
        """
        return {
            "is_running": self._is_running,
            "last_triggered_candle": self._last_triggered_candle,
            "candle_interval_minutes": self.candle_interval_minutes,
            "readiness_check_time": self.readiness_check_time.strftime("%H:%M"),
            "task_done": self._scheduler_task.done() if self._scheduler_task else True,
        }
