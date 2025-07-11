"""
Production-grade market-aware scheduler for trading signal generation.
Handles market hours, readiness checks, and precise timing for predictions.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from src.state.system_state import SystemState
from src.utils.config_loader import SchedulerConfig
from src.utils.logger import LOGGER as logger
from src.utils.market_calendar import MarketCalendar
from src.utils.time_helper import get_next_candle_boundary


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
        scheduler_config: SchedulerConfig,
    ):
        if not scheduler_config:
            raise ValueError("Scheduler configuration is required")

        self.market_calendar = market_calendar
        self.system_state = system_state
        self.prediction_pipeline_callback = prediction_pipeline_callback

        # Configuration-driven parameters
        self.candle_interval_minutes = scheduler_config.prediction_interval_minutes
        self.readiness_check_time = datetime.strptime(scheduler_config.readiness_check_time, "%H:%M").time()

        # State management
        self._scheduler_task: Optional[asyncio.Task] = None
        self._last_triggered_candle = -1
        self._is_running = False
        self._lock = asyncio.Lock()

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
                    if not await self._check_system_readiness(now):
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
                        next_boundary = get_next_candle_boundary(now, self.candle_interval_minutes)
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

    async def _check_system_readiness(self, now: datetime) -> bool:
        """
        Check if system is ready for predictions.
        """
        # Time-based readiness check
        if now.time() < self.readiness_check_time:
            return False

        # 60-min data availability check
        if not self.system_state.is_60_min_data_available():
            logger.info("Performing 60-min data readiness check...")

            try:
                from src.database.ohlcv_repo import OHLCVRepository

                ohlcv_repo = OHLCVRepository()

                # The lookback period should be sufficient for the model's feature calculation.
                # Using a hardcoded but reasonable value here.
                # TODO: Link this to the actual model's lookback requirement from config.
                required_candles = 100
                start_time = now - timedelta(hours=required_candles)

                # Use repository method to check data availability
                candle_count = await ohlcv_repo.check_data_availability("60min", start_time, now)

                # Check if we have sufficient data (e.g., at least 80% of required candles)
                min_required = int(required_candles * 0.8)
                is_available = candle_count >= min_required

                logger.info(
                    f"60-min data check: {candle_count}/{required_candles} candles available, "
                    f"minimum required: {min_required}, status: {is_available}"
                )

                # Update system state with the result
                self.system_state.set_60_min_data_available(is_available)
                return is_available

            except Exception as e:
                logger.error(f"Error checking 60-min data availability: {e}")
                return False

        return self.system_state.is_60_min_data_available()

    def _should_trigger_prediction(self, now: datetime) -> bool:
        """
        Determine if prediction should be triggered based on timing.
        This is a placeholder and needs to be refined based on FLW-022.
        """
        # For now, a simple check: trigger at the start of a new candle interval
        minutes_since_midnight = now.hour * 60 + now.minute
        current_candle_id = minutes_since_midnight // self.candle_interval_minutes
        return current_candle_id != self._last_triggered_candle

    def _get_current_candle_id(self, now: datetime) -> int:
        """
        Get unique identifier for current candle interval.
        This ID changes only when a new candle interval begins.
        """
        minutes_since_midnight = now.hour * 60 + now.minute
        return minutes_since_midnight // self.candle_interval_minutes

    async def start(self) -> None:
        """
        Start the scheduler in a non-blocking way.
        """
        async with self._lock:
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
        async with self._lock:
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
