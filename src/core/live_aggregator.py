"""Production-grade live OHLCV aggregator with comprehensive error handling."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.core.candle_validator import CandleValidator
from src.database.models import OHLCVData
from src.database.ohlcv_repo import OHLCVRepository
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger
from src.utils.performance_metrics import PerformanceMetrics

# Load configuration
config = config_loader.get_config()


class CandleData(BaseModel):
    """Validated candle data model."""

    instrument_token: int = Field(..., gt=0)
    start_time: datetime
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    oi: Optional[float] = Field(None, ge=0)


class AggregationStats(BaseModel):
    """Aggregation performance statistics."""

    total_candles_processed: int = 0
    successful_aggregations: int = 0
    failed_aggregations: int = 0
    validation_failures: int = 0
    storage_failures: int = 0
    avg_processing_time_ms: float = 0.0
    last_processed_timestamp: Optional[datetime] = None


class LiveAggregator:
    """
    Production-grade live OHLCV aggregator with comprehensive error handling.

    Features:
    - Aggregates 1-minute candles into higher timeframes (5, 15, 60 minutes)
    - Comprehensive candle validation using CandleValidator
    - Circuit breaker pattern for error handling
    - Performance metrics and monitoring
    - Thread-safe operations with proper locking
    - Graceful degradation and recovery
    - Memory-efficient partial candle management
    """

    def __init__(
        self,
        ohlcv_repo: OHLCVRepository,
        error_handler: ErrorHandler,
        health_monitor: HealthMonitor,
        performance_metrics: PerformanceMetrics,
        on_15min_candle_complete: Optional[Callable[[dict[str, Any]], Any]] = None,
    ):
        # Core dependencies
        self.ohlcv_repo = ohlcv_repo
        self.error_handler = error_handler
        self.health_monitor = health_monitor
        self.performance_metrics = performance_metrics
        self.candle_validator = CandleValidator()

        # Configuration from config.yaml
        self.timeframes = config.trading.aggregation_timeframes
        self.max_partial_candles = config.live_aggregator.max_partial_candles
        self.partial_candle_cleanup_hours = config.live_aggregator.partial_candle_cleanup_hours
        self.health_check_success_rate_threshold = config.live_aggregator.health_check_success_rate_threshold
        self.health_check_avg_processing_time_ms_threshold = (
            config.live_aggregator.health_check_avg_processing_time_ms_threshold
        )
        self.health_check_validation_failures_threshold = (
            config.live_aggregator.health_check_validation_failures_threshold
        )
        self.validation_enabled = config.data_quality.validation.enabled

        # Callbacks
        self.on_15min_candle_complete = on_15min_candle_complete

        # State management - thread-safe
        self._partial_candles: dict[int, dict[int, dict[str, Any]]] = {}  # {timeframe: {instrument_token: candle_data}}
        self._last_processed_timestamp: dict[int, dict[int, datetime]] = {}  # {timeframe: {instrument_token: last_ts}}
        self._processing_lock = asyncio.Lock()
        self._active_instruments: set[int] = set()

        # Performance tracking
        self.stats = AggregationStats()

        # Initialize timeframe structures
        for tf in self.timeframes:
            self._partial_candles[tf] = {}
            self._last_processed_timestamp[tf] = {}

        logger.info(
            f"LiveAggregator initialized with timeframes {self.timeframes}, "
            f"validation_enabled={self.validation_enabled}"
        )

    async def process_one_minute_candle(self, candle: dict[str, Any]) -> bool:
        """
        Processes an incoming 1-minute candle to build higher timeframe candles.

        Args:
            candle: Dictionary containing candle data

        Returns:
            bool: True if processing was successful, False otherwise
        """
        processing_start = datetime.now()

        try:
            # Validate input candle data
            try:
                candle_data = CandleData(**candle)
            except Exception as e:
                await self.error_handler.handle_error(
                    "live_aggregator", f"Invalid candle data: {e}", {"candle": candle}
                )
                self.stats.validation_failures += 1
                return False

            # Extract validated data
            instrument_token = candle_data.instrument_token
            timestamp = candle_data.start_time
            open_price = candle_data.open
            high_price = candle_data.high
            low_price = candle_data.low
            close_price = candle_data.close
            volume = candle_data.volume
            oi = candle_data.oi or 0.0

            # Track active instruments
            self._active_instruments.add(instrument_token)

            # Memory management check
            total_partial = sum(len(tf_candles) for tf_candles in self._partial_candles.values())
            if total_partial > self.max_partial_candles:
                await self._cleanup_old_partial_candles()

            async with self._processing_lock:
                for tf in self.timeframes:
                    success = await self._process_timeframe(
                        tf, instrument_token, timestamp, open_price, high_price, low_price, close_price, volume, oi
                    )

                    if not success:
                        self.stats.failed_aggregations += 1
                        await self.error_handler.handle_error(
                            "live_aggregator",
                            f"Failed to process {tf}-min timeframe for instrument {instrument_token}",
                            {"timeframe": tf, "instrument_token": instrument_token, "timestamp": timestamp},
                        )

            # Update statistics
            self.stats.total_candles_processed += 1
            self.stats.successful_aggregations += 1
            self.stats.last_processed_timestamp = timestamp

            # Update performance metrics
            processing_time = (datetime.now() - processing_start).total_seconds() * 1000
            self.stats.avg_processing_time_ms = (
                self.stats.avg_processing_time_ms * (self.stats.total_candles_processed - 1) + processing_time
            ) / self.stats.total_candles_processed

            # self.performance_metrics.record_latency(...)

            return True

        except Exception as e:
            self.stats.failed_aggregations += 1
            await self.error_handler.handle_error(
                "live_aggregator", f"Unexpected error processing candle: {e}", {"candle": candle, "error": str(e)}
            )
            return False

    async def _process_timeframe(
        self,
        tf: int,
        instrument_token: int,
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
        oi: float,
    ) -> bool:
        """
        Process a single timeframe aggregation.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Determine the start time of the current higher-timeframe candle
            tf_candle_start_time = self._get_candle_start_time(timestamp, tf)

            current_partial_candle = self._partial_candles[tf].get(instrument_token)

            if current_partial_candle is None or tf_candle_start_time > current_partial_candle["start_time"]:
                # New higher-timeframe candle starts or first candle for this instrument/timeframe
                if current_partial_candle is not None:  # Emit previous completed candle if exists
                    emit_success = await self._emit_and_store_candle(current_partial_candle)
                    if not emit_success:
                        return False

                self._partial_candles[tf][instrument_token] = {
                    "instrument_token": instrument_token,
                    "timeframe": tf,
                    "start_time": tf_candle_start_time,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "oi": oi,
                }
                logger.debug(f"Started new {tf}-min live candle for {instrument_token} at {tf_candle_start_time}")

            elif tf_candle_start_time == current_partial_candle["start_time"]:
                # Update existing higher-timeframe candle
                current_partial_candle["high"] = max(current_partial_candle["high"], high_price)
                current_partial_candle["low"] = min(current_partial_candle["low"], low_price)
                current_partial_candle["close"] = close_price
                current_partial_candle["volume"] += volume
                current_partial_candle["oi"] = oi  # Update OI with the latest 1-min candle's OI
                logger.debug(f"Updated {tf}-min live candle for {instrument_token} at {tf_candle_start_time}")

            else:
                logger.warning(
                    f"Out-of-sequence 1-min candle for {instrument_token} ({tf}-min). "
                    f"Expected: {current_partial_candle['start_time']}, Got: {tf_candle_start_time}"
                )
                return False

            self._last_processed_timestamp[tf][instrument_token] = timestamp
            return True

        except Exception as e:
            logger.error(f"Error processing {tf}-min timeframe for {instrument_token}: {e}")
            return False

    def _get_candle_start_time(self, timestamp: datetime, timeframe_minutes: int) -> datetime:
        """
        Calculates the start time of the higher-timeframe candle for a given timestamp.
        """
        timestamp_in_minutes = timestamp.hour * 60 + timestamp.minute
        floored_minute = (timestamp_in_minutes // timeframe_minutes) * timeframe_minutes
        return timestamp.replace(hour=floored_minute // 60, minute=floored_minute % 60, second=0, microsecond=0)

    async def _emit_and_store_candle(self, candle_data: dict[str, Any]) -> bool:
        """
        Emits the completed candle via callback and stores it in the database.

        Returns:
            bool: True if successful, False otherwise
        """
        tf_str = f"{candle_data['timeframe']}min"
        instrument_token = candle_data["instrument_token"]
        start_time = candle_data["start_time"]

        logger.info(f"Emitting and storing completed {tf_str} candle for {instrument_token} at {start_time}")

        try:
            # Validate candle before storage if enabled
            if self.validation_enabled:
                validation_df = pd.DataFrame(
                    [
                        {
                            "timestamp": start_time,
                            "open": candle_data["open"],
                            "high": candle_data["high"],
                            "low": candle_data["low"],
                            "close": candle_data["close"],
                            "volume": candle_data["volume"],
                        }
                    ]
                )
                validation_result = await self.candle_validator.validate_candles(
                    validation_df, instrument_token, tf_str
                )
                if not validation_result.is_valid:
                    logger.warning(
                        f"Candle validation failed for {tf_str} candle {instrument_token}: "
                        f"{validation_result.validation_errors}"
                    )
                    self.stats.validation_failures += 1
                    return False

            # Store in database
            await self.ohlcv_repo.insert_ohlcv_data(
                instrument_token,
                tf_str,
                [
                    OHLCVData(
                        ts=start_time,
                        open=candle_data["open"],
                        high=candle_data["high"],
                        low=candle_data["low"],
                        close=candle_data["close"],
                        volume=candle_data["volume"],
                        oi=candle_data["oi"],
                    )
                ],
            )
            logger.debug(f"Successfully stored {tf_str} candle for {instrument_token}")

            # self.health_monitor.record_successful_operation("live_aggregator")

            # Trigger 15-min event if applicable
            if candle_data["timeframe"] == 15 and self.on_15min_candle_complete:
                try:
                    result = self.on_15min_candle_complete(candle_data)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Error in 15-min candle complete callback: {e}")

            return True

        except Exception as e:
            self.stats.storage_failures += 1
            await self.error_handler.handle_error(
                "live_aggregator",
                f"Failed to store {tf_str} candle for {instrument_token}: {e}",
                {
                    "instrument_token": instrument_token,
                    "timeframe": tf_str,
                    "candle_data": candle_data,
                    "error": str(e),
                },
            )
            return False

    async def _cleanup_old_partial_candles(self) -> None:
        """Clean up old partial candles to prevent memory leaks."""
        try:
            current_time = datetime.now()
            # --- SUGGESTION IMPLEMENTED: Use configured cleanup hours ---
            cutoff_time = current_time - timedelta(hours=self.partial_candle_cleanup_hours)

            cleaned_count = 0
            for tf in self.timeframes:
                to_remove = [
                    token for token, data in self._partial_candles[tf].items() if data["start_time"] < cutoff_time
                ]
                for instrument_token in to_remove:
                    del self._partial_candles[tf][instrument_token]
                    if instrument_token in self._last_processed_timestamp[tf]:
                        del self._last_processed_timestamp[tf][instrument_token]
                    cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old partial candles")

        except Exception as e:
            logger.error(f"Error during partial candle cleanup: {e}")

    async def flush_all_partial_candles(self) -> int:
        """
        Forces all current partial candles to be completed and emitted.
        Useful at market close or system shutdown.
        """
        logger.info("Flushing all partial live candles...")
        flushed_count = 0

        async with self._processing_lock:
            for tf in self.timeframes:
                candles_to_flush = list(self._partial_candles[tf].items())
                for instrument_token, candle_data in candles_to_flush:
                    try:
                        if await self._emit_and_store_candle(candle_data):
                            flushed_count += 1
                        del self._partial_candles[tf][instrument_token]
                    except Exception as e:
                        logger.error(f"Error flushing {tf}-min candle for {instrument_token}: {e}")

        logger.info(f"Flushed {flushed_count} partial live candles")
        return flushed_count

    def get_aggregation_stats(self) -> dict[str, Any]:
        """Get current aggregation statistics."""
        success_rate = (
            (self.stats.successful_aggregations / self.stats.total_candles_processed * 100)
            if self.stats.total_candles_processed > 0
            else 100.0
        )
        return {
            "total_candles_processed": self.stats.total_candles_processed,
            "successful_aggregations": self.stats.successful_aggregations,
            "failed_aggregations": self.stats.failed_aggregations,
            "validation_failures": self.stats.validation_failures,
            "storage_failures": self.stats.storage_failures,
            "success_rate": success_rate,
            "avg_processing_time_ms": self.stats.avg_processing_time_ms,
            "last_processed_timestamp": self.stats.last_processed_timestamp,
            "active_instruments": len(self._active_instruments),
            "partial_candles_count": sum(len(tf_candles) for tf_candles in self._partial_candles.values()),
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check of the aggregator."""
        stats = self.get_aggregation_stats()
        total_processed = stats["total_candles_processed"]
        validation_failure_rate = (stats["validation_failures"] / total_processed) if total_processed > 0 else 0

        is_healthy = (
            stats["success_rate"] >= self.health_check_success_rate_threshold
            and stats["avg_processing_time_ms"] < self.health_check_avg_processing_time_ms_threshold
            and validation_failure_rate < self.health_check_validation_failures_threshold
        )
        return {
            "is_healthy": is_healthy,
            "status": "healthy" if is_healthy else "degraded",
            "stats": stats,
            "timestamp": datetime.now().isoformat(),
        }
