"""Production-grade live OHLCV aggregator with comprehensive error handling."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.database.models import OHLCVData
from src.database.ohlcv_repo import OHLCVRepository
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger
from src.utils.time_helper import get_candle_start_time
from src.validation.candle_validator import CandleValidator

# Load configuration
config = ConfigLoader().get_config()


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
        on_15min_candle_complete: Optional[Callable[[dict[str, Any]], Any]] = None,
    ):
        if not isinstance(ohlcv_repo, OHLCVRepository):
            raise TypeError("ohlcv_repo must be a valid OHLCVRepository instance.")
        if not isinstance(error_handler, ErrorHandler) or not isinstance(health_monitor, HealthMonitor):
            raise TypeError("error_handler and health_monitor must be valid instances.")

        self.ohlcv_repo = ohlcv_repo
        self.error_handler = error_handler
        self.health_monitor = health_monitor
        self.candle_validator = CandleValidator(config.data_quality)

        if not all(
            [
                config.trading,
                config.live_aggregator,
                config.data_quality,
                config.data_quality.validation,
                config.model_training,
                config.model_training.feature_engineering,
                config.model_training.feature_engineering.cross_asset,
            ]
        ):
            raise ValueError("Trading, live_aggregator, data_quality, and model_training configurations are required.")

        if config.live_aggregator is None:
            raise ValueError("Live aggregator configuration is missing")
        if config.data_quality.validation is None:
            raise ValueError("Data quality validation configuration is missing")

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

        self.on_15min_candle_complete = on_15min_candle_complete

        self._partial_candles: dict[int, dict[int, dict[str, Any]]] = {}
        self._last_processed_timestamp: dict[int, dict[int, datetime]] = {}
        self._processing_lock = asyncio.Lock()
        self._active_instruments: set[int] = set()

        self._cross_asset_instruments = config.model_training.feature_engineering.cross_asset.instruments
        self._cross_asset_data: dict[int, dict[str, Any]] = {}

        self.stats = AggregationStats()

        for tf in self.timeframes:
            self._partial_candles[tf] = {}
            self._last_processed_timestamp[tf] = {}

        logger.info(
            f"LiveAggregator initialized with timeframes {self.timeframes}, "
            f"validation_enabled={self.validation_enabled}, "
            f"cross_asset_instruments={len(self._cross_asset_instruments)}."
        )

    async def process_one_minute_candle(self, candle: dict[str, Any]) -> bool:
        """
        Processes an incoming 1-minute candle to build higher timeframe candles.
        """
        processing_start = datetime.now()
        try:
            candle_data = CandleData(**candle)
        except Exception as e:
            await self.error_handler.handle_error(
                "live_aggregator_validation", f"Invalid candle data: {e}", {"candle": candle}, exc_info=True
            )
            self.stats.validation_failures += 1
            return False

        instrument_token = candle_data.instrument_token
        logger.debug(f"Processing 1-min candle for instrument {instrument_token} at {candle_data.start_time}.")

        try:
            self._active_instruments.add(instrument_token)
            await self._collect_cross_asset_data(instrument_token, candle_data)

            if sum(len(tf_candles) for tf_candles in self._partial_candles.values()) > self.max_partial_candles:
                await self._cleanup_old_partial_candles()

            async with self._processing_lock:
                for tf in self.timeframes:
                    if not await self._process_timeframe(tf, candle_data):
                        self.stats.failed_aggregations += 1
                        await self.error_handler.handle_error(
                            "live_aggregator_timeframe",
                            f"Failed to process {tf}-min timeframe for instrument {instrument_token}",
                            {
                                "timeframe": tf,
                                "instrument_token": instrument_token,
                                "timestamp": candle_data.start_time,
                            },
                        )

            self.stats.total_candles_processed += 1
            self.stats.successful_aggregations += 1
            self.stats.last_processed_timestamp = candle_data.start_time

            processing_time_ms = (datetime.now() - processing_start).total_seconds() * 1000
            total_processed = self.stats.total_candles_processed
            self.stats.avg_processing_time_ms = (
                (self.stats.avg_processing_time_ms * (total_processed - 1) + processing_time_ms) / total_processed
                if total_processed > 0
                else processing_time_ms
            )

            return True

        except Exception as e:
            self.stats.failed_aggregations += 1
            await self.error_handler.handle_error(
                "live_aggregator_processing",
                f"Unexpected error processing candle: {e}",
                {"candle": candle, "error": str(e)},
                exc_info=True,
            )
            return False

    async def _process_timeframe(self, tf: int, candle_data: CandleData) -> bool:
        """
        Process a single timeframe aggregation.
        """
        instrument_token = candle_data.instrument_token
        timestamp = candle_data.start_time
        try:
            tf_candle_start_time = self._get_candle_start_time(timestamp, tf)
            current_partial_candle = self._partial_candles[tf].get(instrument_token)

            if current_partial_candle is None or tf_candle_start_time > current_partial_candle["start_time"]:
                if current_partial_candle and not await self._emit_and_store_candle(current_partial_candle):
                    return False

                self._partial_candles[tf][instrument_token] = {
                    "instrument_token": instrument_token,
                    "timeframe": tf,
                    "start_time": tf_candle_start_time,
                    "open": candle_data.open,
                    "high": candle_data.high,
                    "low": candle_data.low,
                    "close": candle_data.close,
                    "volume": candle_data.volume,
                    "oi": candle_data.oi or 0.0,
                }
                logger.debug(f"Started new {tf}-min live candle for {instrument_token} at {tf_candle_start_time}.")

            elif tf_candle_start_time == current_partial_candle["start_time"]:
                current_partial_candle["high"] = max(current_partial_candle["high"], candle_data.high)
                current_partial_candle["low"] = min(current_partial_candle["low"], candle_data.low)
                current_partial_candle["close"] = candle_data.close
                current_partial_candle["volume"] += candle_data.volume
                current_partial_candle["oi"] = candle_data.oi or 0.0
                logger.debug(f"Updated {tf}-min live candle for {instrument_token} at {tf_candle_start_time}.")

            else:
                logger.error(
                    f"Out-of-sequence 1-min candle for {instrument_token} ({tf}min). "
                    f"Expected start time >= {current_partial_candle['start_time']}, but got {tf_candle_start_time}."
                )
                return False

            self._last_processed_timestamp[tf][instrument_token] = timestamp
            return True

        except Exception as e:
            logger.error(f"Error processing {tf}-min timeframe for {instrument_token}: {e}", exc_info=True)
            return False

    async def _emit_and_store_candle(self, candle_data: dict[str, Any]) -> bool:
        """
        Emits the completed candle via callback and stores it in the database.
        """
        tf_str = f"{candle_data['timeframe']}min"
        instrument_token = candle_data["instrument_token"]
        start_time = candle_data["start_time"]
        logger.info(f"Emitting and storing completed {tf_str} candle for {instrument_token} at {start_time}.")

        try:
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
                is_valid, _, quality_report = self.candle_validator.validate(
                    validation_df, symbol=str(instrument_token), instrument_type="UNKNOWN", timeframe=tf_str
                )
                if not is_valid:
                    self.stats.validation_failures += 1
                    logger.error(
                        f"Candle validation failed for {tf_str} candle {instrument_token}. Issues: {quality_report.issues}"
                    )
                    # Do not store invalid candles
                    return False

            await self.ohlcv_repo.insert_ohlcv_data(
                instrument_token, tf_str, [OHLCVData.model_validate(candle_data, from_attributes=True)]
            )
            logger.debug(f"Successfully stored {tf_str} candle for {instrument_token}.")

            if candle_data["timeframe"] == 15 and self.on_15min_candle_complete:
                logger.debug("Invoking 15-min candle completion callback.")
                try:
                    result = self.on_15min_candle_complete(candle_data)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Error in 15-min candle complete callback: {e}", exc_info=True)

            return True

        except Exception as e:
            self.stats.storage_failures += 1
            await self.error_handler.handle_error(
                "live_aggregator_storage",
                f"Failed to store {tf_str} candle for {instrument_token}: {e}",
                {"instrument_token": instrument_token, "timeframe": tf_str, "candle_data": candle_data},
                exc_info=True,
            )
            return False

    async def _cleanup_old_partial_candles(self) -> None:
        """Clean up old partial candles to prevent memory leaks."""
        logger.debug("Running cleanup of old partial candles.")
        try:
            current_time = datetime.now()
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
                logger.info(f"Cleaned up {cleaned_count} old partial candles older than {cutoff_time}.")

        except Exception as e:
            logger.error(f"Error during partial candle cleanup: {e}", exc_info=True)

    def _get_candle_start_time(self, timestamp: datetime, timeframe_minutes: int) -> datetime:
        """
        Calculate the start time of a candle for the given timestamp and timeframe.
        """
        return get_candle_start_time(timestamp, timeframe_minutes)

    async def flush_all_partial_candles(self) -> int:
        """
        Forces all current partial candles to be completed and emitted.
        Useful at market close or system shutdown.
        """
        logger.info("Flushing all partial live candles...")
        flushed_count = 0
        async with self._processing_lock:
            for tf in self.timeframes:
                candles_to_flush = list(self._partial_candles[tf].values())
                logger.debug(f"Flushing {len(candles_to_flush)} partial candles for timeframe {tf}min.")
                for candle_data in candles_to_flush:
                    try:
                        if await self._emit_and_store_candle(candle_data):
                            flushed_count += 1
                        # Remove after processing, regardless of success, to avoid reprocessing
                        if candle_data["instrument_token"] in self._partial_candles[tf]:
                            del self._partial_candles[tf][candle_data["instrument_token"]]
                    except Exception as e:
                        instrument_token = candle_data.get("instrument_token", "unknown")
                        logger.error(
                            f"Error flushing {tf}-min candle for instrument {instrument_token}: {e}", exc_info=True
                        )

        logger.info(f"Flushed a total of {flushed_count} partial live candles.")
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
        logger.debug("Performing health check on LiveAggregator.")
        stats = self.get_aggregation_stats()
        total_processed = stats["total_candles_processed"]
        validation_failure_rate = (stats["validation_failures"] / total_processed) if total_processed > 0 else 0

        is_healthy = (
            stats["success_rate"] >= self.health_check_success_rate_threshold
            and stats["avg_processing_time_ms"] < self.health_check_avg_processing_time_ms_threshold
            and validation_failure_rate < self.health_check_validation_failures_threshold
        )
        status = "healthy" if is_healthy else "degraded"
        logger.info(f"LiveAggregator health check status: {status}.")
        return {
            "is_healthy": is_healthy,
            "status": status,
            "stats": stats,
            "timestamp": datetime.now().isoformat(),
        }

    async def _collect_cross_asset_data(self, instrument_token: int, candle_data: CandleData) -> None:
        """
        Collect and store cross-asset data for correlation analysis.
        """
        if instrument_token not in self._cross_asset_instruments:
            return

        logger.debug(f"Collecting cross-asset data for instrument {instrument_token}.")
        try:
            symbol_name = self._cross_asset_instruments[instrument_token]
            self._cross_asset_data[instrument_token] = {
                "symbol": symbol_name,
                "timestamp": candle_data.start_time,
                "open": candle_data.open,
                "high": candle_data.high,
                "low": candle_data.low,
                "close": candle_data.close,
                "volume": candle_data.volume,
                "last_updated": datetime.now(),
            }
            await self._store_cross_asset_ohlcv(instrument_token, candle_data)
        except Exception as e:
            await self.error_handler.handle_error(
                "cross_asset_collection",
                f"Failed to collect cross-asset data for {instrument_token}: {e}",
                {"instrument_token": instrument_token},
                exc_info=True,
            )

    async def _store_cross_asset_ohlcv(self, instrument_token: int, candle_data: CandleData) -> None:
        """
        Store cross-asset OHLCV data in the database for correlation analysis.
        """
        logger.debug(f"Storing cross-asset OHLCV data for instrument {instrument_token}.")
        try:
            ohlcv_data = OHLCVData(
                ts=candle_data.start_time,
                open=candle_data.open,
                high=candle_data.high,
                low=candle_data.low,
                close=candle_data.close,
                volume=int(candle_data.volume),
                oi=int(candle_data.oi) if candle_data.oi is not None else None,
            )
            await self.ohlcv_repo.insert_ohlcv_data(instrument_token, "1min", [ohlcv_data])
        except Exception as e:
            await self.error_handler.handle_error(
                "cross_asset_storage",
                f"Failed to store cross-asset OHLCV data for {instrument_token}: {e}",
                {"instrument_token": instrument_token},
                exc_info=True,
            )

    def get_cross_asset_data(self) -> dict[int, dict[str, Any]]:
        """
        Get current cross-asset data for correlation analysis.

        Returns:
            Dictionary mapping instrument_token to latest candle data
        """
        return self._cross_asset_data.copy()

    def get_cross_asset_instruments(self) -> dict[int, str]:
        """
        Get mapping of cross-asset instrument tokens to symbols.

        Returns:
            Dictionary mapping instrument_token to symbol name
        """
        return self._cross_asset_instruments.copy()
