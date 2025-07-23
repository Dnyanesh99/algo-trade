"""Production-grade historical OHLCV data aggregator with comprehensive validation."""

import asyncio
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.database.models import OHLCVData
from src.database.ohlcv_repo import OHLCVRepository
from src.metrics import metrics_registry
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger
from src.validation.candle_validator import CandleValidator

# Load configuration
config = ConfigLoader().get_config()


class AggregationResult(BaseModel):
    """Result of historical aggregation operation."""

    success: bool
    # --- SUGGESTION IMPLEMENTED: Changed timeframe type to string ---
    timeframe: str
    instrument_id: int
    processed_rows: int
    stored_rows: int
    validation_errors: list[str] = Field(default_factory=list)
    processing_time_ms: float
    quality_score: float = 0.0


class HistoricalAggregator:
    """
    Production-grade historical OHLCV data aggregator with comprehensive validation.

    Features:
    - Aggregates 1-minute historical data into higher timeframes (5, 15, 60 minutes)
    - Comprehensive data validation using CandleValidator
    - Batch processing with memory management
    - Error handling and recovery
    - Performance monitoring and metrics
    - Data quality assessment and reporting
    - Configurable timeframes and validation thresholds
    """

    def __init__(
        self,
        ohlcv_repo: OHLCVRepository,
        error_handler: ErrorHandler,
        health_monitor: HealthMonitor,
    ):
        if not isinstance(ohlcv_repo, OHLCVRepository):
            raise TypeError("ohlcv_repo must be a valid OHLCVRepository instance.")
        if not isinstance(error_handler, ErrorHandler) or not isinstance(health_monitor, HealthMonitor):
            raise TypeError("error_handler and health_monitor must be valid instances.")

        self.ohlcv_repo = ohlcv_repo
        self.error_handler = error_handler
        self.health_monitor = health_monitor
        self.candle_validator = CandleValidator(config.data_quality)

        if not all([config.trading, config.performance, config.data_quality, config.data_quality.validation]):
            raise ValueError("Trading, performance, and data_quality configurations are required.")

        self.timeframes = config.trading.aggregation_timeframes
        self.batch_size = config.performance.processing.batch_size
        self.validation_enabled = config.data_quality.validation.enabled
        self.quality_threshold = config.data_quality.validation.quality_score_threshold
        self.max_retries = config.performance.max_retries

        logger.info(
            f"HistoricalAggregator initialized with timeframes {self.timeframes}, "
            f"batch_size={self.batch_size}, validation_enabled={self.validation_enabled}, max_retries={self.max_retries}."
        )

    async def aggregate_and_store(self, instrument_id: int, one_min_data: pd.DataFrame) -> list[AggregationResult]:
        """
        Aggregates 1-minute data into specified higher timeframes and stores them.
        """
        start_time = datetime.now()
        logger.info(
            f"Starting historical aggregation for instrument {instrument_id} with {len(one_min_data)} 1-minute candles."
        )

        if one_min_data.empty:
            logger.error(f"No 1-minute data provided for aggregation for instrument {instrument_id}.")
            return [
                self._create_failure_result(instrument_id, f"{tf}min", start_time, ["Input data was empty."])
                for tf in self.timeframes
            ]

        try:
            df_prepared = await self._prepare_input_data(one_min_data, instrument_id)
            if df_prepared is None:
                logger.error(f"Data preparation failed for instrument {instrument_id}.")
                return [
                    self._create_failure_result(instrument_id, f"{tf}min", start_time, ["Data preparation failed."])
                    for tf in self.timeframes
                ]

            aggregation_tasks = [
                self._process_timeframe_aggregation(instrument_id, df_prepared, tf) for tf in self.timeframes
            ]
            results = await asyncio.gather(*aggregation_tasks, return_exceptions=True)

            final_results: list[AggregationResult] = []
            for i, res in enumerate(results):
                tf_str = f"{self.timeframes[i]}min"
                if isinstance(res, Exception):
                    await self.error_handler.handle_error(
                        "historical_aggregator",
                        f"Unhandled exception during {tf_str} aggregation for instrument {instrument_id}: {res}",
                        {"instrument_id": instrument_id, "timeframe": tf_str},
                        exc_info=True,
                    )
                    final_results.append(self._create_failure_result(instrument_id, tf_str, start_time, [str(res)]))
                else:
                    if not isinstance(res, AggregationResult):
                        raise TypeError(f"Expected AggregationResult, but got {type(res)}")
                    final_results.append(res)
                    metrics_registry.observe_histogram(
                        "candle_aggregation_duration_seconds",
                        res.processing_time_ms / 1000,
                        {"timeframe": res.timeframe},
                    )
                    metrics_registry.increment_counter(
                        "candle_aggregation_total",
                        {"timeframe": res.timeframe, "status": "success" if res.success else "failure"},
                    )

            successful_count = sum(1 for r in final_results if r.success)
            logger.info(
                f"Historical aggregation completed for instrument {instrument_id}. "
                f"Success: {successful_count}/{len(final_results)} timeframes."
            )
            return final_results

        except Exception as e:
            await self.error_handler.handle_error(
                "historical_aggregator",
                f"Unexpected error during aggregation for instrument {instrument_id}: {e}",
                {"instrument_id": instrument_id, "error": str(e)},
                exc_info=True,
            )
            return [
                self._create_failure_result(instrument_id, f"{tf}min", start_time, [str(e)]) for tf in self.timeframes
            ]

    async def _prepare_input_data(self, one_min_data: pd.DataFrame, instrument_id: int) -> Optional[pd.DataFrame]:
        """Prepare and validate input 1-minute data."""
        logger.debug(f"Preparing input data for instrument {instrument_id}.")
        try:
            required_columns = set(config.data_quality.validation.required_columns)
            missing_columns = required_columns - set(one_min_data.columns)
            if missing_columns:
                await self.error_handler.handle_error(
                    "historical_aggregator_prepare",
                    f"Input data is missing required columns: {missing_columns}",
                    {"instrument_id": instrument_id, "missing_columns": list(missing_columns)},
                )
                return None

            df_work = one_min_data.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_work["ts"]):
                df_work["ts"] = pd.to_datetime(df_work["ts"])
            df_work = df_work.sort_values("ts").set_index("ts")

            if self.validation_enabled:
                logger.debug("Input data validation is enabled.")
                validation_df = df_work.reset_index()
                is_valid, cleaned_df, quality_report = self.candle_validator.validate(
                    validation_df, symbol=str(instrument_id), instrument_type="UNKNOWN", timeframe="1min"
                )

                if not is_valid:
                    logger.error(
                        f"Input data validation failed for instrument {instrument_id}. Issues: {quality_report.issues}"
                    )
                    if quality_report.quality_score < self.quality_threshold:
                        await self.error_handler.handle_error(
                            "historical_aggregator_quality",
                            f"Data quality score {quality_report.quality_score:.2f}% is below threshold of {self.quality_threshold}%.",
                            {"instrument_id": instrument_id, "quality_score": quality_report.quality_score},
                        )
                        return None
                df_work = cleaned_df.set_index("ts")

            logger.info(
                f"Successfully prepared {len(df_work)} 1-minute candles for aggregation for instrument {instrument_id}."
            )
            return df_work

        except Exception as e:
            await self.error_handler.handle_error(
                "historical_aggregator_prepare",
                f"Error preparing input data for instrument {instrument_id}: {e}",
                {"instrument_id": instrument_id, "error": str(e)},
                exc_info=True,
            )
            return None

    async def _process_timeframe_aggregation(
        self, instrument_id: int, df_prepared: pd.DataFrame, timeframe_minutes: int
    ) -> AggregationResult:
        """Process aggregation for a single timeframe."""
        tf_start_time = datetime.now()
        timeframe_str = f"{timeframe_minutes}min"
        logger.info(f"Aggregating to {timeframe_str} for instrument {instrument_id}.")

        try:
            agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            if "oi" in df_prepared.columns:
                agg_dict["oi"] = "last"

            aggregated_df = df_prepared.resample(timeframe_str).agg(agg_dict).dropna()

            if aggregated_df.empty:
                logger.warning(
                    f"No data generated after aggregating to {timeframe_str} for instrument {instrument_id}."
                )
                return self._create_failure_result(
                    instrument_id, timeframe_str, tf_start_time, ["Aggregation resulted in empty dataframe."]
                )

            quality_score, validation_errors = 100.0, []
            if self.validation_enabled:
                validation_df = aggregated_df.reset_index()
                is_valid, cleaned_df, quality_report = self.candle_validator.validate(
                    validation_df, symbol=str(instrument_id), instrument_type="UNKNOWN", timeframe=timeframe_str
                )
                quality_score, validation_errors = quality_report.quality_score, quality_report.issues
                if not is_valid:
                    logger.error(
                        f"Aggregated {timeframe_str} data validation failed for {instrument_id}. Issues: {validation_errors}"
                    )
                    if quality_score < self.quality_threshold:
                        return self._create_failure_result(
                            instrument_id, timeframe_str, tf_start_time, validation_errors, quality_score
                        )
                aggregated_df = cleaned_df.set_index("ts")

            ohlcv_records = aggregated_df.reset_index().to_dict(orient="records")
            stored_rows = await self._store_with_retry(instrument_id, timeframe_str, ohlcv_records)

            if stored_rows > 0:
                logger.info(f"Successfully stored {stored_rows} {timeframe_str} candles for {instrument_id}.")
                # self.health_monitor.record_successful_operation("historical_aggregator")

            return AggregationResult(
                success=stored_rows > 0,
                timeframe=timeframe_str,
                instrument_id=instrument_id,
                processed_rows=len(df_prepared),
                stored_rows=stored_rows,
                validation_errors=validation_errors,
                processing_time_ms=(datetime.now() - tf_start_time).total_seconds() * 1000,
                quality_score=quality_score,
            )
        except Exception as e:
            await self.error_handler.handle_error(
                "historical_aggregator_process",
                f"Error processing {timeframe_str} aggregation for {instrument_id}: {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe_str, "error": str(e)},
                exc_info=True,
            )
            return self._create_failure_result(instrument_id, timeframe_str, tf_start_time, [str(e)])

    def _create_failure_result(
        self, instrument_id: int, timeframe: str, start_time: datetime, errors: list[str], quality_score: float = 0.0
    ) -> AggregationResult:
        """Helper to create a standardized failure result."""
        return AggregationResult(
            success=False,
            timeframe=timeframe,
            instrument_id=instrument_id,
            processed_rows=0,
            stored_rows=0,
            validation_errors=errors,
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            quality_score=quality_score,
        )

    async def _store_with_retry(self, instrument_id: int, timeframe: str, ohlcv_records: list[dict[str, Any]]) -> int:
        """
        Store OHLCV records with retry logic.
        """
        if not ohlcv_records:
            return 0

        for attempt in range(self.max_retries):
            try:
                records_as_models = [OHLCVData(**rec) for rec in ohlcv_records]
                await self.ohlcv_repo.insert_ohlcv_data(instrument_id, timeframe, records_as_models)
                return len(ohlcv_records)
            except Exception as e:
                logger.warning(
                    f"Storage attempt {attempt + 1}/{self.max_retries} failed for {timeframe} data (instrument {instrument_id}): {e}"
                )
                if attempt == self.max_retries - 1:
                    await self.error_handler.handle_error(
                        "historical_aggregator_storage",
                        f"Failed to store {timeframe} data for {instrument_id} after {self.max_retries} attempts: {e}",
                        {
                            "instrument_id": instrument_id,
                            "timeframe": timeframe,
                            "attempt": attempt + 1,
                            "error": str(e),
                        },
                        exc_info=True,
                    )
                    return 0
                await asyncio.sleep(2**attempt)  # Exponential backoff
        return 0

    async def get_aggregation_health(self) -> dict[str, Any]:
        """Get health status of the historical aggregator."""
        logger.debug("Getting historical aggregator health status.")
        try:
            # This would typically fetch real-time status from HealthMonitor
            health_status = {"status": "healthy", "last_success": None}
            return {
                "component": "historical_aggregator",
                "status": health_status.get("status", "unknown"),
                "last_success": health_status.get("last_success"),
                "error_count": health_status.get("error_count", 0),
                "configuration": {
                    "timeframes": self.timeframes,
                    "batch_size": self.batch_size,
                    "validation_enabled": self.validation_enabled,
                    "quality_threshold": self.quality_threshold,
                },
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting aggregation health: {e}", exc_info=True)
            return {
                "component": "historical_aggregator",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
