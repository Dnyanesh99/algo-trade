"""Production-grade historical OHLCV data aggregator with comprehensive validation."""

import asyncio
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.core.candle_validator import CandleValidator
from src.database.models import OHLCVData
from src.database.ohlcv_repo import OHLCVRepository
from src.metrics import metrics_registry
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

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
        # Core dependencies
        self.ohlcv_repo = ohlcv_repo
        self.error_handler = error_handler
        self.health_monitor = health_monitor
        self.candle_validator = CandleValidator()

        # Configuration from config.yaml
        if config.trading is None:
            raise ValueError("Trading configuration is required")
        if config.performance is None:
            raise ValueError("Performance configuration is required")
        if config.data_quality is None:
            raise ValueError("Data quality configuration is required")
        self.timeframes = config.trading.aggregation_timeframes
        self.batch_size = config.performance.processing.batch_size
        self.validation_enabled = config.data_quality.validation.enabled
        self.quality_threshold = config.data_quality.validation.quality_score_threshold
        self.max_retries = config.performance.max_retries

        logger.info(
            f"HistoricalAggregator initialized with timeframes {self.timeframes}, "
            f"batch_size={self.batch_size}, validation_enabled={self.validation_enabled}"
        )

    async def aggregate_and_store(self, instrument_id: int, one_min_data: pd.DataFrame) -> list[AggregationResult]:
        """
        Aggregates 1-minute data into specified higher timeframes and stores them.

        Args:
            instrument_id: The ID of the instrument
            one_min_data: DataFrame containing 1-minute OHLCV data

        Returns:
            List of AggregationResult objects for each timeframe
        """
        start_time = datetime.now()
        results: list[AggregationResult] = []

        try:
            if one_min_data.empty:
                logger.warning(f"No 1-minute data provided for aggregation for instrument {instrument_id}")
                return []

            logger.info(
                f"Starting historical aggregation for instrument {instrument_id} "
                f"with {len(one_min_data)} 1-minute candles"
            )

            # Prepare and validate input data
            df_prepared = await self._prepare_input_data(one_min_data, instrument_id)
            if df_prepared is None:
                return []

            # Process each timeframe
            for tf in self.timeframes:
                tf_result = await self._process_timeframe_aggregation(instrument_id, df_prepared, tf)
                results.append(tf_result)

                # Record aggregation metrics
                metrics_registry.observe_histogram(
                    "candle_aggregation_duration_seconds",
                    tf_result.processing_time_ms / 1000,
                    {"timeframe": tf_result.timeframe},
                )
                metrics_registry.increment_counter(
                    "candle_aggregation_total",
                    {"timeframe": tf_result.timeframe, "status": "success" if tf_result.success else "failure"},
                )

            # Log summary
            successful_results = [r for r in results if r.success]
            logger.info(
                f"Historical aggregation completed for instrument {instrument_id}. "
                f"Success: {len(successful_results)}/{len(results)} timeframes"
            )
            return results

        except Exception as e:
            await self.error_handler.handle_error(
                "historical_aggregator",
                f"Unexpected error during aggregation for instrument {instrument_id}: {e}",
                {"instrument_id": instrument_id, "error": str(e)},
            )
            # Return failed results for all timeframes
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return [
                AggregationResult(
                    success=False,
                    timeframe=f"{tf}min",
                    instrument_id=instrument_id,
                    processed_rows=0,
                    stored_rows=0,
                    validation_errors=[str(e)],
                    processing_time_ms=processing_time,
                    quality_score=0.0,
                )
                for tf in self.timeframes
            ]

    async def _prepare_input_data(self, one_min_data: pd.DataFrame, instrument_id: int) -> Optional[pd.DataFrame]:
        """Prepare and validate input 1-minute data."""
        try:
            required_columns = config.data_quality.validation.required_columns
            missing_columns = [col for col in required_columns if col not in one_min_data.columns]
            if missing_columns:
                await self.error_handler.handle_error(
                    "historical_aggregator",
                    f"Missing required columns for instrument {instrument_id}: {missing_columns}",
                    {"instrument_id": instrument_id, "missing_columns": missing_columns},
                )
                return None

            df_work = one_min_data.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_work["ts"]):
                df_work["ts"] = pd.to_datetime(df_work["ts"])
            df_work = df_work.sort_values("ts").set_index("ts")

            if self.validation_enabled:
                validation_df = df_work.reset_index().rename(columns={"ts": "timestamp"})
                validation_result = await self.candle_validator.validate_candles(validation_df, instrument_id, "1min")
                if not validation_result.is_valid:
                    logger.warning(
                        f"Input data validation failed for instrument {instrument_id}: "
                        f"Quality score: {validation_result.quality_report.quality_score:.2f}%"
                    )
                    if validation_result.quality_report.quality_score < self.quality_threshold:
                        logger.error(
                            f"Data quality below threshold for instrument {instrument_id}. Aggregation aborted."
                        )
                        return None
                df_work = validation_result.cleaned_data.set_index("timestamp")

            logger.info(f"Prepared {len(df_work)} 1-minute candles for aggregation (instrument {instrument_id})")
            return df_work

        except Exception as e:
            await self.error_handler.handle_error(
                "historical_aggregator",
                f"Error preparing input data for instrument {instrument_id}: {e}",
                {"instrument_id": instrument_id, "error": str(e)},
            )
            return None

    async def _process_timeframe_aggregation(
        self, instrument_id: int, df_prepared: pd.DataFrame, timeframe_minutes: int
    ) -> AggregationResult:
        """Process aggregation for a single timeframe."""
        tf_start_time = datetime.now()
        timeframe_str = f"{timeframe_minutes}min"

        try:
            logger.info(f"Aggregating to {timeframe_str} for instrument {instrument_id}")
            # Build aggregation dict based on available columns
            agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

            # Only include 'oi' if it exists in the dataframe
            if "oi" in df_prepared.columns:
                agg_dict["oi"] = "last"

            aggregated_df = df_prepared.resample(timeframe_str, on="ts").agg(agg_dict).dropna()

            if aggregated_df.empty:
                processing_time = (datetime.now() - tf_start_time).total_seconds() * 1000
                return AggregationResult(
                    success=False,
                    timeframe=timeframe_str,
                    instrument_id=instrument_id,
                    processed_rows=len(df_prepared),
                    stored_rows=0,
                    validation_errors=["No data generated after aggregation"],
                    processing_time_ms=processing_time,
                    quality_score=0.0,
                )

            quality_score, validation_errors = 100.0, []
            if self.validation_enabled:
                validation_df = aggregated_df.reset_index().rename(columns={"date": "timestamp"})
                validation_result = await self.candle_validator.validate_candles(
                    validation_df, instrument_id, timeframe_str
                )
                quality_score, validation_errors = (
                    validation_result.quality_report.quality_score,
                    validation_result.validation_errors,
                )
                if not validation_result.is_valid:
                    logger.warning(
                        f"Aggregated {timeframe_str} data validation failed for {instrument_id} "
                        f"with quality score: {quality_score:.2f}%"
                    )
                    if quality_score < self.quality_threshold:
                        processing_time = (datetime.now() - tf_start_time).total_seconds() * 1000
                        return AggregationResult(
                            success=False,
                            timeframe=timeframe_str,
                            instrument_id=instrument_id,
                            processed_rows=len(df_prepared),
                            stored_rows=0,
                            validation_errors=validation_errors,
                            processing_time_ms=processing_time,
                            quality_score=quality_score,
                        )
                aggregated_df = validation_result.cleaned_data.set_index("timestamp")

            ohlcv_records = aggregated_df.reset_index().rename(columns={"timestamp": "ts"}).to_dict(orient="records")
            stored_rows = await self._store_with_retry(instrument_id, timeframe_str, ohlcv_records)
            processing_time = (datetime.now() - tf_start_time).total_seconds() * 1000

            if stored_rows > 0:
                logger.info(f"Successfully stored {stored_rows} {timeframe_str} candles for {instrument_id}")
                # self.health_monitor.record_successful_operation("historical_aggregator")

            return AggregationResult(
                success=stored_rows > 0,
                timeframe=timeframe_str,
                instrument_id=instrument_id,
                processed_rows=len(df_prepared),
                stored_rows=stored_rows,
                validation_errors=validation_errors,
                processing_time_ms=processing_time,
                quality_score=quality_score,
            )
        except Exception as e:
            processing_time = (datetime.now() - tf_start_time).total_seconds() * 1000
            await self.error_handler.handle_error(
                "historical_aggregator",
                f"Error processing {timeframe_str} aggregation for {instrument_id}: {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe_str, "error": str(e)},
            )
            return AggregationResult(
                success=False,
                timeframe=timeframe_str,
                instrument_id=instrument_id,
                processed_rows=len(df_prepared) if "df_prepared" in locals() else 0,
                stored_rows=0,
                validation_errors=[str(e)],
                processing_time_ms=processing_time,
                quality_score=0.0,
            )

    async def _store_with_retry(self, instrument_id: int, timeframe: str, ohlcv_records: list[dict]) -> int:
        """
        Store OHLCV records with retry logic.

        Args:
            instrument_id: Instrument identifier
            timeframe: Timeframe string (e.g., "5min")
            ohlcv_records: List of OHLCV records to store
        """
        for attempt in range(self.max_retries):
            try:
                records_as_models = [OHLCVData(**rec) for rec in ohlcv_records]
                await self.ohlcv_repo.insert_ohlcv_data(instrument_id, timeframe, records_as_models)
                return len(ohlcv_records)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    await self.error_handler.handle_error(
                        "historical_aggregator",
                        f"Failed to store {timeframe} data for {instrument_id} after {self.max_retries} attempts: {e}",
                        {
                            "instrument_id": instrument_id,
                            "timeframe": timeframe,
                            "attempt": attempt + 1,
                            "error": str(e),
                        },
                    )
                    return 0
                logger.warning(
                    f"Storage attempt {attempt + 1} failed for {timeframe} data (instrument {instrument_id}). Retrying..."
                )
                await asyncio.sleep(2**attempt)
        return 0

    async def get_aggregation_health(self) -> dict[str, Any]:
        """Get health status of the historical aggregator."""
        try:
            # This would typically fetch real-time status from HealthMonitor
            health_status: dict[str, Any] = {}
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
            logger.error(f"Error getting aggregation health: {e}")
            return {
                "component": "historical_aggregator",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
