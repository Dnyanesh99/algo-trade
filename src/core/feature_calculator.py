import asyncio
from datetime import datetime
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.feature_validator import FeatureValidator
from src.database.feature_repo import FeatureRepository
from src.database.models import FeatureData
from src.database.ohlcv_repo import OHLCVRepository
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger
from src.utils.performance_metrics import PerformanceMetrics

talib: Optional[Any]
try:
    import talib
except ImportError:
    logger.warning("TA-Lib not found. Using pandas-based TA calculations, which may be slower.")
    talib = None

config = config_loader.get_config()


class FeatureCalculationResult(BaseModel):
    success: bool
    instrument_id: int
    timeframe: str  # Standardized to string
    timestamp: datetime
    features_calculated: int
    features_stored: int
    validation_errors: list[str] = Field(default_factory=list)
    processing_time_ms: float
    quality_score: float = 0.0


class FeatureConfig(BaseModel):
    name: str
    function: str
    params: dict[str, Any] = Field(default_factory=dict)


class FeatureCalculator:
    """Production-grade technical feature calculator with comprehensive validation."""

    def __init__(
        self,
        ohlcv_repo: OHLCVRepository,
        feature_repo: FeatureRepository,
        error_handler: ErrorHandler,
        health_monitor: HealthMonitor,
        performance_metrics: PerformanceMetrics,
    ):
        self.ohlcv_repo = ohlcv_repo
        self.feature_repo = feature_repo
        self.error_handler = error_handler
        self.health_monitor = health_monitor
        self.performance_metrics = performance_metrics
        self.feature_validator = FeatureValidator()
        assert config.trading is not None
        assert config.data_quality is not None
        self.timeframes = config.trading.feature_timeframes
        self.lookback_period = config.trading.features.lookback_period
        self.validation_enabled = config.data_quality.validation.enabled
        self.feature_configs = self._load_feature_configs()
        logger.info(
            f"FeatureCalculator initialized for timeframes {self.timeframes} with {len(self.feature_configs)} features."
        )

    def _load_feature_configs(self) -> list[FeatureConfig]:
        """Loads and parses feature configurations from config.yaml."""
        try:
            assert config.trading is not None and config.trading.features is not None
            features_list = config.trading.features.configurations
            return [FeatureConfig(**fc) for fc in features_list]
        except Exception as e:
            logger.error(f"Error loading feature configurations: {e}")
            return []

    def _get_params_for_timeframe(self, feature_config: FeatureConfig, timeframe: int) -> dict[str, Any]:
        """Merges default and timeframe-specific parameters."""
        tf_key = f"{timeframe}m"
        default_params: dict[str, Any] = feature_config.params.get("default", {})
        tf_params: dict[str, Any] = feature_config.params.get(tf_key, {})
        final_params = default_params.copy()
        final_params.update(tf_params)
        return final_params

    async def calculate_and_store_features(
        self, instrument_id: int, latest_candle_timestamp: datetime
    ) -> list[FeatureCalculationResult]:
        """Calculates features for an instrument across all configured timeframes."""
        tasks = [
            self._calculate_timeframe_features(instrument_id, tf, latest_candle_timestamp) for tf in self.timeframes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results: list[FeatureCalculationResult] = []
        for res in results:
            if isinstance(res, Exception):
                await self.error_handler.handle_error(
                    "feature_calculator",
                    f"Unhandled exception in timeframe calculation for instrument {instrument_id}: {res}",
                    {"instrument_id": instrument_id},
                )
            elif isinstance(res, FeatureCalculationResult):
                final_results.append(res)
        return final_results

    async def _calculate_timeframe_features(
        self, instrument_id: int, timeframe_minutes: int, latest_candle_timestamp: datetime
    ) -> FeatureCalculationResult:
        """Calculates and stores features for a single timeframe."""
        tf_start_time = datetime.now()
        timeframe_str = f"{timeframe_minutes}min"

        max_period = 0
        for fc in self.feature_configs:
            params = self._get_params_for_timeframe(fc, timeframe_minutes)
            period_keys = ["timeperiod", "period", "slowperiod", "timeperiod3"]
            max_period_for_feature = max([params.get(k, 0) for k in period_keys] + [0])
            if max_period_for_feature > max_period:
                max_period = max_period_for_feature
        min_required_candles = max_period + 50

        try:
            ohlcv_records = await self.ohlcv_repo.get_ohlcv_data_for_features(
                instrument_id, timeframe_minutes, self.lookback_period
            )
            if len(ohlcv_records) < min_required_candles:
                msg = f"Insufficient data for {timeframe_str}: Required {min_required_candles}, found {len(ohlcv_records)}"
                return self._create_failure_result(
                    instrument_id, timeframe_str, latest_candle_timestamp, [msg], tf_start_time
                )

            df = pd.DataFrame([o.model_dump() for o in ohlcv_records])
            df["timestamp"] = pd.to_datetime(df["ts"])
            df = df.set_index("timestamp").sort_index()
            ohlcv_df = df[["open", "high", "low", "close", "volume"]].astype(float)

            features_df = await self._calculate_all_features(ohlcv_df, timeframe_minutes)
            if features_df.empty:
                return self._create_failure_result(
                    instrument_id,
                    timeframe_str,
                    latest_candle_timestamp,
                    ["No features were calculated."],
                    tf_start_time,
                )

            # Add timestamp column and handle NaN values
            features_df["timestamp"] = df.index
            features_df = features_df.dropna().reset_index(drop=True)

            quality_score = 100.0
            validation_errors = []
            if self.validation_enabled:
                is_valid, cleaned_df, report = self.feature_validator.validate_features(features_df, instrument_id)
                quality_score, validation_errors = report.quality_score, report.issues
                if not is_valid and cleaned_df.empty:
                    return self._create_failure_result(
                        instrument_id, timeframe_str, latest_candle_timestamp, validation_errors, tf_start_time
                    )
                features_df = cleaned_df

            # Handle different modes: HISTORICAL_MODE stores all features, LIVE_MODE stores only latest
            system_config = config.get_system_config()
            
            if system_config.mode == "HISTORICAL_MODE":
                # For historical/training mode, store ALL features for all timestamps
                if features_df.empty:
                    return self._create_failure_result(
                        instrument_id,
                        timeframe_str,
                        latest_candle_timestamp,
                        ["No features available after cleaning."],
                        tf_start_time,
                    )

                # Melt all features for all timestamps
                features_to_insert = features_df.melt(
                    id_vars=["timestamp"], 
                    var_name="feature_name", 
                    value_name="feature_value"
                )
                features_to_insert = features_to_insert.rename(columns={"timestamp": "ts"})
                features_to_insert["feature_value"] = features_to_insert["feature_value"].astype(float)
                features_to_insert = features_to_insert.dropna()
                
            else:
                # For live mode, store only the latest features
                latest_features = features_df[features_df.timestamp == features_df.timestamp.max()]
                if latest_features.empty:
                    return self._create_failure_result(
                        instrument_id,
                        timeframe_str,
                        latest_candle_timestamp,
                        ["No features available for the latest timestamp after cleaning."],
                        tf_start_time,
                    )

                features_to_insert = latest_features.drop(columns=["timestamp"]).melt(
                    var_name="feature_name", value_name="feature_value"
                )
                features_to_insert["ts"] = latest_features["timestamp"].iloc[0]
                features_to_insert["feature_value"] = features_to_insert["feature_value"].astype(float)

            # Convert to FeatureData models
            feature_models = [
                FeatureData(
                    ts=row["ts"],
                    feature_name=row["feature_name"],
                    feature_value=row["feature_value"],
                )
                for _, row in features_to_insert.iterrows()
            ]

            stored_count = 0
            if feature_models:
                await self.feature_repo.insert_features(instrument_id, timeframe_str, feature_models)
                stored_count = len(feature_models)

            processing_time = (datetime.now() - tf_start_time).total_seconds() * 1000
            return FeatureCalculationResult(
                success=stored_count > 0,
                instrument_id=instrument_id,
                timeframe=timeframe_str,
                timestamp=latest_candle_timestamp,
                features_calculated=len(features_df.columns) - 1,
                features_stored=stored_count,
                validation_errors=validation_errors,
                processing_time_ms=processing_time,
                quality_score=quality_score,
            )
        except Exception as e:
            await self.error_handler.handle_error(
                "feature_calculator",
                f"Error in _calculate_timeframe_features for {timeframe_str}: {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe_str},
            )
            return self._create_failure_result(
                instrument_id, timeframe_str, latest_candle_timestamp, [str(e)], tf_start_time
            )

    async def _calculate_all_features(self, df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
        loop = asyncio.get_running_loop()

        def _run_calculations() -> pd.DataFrame:
            features_dict = {}
            expected_length = len(df)
            
            for fc in self.feature_configs:
                try:
                    params = self._get_params_for_timeframe(fc, timeframe_minutes)
                    result = self._calculate_single_feature(df, fc.function, params)
                    if result is not None:
                        if isinstance(result, pd.DataFrame):
                            for col in result.columns:
                                # Ensure the Series has the same length as the DataFrame
                                series_data = result[col]
                                if len(series_data) != expected_length:
                                    # Reindex to match the expected length
                                    series_data = series_data.reindex(range(expected_length))
                                features_dict[f"{fc.name}_{col}"] = series_data
                        else:
                            # Ensure the Series has the same length as the DataFrame
                            if len(result) != expected_length:
                                # Reindex to match the expected length
                                result = result.reindex(range(expected_length))
                            features_dict[fc.name] = result
                except Exception as e:
                    logger.error(f"Failed to calculate feature '{fc.name}' for {timeframe_minutes}m: {e}")
            
            if features_dict:
                features_df = pd.DataFrame(features_dict, index=df.index)
                logger.debug(f"Calculated {len(features_dict)} features with {len(features_df)} rows for {timeframe_minutes}m")
                return features_df
            else:
                logger.warning(f"No features calculated for {timeframe_minutes}m timeframe")
                return pd.DataFrame()

        return await loop.run_in_executor(None, _run_calculations)

    def _calculate_single_feature(
        self, df: pd.DataFrame, func_name: str, params: dict
    ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        try:
            open, high, low, close, volume = (df["open"], df["high"], df["low"], df["close"], df["volume"])
            if talib and hasattr(talib, func_name):
                func = getattr(talib, func_name)
                if func_name == "SAR":
                    return func(high, low, acceleration=params["acceleration"], maximum=params["maximum"])
                if func_name == "AROON":
                    aroon_down, aroon_up = func(high, low, timeperiod=params["timeperiod"])
                    return pd.DataFrame({"down": aroon_down, "up": aroon_up})
                if func_name == "WILLR":
                    return func(high, low, close, timeperiod=params["timeperiod"])
                if func_name in ["ADX", "ATR", "CCI", "TRANGE", "ULTOSC"]:
                    return func(high, low, close, **params)
                if func_name in ["CMO", "RSI"]:
                    return func(close, **params)
                if func_name == "OBV":
                    return func(close, volume, **params)
                if func_name == "ADOSC":
                    return func(high, low, close, volume, **params)
                if func_name == "STOCH":
                    k, d = func(high, low, close, **params)
                    return pd.DataFrame({"k": k, "d": d})
                if func_name == "MACD":
                    macd, macdsignal, macdhist = func(close, **params)
                    return pd.DataFrame({"line": macd, "signal": macdsignal, "hist": macdhist})
                if func_name == "BBANDS":
                    upper, middle, lower = func(close, **params)
                    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})
                if func_name in ["HT_TRENDLINE", "HT_DCPERIOD", "BOP", "HT_PHASOR"]:
                    if func_name == "BOP":
                        return func(open, high, low, close)
                    if func_name == "HT_PHASOR":
                        inphase, quadrature = func(close)
                        return pd.DataFrame({"inphase": inphase, "quadrature": quadrature})
                    return func(close)
                return func(close, **params)
            logger.debug(f"Function {func_name} not found in TA-Lib or TA-Lib is not installed.")
            return None
        except Exception as e:
            logger.error(f"Error calculating feature {func_name}: {e}")
            return None

    def _create_failure_result(self, instrument_id: int, timeframe_str: str, timestamp: datetime, errors: list[str], start_time: datetime) -> FeatureCalculationResult:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return FeatureCalculationResult(
            success=False,
            instrument_id=instrument_id,
            timeframe=timeframe_str,
            timestamp=timestamp,
            features_calculated=0,
            features_stored=0,
            validation_errors=errors,
            processing_time_ms=processing_time,
            quality_score=0.0,
        )
