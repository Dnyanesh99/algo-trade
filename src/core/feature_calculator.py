import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union, cast

if TYPE_CHECKING:
    from src.database.instrument_repo import InstrumentRepository
    from src.validation.feature_validator import FeatureValidator

import pandas as pd
from pydantic import BaseModel, Field

# Import modular indicator system
from src.core.indicators.main import IndicatorCalculator
from src.database.feature_repo import FeatureRepository
from src.database.instrument_repo import InstrumentRepository
from src.database.models import FeatureData
from src.database.ohlcv_repo import OHLCVRepository
from src.metrics import metrics_registry
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

talib: Optional[Any]
try:
    import talib  # noqa: F401
except ImportError as e:
    logger.error("🚨 CRITICAL: TA-Lib not found. This system requires TA-Lib for optimal performance.")
    logger.error("📋 Install TA-Lib: pip install TA-Lib or follow: https://github.com/mrjbq7/ta-lib")
    raise RuntimeError("❌ SYSTEM HALT: TA-Lib dependency missing - cannot proceed with suboptimal calculations") from e

config = ConfigLoader().get_config()


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
        instrument_repo: InstrumentRepository,
        error_handler: ErrorHandler,
        health_monitor: HealthMonitor,
    ):
        if not all(
            isinstance(repo, (OHLCVRepository, FeatureRepository, InstrumentRepository))
            for repo in [ohlcv_repo, feature_repo, instrument_repo]
        ):
            raise TypeError("All repository arguments must be valid repository instances.")
        if not isinstance(error_handler, ErrorHandler) or not isinstance(health_monitor, HealthMonitor):
            raise TypeError("error_handler and health_monitor must be valid instances.")

        self.ohlcv_repo = ohlcv_repo
        self.feature_repo = feature_repo
        self.instrument_repo = instrument_repo
        self.error_handler = error_handler
        self.health_monitor = health_monitor

        from src.validation.factory import ValidationFactory

        validation_factory = ValidationFactory(config)
        self.feature_validator = validation_factory.get_validator("feature")

        if not config.trading or not config.trading.features or not config.trading.feature_timeframes:
            raise ValueError("trading.features and trading.feature_timeframes configurations are required.")
        if not config.data_quality or not config.data_quality.validation:
            raise ValueError("data_quality.validation configuration is required.")

        self.timeframes = config.trading.feature_timeframes
        self.lookback_period = config.trading.features.lookback_period
        self.validation_enabled = config.data_quality.validation.enabled

        # Initialize modular indicator flag early to ensure it always exists
        self.use_modular_indicators = False
        self.modular_calculator = None

        from src.core.feature_engineering_pipeline import FeatureEngineeringPipeline

        self.feature_engineering = FeatureEngineeringPipeline(
            ohlcv_repo, feature_repo, cast("FeatureValidator", self.feature_validator), error_handler, health_monitor
        )

        try:
            logger.info("Initializing modular indicator system...")
            self.modular_calculator = IndicatorCalculator()
            if not self._validate_modular_system_health():
                raise RuntimeError("Modular indicator system health check failed.")

            self.use_modular_indicators = True
            total_indicators = len(self.modular_calculator.registry.get_all_indicators())
            enabled_indicators = len(self.modular_calculator.registry.get_enabled_indicators())
            logger.info(
                f"Modular indicator system initialized successfully with {total_indicators} indicators ({enabled_indicators} enabled)."
            )

            self.feature_configs: list[Any] = []
            self._feature_dispatch: dict[str, Any] = {}
            logger.info("Production mode: Using modular indicators exclusively.")

        except Exception as e:
            logger.critical(f"Failed to initialize modular indicator system: {e}", exc_info=True)
            raise RuntimeError("Feature calculator initialization failed due to indicator system error.") from e

        logger.info(
            f"FeatureCalculator initialized for timeframes {self.timeframes} with {len(self.modular_calculator.registry.get_all_indicators())} modular indicators."
        )

    def _get_params_for_timeframe(self, feature_config: Any, timeframe_minutes: int) -> dict[str, Any]:
        """Legacy method placeholder - not used in modular system."""
        return {}

    async def calculate_and_store_features(
        self, instrument_id: int, latest_candle_timestamp: datetime
    ) -> list[FeatureCalculationResult]:
        """Calculates features for an instrument across all configured timeframes in parallel and stores them sequentially."""

        # Step 1: Calculate features for all timeframes in parallel
        calculation_tasks = [
            self._calculate_timeframe_features(instrument_id, tf, latest_candle_timestamp) for tf in self.timeframes
        ]

        # This will run all calculations concurrently
        results_with_data = await asyncio.gather(*calculation_tasks, return_exceptions=True)

        final_results: list[FeatureCalculationResult] = []
        all_features_to_store: list[FeatureData] = []

        # Step 2: Process results and collect all features to be stored
        for i, result in enumerate(results_with_data):
            timeframe_minutes = self.timeframes[i]
            timeframe_str = f"{timeframe_minutes}min"

            if isinstance(result, Exception):
                await self.error_handler.handle_error(
                    "feature_calculator",
                    f"Unhandled exception in timeframe calculation for instrument {instrument_id}: {result}",
                    {"instrument_id": instrument_id, "timeframe": timeframe_str},
                )
                failure_result, _ = self._create_failure_result(
                    instrument_id, timeframe_str, latest_candle_timestamp, [str(result)], datetime.now()
                )
                final_results.append(failure_result)
            else:
                # Unpack the tuple of (FeatureCalculationResult, list[FeatureData])
                # At this point, result is guaranteed to be a tuple, not an Exception
                calc_result, features_to_store = cast(
                    "tuple[FeatureCalculationResult, Optional[list[FeatureData]]]", result
                )
                final_results.append(calc_result)
                if calc_result.success and features_to_store:
                    all_features_to_store.extend(features_to_store)

        # Step 3: Perform a single batch insert to the database
        if all_features_to_store:
            try:
                await self.feature_repo.insert_features(instrument_id, "batch", all_features_to_store)
                logger.info(
                    f"Successfully stored a batch of {len(all_features_to_store)} features for instrument {instrument_id}."
                )
            except Exception as e:
                await self.error_handler.handle_error(
                    "feature_storage",
                    f"Batch feature storage failed for instrument {instrument_id}: {e}",
                    {"instrument_id": instrument_id, "total_features": len(all_features_to_store)},
                )
                # Optionally update results to reflect storage failure
                for res in final_results:
                    res.success = False
                    res.validation_errors.append("Database storage failed.")

        return final_results

    async def _calculate_timeframe_features(
        self, instrument_id: int, timeframe_minutes: int, latest_candle_timestamp: datetime
    ) -> tuple[FeatureCalculationResult, Optional[list[FeatureData]]]:
        """Calculates and stores features for a single timeframe."""
        tf_start_time = datetime.now()
        timeframe_str = f"{timeframe_minutes}min"

        # Get system config for mode checking
        system_config = config.system

        # Get instrument segment to determine if it's an index
        instrument = await self.instrument_repo.get_instrument_by_id(instrument_id)
        segment = (instrument.segment or "") if instrument else ""

        max_period = 0
        for fc in self.feature_configs:
            params = self._get_params_for_timeframe(fc, timeframe_minutes)
            period_keys = ["timeperiod", "period", "slowperiod", "timeperiod3"]
            max_period_for_feature = max([params.get(k, 0) for k in period_keys] + [0])
            if max_period_for_feature > max_period:
                max_period = max_period_for_feature
        min_required_candles = max_period + 50

        try:
            # In HISTORICAL_MODE, get ALL available OHLCV data for complete feature calculation
            # In LIVE_MODE, use configured lookback_period for recent data only
            if system_config and system_config.mode == "HISTORICAL_MODE":
                # For historical processing, get all data from earliest to latest_candle_timestamp

                # Get earliest available data point (use a very early date)
                earliest_date = datetime(2020, 1, 1)  # Reasonable start date for historical data

                # Get all historical data up to latest_candle_timestamp
                ohlcv_records = await self.ohlcv_repo.get_ohlcv_data(
                    instrument_id, timeframe_str, earliest_date, latest_candle_timestamp
                )
                logger.debug(
                    f"HISTORICAL_MODE: Fetched {len(ohlcv_records)} candles for {instrument_id} ({timeframe_str})"
                )
            else:
                # Live mode: use recent data with configured lookback
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

            features_df = await self._calculate_all_features(ohlcv_df, timeframe_minutes, segment)
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
                is_valid, cleaned_df, report = self.feature_validator.validate(features_df, instrument_id=instrument_id)
                quality_score, validation_errors = report.quality_score, report.issues
                if not is_valid and cleaned_df.empty:
                    return self._create_failure_result(
                        instrument_id, timeframe_str, latest_candle_timestamp, validation_errors, tf_start_time
                    )
                features_df = cleaned_df

            # NEW: Feature Engineering Integration
            enhanced_features_df = await self._enhance_with_feature_engineering(
                features_df, instrument_id, timeframe_str, latest_candle_timestamp, segment, ohlcv_df
            )

            # Handle different modes: HISTORICAL_MODE stores all features, LIVE_MODE stores only latest
            system_config = config.system

            if system_config and system_config.mode == "HISTORICAL_MODE":
                # For historical/training mode, store ALL features for all timestamps
                if enhanced_features_df.empty:
                    return self._create_failure_result(
                        instrument_id,
                        timeframe_str,
                        latest_candle_timestamp,
                        ["No features available after cleaning."],
                        tf_start_time,
                    )

                # Melt all features for all timestamps
                features_to_insert = enhanced_features_df.melt(
                    id_vars=["timestamp"], var_name="feature_name", value_name="feature_value"
                )
                features_to_insert = features_to_insert.rename(columns={"timestamp": "ts"})
                features_to_insert["feature_value"] = features_to_insert["feature_value"].astype(float)
                features_to_insert = features_to_insert.dropna()

            else:
                # For live mode, store only the latest features (with feature selection)
                latest_features = enhanced_features_df[
                    enhanced_features_df.timestamp == enhanced_features_df.timestamp.max()
                ]
                if latest_features.empty:
                    return self._create_failure_result(
                        instrument_id,
                        timeframe_str,
                        latest_candle_timestamp,
                        ["No features available for the latest timestamp after cleaning."],
                        tf_start_time,
                    )

                # NEW: Apply feature selection in live mode
                latest_features = await self._apply_feature_selection(latest_features, instrument_id, timeframe_str)

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
            processing_duration = processing_time / 1000  # Convert to seconds for metrics

            # Record metrics for feature calculation
            success = stored_count > 0
            if metrics_registry.is_enabled():
                metrics_registry.increment_counter(
                    "feature_calculations_total",
                    {
                        "instrument": str(instrument_id),
                        "timeframe": timeframe_str,
                        "feature": "aggregate",
                        "status": str(success),
                    },
                )
                metrics_registry.observe_histogram(
                    "feature_calculation_duration_seconds",
                    processing_duration,
                    {"instrument": str(instrument_id), "timeframe": timeframe_str, "feature": "aggregate"},
                )

            return FeatureCalculationResult(
                success=success,
                instrument_id=instrument_id,
                timeframe=timeframe_str,
                timestamp=latest_candle_timestamp,
                features_calculated=len(features_df.columns) - 1,
                features_stored=stored_count,
                validation_errors=validation_errors,
                processing_time_ms=processing_time,
                quality_score=quality_score,
            ), feature_models
        except Exception as e:
            await self.error_handler.handle_error(
                "feature_calculator",
                f"Error in _calculate_timeframe_features for {timeframe_str}: {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe_str},
            )

            # Record failure metrics
            processing_duration = (datetime.now() - tf_start_time).total_seconds()
            if metrics_registry.is_enabled():
                metrics_registry.increment_counter(
                    "feature_calculations_total",
                    {
                        "instrument": str(instrument_id),
                        "timeframe": timeframe_str,
                        "feature": "aggregate",
                        "status": "False",
                    },
                )
                metrics_registry.observe_histogram(
                    "feature_calculation_duration_seconds",
                    processing_duration,
                    {"instrument": str(instrument_id), "timeframe": timeframe_str, "feature": "aggregate"},
                )

            return self._create_failure_result(
                instrument_id, timeframe_str, latest_candle_timestamp, [str(e)], tf_start_time
            )

    async def _calculate_all_features(self, df: pd.DataFrame, timeframe_minutes: int, segment: str) -> pd.DataFrame:
        loop = asyncio.get_running_loop()

        def _run_calculations() -> pd.DataFrame:
            # MODULAR SYSTEM ONLY - NO FALLBACKS
            if not self.use_modular_indicators or self.modular_calculator is None:
                raise RuntimeError("❌ CRITICAL: Modular indicator system not available - initialization failed")

            try:
                logger.debug(f"🔄 Calculating features using modular system for {timeframe_minutes}m")
                return self._calculate_features_modular(df, timeframe_minutes, segment)
            except Exception as e:
                logger.error(f"🚨 FEATURE CALCULATION FAILED for {timeframe_minutes}m: {e}")
                logger.error("💀 NO FALLBACK: Fix the modular indicator system!")
                raise RuntimeError(f"Feature calculation failed: {e}") from e

        return await loop.run_in_executor(None, _run_calculations)

    def _calculate_features_modular(self, df: pd.DataFrame, timeframe_minutes: int, segment: str) -> pd.DataFrame:
        """Calculate features using the modular indicator system with enhanced indicators support."""
        try:
            # Skip volume-based indicators if segment is an index
            is_index = segment in config.broker.segment_types and config.broker.segment_types.index(segment) == 0

            # Get all features using modular system (now includes enhanced indicators)
            if not self.modular_calculator:
                raise RuntimeError("Modular calculator not available")

            logger.debug(
                f"🔄 Calculating modular features (base + enhanced) for {timeframe_minutes}m with {len(df)} candles"
            )
            features_df = self.modular_calculator.calculate_all_features(df, timeframe_minutes)

            if features_df.empty:
                raise RuntimeError(
                    f"🚨 CRITICAL FEATURE FAILURE: Modular system returned empty DataFrame for {timeframe_minutes}m. Input data shape: {df.shape}. Check indicator configurations and data quality - Trading system compromised"
                )

            # Filter out volume-based features for index instruments
            if is_index:
                volume_features = ["obv", "adosc", "ultosc", "volume_confirmed_breakout"]
                cols_to_drop = [col for col in features_df.columns if any(vf in col.lower() for vf in volume_features)]
                if cols_to_drop:
                    features_df = features_df.drop(columns=cols_to_drop)
                    logger.debug(f"Dropped {len(cols_to_drop)} volume-based features for index instrument")

            # Log feature breakdown for enhanced visibility
            base_features = [
                col
                for col in features_df.columns
                if not any(
                    enhanced in col
                    for enhanced in [
                        "momentum_ratio",
                        "volatility_adjusted",
                        "trend_strength",
                        "trend_consensus",
                        "mean_reversion",
                        "breakout_strength",
                        "crossover_signal",
                    ]
                )
            ]
            enhanced_features = [col for col in features_df.columns if col not in base_features]

            logger.info(
                f"✅ Modular system calculated {len(features_df.columns)} total features for {timeframe_minutes}m:"
            )
            logger.info(f"   📊 Base indicators: {len(base_features)} features")
            logger.info(f"   🚀 Enhanced indicators: {len(enhanced_features)} features")
            logger.debug(f"Base features: {base_features}")
            logger.debug(f"Enhanced features: {enhanced_features}")

            return features_df

        except Exception as e:
            logger.error(f"❌ Modular feature calculation failed: {e}")
            raise

    def _calculate_features_legacy(self, df: pd.DataFrame, timeframe_minutes: int, segment: str) -> pd.DataFrame:
        """Calculate features using the legacy dispatch system."""
        features_dict = {}
        expected_length = len(df)

        # Skip volume-based indicators if segment is an index
        is_index = segment in config.broker.segment_types and config.broker.segment_types.index(segment) == 0

        for fc in self.feature_configs:
            try:
                # Skip volume-based features if no volume data
                if is_index and fc.function in [
                    "OBV",
                    "ADOSC",
                    "ULTOSC",
                ]:
                    logger.debug(f"Skipping volume-based feature '{fc.name}' for index instrument.")
                    continue

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
            logger.debug(
                f"Legacy system calculated {len(features_dict)} features with {len(features_df)} rows for {timeframe_minutes}m"
            )
            return features_df
        raise RuntimeError(
            f"🚨 CRITICAL FEATURE FAILURE: No features calculated for {timeframe_minutes}m timeframe. Input data shape: {df.shape if df is not None else 'None'} - Trading system completely compromised"
        )

    def _calculate_single_feature(
        self, df: pd.DataFrame, func_name: str, params: dict[str, Any]
    ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        try:
            if func_name in self._feature_dispatch:
                return self._feature_dispatch[func_name](df, params)
            raise RuntimeError(
                f"🚨 CRITICAL CONFIGURATION ERROR: Feature function '{func_name}' not found in dispatch table. Available functions: {list(self._feature_dispatch.keys())} - Configuration corrupted"
            )
        except Exception as e:
            logger.error(f"Error calculating feature {func_name}: {e}")
            return None

    def _create_failure_result(
        self, instrument_id: int, timeframe_str: str, timestamp: datetime, errors: list[str], start_time: datetime
    ) -> tuple[FeatureCalculationResult, None]:
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
        ), None

    async def _enhance_with_feature_engineering(
        self,
        features_df: pd.DataFrame,
        instrument_id: int,
        timeframe_str: str,
        latest_candle_timestamp: datetime,
        segment: str,
        ohlcv_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Enhanced feature integration - uses modular system when available, falls back to pipeline
        """
        try:
            if not config.model_training or not config.model_training.feature_engineering.enabled:
                logger.debug("Feature engineering disabled, returning original features")
                return features_df

            # Check if modular enhanced indicators are already included
            enhanced_feature_patterns = [
                "momentum_ratio",
                "volatility_adjusted",
                "trend_strength",
                "trend_consensus",
                "mean_reversion",
                "breakout_strength",
                "crossover_signal",
            ]

            existing_enhanced_features = [
                col for col in features_df.columns if any(pattern in col for pattern in enhanced_feature_patterns)
            ]

            if existing_enhanced_features:
                logger.info(
                    f"🚀 Modular enhanced indicators already included: {len(existing_enhanced_features)} features"
                )
                logger.debug(f"Enhanced features from modular system: {existing_enhanced_features}")

                # Enhanced features are already calculated by modular system - no additional processing needed
                return features_df

            # No enhanced features detected from modular system - this indicates configuration issue
            logger.warning(
                f"🚨 ENHANCED INDICATORS MISSING: No enhanced features detected for {instrument_id} ({timeframe_str}). "
                f"Check that enhanced indicators are enabled in config.yaml under trading.features.indicator_control.categories.enhanced"
            )

            # Return original features - enhanced indicators should be handled by modular system
            logger.info(
                "📊 Continuing with base features only - enhanced features should be configured in modular system"
            )
            return features_df

        except Exception as e:
            logger.error(f"Feature engineering enhancement failed: {e}")
            await self.error_handler.handle_error(
                "feature_engineering",
                f"Enhancement failed for {instrument_id} ({timeframe_str}): {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe_str},
            )
            return features_df

    async def _apply_feature_selection(
        self, features_df: pd.DataFrame, instrument_id: int, timeframe_str: str
    ) -> pd.DataFrame:
        """
        Apply feature selection in live mode to use only selected features
        """
        try:
            if not config.model_training or not config.model_training.feature_engineering.feature_selection.enabled:
                logger.debug("Feature selection disabled, returning all features")
                return features_df

            # Get selected features for this instrument/timeframe
            selected_features = await self.feature_engineering.get_selected_features(instrument_id, timeframe_str)

            if not selected_features:
                logger.debug(f"No feature selection found for {instrument_id}_{timeframe_str}, using all features")
                return features_df

            # Filter features to only include selected ones + timestamp
            available_features = set(features_df.columns) - {"timestamp"}
            features_to_keep = available_features.intersection(selected_features)

            if not features_to_keep:
                raise RuntimeError(
                    f"🚨 CRITICAL SELECTION FAILURE: No selected features available for {instrument_id}_{timeframe_str}. Available: {list(available_features)}, Selected: {list(selected_features)} - Feature selection system broken"
                )

            # Keep timestamp column and selected features
            columns_to_keep = ["timestamp"] + list(features_to_keep)
            selected_df = features_df[columns_to_keep].copy()

            logger.debug(
                f"Applied feature selection: {len(features_to_keep)}/{len(available_features)} features selected"
            )
            return selected_df

        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            await self.error_handler.handle_error(
                "feature_selection",
                f"Selection failed for {instrument_id} ({timeframe_str}): {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe_str},
            )
            return features_df

    def _normalize_feature_name(self, column_name: str) -> str:
        """
        Normalize feature names to match expected naming conventions
        """
        # Handle TALIB multi-output features
        name_mapping = config.trading.features.name_mapping

        normalized = name_mapping.get(column_name.lower(), column_name.lower())

        # Add period suffix for common indicators if not present
        if normalized in ["rsi", "atr"] and not any(char.isdigit() for char in normalized):
            normalized += "_14"  # Default period

        return normalized

    def _normalize_modular_feature_name(self, column_name: str) -> str:
        """
        Normalize feature names from modular system for consistent naming
        """
        # Apply existing name mapping first
        normalized = self._normalize_feature_name(column_name)

        # Handle modular system specific naming patterns
        modular_mappings = {
            "macdsignal": "macd_signal",
            "macdhist": "macd_hist",
            "upperband": "bb_upper",
            "middleband": "bb_middle",
            "lowerband": "bb_lower",
            "aroonup": "aroon_up",
            "aroondown": "aroon_down",
            "slowk": "stoch_k",
            "slowd": "stoch_d",
            # Additional mappings for better compatibility
            "rsi": "rsi_14",  # Default RSI period mapping
            "atr": "atr_14",  # Default ATR period mapping
        }

        return modular_mappings.get(normalized.lower(), normalized)

    def _validate_modular_system_health(self) -> bool:
        """
        Validate that the modular indicator system is functioning properly
        """
        logger.info("Starting modular system health check...")
        if not self.use_modular_indicators:
            logger.info("Modular indicators disabled (use_modular_indicators=False), skipping health check")
            return True

        if self.modular_calculator is None:
            logger.error("Health check failed: use_modular_indicators=True but modular_calculator=None")
            return False

        try:
            # Check if indicators are properly loaded
            indicators = self.modular_calculator.registry.get_all_indicators()
            if len(indicators) == 0:
                raise RuntimeError(
                    "🚨 CRITICAL INDICATOR FAILURE: No indicators loaded in modular system - Zero indicators registered, trading system inoperable"
                )

            # Get registry stats for detailed info
            stats = self.modular_calculator.registry.get_registry_stats()
            logger.info(f"Modular system stats: {stats}")

            # Check if enabled indicators from config are available in the system
            enabled_indicators_from_config = []
            if (
                hasattr(config, "trading")
                and hasattr(config.trading, "features")
                and hasattr(config.trading.features, "configurations")
            ):
                for indicator_config in config.trading.features.configurations:
                    # Check if indicator is not in disabled list
                    disabled_indicators = getattr(config.trading.features.indicator_control, "disabled_indicators", [])
                    if indicator_config.name not in disabled_indicators:
                        enabled_indicators_from_config.append(indicator_config.name)

            logger.info(f"Enabled indicators from config: {enabled_indicators_from_config}")
            available_indicators = [name.lower() for name in indicators]
            logger.info(f"Available indicators in system: {list(available_indicators)}")

            if enabled_indicators_from_config:
                missing_from_config = [ind for ind in enabled_indicators_from_config if ind not in available_indicators]
                logger.info(f"Missing indicators: {missing_from_config}")

                if missing_from_config:
                    raise RuntimeError(
                        f"🚨 CRITICAL INDICATOR FAILURE: Enabled indicators from config missing in modular system: {missing_from_config}. Available indicators: {list(available_indicators)} - Configured indicators unavailable"
                    )

            logger.info(f"✅ Modular system health check passed: {len(indicators)} indicators available")
            return True

        except Exception as e:
            logger.error(f"Error validating modular system: {e}")
            return False
