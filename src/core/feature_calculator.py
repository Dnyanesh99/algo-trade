import asyncio
from datetime import datetime
from typing import Any, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

from src.core.feature_validator import FeatureValidator
from src.database.feature_repo import FeatureRepository
from src.database.models import FeatureData
from src.database.ohlcv_repo import OHLCVRepository
from src.metrics import metrics_registry
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

talib: Optional[Any]
try:
    import talib
except ImportError:
    logger.warning("TA-Lib not found. Using pandas-based TA calculations, which may be slower.")
    talib = None

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
        error_handler: ErrorHandler,
        health_monitor: HealthMonitor,
    ):
        self.ohlcv_repo = ohlcv_repo
        self.feature_repo = feature_repo
        self.error_handler = error_handler
        self.health_monitor = health_monitor
        self.feature_validator = FeatureValidator()
        if config.trading is None:
            raise ValueError("Trading configuration is required")
        if config.data_quality is None:
            raise ValueError("Data quality configuration is required")
        self.timeframes = config.trading.feature_timeframes
        self.lookback_period = config.trading.features.lookback_period
        self.validation_enabled = config.data_quality.validation.enabled

        # NEW: Feature Engineering Pipeline Integration
        from src.core.feature_engineering_pipeline import FeatureEngineeringPipeline

        self.feature_engineering = FeatureEngineeringPipeline(
            ohlcv_repo, feature_repo, self.feature_validator, error_handler, health_monitor
        )
        self.feature_configs = self._load_feature_configs()
        self._feature_dispatch = self._build_feature_dispatch()
        logger.info(
            f"FeatureCalculator initialized for timeframes {self.timeframes} with {len(self.feature_configs)} features."
        )

    def _load_feature_configs(self) -> list[FeatureConfig]:
        """Loads and parses feature configurations from config.yaml."""
        try:
            if config.trading is None or config.trading.features is None:
                raise ValueError("Trading feature configuration is required")
            features_list = config.trading.features.configurations
            return [FeatureConfig(name=fc.name, function=fc.function, params=fc.params) for fc in features_list]
        except Exception as e:
            logger.error(f"Error loading feature configurations: {e}")
            return []

    def _build_feature_dispatch(self) -> dict[str, Any]:
        """Builds the feature dispatch table mapping function names to implementations."""
        dispatch = {}

        # Use TA-Lib functions if available, otherwise use pandas-based implementations
        if talib is not None:
            dispatch.update(
                {
                    # Trend-Following Indicators
                    "MACD": lambda df, params: self._calculate_talib_macd(df, params),
                    "ADX": lambda df, params: self._calculate_talib_adx(df, params),
                    "SAR": lambda df, params: self._calculate_talib_sar(df, params),
                    "AROON": lambda df, params: self._calculate_talib_aroon(df, params),
                    "HT_TRENDLINE": lambda df, params: self._calculate_talib_ht_trendline(df, params),
                    # Momentum Indicators
                    "RSI": lambda df, params: self._calculate_talib_rsi(df, params),
                    "STOCH": lambda df, params: self._calculate_talib_stoch(df, params),
                    "WILLR": lambda df, params: self._calculate_talib_willr(df, params),
                    "CMO": lambda df, params: self._calculate_talib_cmo(df, params),
                    "BOP": lambda df, params: self._calculate_talib_bop(df, params),
                    # Volatility Indicators
                    "ATR": lambda df, params: self._calculate_talib_atr(df, params),
                    "BBANDS": lambda df, params: self._calculate_talib_bbands(df, params),
                    "STDDEV": lambda df, params: self._calculate_talib_stddev(df, params),
                    # Volume Indicators
                    "OBV": lambda df, params: self._calculate_talib_obv(df, params),
                    "ADOSC": lambda df, params: self._calculate_talib_adosc(df, params),
                    # Cycle Indicators
                    "HT_DCPERIOD": lambda df, params: self._calculate_talib_ht_dcperiod(df, params),
                    "HT_PHASOR": lambda df, params: self._calculate_talib_ht_phasor(df, params),
                    # Other/Price Transformation Indicators
                    "ULTOSC": lambda df, params: self._calculate_talib_ultosc(df, params),
                    "TRANGE": lambda df, params: self._calculate_talib_trange(df, params),
                    "CCI": lambda df, params: self._calculate_talib_cci(df, params),
                }
            )
        else:
            # Pandas-based fallback implementations
            dispatch.update(
                {
                    # Trend-Following Indicators
                    "MACD": lambda df, params: self._calculate_pandas_macd(df, params),
                    "ADX": lambda df, params: self._calculate_pandas_adx(df, params),
                    "SAR": lambda df, params: self._calculate_pandas_sar(df, params),
                    "AROON": lambda df, params: self._calculate_pandas_aroon(df, params),
                    "HT_TRENDLINE": lambda df, params: self._calculate_pandas_ht_trendline(df, params),
                    # Momentum Indicators
                    "RSI": lambda df, params: self._calculate_pandas_rsi(df, params),
                    "STOCH": lambda df, params: self._calculate_pandas_stoch(df, params),
                    "WILLR": lambda df, params: self._calculate_pandas_willr(df, params),
                    "CMO": lambda df, params: self._calculate_pandas_cmo(df, params),
                    "BOP": lambda df, params: self._calculate_pandas_bop(df, params),
                    # Volatility Indicators
                    "ATR": lambda df, params: self._calculate_pandas_atr(df, params),
                    "BBANDS": lambda df, params: self._calculate_pandas_bbands(df, params),
                    "STDDEV": lambda df, params: self._calculate_pandas_stddev(df, params),
                    # Volume Indicators
                    "OBV": lambda df, params: self._calculate_pandas_obv(df, params),
                    "ADOSC": lambda df, params: self._calculate_pandas_adosc(df, params),
                    # Cycle Indicators
                    "HT_DCPERIOD": lambda df, params: self._calculate_pandas_ht_dcperiod(df, params),
                    "HT_PHASOR": lambda df, params: self._calculate_pandas_ht_phasor(df, params),
                    # Other/Price Transformation Indicators
                    "ULTOSC": lambda df, params: self._calculate_pandas_ultosc(df, params),
                    "TRANGE": lambda df, params: self._calculate_pandas_trange(df, params),
                    "CCI": lambda df, params: self._calculate_pandas_cci(df, params),
                }
            )

        return dispatch

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

            # NEW: Feature Engineering Integration
            enhanced_features_df = await self._enhance_with_feature_engineering(
                features_df, instrument_id, timeframe_str, latest_candle_timestamp
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
                    {"instrument": str(instrument_id), "timeframe": timeframe_str, "success": str(success)},
                )
                metrics_registry.observe_histogram(
                    "feature_calculation_duration_seconds",
                    processing_duration,
                    {"instrument": str(instrument_id), "timeframe": timeframe_str},
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
            )
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
                    {"instrument": str(instrument_id), "timeframe": timeframe_str, "success": "False"},
                )
                metrics_registry.observe_histogram(
                    "feature_calculation_duration_seconds",
                    processing_duration,
                    {"instrument": str(instrument_id), "timeframe": timeframe_str},
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
                logger.debug(
                    f"Calculated {len(features_dict)} features with {len(features_df)} rows for {timeframe_minutes}m"
                )
                return features_df
            logger.warning(f"No features calculated for {timeframe_minutes}m timeframe")
            return pd.DataFrame()

        return await loop.run_in_executor(None, _run_calculations)

    def _calculate_single_feature(
        self, df: pd.DataFrame, func_name: str, params: dict
    ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        try:
            if func_name in self._feature_dispatch:
                return self._feature_dispatch[func_name](df, params)
            logger.warning(f"Feature function '{func_name}' not found in dispatch table.")
            return None
        except Exception as e:
            logger.error(f"Error calculating feature {func_name}: {e}")
            return None

    def _create_failure_result(
        self, instrument_id: int, timeframe_str: str, timestamp: datetime, errors: list[str], start_time: datetime
    ) -> FeatureCalculationResult:
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

    async def _enhance_with_feature_engineering(
        self, features_df: pd.DataFrame, instrument_id: int, timeframe_str: str, latest_candle_timestamp: datetime
    ) -> pd.DataFrame:
        """
        Enhance base features with engineered features from FeatureEngineeringPipeline
        """
        try:
            if not config.model_training or not config.model_training.feature_engineering.enabled:
                logger.debug("Feature engineering disabled, returning original features")
                return features_df

            # Get the latest timestamp's features for engineering
            latest_features = features_df[features_df.timestamp == features_df.timestamp.max()]
            if latest_features.empty:
                logger.warning("No latest features found for engineering")
                return features_df

            # Convert latest features to dictionary for pipeline
            base_features_dict = {}
            for col in latest_features.columns:
                if col != "timestamp" and not pd.isna(latest_features[col].iloc[0]):
                    # Convert column names to match expected feature names (handle TALIB naming)
                    feature_name = self._normalize_feature_name(col)
                    base_features_dict[feature_name] = float(latest_features[col].iloc[0])

            logger.debug(f"Extracted {len(base_features_dict)} base features for engineering")

            # Generate engineered features
            engineered_features = await self.feature_engineering.generate_engineered_features(
                instrument_id, timeframe_str, latest_candle_timestamp, base_features_dict
            )

            if not engineered_features:
                logger.debug("No engineered features generated")
                return features_df

            # Add engineered features to the latest row of features_df
            enhanced_df = features_df.copy()
            latest_timestamp = enhanced_df.timestamp.max()
            latest_idx = enhanced_df[enhanced_df.timestamp == latest_timestamp].index

            for feature_name, eng_feature in engineered_features.items():
                if (
                    eng_feature.quality_score
                    >= config.model_training.feature_engineering.feature_generation.min_quality_score
                ):
                    enhanced_df.loc[latest_idx, f"eng_{feature_name}"] = eng_feature.value

            logger.debug(f"Added {len(engineered_features)} engineered features to feature set")
            return enhanced_df

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
                logger.warning(
                    f"No selected features available in current feature set for {instrument_id}_{timeframe_str}"
                )
                return features_df

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

    # TA-Lib implementations
    def _calculate_talib_macd(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Calculate MACD using TA-Lib."""
        fastperiod = params.get("fastperiod", 12)
        slowperiod = params.get("slowperiod", 26)
        signalperiod = params.get("signalperiod", 9)

        if talib is None:
            raise RuntimeError("TA-Lib not available for MACD calculation")
        macd, macdsignal, macdhist = talib.MACD(
            df["close"].values, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod
        )

        return pd.DataFrame({"macd": macd, "macdsignal": macdsignal, "macdhist": macdhist}, index=df.index)

    def _calculate_talib_rsi(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate RSI using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        if talib is None:
            raise RuntimeError("TA-Lib not available for RSI calculation")
        rsi = talib.RSI(df["close"].values, timeperiod=timeperiod)
        return pd.Series(rsi, index=df.index)

    def _calculate_talib_bbands(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Calculate Bollinger Bands using TA-Lib."""
        timeperiod = params.get("timeperiod", 20)
        nbdevup = params.get("nbdevup", 2.0)
        nbdevdn = params.get("nbdevdn", 2.0)

        if talib is None:
            raise RuntimeError("TA-Lib not available for BBANDS calculation")
        upperband, middleband, lowerband = talib.BBANDS(
            df["close"].values, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn
        )

        return pd.DataFrame({"upperband": upperband, "middleband": middleband, "lowerband": lowerband}, index=df.index)

    def _calculate_talib_atr(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate ATR using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        if talib is None:
            raise RuntimeError("TA-Lib not available for ATR calculation")
        atr = talib.ATR(df["high"].values, df["low"].values, df["close"].values, timeperiod=timeperiod)
        return pd.Series(atr, index=df.index)

    def _calculate_talib_adx(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate ADX using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        if talib is None:
            raise RuntimeError("TA-Lib not available for ADX calculation")
        adx = talib.ADX(df["high"].values, df["low"].values, df["close"].values, timeperiod=timeperiod)
        return pd.Series(adx, index=df.index)

    def _calculate_talib_sar(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate SAR using TA-Lib."""
        acceleration = params.get("acceleration", 0.02)
        maximum = params.get("maximum", 0.2)
        if talib is None:
            raise RuntimeError("TA-Lib not available for SAR calculation")
        sar = talib.SAR(df["high"].values, df["low"].values, acceleration=acceleration, maximum=maximum)
        return pd.Series(sar, index=df.index)

    def _calculate_talib_aroon(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Calculate AROON using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        if talib is None:
            raise RuntimeError("TA-Lib not available for AROON calculation")
        aroondown, aroonup = talib.AROON(df["high"].values, df["low"].values, timeperiod=timeperiod)
        return pd.DataFrame({"aroondown": aroondown, "aroonup": aroonup}, index=df.index)

    def _calculate_talib_ht_trendline(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate HT_TRENDLINE using TA-Lib."""
        if talib is None:
            raise RuntimeError("TA-Lib not available for HT_TRENDLINE calculation")
        ht_trendline = talib.HT_TRENDLINE(df["close"].values)
        return pd.Series(ht_trendline, index=df.index)

    def _calculate_talib_stoch(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Calculate STOCH using TA-Lib."""
        fastk_period = params.get("fastk_period", 14)
        slowk_period = params.get("slowk_period", 3)
        slowd_period = params.get("slowd_period", 3)

        if talib is None:
            raise RuntimeError("TA-Lib not available for STOCH calculation")
        slowk, slowd = talib.STOCH(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowd_period=slowd_period,
        )

        return pd.DataFrame({"slowk": slowk, "slowd": slowd}, index=df.index)

    def _calculate_talib_willr(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate WILLR using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        if talib is None:
            raise RuntimeError("TA-Lib not available for WILLR calculation")
        willr = talib.WILLR(df["high"].values, df["low"].values, df["close"].values, timeperiod=timeperiod)
        return pd.Series(willr, index=df.index)

    def _calculate_talib_cmo(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate CMO using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        if talib is None:
            raise RuntimeError("TA-Lib not available for CMO calculation")
        cmo = talib.CMO(df["close"].values, timeperiod=timeperiod)
        return pd.Series(cmo, index=df.index)

    def _calculate_talib_bop(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate BOP using TA-Lib."""
        if talib is None:
            raise RuntimeError("TA-Lib not available for BOP calculation")
        bop = talib.BOP(df["open"].values, df["high"].values, df["low"].values, df["close"].values)
        return pd.Series(bop, index=df.index)

    def _calculate_talib_stddev(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate STDDEV using TA-Lib."""
        timeperiod = params.get("timeperiod", 5)
        nbdev = params.get("nbdev", 1.0)
        if talib is None:
            raise RuntimeError("TA-Lib not available for STDDEV calculation")
        stddev = talib.STDDEV(df["close"].values, timeperiod=timeperiod, nbdev=nbdev)
        return pd.Series(stddev, index=df.index)

    def _calculate_talib_obv(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate OBV using TA-Lib."""
        if talib is None:
            raise RuntimeError("TA-Lib not available for OBV calculation")
        obv = talib.OBV(df["close"].values, df["volume"].values)
        return pd.Series(obv, index=df.index)

    def _calculate_talib_adosc(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate ADOSC using TA-Lib."""
        fastperiod = params.get("fastperiod", 3)
        slowperiod = params.get("slowperiod", 10)
        if talib is None:
            raise RuntimeError("TA-Lib not available for ADOSC calculation")
        adosc = talib.ADOSC(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            df["volume"].values,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
        )
        return pd.Series(adosc, index=df.index)

    def _calculate_talib_ht_dcperiod(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate HT_DCPERIOD using TA-Lib."""
        if talib is None:
            raise RuntimeError("TA-Lib not available for HT_DCPERIOD calculation")
        ht_dcperiod = talib.HT_DCPERIOD(df["close"].values)
        return pd.Series(ht_dcperiod, index=df.index)

    def _calculate_talib_ht_phasor(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Calculate HT_PHASOR using TA-Lib."""
        if talib is None:
            raise RuntimeError("TA-Lib not available for HT_PHASOR calculation")
        inphase, quadrature = talib.HT_PHASOR(df["close"].values)
        return pd.DataFrame({"inphase": inphase, "quadrature": quadrature}, index=df.index)

    def _calculate_talib_ultosc(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate ULTOSC using TA-Lib."""
        timeperiod1 = params.get("timeperiod1", 7)
        timeperiod2 = params.get("timeperiod2", 14)
        timeperiod3 = params.get("timeperiod3", 28)
        if talib is None:
            raise RuntimeError("TA-Lib not available for ULTOSC calculation")
        ultosc = talib.ULTOSC(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            timeperiod1=timeperiod1,
            timeperiod2=timeperiod2,
            timeperiod3=timeperiod3,
        )
        return pd.Series(ultosc, index=df.index)

    def _calculate_talib_trange(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate TRANGE using TA-Lib."""
        if talib is None:
            raise RuntimeError("TA-Lib not available for TRANGE calculation")
        trange = talib.TRANGE(df["high"].values, df["low"].values, df["close"].values)
        return pd.Series(trange, index=df.index)

    def _calculate_talib_cci(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate CCI using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        if talib is None:
            raise RuntimeError("TA-Lib not available for CCI calculation")
        cci = talib.CCI(df["high"].values, df["low"].values, df["close"].values, timeperiod=timeperiod)
        return pd.Series(cci, index=df.index)

    # Pandas-based fallback implementations
    def _calculate_pandas_macd(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Calculate MACD using pandas."""
        fastperiod = params.get("fastperiod", 12)
        slowperiod = params.get("slowperiod", 26)
        signalperiod = params.get("signalperiod", 9)

        exp1 = df["close"].ewm(span=fastperiod).mean()
        exp2 = df["close"].ewm(span=slowperiod).mean()
        macd = exp1 - exp2
        macdsignal = macd.ewm(span=signalperiod).mean()
        macdhist = macd - macdsignal

        return pd.DataFrame({"macd": macd, "macdsignal": macdsignal, "macdhist": macdhist}, index=df.index)

    def _calculate_pandas_rsi(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate RSI using pandas."""
        timeperiod = params.get("timeperiod", 14)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_pandas_bbands(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Calculate Bollinger Bands using pandas."""
        timeperiod = params.get("timeperiod", 20)
        nbdevup = params.get("nbdevup", 2.0)
        nbdevdn = params.get("nbdevdn", 2.0)

        middleband = df["close"].rolling(window=timeperiod).mean()
        std = df["close"].rolling(window=timeperiod).std()
        upperband = middleband + (std * nbdevup)
        lowerband = middleband - (std * nbdevdn)

        return pd.DataFrame({"upperband": upperband, "middleband": middleband, "lowerband": lowerband}, index=df.index)

    def _calculate_pandas_atr(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate ATR using pandas."""
        timeperiod = params.get("timeperiod", 14)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=timeperiod).mean()

    def _calculate_pandas_adx(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate ADX using pandas (simplified version)."""
        timeperiod = params.get("timeperiod", 14)
        # Simplified ADX calculation
        high_diff = df["high"].diff()
        low_diff = df["low"].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        atr = self._calculate_pandas_atr(df, {"timeperiod": timeperiod})
        plus_di = 100 * (plus_dm.rolling(window=timeperiod).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=timeperiod).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        return dx.rolling(window=timeperiod).mean()

    def _calculate_pandas_sar(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate SAR using pandas (simplified version)."""
        # Simplified SAR calculation - return close price for now
        # Parameters are available but not used in this simplified implementation
        _ = params.get("acceleration", 0.02)
        _ = params.get("maximum", 0.2)
        return df["close"]

    def _calculate_pandas_aroon(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Calculate AROON using pandas."""
        timeperiod = params.get("timeperiod", 14)
        aroonup = df["high"].rolling(window=timeperiod).apply(lambda x: (timeperiod - x.argmax()) / timeperiod * 100)
        aroondown = df["low"].rolling(window=timeperiod).apply(lambda x: (timeperiod - x.argmin()) / timeperiod * 100)
        return pd.DataFrame({"aroondown": aroondown, "aroonup": aroonup}, index=df.index)

    def _calculate_pandas_ht_trendline(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate HT_TRENDLINE using pandas (simplified version)."""
        # Simplified implementation - return EMA
        return df["close"].ewm(span=14).mean()

    def _calculate_pandas_stoch(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Calculate STOCH using pandas."""
        fastk_period = params.get("fastk_period", 14)
        slowk_period = params.get("slowk_period", 3)
        slowd_period = params.get("slowd_period", 3)

        lowest_low = df["low"].rolling(window=fastk_period).min()
        highest_high = df["high"].rolling(window=fastk_period).max()
        fastk = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        slowk = fastk.rolling(window=slowk_period).mean()
        slowd = slowk.rolling(window=slowd_period).mean()

        return pd.DataFrame({"slowk": slowk, "slowd": slowd}, index=df.index)

    def _calculate_pandas_willr(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate WILLR using pandas."""
        timeperiod = params.get("timeperiod", 14)
        highest_high = df["high"].rolling(window=timeperiod).max()
        lowest_low = df["low"].rolling(window=timeperiod).min()
        return -100 * (highest_high - df["close"]) / (highest_high - lowest_low)

    def _calculate_pandas_cmo(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate CMO using pandas."""
        timeperiod = params.get("timeperiod", 14)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=timeperiod).sum()
        loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).sum()
        return 100 * (gain - loss) / (gain + loss)

    def _calculate_pandas_bop(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate BOP using pandas."""
        return (df["close"] - df["open"]) / (df["high"] - df["low"])

    def _calculate_pandas_stddev(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate STDDEV using pandas."""
        timeperiod = params.get("timeperiod", 5)
        return df["close"].rolling(window=timeperiod).std()

    def _calculate_pandas_obv(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate OBV using pandas."""
        return (df["volume"] * df["close"].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)).cumsum()

    def _calculate_pandas_adosc(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate ADOSC using pandas (simplified version)."""
        fastperiod = params.get("fastperiod", 3)
        slowperiod = params.get("slowperiod", 10)
        # Simplified implementation
        ad = df["volume"] * (df["close"] - df["low"] - df["high"] + df["close"]) / (df["high"] - df["low"])
        return ad.ewm(span=fastperiod).mean() - ad.ewm(span=slowperiod).mean()

    def _calculate_pandas_ht_dcperiod(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate HT_DCPERIOD using pandas (simplified version)."""
        # Simplified implementation - return constant
        return pd.Series([30] * len(df), index=df.index)

    def _calculate_pandas_ht_phasor(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Calculate HT_PHASOR using pandas (simplified version)."""
        # Simplified implementation
        return pd.DataFrame({"inphase": df["close"] * 0.5, "quadrature": df["close"] * 0.5}, index=df.index)

    def _calculate_pandas_ultosc(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate ULTOSC using pandas."""
        timeperiod1 = params.get("timeperiod1", 7)
        timeperiod2 = params.get("timeperiod2", 14)
        timeperiod3 = params.get("timeperiod3", 28)

        # Simplified Ultimate Oscillator
        bp = df["close"] - df[["low", "close"]].shift().min(axis=1)
        tr = df[["high", "close"]].shift().max(axis=1) - df[["low", "close"]].shift().min(axis=1)

        avg1 = bp.rolling(window=timeperiod1).sum() / tr.rolling(window=timeperiod1).sum()
        avg2 = bp.rolling(window=timeperiod2).sum() / tr.rolling(window=timeperiod2).sum()
        avg3 = bp.rolling(window=timeperiod3).sum() / tr.rolling(window=timeperiod3).sum()

        return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

    def _calculate_pandas_trange(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate TRANGE using pandas."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    def _calculate_pandas_cci(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate CCI using pandas."""
        timeperiod = params.get("timeperiod", 14)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma = typical_price.rolling(window=timeperiod).mean()
        mad = typical_price.rolling(window=timeperiod).apply(lambda x: (x - x.mean()).abs().mean())
        return (typical_price - sma) / (0.015 * mad)
