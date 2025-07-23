"""
Complete Feature Engineering Pipeline Implementation
Integrates automated feature generation, selection, and cross-asset features
with existing FeatureCalculator using all current indicators from config.yaml
"""

import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytz
from pydantic import BaseModel

from src.database.feature_repo import FeatureRepository
from src.database.ohlcv_repo import OHLCVRepository
from src.metrics.registry import metrics_registry
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import ConfigLoader, FeatureEngineeringConfig
from src.utils.logger import LOGGER as logger
from src.validation.feature_validator import FeatureValidator

config = ConfigLoader().get_config()


class EngineeredFeature(BaseModel):
    """Model for engineered feature data"""

    name: str
    value: float
    generation_method: str
    source_features: list[str]
    quality_score: float


class FeatureScore(BaseModel):
    """Model for feature selection scoring"""

    feature_name: str
    importance_score: float
    stability_score: float
    consistency_score: float
    composite_score: float


class FeatureEngineeringPipeline:
    """
    Production-grade feature engineering pipeline that integrates with existing codebase.
    Uses actual indicators from trading.features.configurations in config.yaml
    """

    def __init__(
        self,
        ohlcv_repo: OHLCVRepository,
        feature_repo: FeatureRepository,
        feature_validator: FeatureValidator,
        error_handler: ErrorHandler,
        health_monitor: HealthMonitor,
    ):
        if not all(isinstance(repo, (OHLCVRepository, FeatureRepository)) for repo in [ohlcv_repo, feature_repo]):
            raise TypeError("ohlcv_repo and feature_repo must be valid repository instances.")
        if not isinstance(feature_validator, FeatureValidator):
            raise TypeError("feature_validator must be a valid FeatureValidator instance.")
        if not isinstance(error_handler, ErrorHandler) or not isinstance(health_monitor, HealthMonitor):
            raise TypeError("error_handler and health_monitor must be valid instances.")

        self.ohlcv_repo = ohlcv_repo
        self.feature_repo = feature_repo
        self.feature_validator = feature_validator
        self.error_handler = error_handler
        self.health_monitor = health_monitor

        if not config.model_training or not config.model_training.feature_engineering:
            raise ValueError("model_training.feature_engineering configuration is required.")
        self.config: FeatureEngineeringConfig = config.model_training.feature_engineering

        if not config.trading or not config.trading.features:
            raise ValueError("trading.features configuration is required.")
        self.base_feature_configs = config.trading.features.configurations
        self.feature_timeframes = config.trading.feature_timeframes

        self.importance_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.correlation_cache: dict[str, tuple[float, datetime]] = {}
        self.selected_features_cache: dict[str, set[str]] = {}

        logger.info("FeatureEngineeringPipeline initialized with production configuration.")

    async def generate_engineered_features(
        self,
        instrument_id: int,
        timeframe: str,
        timestamp: datetime,
        base_features: dict[str, float],
        ohlcv_df: pd.DataFrame,
        segment: str,
    ) -> dict[str, EngineeredFeature]:
        """
        Generate engineered features using existing base features from FeatureCalculator.
        """
        if not self.config.feature_generation.enabled:
            logger.debug("Feature engineering is disabled in the configuration.")
            return {}

        start_time = time.time()
        engineered_features: dict[str, EngineeredFeature] = {}
        logger.debug(f"Starting engineered feature generation for instrument {instrument_id} ({timeframe}).")

        try:
            if ohlcv_df.empty:
                logger.warning(
                    f"OHLCV DataFrame is empty for instrument {instrument_id}. Cannot generate engineered features."
                )
                return {}

            for pattern in self.config.feature_generation.patterns:
                pattern_start = time.time()
                try:
                    pattern_features = await self._generate_pattern_features(pattern, base_features, ohlcv_df, segment)
                    engineered_features.update(pattern_features)
                    pattern_duration = (time.time() - pattern_start) * 1000
                    metrics_registry.record_feature_generation(
                        str(instrument_id), timeframe, pattern, pattern_duration, True
                    )
                    logger.debug(
                        f"Successfully generated {len(pattern_features)} features for pattern '{pattern}' in {pattern_duration:.2f}ms."
                    )
                except Exception as e:
                    pattern_duration = (time.time() - pattern_start) * 1000
                    metrics_registry.record_feature_generation(
                        str(instrument_id), timeframe, pattern, pattern_duration, False
                    )
                    await self.error_handler.handle_error(
                        "feature_engineering_pattern",
                        f"Pattern generation failed for '{pattern}': {e}",
                        {"instrument_id": instrument_id, "timeframe": timeframe, "pattern": pattern},
                        exc_info=True,
                    )

            if engineered_features:
                await self._store_engineered_features(instrument_id, timeframe, timestamp, engineered_features)

            total_duration = (time.time() - start_time) * 1000
            logger.info(
                f"Generated and stored {len(engineered_features)} engineered features for instrument {instrument_id} "
                f"({timeframe}) in {total_duration:.2f}ms."
            )
            return engineered_features

        except Exception as e:
            await self.error_handler.handle_error(
                "feature_engineering_pipeline",
                f"Feature generation pipeline failed: {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe},
                exc_info=True,
            )
            return {}

    async def _generate_pattern_features(
        self,
        pattern: str,
        base_features: dict[str, float],
        ohlcv_df: pd.DataFrame,
        segment: str,
    ) -> dict[str, EngineeredFeature]:
        """
        DEPRECATED: Pattern generation has been migrated to modular indicators system.
        This method is kept for backward compatibility only.
        """
        logger.warning(
            f"⚠️ DEPRECATED: Pattern generation for '{pattern}' should now use modular indicators system. "
            f"This legacy method will be removed in a future version."
        )
        # Return empty features - modular system handles this now
        return {}

    def _generate_momentum_ratios(
        self, base_features: dict[str, float], ohlcv_df: pd.DataFrame
    ) -> dict[str, EngineeredFeature]:
        """
        DEPRECATED: Migrated to MomentumRatiosIndicator in modular system.
        Use src/core/indicators/enhanced/momentum_ratios.py instead.
        """
        logger.warning("⚠️ DEPRECATED: _generate_momentum_ratios migrated to modular indicators")
        return {}

    def _generate_volatility_adjustments(
        self, base_features: dict[str, float], ohlcv_df: pd.DataFrame
    ) -> dict[str, EngineeredFeature]:
        """
        DEPRECATED: Migrated to VolatilityAdjustmentsIndicator in modular system.
        Use src/core/indicators/enhanced/volatility_adjustments.py instead.
        """
        logger.warning("⚠️ DEPRECATED: _generate_volatility_adjustments migrated to modular indicators")
        return {}

    def _generate_trend_confirmations(
        self, base_features: dict[str, float], ohlcv_df: pd.DataFrame
    ) -> dict[str, EngineeredFeature]:
        """
        DEPRECATED: Migrated to TrendConfirmationsIndicator in modular system.
        Use src/core/indicators/enhanced/trend_confirmations.py instead.
        """
        logger.warning("⚠️ DEPRECATED: _generate_trend_confirmations migrated to modular indicators")
        return {}

    def _generate_mean_reversion_signals(
        self, base_features: dict[str, float], ohlcv_df: pd.DataFrame
    ) -> dict[str, EngineeredFeature]:
        """
        DEPRECATED: Migrated to MeanReversionSignalsIndicator in modular system.
        Use src/core/indicators/enhanced/mean_reversion_signals.py instead.
        """
        logger.warning("⚠️ DEPRECATED: _generate_mean_reversion_signals migrated to modular indicators")
        return {}

    def _generate_breakout_indicators(
        self, base_features: dict[str, float], ohlcv_df: pd.DataFrame, segment: str
    ) -> dict[str, EngineeredFeature]:
        """
        DEPRECATED: Migrated to BreakoutIndicatorsIndicator in modular system.
        Use src/core/indicators/enhanced/breakout_indicators.py instead.
        """
        logger.warning("⚠️ DEPRECATED: _generate_breakout_indicators migrated to modular indicators")
        return {}

    async def update_feature_selection(
        self, instrument_id: int, timeframe: str, model_metadata: dict[str, Any]
    ) -> Optional[list[str]]:
        """
        Update feature selection based on model training results.
        Called by LGBMTrainer after training completion.
        """
        if not self.config.feature_selection.enabled:
            logger.info("Feature selection is disabled in the configuration.")
            return None

        logger.info(f"Updating feature selection for instrument {instrument_id} ({timeframe}).")
        try:
            model_key = f"{instrument_id}_{timeframe}"
            feature_importance = model_metadata.get("features", {}).get("importance", {})
            model_accuracy = model_metadata.get("metrics", {}).get("accuracy", 0.0)
            model_version = model_metadata.get("version_id", "unknown")

            if not feature_importance:
                logger.error(f"No feature importance data provided for {model_key}. Cannot update feature selection.")
                return None

            await self._store_feature_scores(
                instrument_id, timeframe, feature_importance, model_accuracy, model_version
            )
            selected_features = await self._calculate_optimal_feature_selection(instrument_id, timeframe)

            if selected_features:
                self.selected_features_cache[model_key] = set(selected_features)
                metrics_registry.record_feature_selection_change(str(instrument_id), timeframe, "model_update")
                metrics_registry.record_selected_features_count(str(instrument_id), timeframe, len(selected_features))
                logger.info(
                    f"Successfully updated feature selection for {model_key}: {len(selected_features)} features selected."
                )
            else:
                logger.warning(f"Optimal feature selection returned no features for {model_key}.")

            return selected_features

        except Exception as e:
            await self.error_handler.handle_error(
                "feature_selection_update",
                f"Feature selection update failed: {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe},
                exc_info=True,
            )
            return None

    async def get_selected_features(self, instrument_id: int, timeframe: str) -> Optional[set[str]]:
        """Get currently selected features for an instrument/timeframe."""
        model_key = f"{instrument_id}_{timeframe}"
        logger.debug(f"Getting selected features for {model_key}.")

        if model_key in self.selected_features_cache:
            logger.debug(f"Returning cached feature selection for {model_key}.")
            return self.selected_features_cache[model_key]

        try:
            logger.info(f"No cached selection found for {model_key}. Loading from database.")
            selected_features = await self._load_latest_feature_selection(instrument_id, timeframe)
            if selected_features:
                self.selected_features_cache[model_key] = set(selected_features)
                logger.info(f"Loaded and cached {len(selected_features)} features for {model_key}.")
                return set(selected_features)
            logger.warning(f"No feature selection found in the database for {model_key}.")
            return None
        except Exception as e:
            logger.error(f"Failed to get selected features for {model_key}: {e}", exc_info=True)
            # Re-raise as a runtime error to indicate a critical failure in getting model configuration
            raise RuntimeError(f"Could not load feature selection for {model_key}.") from e

    async def calculate_cross_asset_features(
        self, primary_instrument_id: int, timeframe: str, timestamp: datetime
    ) -> dict[str, float]:
        """Calculate cross-asset correlation features."""
        if not self.config.cross_asset.enabled:
            logger.debug("Cross-asset feature calculation is disabled.")
            return {}

        logger.debug(f"Calculating cross-asset features for instrument {primary_instrument_id} ({timeframe}).")
        try:
            cache_key = f"{primary_instrument_id}_{timeframe}_{timestamp.strftime('%Y%m%d_%H%M')}"
            if cache_key in self.correlation_cache:
                cached_value, cached_time = self.correlation_cache[cache_key]
                if (timestamp - cached_time).total_seconds() < self.config.cross_asset.cache_duration_minutes * 60:
                    logger.debug(f"Returning cached cross-asset features for {cache_key}.")
                    return {"cached_correlation": cached_value}

            related_instruments = await self._get_related_instruments(primary_instrument_id)
            if not related_instruments:
                logger.debug("No related instruments configured for cross-asset analysis.")
                return {}

            cross_features: dict[str, float] = {}
            for related_id, related_symbol in related_instruments[: self.config.cross_asset.max_related_instruments]:
                correlation = await self._calculate_instrument_correlation(
                    primary_instrument_id, related_id, timeframe, timestamp
                )
                if abs(correlation) >= self.config.cross_asset.minimum_correlation_threshold:
                    feature_name = f"correlation_{related_symbol.lower()}"
                    cross_features[feature_name] = correlation
                    metrics_registry.record_cross_asset_correlation(
                        str(primary_instrument_id), related_symbol, timeframe, correlation
                    )

            if cross_features:
                avg_correlation = float(np.mean(list(cross_features.values())))
                self.correlation_cache[cache_key] = (avg_correlation, timestamp)
                logger.info(
                    f"Calculated {len(cross_features)} cross-asset features for instrument {primary_instrument_id}."
                )

            return cross_features

        except Exception as e:
            await self.error_handler.handle_error(
                "cross_asset_feature_calculation",
                f"Cross-asset feature calculation failed: {e}",
                {"instrument_id": primary_instrument_id, "timeframe": timeframe},
                exc_info=True,
            )
            return {}

    async def _store_engineered_features(
        self, instrument_id: int, timeframe: str, timestamp: datetime, features: dict[str, EngineeredFeature]
    ) -> None:
        """Store engineered features in the database."""
        logger.debug(f"Storing {len(features)} engineered features for instrument {instrument_id} ({timeframe}).")
        try:
            feature_records = [
                {
                    "timestamp": timestamp,
                    "feature_name": feature.name,
                    "feature_value": feature.value,
                    "generation_method": feature.generation_method,
                    "source_features": feature.source_features,
                    "quality_score": feature.quality_score,
                }
                for feature in features.values()
                if feature.quality_score >= self.config.feature_generation.min_quality_score
            ]

            if feature_records:
                await self.feature_repo.insert_engineered_features(instrument_id, timeframe, feature_records)
                logger.info(f"Successfully stored {len(feature_records)} engineered features.")
            else:
                logger.debug("No engineered features met the quality score threshold for storage.")

        except Exception as e:
            logger.error(f"Failed to store engineered features for instrument {instrument_id}: {e}", exc_info=True)
            # Do not re-raise here to avoid halting the entire pipeline for a storage error.
            await self.error_handler.handle_error(
                "feature_storage",
                f"Failed to store engineered features: {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe},
            )

    def _calculate_feature_scores(self, history: list[dict[str, Any]]) -> dict[str, FeatureScore]:
        """Calculates feature scores based on importance history."""
        if not history:
            logger.warning("Cannot calculate feature scores: importance history is empty.")
            return {}

        logger.debug(f"Calculating feature scores from a history of {len(history)} entries.")
        feature_scores: dict[str, FeatureScore] = {}
        all_features = {k for entry in history for k in entry.get("importance", {})}

        for feature_name in all_features:
            importance_scores = [
                entry["importance"][feature_name] for entry in history if feature_name in entry.get("importance", {})
            ]

            if not importance_scores:
                continue

            avg_importance = float(np.mean(importance_scores))
            stability = 1.0 / (np.std(importance_scores) + 1e-9) if len(importance_scores) >= 2 else 1.0
            consistency = len(importance_scores) / len(history)

            composite_score = (
                avg_importance * self.config.feature_selection.importance_weight
                + min(stability, 1.0) * self.config.feature_selection.stability_weight
                + consistency * self.config.feature_selection.consistency_weight
            )

            feature_scores[feature_name] = FeatureScore(
                feature_name=feature_name,
                importance_score=avg_importance,
                stability_score=float(min(stability, 1.0)),
                consistency_score=consistency,
                composite_score=float(composite_score),
            )
        logger.info(f"Calculated scores for {len(feature_scores)} unique features.")
        return feature_scores

    async def _store_feature_scores(
        self,
        instrument_id: int,
        timeframe: str,
        feature_importance: dict[str, float],
        model_accuracy: float,
        model_version: str,
    ) -> None:
        """Store feature importance scores for the selection algorithm."""
        logger.debug(f"Storing feature scores for instrument {instrument_id} ({timeframe}), model '{model_version}'.")
        try:
            model_key = f"{instrument_id}_{timeframe}"
            history = self.importance_history[model_key]

            history.append(
                {
                    "timestamp": datetime.now(),
                    "importance": feature_importance,
                    "accuracy": model_accuracy,
                    "version": model_version,
                }
            )

            max_history = self.config.feature_selection.importance_history_length
            if len(history) > max_history:
                self.importance_history[model_key] = history[-max_history:]

            feature_scores = self._calculate_feature_scores(self.importance_history[model_key])
            if not feature_scores:
                logger.warning("No feature scores were calculated, nothing to store.")
                return

            feature_score_records = [
                {
                    "training_timestamp": datetime.now(),
                    "feature_name": scores.feature_name,
                    "importance_score": scores.importance_score,
                    "stability_score": scores.stability_score,
                    "consistency_score": scores.consistency_score,
                    "composite_score": scores.composite_score,
                    "model_version": model_version,
                }
                for scores in feature_scores.values()
            ]

            await self.feature_repo.insert_feature_scores(instrument_id, timeframe, feature_score_records)
            logger.info(f"Successfully stored {len(feature_score_records)} feature scores.")

        except Exception as e:
            logger.error(f"Failed to store feature scores for instrument {instrument_id}: {e}", exc_info=True)
            await self.error_handler.handle_error(
                "feature_score_storage",
                f"Failed to store feature scores: {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe},
            )

    async def _calculate_optimal_feature_selection(self, instrument_id: int, timeframe: str) -> Optional[list[str]]:
        """Calculate optimal feature selection using composite scoring."""
        logger.info(f"Calculating optimal feature selection for instrument {instrument_id} ({timeframe}).")
        try:
            model_key = f"{instrument_id}_{timeframe}"
            history = self.importance_history.get(model_key, [])

            if len(history) < self.config.feature_selection.min_history_for_selection:
                logger.warning(
                    f"Not enough history for feature selection for {model_key}: "
                    f"{len(history)} < {self.config.feature_selection.min_history_for_selection} required. Using all features."
                )
                return None

            feature_scores = self._calculate_feature_scores(history)
            if not feature_scores:
                logger.error(f"No feature scores could be calculated for {model_key}. Cannot perform selection.")
                return None

            end_time = datetime.now(pytz.timezone(config.system.timezone))
            start_time = end_time - timedelta(
                days=self.config.feature_selection.correlation_data_lookback_multiplier * 30
            )
            historical_features_df = await self.feature_repo.get_features_for_correlation(
                instrument_id, timeframe, start_time, end_time
            )

            selected_feature_scores = await self._remove_correlated_features(
                feature_scores, historical_features_df, instrument_id, timeframe
            )

            sorted_features = sorted(selected_feature_scores.values(), key=lambda x: x.composite_score, reverse=True)
            target_count = self.config.feature_selection.target_feature_count
            final_selection = [f.feature_name for f in sorted_features[:target_count]]

            if final_selection:
                await self.feature_repo.insert_feature_selection_history(
                    instrument_id,
                    timeframe,
                    final_selection,
                    {"method": "composite_scoring", "threshold": target_count},
                    len(feature_scores),
                    "composite_scoring",
                    f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
                logger.info(f"Final feature selection for {model_key} has {len(final_selection)} features.")
            else:
                logger.warning(f"Feature selection process resulted in an empty feature set for {model_key}.")

            return final_selection

        except Exception as e:
            logger.error(f"Feature selection calculation failed for {instrument_id} ({timeframe}): {e}", exc_info=True)
            return None

    async def _remove_correlated_features(
        self,
        feature_scores: dict[str, FeatureScore],
        historical_features_df: pd.DataFrame,
        instrument_id: int,
        timeframe: str,
    ) -> dict[str, FeatureScore]:
        """Remove highly correlated features to reduce redundancy, prioritizing higher composite scores."""
        if not feature_scores:
            return {}

        logger.debug(f"Removing correlated features for instrument {instrument_id} ({timeframe}).")
        try:
            sorted_features = sorted(feature_scores.values(), key=lambda x: x.composite_score, reverse=True)
            selected_features_names = [f.feature_name for f in sorted_features]

            if historical_features_df.empty or len(historical_features_df.columns) < 2:
                logger.warning(
                    f"Insufficient historical data for correlation analysis for {instrument_id} ({timeframe}). Skipping correlation removal."
                )
                return feature_scores

            cols_to_correlate = [
                col
                for col in selected_features_names
                if col in historical_features_df.columns and pd.api.types.is_numeric_dtype(historical_features_df[col])
            ]
            if len(cols_to_correlate) < 2:
                logger.warning(
                    f"Not enough numeric features for correlation analysis for {instrument_id} ({timeframe})."
                )
                return feature_scores

            filtered_df = historical_features_df[cols_to_correlate].dropna()
            if filtered_df.empty or len(filtered_df.columns) < 2:
                logger.warning(f"Not enough clean data for correlation analysis for {instrument_id} ({timeframe}).")
                return feature_scores

            correlation_matrix = filtered_df.corr().abs()
            features_to_keep_names: set[str] = set()

            for feature_score in sorted_features:
                current_feature_name = feature_score.feature_name
                if (
                    current_feature_name in correlation_matrix.columns
                    and current_feature_name not in features_to_keep_names
                ):
                    is_highly_correlated = False
                    for kept_feature_name in features_to_keep_names:
                        if (
                            kept_feature_name in correlation_matrix.columns
                            and current_feature_name in correlation_matrix.index
                            and correlation_matrix.loc[current_feature_name, kept_feature_name]
                            > self.config.feature_selection.correlation_threshold
                        ):
                            is_highly_correlated = True
                            break
                    if not is_highly_correlated:
                        features_to_keep_names.add(current_feature_name)

            reduced_feature_scores = {name: feature_scores[name] for name in features_to_keep_names}
            logger.info(
                f"Reduced features for {instrument_id} ({timeframe}) from {len(feature_scores)} to {len(reduced_feature_scores)} after correlation analysis."
            )
            return reduced_feature_scores

        except Exception as e:
            logger.error(
                f"Error during correlation-based feature removal for {instrument_id} ({timeframe}): {e}", exc_info=True
            )
            return feature_scores

    async def _get_related_instruments(self, primary_instrument_id: int) -> list[tuple[int, str]]:
        """Get related instruments for cross-asset features."""
        related_instruments = self.config.cross_asset.instruments
        if not related_instruments:
            logger.warning("No cross-asset instruments configured.")
            return []
        return [(int(rid), name) for rid, name in related_instruments.items() if int(rid) != primary_instrument_id]

    async def _calculate_instrument_correlation(
        self, primary_id: int, related_id: int, timeframe: str, timestamp: datetime
    ) -> float:
        """Calculate correlation between two instruments."""
        logger.debug(f"Calculating correlation between {primary_id} and {related_id} for timeframe {timeframe}.")
        try:
            lookback_days = self.config.cross_asset.correlation_lookback_days
            start_time = timestamp - timedelta(days=lookback_days)

            primary_data = await self.ohlcv_repo.get_ohlcv_data(primary_id, timeframe, start_time, timestamp)
            related_data = await self.ohlcv_repo.get_ohlcv_data(related_id, timeframe, start_time, timestamp)

            if len(primary_data) < 10 or len(related_data) < 10:
                logger.warning(
                    f"Insufficient data to correlate {primary_id} and {related_id}. Need at least 10 data points."
                )
                return 0.0

            primary_returns = pd.Series([c.close for c in primary_data]).pct_change().dropna()
            related_returns = pd.Series([c.close for c in related_data]).pct_change().dropna()

            min_length = min(len(primary_returns), len(related_returns))
            if min_length < 5:
                logger.warning(f"Insufficient overlapping returns to correlate {primary_id} and {related_id}.")
                return 0.0

            correlation = np.corrcoef(
                primary_returns.iloc[-min_length:].values, related_returns.iloc[-min_length:].values
            )[0, 1]

            return float(correlation) if np.isfinite(correlation) else 0.0

        except Exception as e:
            logger.error(f"Error calculating correlation between {primary_id} and {related_id}: {e}", exc_info=True)
            raise RuntimeError("Instrument correlation calculation failed.") from e

    async def _load_latest_feature_selection(self, instrument_id: int, timeframe: str) -> Optional[list[str]]:
        """Load latest feature selection from the database."""
        logger.debug(f"Loading latest feature selection for instrument {instrument_id} ({timeframe}).")
        try:
            selection = await self.feature_repo.get_latest_feature_selection(instrument_id, timeframe)
            if selection:
                logger.info(f"Successfully loaded {len(selection)} selected features from database.")
            else:
                logger.info("No feature selection found in database.")
            return selection
        except Exception as e:
            logger.error(f"Database error loading feature selection: {e}", exc_info=True)
            raise RuntimeError("Failed to load feature selection from database.") from e
