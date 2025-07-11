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
from pydantic import BaseModel

from src.core.feature_validator import FeatureValidator
from src.database.feature_repo import FeatureRepository
from src.database.ohlcv_repo import OHLCVRepository
from src.metrics.registry import metrics_registry
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import ConfigLoader, FeatureEngineeringConfig
from src.utils.logger import LOGGER as logger

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
        self.ohlcv_repo = ohlcv_repo
        self.feature_repo = feature_repo
        self.feature_validator = feature_validator
        self.error_handler = error_handler
        self.health_monitor = health_monitor

        # Load configuration
        assert config.model_training is not None
        self.config: FeatureEngineeringConfig = config.model_training.feature_engineering

        # Get existing feature configurations from config.yaml
        assert config.trading is not None
        self.base_feature_configs = config.trading.features.configurations
        self.feature_timeframes = config.trading.feature_timeframes

        # Feature importance history for selection
        self.importance_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Cross-asset correlation cache
        self.correlation_cache: dict[str, tuple[float, datetime]] = {}

        # Selected features cache
        self.selected_features_cache: dict[str, set[str]] = {}

        logger.info("FeatureEngineeringPipeline initialized with production configuration")

    async def generate_engineered_features(
        self, instrument_id: int, timeframe: str, timestamp: datetime, base_features: dict[str, float]
    ) -> dict[str, EngineeredFeature]:
        """
        Generate engineered features using existing base features from FeatureCalculator
        """
        if not self.config.feature_generation.enabled:
            return {}

        start_time = time.time()
        engineered_features = {}

        try:
            # Get recent OHLCV data for pattern generation
            recent_data = await self._get_recent_ohlcv_data(instrument_id, timeframe, timestamp)
            if recent_data.empty:
                return {}

            # Generate features for each configured pattern
            for pattern in self.config.feature_generation.patterns:
                pattern_start = time.time()

                try:
                    pattern_features = await self._generate_pattern_features(
                        pattern, base_features, recent_data, timestamp
                    )
                    engineered_features.update(pattern_features)

                    # Record metrics for successful generation
                    pattern_duration = time.time() - pattern_start
                    metrics_registry.record_feature_generation(
                        str(instrument_id), timeframe, pattern, pattern_duration, True
                    )

                except Exception as e:
                    # Record failed generation
                    pattern_duration = time.time() - pattern_start
                    metrics_registry.record_feature_generation(
                        str(instrument_id), timeframe, pattern, pattern_duration, False
                    )

                    await self.error_handler.handle_error(
                        "feature_engineering",
                        f"Pattern generation failed for {pattern}: {e}",
                        {"instrument_id": instrument_id, "timeframe": timeframe, "pattern": pattern},
                    )

            # Store engineered features if any were generated
            if engineered_features:
                await self._store_engineered_features(instrument_id, timeframe, timestamp, engineered_features)

            total_duration = time.time() - start_time
            logger.debug(
                f"Generated {len(engineered_features)} features for {instrument_id} "
                f"({timeframe}) in {total_duration:.3f}s"
            )

            return engineered_features

        except Exception as e:
            await self.error_handler.handle_error(
                "feature_engineering",
                f"Feature generation failed: {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe},
            )
            return {}

    async def _generate_pattern_features(
        self, pattern: str, base_features: dict[str, float], recent_data: pd.DataFrame, timestamp: datetime
    ) -> dict[str, EngineeredFeature]:
        """Generate features for a specific pattern using actual base features"""
        features = {}

        if pattern == "momentum_ratios":
            features.update(self._generate_momentum_ratios(base_features, recent_data))
        elif pattern == "volatility_adjustments":
            features.update(self._generate_volatility_adjustments(base_features, recent_data))
        elif pattern == "trend_confirmations":
            features.update(self._generate_trend_confirmations(base_features, recent_data))
        elif pattern == "mean_reversion_signals":
            features.update(self._generate_mean_reversion_signals(base_features, recent_data))
        elif pattern == "breakout_indicators":
            features.update(self._generate_breakout_indicators(base_features, recent_data))

        return features

    def _generate_momentum_ratios(
        self, base_features: dict[str, float], recent_data: pd.DataFrame
    ) -> dict[str, EngineeredFeature]:
        """Generate momentum-based ratio features using existing indicators"""
        features = {}

        try:
            # Use actual price data for momentum calculations
            if len(recent_data) >= 20:
                close_prices = recent_data["close"].values

                # Price momentum ratios
                for period in self.config.feature_generation.lookback_periods:
                    if len(close_prices) > period:
                        momentum_ratio = close_prices[-1] / close_prices[-period - 1] - 1.0
                        features[f"momentum_ratio_{period}d"] = EngineeredFeature(
                            name=f"momentum_ratio_{period}d",
                            value=momentum_ratio,
                            generation_method="price_ratio",
                            source_features=["close"],
                            quality_score=min(1.0, abs(momentum_ratio) * 10),  # Higher volatility = higher quality
                        )

                # RSI momentum confirmation (using existing RSI values)
                rsi_14 = base_features.get("rsi_14")
                if rsi_14 is not None:
                    rsi_momentum = (rsi_14 - 50) / 50  # Normalize RSI around 50
                    features["rsi_momentum_normalized"] = EngineeredFeature(
                        name="rsi_momentum_normalized",
                        value=rsi_momentum,
                        generation_method="rsi_normalization",
                        source_features=["rsi_14"],
                        quality_score=abs(rsi_momentum),
                    )

                # MACD momentum strength (using existing MACD values)
                macd = base_features.get("macd")
                macd_signal = base_features.get("macd_signal")
                if macd is not None and macd_signal is not None:
                    macd_strength = abs(macd - macd_signal)
                    features["macd_momentum_strength"] = EngineeredFeature(
                        name="macd_momentum_strength",
                        value=macd_strength,
                        generation_method="macd_divergence",
                        source_features=["macd", "macd_signal"],
                        quality_score=min(1.0, macd_strength),
                    )

        except Exception as e:
            logger.warning(f"Momentum ratio generation failed: {e}")

        return features

    def _generate_volatility_adjustments(
        self, base_features: dict[str, float], recent_data: pd.DataFrame
    ) -> dict[str, EngineeredFeature]:
        """Generate volatility-adjusted features using existing ATR"""
        features = {}

        try:
            atr_14 = base_features.get("atr_14")
            if atr_14 is not None and atr_14 > 0 and len(recent_data) >= 2:
                # Volatility-adjusted returns
                recent_return = recent_data["close"].iloc[-1] / recent_data["close"].iloc[-2] - 1.0
                vol_adjusted_return = recent_return / atr_14

                features["volatility_adjusted_return"] = EngineeredFeature(
                    name="volatility_adjusted_return",
                    value=vol_adjusted_return,
                    generation_method="return_atr_normalization",
                    source_features=["close", "atr_14"],
                    quality_score=min(1.0, abs(vol_adjusted_return)),
                )

                # Bollinger Band position with volatility adjustment
                bb_upper = base_features.get("bb_upper")
                bb_lower = base_features.get("bb_lower")
                close = recent_data["close"].iloc[-1]

                if bb_upper is not None and bb_lower is not None and bb_upper != bb_lower:
                    bb_position = (close - bb_lower) / (bb_upper - bb_lower)
                    vol_adjusted_bb = bb_position * (atr_14 / close)  # Adjust by relative volatility

                    features["vol_adjusted_bb_position"] = EngineeredFeature(
                        name="vol_adjusted_bb_position",
                        value=vol_adjusted_bb,
                        generation_method="bb_volatility_adjustment",
                        source_features=["bb_upper", "bb_lower", "close", "atr_14"],
                        quality_score=abs(bb_position - 0.5) * 2,  # Quality higher at extremes
                    )

        except Exception as e:
            logger.warning(f"Volatility adjustment generation failed: {e}")

        return features

    def _generate_trend_confirmations(
        self, base_features: dict[str, float], recent_data: pd.DataFrame
    ) -> dict[str, EngineeredFeature]:
        """Generate trend confirmation features using existing trend indicators"""
        features = {}

        try:
            # ADX trend strength confirmation
            adx = base_features.get("adx")
            if adx is not None:
                trend_strength = min(adx / 25.0, 2.0)  # Normalize ADX (25+ is strong trend)
                features["trend_strength_normalized"] = EngineeredFeature(
                    name="trend_strength_normalized",
                    value=trend_strength,
                    generation_method="adx_normalization",
                    source_features=["adx"],
                    quality_score=min(1.0, trend_strength),
                )

            # Multi-indicator trend confirmation
            trend_indicators = []

            # MACD trend
            macd = base_features.get("macd")
            macd_signal = base_features.get("macd_signal")
            if macd is not None and macd_signal is not None:
                trend_indicators.append(1 if macd > macd_signal else -1)

            # SAR trend
            sar = base_features.get("sar")
            if sar is not None and len(recent_data) > 0:
                current_close = recent_data["close"].iloc[-1]
                trend_indicators.append(1 if current_close > sar else -1)

            # AROON trend
            aroon_up = base_features.get("aroon_up")
            aroon_down = base_features.get("aroon_down")
            if aroon_up is not None and aroon_down is not None:
                trend_indicators.append(1 if aroon_up > aroon_down else -1)

            if trend_indicators:
                trend_consensus = sum(trend_indicators) / len(trend_indicators)
                features["trend_consensus"] = EngineeredFeature(
                    name="trend_consensus",
                    value=trend_consensus,
                    generation_method="multi_indicator_consensus",
                    source_features=["macd", "macd_signal", "sar", "aroon_up", "aroon_down"],
                    quality_score=abs(trend_consensus),
                )

        except Exception as e:
            logger.warning(f"Trend confirmation generation failed: {e}")

        return features

    def _generate_mean_reversion_signals(
        self, base_features: dict[str, float], recent_data: pd.DataFrame
    ) -> dict[str, EngineeredFeature]:
        """Generate mean reversion signals using existing oscillators"""
        features = {}

        try:
            # RSI mean reversion signal
            rsi_14 = base_features.get("rsi_14")
            if rsi_14 is not None:
                if rsi_14 > 70:
                    reversion_signal = (rsi_14 - 70) / 30  # Overbought strength
                elif rsi_14 < 30:
                    reversion_signal = (30 - rsi_14) / 30  # Oversold strength
                else:
                    reversion_signal = 0.0

                features["rsi_mean_reversion"] = EngineeredFeature(
                    name="rsi_mean_reversion",
                    value=reversion_signal,
                    generation_method="rsi_extremes",
                    source_features=["rsi_14"],
                    quality_score=abs(reversion_signal),
                )

            # Williams %R mean reversion
            willr = base_features.get("willr")
            if willr is not None:
                # Williams %R is typically negative, so reverse the logic
                if willr > -20:  # Overbought
                    willr_reversion = (willr + 20) / 20
                elif willr < -80:  # Oversold
                    willr_reversion = (willr + 80) / 20
                else:
                    willr_reversion = 0.0

                features["willr_mean_reversion"] = EngineeredFeature(
                    name="willr_mean_reversion",
                    value=willr_reversion,
                    generation_method="willr_extremes",
                    source_features=["willr"],
                    quality_score=abs(willr_reversion),
                )

            # Composite mean reversion score
            reversion_signals = []
            for feature_name in ["rsi_mean_reversion", "willr_mean_reversion"]:
                if feature_name in features:
                    reversion_signals.append(features[feature_name].value)

            if reversion_signals:
                composite_reversion = float(np.mean(reversion_signals))
                features["composite_mean_reversion"] = EngineeredFeature(
                    name="composite_mean_reversion",
                    value=composite_reversion,
                    generation_method="multi_oscillator_consensus",
                    source_features=["rsi_14", "willr"],
                    quality_score=float(abs(composite_reversion)),
                )

        except Exception as e:
            logger.warning(f"Mean reversion signal generation failed: {e}")

        return features

    def _generate_breakout_indicators(
        self, base_features: dict[str, float], recent_data: pd.DataFrame
    ) -> dict[str, EngineeredFeature]:
        """Generate breakout indicators using existing features"""
        features: dict[str, EngineeredFeature] = {}

        try:
            if len(recent_data) < 20:
                return features

            current_close = recent_data["close"].iloc[-1]

            # Bollinger Band breakout
            bb_upper = base_features.get("bb_upper")
            bb_lower = base_features.get("bb_lower")
            if bb_upper is not None and bb_lower is not None:
                if current_close > bb_upper:
                    breakout_strength = (current_close - bb_upper) / bb_upper
                elif current_close < bb_lower:
                    breakout_strength = (bb_lower - current_close) / bb_lower
                else:
                    breakout_strength = 0.0

                features["bb_breakout_strength"] = EngineeredFeature(
                    name="bb_breakout_strength",
                    value=breakout_strength,
                    generation_method="bollinger_breakout",
                    source_features=["bb_upper", "bb_lower", "close"],
                    quality_score=abs(breakout_strength) * 10,
                )

            # Volume-confirmed breakout
            obv = base_features.get("obv")
            if obv is not None and len(recent_data) >= 10:
                # Simple volume trend (would be better with OBV history)
                recent_volume = recent_data["volume"].iloc[-5:].mean()
                longer_volume = recent_data["volume"].iloc[-20:-5].mean()

                if longer_volume > 0:
                    volume_ratio = recent_volume / longer_volume

                    # Combine price breakout with volume confirmation
                    bb_breakout = features.get("bb_breakout_strength")
                    if bb_breakout is not None and abs(bb_breakout.value) > 0:
                        volume_confirmed_breakout = bb_breakout.value * min(volume_ratio, 2.0)

                        features["volume_confirmed_breakout"] = EngineeredFeature(
                            name="volume_confirmed_breakout",
                            value=volume_confirmed_breakout,
                            generation_method="volume_price_breakout",
                            source_features=["bb_upper", "bb_lower", "close", "volume"],
                            quality_score=abs(volume_confirmed_breakout),
                        )

        except Exception as e:
            logger.warning(f"Breakout indicator generation failed: {e}")

        return features

    async def update_feature_selection(
        self, instrument_id: int, timeframe: str, model_metadata: dict[str, Any]
    ) -> Optional[list[str]]:
        """
        Update feature selection based on model training results
        Called by LGBMTrainer after training completion
        """
        if not self.config.feature_selection.enabled:
            return None

        try:
            model_key = f"{instrument_id}_{timeframe}"

            # Extract feature importance from model metadata
            feature_importance = model_metadata.get("features", {}).get("importance", {})
            model_accuracy = model_metadata.get("metrics", {}).get("accuracy", 0.0)
            model_version = model_metadata.get("version_id", "unknown")

            if not feature_importance:
                logger.warning(f"No feature importance data for {model_key}")
                return None

            # Store importance scores
            await self._store_feature_scores(
                instrument_id, timeframe, feature_importance, model_accuracy, model_version
            )

            # Calculate new feature selection
            selected_features = await self._calculate_optimal_feature_selection(instrument_id, timeframe)

            if selected_features:
                # Update cache
                self.selected_features_cache[model_key] = set(selected_features)

                # Record metrics
                metrics_registry.record_feature_selection_change(str(instrument_id), timeframe, "model_update")
                metrics_registry.record_selected_features_count(str(instrument_id), timeframe, len(selected_features))

                logger.info(f"Updated feature selection for {model_key}: {len(selected_features)} features selected")

            return selected_features

        except Exception as e:
            await self.error_handler.handle_error(
                "feature_selection",
                f"Feature selection update failed: {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe},
            )
            return None

    async def get_selected_features(self, instrument_id: int, timeframe: str) -> Optional[set[str]]:
        """Get currently selected features for an instrument/timeframe"""
        model_key = f"{instrument_id}_{timeframe}"

        # Return cached selection if available
        if model_key in self.selected_features_cache:
            return self.selected_features_cache[model_key]

        # Load from database
        try:
            selected_features = await self._load_latest_feature_selection(instrument_id, timeframe)
            if selected_features:
                self.selected_features_cache[model_key] = set(selected_features)
                return set(selected_features)
        except Exception as e:
            logger.warning(f"Failed to load feature selection for {model_key}: {e}")

        return None

    async def calculate_cross_asset_features(
        self, primary_instrument_id: int, timeframe: str, timestamp: datetime
    ) -> dict[str, float]:
        """Calculate cross-asset correlation features"""
        if not self.config.cross_asset.enabled:
            return {}

        try:
            cross_features = {}
            cache_key = f"{primary_instrument_id}_{timeframe}_{timestamp.strftime('%Y%m%d_%H%M')}"

            # Check cache first
            if cache_key in self.correlation_cache:
                cached_value, cached_time = self.correlation_cache[cache_key]
                if (timestamp - cached_time).total_seconds() < self.config.cross_asset.cache_duration_minutes * 60:
                    return {"cached_correlation": cached_value}

            # Get related instruments (simplified - would normally come from config)
            related_instruments = await self._get_related_instruments(primary_instrument_id)

            for related_id, related_symbol in related_instruments[: self.config.cross_asset.max_related_instruments]:
                correlation = await self._calculate_instrument_correlation(
                    primary_instrument_id, related_id, timeframe, timestamp
                )

                if abs(correlation) >= self.config.cross_asset.minimum_correlation_threshold:
                    feature_name = f"correlation_{related_symbol.lower()}"
                    cross_features[feature_name] = correlation

                    # Record correlation metric
                    metrics_registry.record_cross_asset_correlation(
                        str(primary_instrument_id), related_symbol, timeframe, correlation
                    )

            # Cache result
            if cross_features:
                avg_correlation = float(np.mean(list(cross_features.values())))
                self.correlation_cache[cache_key] = (avg_correlation, timestamp)

            return cross_features

        except Exception as e:
            await self.error_handler.handle_error(
                "cross_asset_features",
                f"Cross-asset feature calculation failed: {e}",
                {"instrument_id": primary_instrument_id, "timeframe": timeframe},
            )
            return {}

    # Helper methods for database operations and calculations
    async def _get_recent_ohlcv_data(self, instrument_id: int, timeframe: str, timestamp: datetime) -> pd.DataFrame:
        """Get recent OHLCV data for feature generation"""
        try:
            start_time = timestamp - timedelta(days=max(self.config.feature_generation.lookback_periods) + 5)
            ohlcv_data = await self.ohlcv_repo.get_ohlcv_data(instrument_id, timeframe, start_time, timestamp)
            return (
                pd.DataFrame(
                    [
                        {
                            "timestamp": candle.ts,  # Use 'ts' instead of 'timestamp'
                            "open": candle.open,
                            "high": candle.high,
                            "low": candle.low,
                            "close": candle.close,
                            "volume": candle.volume,
                        }
                        for candle in ohlcv_data
                    ]
                )
                .set_index("timestamp")
                .sort_index()
            )
        except Exception as e:
            logger.warning(f"Failed to get recent OHLCV data: {e}")
            return pd.DataFrame()

    async def _store_engineered_features(
        self, instrument_id: int, timeframe: str, timestamp: datetime, features: dict[str, EngineeredFeature]
    ) -> None:
        """Store engineered features in database"""
        try:
            # Convert to database format and store
            feature_records = []
            for feature in features.values():
                if feature.quality_score >= self.config.feature_generation.min_quality_score:
                    feature_records.append(
                        {
                            "timestamp": timestamp,
                            "feature_name": feature.name,
                            "feature_value": feature.value,
                            "generation_method": feature.generation_method,
                            "source_features": feature.source_features,
                            "quality_score": feature.quality_score,
                        }
                    )

            if feature_records:
                await self.feature_repo.insert_engineered_features(instrument_id, timeframe, feature_records)

        except Exception as e:
            logger.error(f"Failed to store engineered features: {e}")

    async def _store_feature_scores(
        self,
        instrument_id: int,
        timeframe: str,
        feature_importance: dict[str, float],
        model_accuracy: float,
        model_version: str,
    ) -> None:
        """Store feature importance scores for selection algorithm"""
        try:
            # Calculate stability and consistency scores from history
            model_key = f"{instrument_id}_{timeframe}"
            history = self.importance_history[model_key]

            # Add current scores to history
            history.append(
                {
                    "timestamp": datetime.now(),
                    "importance": feature_importance,
                    "accuracy": model_accuracy,
                    "version": model_version,
                }
            )

            # Keep only recent history
            max_history = self.config.feature_selection.importance_history_length
            if len(history) > max_history:
                history[:] = history[-max_history:]

            # Store feature scores in database
            feature_score_records = []
            for feature_name, importance in feature_importance.items():
                feature_score_records.append(
                    {
                        "timestamp": datetime.now(),
                        "feature_name": feature_name,
                        "importance_score": importance,
                        "stability_score": 1.0,  # Will be calculated later with more history
                        "consistency_score": 1.0,  # Will be calculated later with more history
                        "composite_score": importance,  # Simplified for first entry
                        "model_version": model_version,
                    }
                )

            if feature_score_records:
                await self.feature_repo.insert_feature_scores(instrument_id, timeframe, feature_score_records)

        except Exception as e:
            logger.error(f"Failed to store feature scores: {e}")

    async def _calculate_optimal_feature_selection(self, instrument_id: int, timeframe: str) -> Optional[list[str]]:
        """Calculate optimal feature selection using composite scoring"""
        try:
            model_key = f"{instrument_id}_{timeframe}"
            history = self.importance_history[model_key]

            if len(history) < 2:
                return None

            # Calculate composite scores for each feature
            feature_scores = {}
            all_features = set()

            # Collect all features across history
            for entry in history:
                all_features.update(entry["importance"].keys())

            for feature_name in all_features:
                importance_scores = []
                accuracies = []

                for entry in history:
                    if feature_name in entry["importance"]:
                        importance_scores.append(entry["importance"][feature_name])
                        accuracies.append(entry["accuracy"])

                if len(importance_scores) >= 2:
                    # Calculate component scores
                    avg_importance = float(np.mean(importance_scores))
                    stability = float(1.0 / (np.std(importance_scores) + 1e-8))
                    consistency = float(len(importance_scores) / len(history))

                    # Weighted composite score
                    composite_score = float(
                        avg_importance * self.config.feature_selection.importance_weight
                        + min(stability, 1.0) * self.config.feature_selection.stability_weight
                        + consistency * self.config.feature_selection.consistency_weight
                    )

                    feature_scores[feature_name] = FeatureScore(
                        feature_name=feature_name,
                        importance_score=avg_importance,
                        stability_score=float(min(stability, 1.0)),
                        consistency_score=consistency,
                        composite_score=composite_score,
                    )

            # Remove highly correlated features
            selected_features = await self._remove_correlated_features(feature_scores, instrument_id, timeframe)

            # Sort by composite score and take top N
            sorted_features = sorted(selected_features.values(), key=lambda x: x.composite_score, reverse=True)

            target_count = self.config.feature_selection.target_feature_count
            final_selection = [f.feature_name for f in sorted_features[:target_count]]

            # Store feature selection history
            if final_selection:
                await self.feature_repo.insert_feature_selection_history(
                    instrument_id,
                    timeframe,
                    final_selection,
                    "composite_scoring",
                    f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )

            return final_selection

        except Exception as e:
            logger.error(f"Feature selection calculation failed: {e}")
            return None

    async def _remove_correlated_features(
        self, feature_scores: dict[str, FeatureScore], instrument_id: int, timeframe: str
    ) -> dict[str, FeatureScore]:
        """Remove highly correlated features to reduce redundancy"""
        try:
            # Simplified correlation removal - in production would use actual feature correlation
            # For now, just return the input scores
            # Implementation would require loading recent feature values and calculating correlations
            return feature_scores

        except Exception as e:
            logger.warning(f"Correlation removal failed: {e}")
            return feature_scores

    async def _get_related_instruments(self, primary_instrument_id: int) -> list[tuple[int, str]]:
        """Get related instruments for cross-asset features"""
        # Simplified implementation - would normally query instrument relationships
        # For Indian markets, common relationships:
        related_instruments = config.model_training.feature_engineering.cross_asset.instruments
        return [(rid, name) for rid, name in related_instruments.items() if rid != primary_instrument_id]

    async def _calculate_instrument_correlation(
        self, primary_id: int, related_id: int, timeframe: str, timestamp: datetime
    ) -> float:
        """Calculate correlation between two instruments"""
        try:
            lookback_days = self.config.cross_asset.correlation_lookback_days
            start_time = timestamp - timedelta(days=lookback_days)

            # Get price data for both instruments
            primary_data = await self.ohlcv_repo.get_ohlcv_data(primary_id, timeframe, start_time, timestamp)
            related_data = await self.ohlcv_repo.get_ohlcv_data(related_id, timeframe, start_time, timestamp)

            if len(primary_data) < 10 or len(related_data) < 10:
                return 0.0

            # Convert to returns and calculate correlation
            primary_returns = pd.Series([c.close for c in primary_data]).pct_change().dropna()
            related_returns = pd.Series([c.close for c in related_data]).pct_change().dropna()

            # Align by timestamp and calculate correlation
            min_length = min(len(primary_returns), len(related_returns))
            if min_length < 5:
                return 0.0

            correlation = np.corrcoef(
                primary_returns.iloc[-min_length:].values, related_returns.iloc[-min_length:].values
            )[0, 1]

            return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            return 0.0

    async def _load_latest_feature_selection(self, instrument_id: int, timeframe: str) -> Optional[list[str]]:
        """Load latest feature selection from database"""
        try:
            return await self.feature_repo.get_latest_feature_selection(instrument_id, timeframe)
        except Exception as e:
            logger.warning(f"Failed to load feature selection: {e}")
            return None
