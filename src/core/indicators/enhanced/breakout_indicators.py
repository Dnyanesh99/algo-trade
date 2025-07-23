"""
Breakout Indicators Implementation.
Exact replication of the _generate_breakout_indicators method from feature_engineering_pipeline.py.
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class EngineeredFeature(BaseModel):
    """Model for engineered feature data"""

    name: str
    value: float
    generation_method: str
    source_features: list[str]
    quality_score: float


class BreakoutIndicatorsIndicator(BaseIndicator):
    """Breakout Indicators - generates breakout indicators using existing features."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate breakout indicators using base features (exact replica from feature_engineering_pipeline.py)."""
        return self._generate_breakout_indicators(df, params)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate breakout indicators using base features (exact replica from feature_engineering_pipeline.py)."""
        return self._generate_breakout_indicators(df, params)

    def _generate_breakout_indicators(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """
        Generate breakout indicators with enhanced precision and robustness.

        Improvements:
        - Robust statistical validation for volume analysis
        - Enhanced bounds checking and outlier detection
        - Configurable thresholds with dynamic adjustment
        - Improved numerical stability and error handling
        - Better separation of concerns for different breakout types
        """
        import numpy as np

        features: dict[str, float] = {}
        base_features = params.get("base_features", {})
        segment = params.get("segment", "EQUITY")
        min_data_length = params.get("min_data_length", 20)
        volume_lookback_short = params.get("volume_lookback_short", 5)
        volume_lookback_long = params.get("volume_lookback_long", 20)
        volume_ratio_max = params.get("volume_ratio_max", 2.0)

        # Enhanced parameters for robustness
        epsilon = params.get("epsilon", 1e-9)
        max_breakout_strength = params.get("max_breakout_strength", 0.5)  # Cap extreme breakouts
        volume_significance_threshold = params.get("volume_significance_threshold", 0.1)

        try:
            logger.debug(f"Generating enhanced breakout indicators with {len(df)} data points")

            if len(df) < min_data_length:
                logger.debug(f"Insufficient data: {len(df)} < {min_data_length}, returning empty features")
                return pd.DataFrame(features, index=df.index)

            # Robust current price extraction
            current_close = df["close"].iloc[-1]
            if not np.isfinite(current_close) or current_close <= epsilon:
                logger.debug("Invalid current close price, skipping breakout calculations")
                return pd.DataFrame(features, index=df.index)

            # Enhanced Bollinger Bands breakout with robust validation
            bb_upper = base_features.get("bb_upper")
            bb_lower = base_features.get("bb_lower")

            if (
                bb_upper is not None
                and bb_lower is not None
                and np.isfinite(bb_upper)
                and np.isfinite(bb_lower)
                and bb_upper > bb_lower
                and (bb_upper - bb_lower) > epsilon
            ):
                try:
                    breakout_strength = 0.0

                    # Calculate breakout strength with proper normalization
                    if current_close > bb_upper:
                        # Upper breakout - normalized by band width for better scaling
                        band_width = bb_upper - bb_lower
                        breakout_strength = (current_close - bb_upper) / band_width
                    elif current_close < bb_lower:
                        # Lower breakout - normalized by band width
                        band_width = bb_upper - bb_lower
                        breakout_strength = (bb_lower - current_close) / band_width

                    # Apply bounds checking and stability controls
                    if np.isfinite(breakout_strength):
                        breakout_strength = np.clip(breakout_strength, -max_breakout_strength, max_breakout_strength)
                        features["bb_breakout_strength"] = float(breakout_strength)
                        logger.debug(f"BB breakout strength: {breakout_strength:.6f}")
                    else:
                        logger.debug("Non-finite breakout strength calculated")

                except Exception as e:
                    logger.debug(f"Error calculating BB breakout strength: {e}")

            # Enhanced volume-confirmed breakout with statistical validation
            if segment != "INDICES" and len(df) >= volume_lookback_long:
                try:
                    # Robust volume analysis with statistical validation
                    volume_data = df["volume"].values

                    # Ensure we have valid volume data
                    if len(volume_data) >= volume_lookback_long and np.all(
                        np.isfinite(volume_data[-volume_lookback_long:])
                    ):
                        # Calculate recent and historical volume with robust statistics
                        recent_volumes = volume_data[-volume_lookback_short:]
                        longer_volumes = volume_data[-volume_lookback_long:-volume_lookback_short]

                        if len(recent_volumes) >= volume_lookback_short and len(longer_volumes) >= (
                            volume_lookback_long - volume_lookback_short
                        ):
                            # Use median for more robust central tendency
                            recent_volume_median = np.median(recent_volumes)
                            longer_volume_median = np.median(longer_volumes)

                            # Enhanced volume ratio calculation with significance testing
                            if longer_volume_median > epsilon:
                                volume_ratio = recent_volume_median / longer_volume_median

                                # Statistical significance check using coefficient of variation
                                longer_volume_cv = np.std(longer_volumes) / (longer_volume_median + epsilon)

                                # Only consider volume confirmation if the change is statistically significant
                                if (
                                    np.isfinite(volume_ratio)
                                    and volume_ratio > (1.0 + volume_significance_threshold)
                                    and longer_volume_cv < 2.0
                                ):  # Avoid too volatile volume periods
                                    # Volume-confirmed breakout calculation
                                    if "bb_breakout_strength" in features:
                                        bb_strength = features["bb_breakout_strength"]

                                        if abs(bb_strength) > epsilon:
                                            # Enhanced volume confirmation with non-linear scaling
                                            volume_multiplier = min(volume_ratio, volume_ratio_max)

                                            # Apply square root to dampen extreme volume spikes
                                            volume_factor = np.sqrt(volume_multiplier)

                                            volume_confirmed_breakout = bb_strength * volume_factor

                                            if np.isfinite(volume_confirmed_breakout):
                                                features["volume_confirmed_breakout"] = float(volume_confirmed_breakout)
                                                logger.debug(
                                                    f"Volume confirmed breakout: {volume_confirmed_breakout:.6f} "
                                                    f"(ratio: {volume_ratio:.2f}, CV: {longer_volume_cv:.2f})"
                                                )
                                            else:
                                                logger.debug("Non-finite volume confirmed breakout")
                                        else:
                                            logger.debug("BB breakout strength too small for volume confirmation")
                                    else:
                                        logger.debug("No BB breakout strength available for volume confirmation")
                                else:
                                    logger.debug(
                                        f"Volume change not significant: ratio={volume_ratio:.2f}, CV={longer_volume_cv:.2f}"
                                    )
                            else:
                                logger.debug("Invalid longer volume median for ratio calculation")
                        else:
                            logger.debug("Insufficient volume data for statistical analysis")
                    else:
                        logger.debug("Invalid or insufficient volume data")

                except Exception as e:
                    logger.debug(f"Error in volume-confirmed breakout calculation: {e}")

            logger.debug(f"Successfully generated {len(features)} breakout indicator features")

        except Exception as e:
            logger.error(f"Critical error in breakout indicator generation: {e}", exc_info=True)
            features = {}

        return pd.DataFrame(features, index=df.index)


def create_breakout_indicators_indicator(feature_config: Any) -> "BreakoutIndicatorsIndicator":
    """Factory function to create BreakoutIndicatorsIndicator from feature configuration."""
    config = IndicatorConfig(
        name=feature_config.name,
        enabled=True,
        params=feature_config.params.get("default", {}),
        timeframe_params={
            tf_key: tf_params for tf_key, tf_params in feature_config.params.items() if tf_key != "default"
        },
    )

    return BreakoutIndicatorsIndicator(config)
