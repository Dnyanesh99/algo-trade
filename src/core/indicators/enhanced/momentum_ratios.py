"""
Momentum Ratios Indicator Implementation.
Exact replication of the _generate_momentum_ratios method from feature_engineering_pipeline.py.
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


class MomentumRatiosIndicator(BaseIndicator):
    """Momentum Ratios indicator - generates momentum-based ratio features using existing indicators."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate momentum ratios using base features (exact replica from feature_engineering_pipeline.py)."""
        return self._generate_momentum_ratios(df, params)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate momentum ratios using base features (exact replica from feature_engineering_pipeline.py)."""
        return self._generate_momentum_ratios(df, params)

    def _generate_momentum_ratios(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """
        Generate momentum-based ratio features with enhanced precision and robustness.

        Improvements:
        - Robust null/infinity checks with finite validation
        - Enhanced error handling with graceful degradation
        - Configurable bounds checking and clamping
        - Improved numerical precision using numpy operations
        - Better logging for debugging and monitoring
        """
        import numpy as np

        features: dict[str, float] = {}
        base_features = params.get("base_features", {})
        lookback_periods = params.get("lookback_periods", [5, 10, 20])
        min_data_length = params.get("min_data_length", 20)
        normalization_constant = params.get("normalization_constant", 50)

        # Enhanced parameters for robustness
        max_momentum_ratio = params.get("max_momentum_ratio", 1.0)  # Cap extreme values
        epsilon = params.get("epsilon", 1e-9)  # Avoid division by zero

        try:
            logger.debug(f"Generating enhanced momentum ratios with {len(df)} data points")

            if len(df) < min_data_length:
                logger.debug(f"Insufficient data: {len(df)} < {min_data_length}, returning empty features")
                return pd.DataFrame(features, index=df.index)

            close_prices = df["close"].values

            # Enhanced momentum ratios with robust calculations
            for period in lookback_periods:
                if len(close_prices) > period:
                    try:
                        current_price = close_prices[-1]
                        past_price = close_prices[-period - 1]

                        # Robust division with bounds checking
                        if abs(past_price) > epsilon and np.isfinite(current_price) and np.isfinite(past_price):
                            momentum_ratio = (current_price / past_price) - 1.0

                            # Clamp extreme values for stability
                            momentum_ratio = np.clip(momentum_ratio, -max_momentum_ratio, max_momentum_ratio)

                            if np.isfinite(momentum_ratio):
                                features[f"momentum_ratio_{period}d"] = float(momentum_ratio)
                            else:
                                logger.debug(f"Non-finite momentum ratio for period {period}, skipping")
                        else:
                            logger.debug(
                                f"Invalid price data for period {period}: current={current_price}, past={past_price}"
                            )
                    except Exception as e:
                        logger.debug(f"Error calculating momentum ratio for period {period}: {e}")
                        continue

            # Enhanced RSI momentum with robust validation
            rsi_14 = base_features.get("rsi_14") or base_features.get("rsi")
            if rsi_14 is not None and np.isfinite(rsi_14) and 0 <= rsi_14 <= 100:
                try:
                    # More robust RSI normalization with bounds checking
                    rsi_momentum = (rsi_14 - normalization_constant) / normalization_constant
                    if np.isfinite(rsi_momentum):
                        features["rsi_momentum_normalized"] = float(rsi_momentum)
                    else:
                        logger.debug("Non-finite RSI momentum, skipping")
                except Exception as e:
                    logger.debug(f"Error calculating RSI momentum: {e}")

            # Enhanced MACD momentum with validation
            macd = base_features.get("macd")
            macd_signal = base_features.get("macd_signal")
            if macd is not None and macd_signal is not None and np.isfinite(macd) and np.isfinite(macd_signal):
                try:
                    macd_strength = abs(macd - macd_signal)
                    if np.isfinite(macd_strength):
                        features["macd_momentum_strength"] = float(macd_strength)
                    else:
                        logger.debug("Non-finite MACD strength, skipping")
                except Exception as e:
                    logger.debug(f"Error calculating MACD momentum: {e}")

            # Enhanced Stochastic momentum with comprehensive validation
            stoch_k = base_features.get("stoch_k")
            stoch_d = base_features.get("stoch_d")
            if (
                stoch_k is not None
                and stoch_d is not None
                and np.isfinite(stoch_k)
                and np.isfinite(stoch_d)
                and 0 <= stoch_k <= 100
                and 0 <= stoch_d <= 100
            ):
                try:
                    # Stochastic momentum divergence
                    stoch_divergence = stoch_k - stoch_d
                    if np.isfinite(stoch_divergence):
                        features["stoch_momentum_divergence"] = float(stoch_divergence)

                    # Stochastic momentum strength with robust calculation
                    if normalization_constant > epsilon:
                        stoch_strength = abs(stoch_k - normalization_constant) / normalization_constant
                        if np.isfinite(stoch_strength):
                            features["stoch_momentum_strength"] = float(stoch_strength)

                        # Stochastic position normalization
                        stoch_position = (stoch_k - normalization_constant) / normalization_constant
                        if np.isfinite(stoch_position):
                            features["stoch_position_normalized"] = float(stoch_position)
                    else:
                        logger.debug("Invalid normalization constant for stochastic calculations")

                except Exception as e:
                    logger.debug(f"Error calculating stochastic momentum: {e}")

            logger.debug(f"Successfully generated {len(features)} momentum ratio features")

        except Exception as e:
            logger.error(f"Critical error in momentum ratio generation: {e}", exc_info=True)
            # Return empty features on critical error
            features = {}

        return pd.DataFrame(features, index=df.index)


def create_momentum_ratios_indicator(feature_config: Any) -> "MomentumRatiosIndicator":
    """Factory function to create MomentumRatiosIndicator from feature configuration."""
    config = IndicatorConfig(
        name=feature_config.name,
        enabled=True,
        params=feature_config.params.get("default", {}),
        timeframe_params={
            tf_key: tf_params for tf_key, tf_params in feature_config.params.items() if tf_key != "default"
        },
    )

    return MomentumRatiosIndicator(config)
