"""
Mean Reversion Signals Indicator Implementation.
Exact replication of the _generate_mean_reversion_signals method from feature_engineering_pipeline.py.
"""

from typing import Any

import numpy as np
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


class MeanReversionSignalsIndicator(BaseIndicator):
    """Mean Reversion Signals indicator - generates mean reversion signals using existing oscillators."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate mean reversion signals using base features (exact replica from feature_engineering_pipeline.py)."""
        return self._generate_mean_reversion_signals(df, params)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate mean reversion signals using base features (exact replica from feature_engineering_pipeline.py)."""
        return self._generate_mean_reversion_signals(df, params)

    def _generate_mean_reversion_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Generate mean reversion signals using existing oscillators with config-driven parameters."""
        features = {}
        base_features = params.get("base_features", {})
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        willr_overbought = params.get("willr_overbought", -20)
        willr_oversold = params.get("willr_oversold", -80)
        stoch_overbought = params.get("stoch_overbought", 80)
        stoch_oversold = params.get("stoch_oversold", 20)

        try:
            logger.debug(f"Generating mean reversion signals with params: {params}")
            # RSI mean reversion signal
            rsi_14 = base_features.get("rsi_14")
            if rsi_14 is not None:
                if rsi_14 > rsi_overbought:
                    reversion_signal = (rsi_14 - rsi_overbought) / (100 - rsi_overbought)
                elif rsi_14 < rsi_oversold:
                    reversion_signal = (rsi_oversold - rsi_14) / rsi_oversold
                else:
                    reversion_signal = 0.0

                features["rsi_mean_reversion"] = reversion_signal

            # Williams %R mean reversion signal
            willr = base_features.get("willr")
            if willr is not None:
                if willr > willr_overbought:
                    willr_reversion = (willr - willr_overbought) / (0 - willr_overbought)
                elif willr < willr_oversold:
                    willr_reversion = (willr_oversold - willr) / (willr_oversold - (-100))
                else:
                    willr_reversion = 0.0

                features["willr_mean_reversion"] = willr_reversion

            # Enhanced Stochastic Mean Reversion Signals
            stoch_k = base_features.get("stoch_k")
            stoch_d = base_features.get("stoch_d")
            if stoch_k is not None and stoch_d is not None:
                # Stochastic overbought/oversold signals
                if stoch_k > stoch_overbought and stoch_d > stoch_overbought:
                    stoch_reversion = (stoch_k - stoch_overbought) / (100 - stoch_overbought)
                elif stoch_k < stoch_oversold and stoch_d < stoch_oversold:
                    stoch_reversion = (stoch_oversold - stoch_k) / stoch_oversold
                else:
                    stoch_reversion = 0.0

                features["stoch_mean_reversion"] = stoch_reversion

                # Stochastic crossover reversion signals
                stoch_crossover = 1.0 if stoch_k > stoch_d else -1.0
                crossover_strength = abs(stoch_k - stoch_d) / 100
                stoch_crossover_signal = stoch_crossover * crossover_strength

                features["stoch_crossover_signal"] = stoch_crossover_signal

            # Composite mean reversion signal
            reversion_signals = []
            for feature_name in ["rsi_mean_reversion", "willr_mean_reversion", "stoch_mean_reversion"]:
                if feature_name in features:
                    reversion_signals.append(features[feature_name])

            if reversion_signals:
                composite_reversion = float(np.mean(reversion_signals))
                features["composite_mean_reversion"] = composite_reversion

        except Exception as e:
            logger.warning(f"Mean reversion signal generation failed: {e}")

        return pd.DataFrame(features, index=df.index)


def create_mean_reversion_signals_indicator(feature_config: Any) -> "MeanReversionSignalsIndicator":
    """Factory function to create MeanReversionSignalsIndicator from feature configuration."""
    config = IndicatorConfig(
        name=feature_config.name,
        enabled=True,
        params=feature_config.params.get("default", {}),
        timeframe_params={
            tf_key: tf_params for tf_key, tf_params in feature_config.params.items() if tf_key != "default"
        },
    )

    return MeanReversionSignalsIndicator(config)
