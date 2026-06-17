"""
Trend Confirmations Indicator Implementation.
Exact replication of the _generate_trend_confirmations method from feature_engineering_pipeline.py.
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


class TrendConfirmationsIndicator(BaseIndicator):
    """Trend Confirmations indicator - generates trend confirmation features using existing trend indicators."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate trend confirmations using base features (exact replica from feature_engineering_pipeline.py)."""
        return self._generate_trend_confirmations(df, params)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate trend confirmations using base features (exact replica from feature_engineering_pipeline.py)."""
        return self._generate_trend_confirmations(df, params)

    def _generate_trend_confirmations(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Generate trend confirmation features using existing trend indicators with config-driven parameters."""
        features = {}
        base_features = params.get("base_features", {})
        adx_normalization_factor = params.get("adx_normalization_factor", 25.0)
        trend_strength_max = params.get("trend_strength_max", 2.0)

        try:
            logger.debug(f"Generating trend confirmations with params: {params}")
            # ADX trend strength normalization
            adx = base_features.get("adx")
            if adx is not None:
                trend_strength = min(adx / adx_normalization_factor, trend_strength_max)
                features["trend_strength_normalized"] = trend_strength

            # Multi-indicator trend consensus
            trend_indicators = []

            # MACD trend signal
            macd = base_features.get("macd")
            macd_signal = base_features.get("macd_signal")
            if macd is not None and macd_signal is not None:
                trend_indicators.append(1 if macd > macd_signal else -1)

            # SAR trend signal
            sar = base_features.get("sar")
            if sar is not None and len(df) > 0:
                current_close = df["close"].iloc[-1]
                trend_indicators.append(1 if current_close > sar else -1)

            # Aroon trend signal
            aroon_up = base_features.get("aroon_up")
            aroon_down = base_features.get("aroon_down")
            if aroon_up is not None and aroon_down is not None:
                trend_indicators.append(1 if aroon_up > aroon_down else -1)

            # Calculate trend consensus
            if trend_indicators:
                trend_consensus = sum(trend_indicators) / len(trend_indicators)
                features["trend_consensus"] = trend_consensus

        except Exception as e:
            logger.warning(f"Trend confirmation generation failed: {e}")

        return pd.DataFrame(features, index=df.index)


def create_trend_confirmations_indicator(feature_config: Any) -> "TrendConfirmationsIndicator":
    """Factory function to create TrendConfirmationsIndicator from feature configuration."""
    config = IndicatorConfig(
        name=feature_config.name,
        enabled=True,
        params=feature_config.params.get("default", {}),
        timeframe_params={
            tf_key: tf_params for tf_key, tf_params in feature_config.params.items() if tf_key != "default"
        },
    )

    return TrendConfirmationsIndicator(config)
