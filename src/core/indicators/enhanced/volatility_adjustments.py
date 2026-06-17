"""
Volatility Adjustments Indicator Implementation.
Exact replication of the _generate_volatility_adjustments method from feature_engineering_pipeline.py.
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


class VolatilityAdjustmentsIndicator(BaseIndicator):
    """Volatility Adjustments indicator - generates volatility-adjusted features using existing ATR."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate volatility adjustments using base features (exact replica from feature_engineering_pipeline.py)."""
        return self._generate_volatility_adjustments(df, params)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate volatility adjustments using base features (exact replica from feature_engineering_pipeline.py)."""
        return self._generate_volatility_adjustments(df, params)

    def _generate_volatility_adjustments(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Generate volatility-adjusted features using existing ATR with config-driven parameters."""
        features = {}
        base_features = params.get("base_features", {})
        min_data_length = params.get("min_data_length", 2)

        try:
            logger.debug(f"Generating volatility adjustments with params: {params}")
            atr_14 = base_features.get("atr_14")
            if atr_14 is not None and atr_14 > 0 and len(df) >= min_data_length:
                recent_return = df["close"].iloc[-1] / df["close"].iloc[-2] - 1.0
                vol_adjusted_return = recent_return / atr_14

                features["volatility_adjusted_return"] = vol_adjusted_return

                bb_upper = base_features.get("bb_upper")
                bb_lower = base_features.get("bb_lower")
                close = df["close"].iloc[-1]

                if bb_upper is not None and bb_lower is not None and bb_upper != bb_lower:
                    bb_position = (close - bb_lower) / (bb_upper - bb_lower)
                    vol_adjusted_bb = bb_position * (atr_14 / close)

                    features["vol_adjusted_bb_position"] = vol_adjusted_bb

        except Exception as e:
            logger.warning(f"Volatility adjustment generation failed: {e}")

        return pd.DataFrame(features, index=df.index)


def create_volatility_adjustments_indicator(feature_config: Any) -> "VolatilityAdjustmentsIndicator":
    """Factory function to create VolatilityAdjustmentsIndicator from feature configuration."""
    config = IndicatorConfig(
        name=feature_config.name,
        enabled=True,
        params=feature_config.params.get("default", {}),
        timeframe_params={
            tf_key: tf_params for tf_key, tf_params in feature_config.params.items() if tf_key != "default"
        },
    )

    return VolatilityAdjustmentsIndicator(config)
