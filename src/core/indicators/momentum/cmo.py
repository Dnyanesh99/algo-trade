"""
CMO (Chande Momentum Oscillator) Indicator Implementation.
"""

from typing import Any

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class CMOIndicator(BaseIndicator):
    """
    Chande Momentum Oscillator implementation.
    Matches TradingView Pine Script logic exactly.
    """

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Validate CMO-specific parameters."""
        timeperiod = params.get("timeperiod", 9)
        if not isinstance(timeperiod, (int, float)) or timeperiod <= 0:
            raise RuntimeError(f"{self.name}: timeperiod must be a positive number - trading system compromised")

        # Validate source column
        source = params.get("source", "close")
        if not isinstance(source, str):
            raise RuntimeError(f"{self.name}: source must be a string - trading system compromised")

        return True

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate CMO using TA-Lib."""
        if self.talib is None:
            raise RuntimeError("TA-Lib not available for CMO calculation")

        timeperiod = max(1, int(params.get("timeperiod", 9)))
        source = params.get("source", "close")

        if len(df) < timeperiod:
            logger.warning(f"Insufficient data for CMO calculation: {len(df)} < {timeperiod}")
            return pd.Series([float("nan")] * len(df), index=df.index)

        try:
            cmo = self.talib.CMO(df[source].values, timeperiod=timeperiod)
            return pd.Series(cmo, index=df.index).clip(-100, 100)
        except Exception as e:
            raise RuntimeError(
                f"CMO calculation failed for {len(df)} data points with params {params}: {e} - trading system compromised"
            ) from e

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate CMO using pandas - matches TradingView Pine Script exactly."""
        timeperiod = max(1, int(params.get("timeperiod", 9)))
        source = params.get("source", "close")

        if len(df) < timeperiod:
            logger.warning(f"Insufficient data for CMO calculation: {len(df)} < {timeperiod}")
            return pd.Series([float("nan")] * len(df), index=df.index)

        try:
            # Pine Script logic: momm = ta.change(src)
            momm = df[source].diff()

            # Pine Script: f1(m) => m >= 0.0 ? m : 0.0 (positive changes only)
            m1 = momm.where(momm >= 0.0, 0.0)

            # Pine Script: f2(m) => m >= 0.0 ? 0.0 : -m (negative changes as positive)
            m2 = (-momm).where(momm < 0.0, 0.0)

            # Pine Script: sm1 = math.sum(m1, length) - Simple rolling sum
            sm1 = m1.rolling(window=timeperiod).sum()

            # Pine Script: sm2 = math.sum(m2, length) - Simple rolling sum
            sm2 = m2.rolling(window=timeperiod).sum()

            # Pine Script: chandeMO = percent(sm1-sm2, sm1+sm2) = 100 * (sm1-sm2) / (sm1+sm2)
            total_movement = sm1 + sm2
            cmo = 100 * (sm1 - sm2) / total_movement.replace(0, float("nan"))

            return cmo.clip(-100, 100)
        except Exception as e:
            raise RuntimeError(f"Pandas CMO calculation failed: {e} - trading system compromised") from e
