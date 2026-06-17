"""
ADX (Average Directional Index) Indicator Implementation.
"""

from typing import Any

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class ADXIndicator(BaseIndicator):
    """ADX trend strength indicator implementation."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate ADX using TA-Lib."""
        if self.talib is None:
            raise RuntimeError("TA-Lib not available for ADX calculation")

        dilen = max(1, int(params.get("dilen", 14)))
        adxlen = max(1, int(params.get("adxlen", 14)))

        if dilen != adxlen:
            logger.warning(f"TA-Lib ADX uses a single timeperiod. Using dilen={dilen} and ignoring adxlen={adxlen}.")

        min_required = dilen * 2
        if len(df) < min_required:
            logger.warning(f"Insufficient data for ADX calculation: {len(df)} < {min_required}")
            return pd.Series([float("nan")] * len(df), index=df.index)

        try:
            adx = self.talib.ADX(df["high"].values, df["low"].values, df["close"].values, timeperiod=dilen)
            return pd.Series(adx, index=df.index).clip(0, 100)
        except Exception as e:
            logger.error(f"ADX calculation failed for {len(df)} data points with params {params}: {e}")
            raise

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate ADX using pandas, matching Pine Script's separate DI and ADX lengths."""
        dilen = max(1, int(params.get("dilen", 14)))
        adxlen = max(1, int(params.get("adxlen", 14)))

        min_required = dilen + adxlen
        if len(df) < min_required:
            logger.warning(f"Insufficient data for ADX calculation: {len(df)} < {min_required}")
            return pd.Series([float("nan")] * len(df), index=df.index)

        try:
            high = df["high"]
            low = df["low"]
            close = df["close"]

            # Wilder's smoothing for True Range
            tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1 / dilen, adjust=False).mean()

            # Directional Movement
            up = high.diff()
            down = -low.diff()
            plus_dm = ((up > down) & (up > 0)) * up
            minus_dm = ((down > up) & (down > 0)) * down

            # Wilder's smoothing for Directional Movement
            plus_di = 100 * plus_dm.ewm(alpha=1 / dilen, adjust=False).mean() / atr
            minus_di = 100 * minus_dm.ewm(alpha=1 / dilen, adjust=False).mean() / atr

            # ADX calculation
            dx_sum = (plus_di + minus_di).replace(0, 1)  # Avoid division by zero
            dx = 100 * (abs(plus_di - minus_di) / dx_sum)
            adx = dx.ewm(alpha=1 / adxlen, adjust=False).mean()

            return adx.clip(0, 100)
        except Exception as e:
            logger.error(f"Pandas ADX calculation failed: {e}")
            raise
