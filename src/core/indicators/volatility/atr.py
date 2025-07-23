"""
ATR (Average True Range) Indicator Implementation.
Exact replication of the logic from feature_calculator.py.
"""

from typing import Any

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class ATRIndicator(BaseIndicator):
    """
    Average True Range (ATR) indicator implementation.
    """

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate ATR using TA-Lib. Note: TA-Lib always uses RMA smoothing."""
        length = params.get("length", 14)
        smoothing = params.get("smoothing", "RMA").upper()

        if self.talib is None:
            raise RuntimeError("TA-Lib not available for ATR calculation")

        if smoothing != "RMA":
            logger.warning(f"TA-Lib ATR only supports RMA smoothing. Pandas will be used for {smoothing}.")

        try:
            atr = self.talib.ATR(df["high"].values, df["low"].values, df["close"].values, timeperiod=length)
        except Exception as e:
            logger.error(f"ATR calculation failed for {len(df)} data points with params {params}: {e}")
            raise
        return pd.Series(atr, index=df.index)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate ATR using pandas with configurable smoothing."""
        length = params.get("length", 14)
        smoothing = params.get("smoothing", "RMA").upper()

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        if smoothing == "SMA":
            return true_range.rolling(window=length).mean()
        if smoothing == "EMA":
            return true_range.ewm(span=length, adjust=False).mean()
        if smoothing == "WMA":
            weights = pd.Series(range(1, length + 1))
            return true_range.rolling(window=length).apply(lambda x: (x * weights).sum() / weights.sum(), raw=False)
        # Default to RMA
        logger.debug(f"Calculating ATR with RMA smoothing for length {length}")
        return true_range.ewm(alpha=1 / length, adjust=False).mean()
