from typing import Any

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class TrueRangeIndicator(BaseIndicator):
    """True Range (TRANGE) indicator implementation."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate TRANGE using TA-Lib."""
        if self.talib is None:
            raise RuntimeError("TA-Lib not available for TRANGE calculation")

        try:
            trange = self.talib.TRANGE(df["high"].values, df["low"].values, df["close"].values)
        except Exception as e:
            logger.error(f"TRANGE calculation failed for {len(df)} data points with params {params}: {e}")
            raise
        return pd.Series(trange, index=df.index)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate TRANGE using pandas."""
        try:
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        except Exception as e:
            logger.error(f"Pandas TRANGE calculation failed for {len(df)} data points with params {params}: {e}")
            raise
