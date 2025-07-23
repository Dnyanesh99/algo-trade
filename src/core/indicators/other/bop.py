from typing import Any

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class BOPIndicator(BaseIndicator):
    """Balance of Power (BOP) indicator implementation."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate BOP using TA-Lib."""
        if self.talib is None:
            raise RuntimeError("TA-Lib not available for BOP calculation")

        try:
            bop = self.talib.BOP(df["open"].values, df["high"].values, df["low"].values, df["close"].values)
        except Exception as e:
            logger.error(f"BOP calculation failed for {len(df)} data points with params {params}: {e}")
            raise
        return pd.Series(bop, index=df.index)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate BOP using pandas."""
        try:
            return (df["close"] - df["open"]) / (df["high"] - df["low"]).replace(0, float("nan"))
        except Exception as e:
            logger.error(f"Pandas BOP calculation failed for {len(df)} data points with params {params}: {e}")
            raise
