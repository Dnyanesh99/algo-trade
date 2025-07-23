from typing import Any

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class CCIIndicator(BaseIndicator):
    """Commodity Channel Index (CCI) indicator implementation."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate CCI using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)

        if self.talib is None:
            raise RuntimeError("TA-Lib not available for CCI calculation")

        try:
            cci = self.talib.CCI(df["high"].values, df["low"].values, df["close"].values, timeperiod=timeperiod)
        except Exception as e:
            logger.error(f"CCI calculation failed for {len(df)} data points with params {params}: {e}")
            raise
        return pd.Series(cci, index=df.index)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate CCI using pandas."""
        timeperiod = params.get("timeperiod", 14)

        try:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            sma = typical_price.rolling(window=timeperiod).mean()
            mad = typical_price.rolling(window=timeperiod).apply(lambda x: (x - x.mean()).abs().mean(), raw=True)

            return (typical_price - sma) / (0.015 * mad).replace(0, float("nan"))
        except Exception as e:
            logger.error(f"Pandas CCI calculation failed for {len(df)} data points with params {params}: {e}")
            raise
