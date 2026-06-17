"""
Bollinger Bands Indicator Implementation.
Exact replication of the logic from feature_calculator.py.
"""

from typing import Any

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class BollingerBandsIndicator(BaseIndicator):
    """Bollinger Bands indicator implementation."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate Bollinger Bands using TA-Lib."""
        length = params.get("length", 34)
        mult = params.get("mult", 2.0)
        source = params.get("source", "close")

        if self.talib is None:
            raise RuntimeError("TA-Lib not available for BBANDS calculation")

        try:
            upperband, middleband, lowerband = self.talib.BBANDS(
                df[source].values, timeperiod=length, nbdevup=mult, nbdevdn=mult
            )
        except Exception as e:
            logger.error(f"BBANDS calculation failed for {len(df)} data points with params {params}: {e}")
            raise
        return pd.DataFrame({"upperband": upperband, "middleband": middleband, "lowerband": lowerband}, index=df.index)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate Bollinger Bands using pandas."""
        length = params.get("length", 34)
        mult = params.get("mult", 2.0)
        source = params.get("source", "close")

        middleband = df[source].rolling(window=length).mean()
        std = df[source].rolling(window=length).std()
        upperband = middleband + (std * mult)
        lowerband = middleband - (std * mult)

        return pd.DataFrame({"upperband": upperband, "middleband": middleband, "lowerband": lowerband}, index=df.index)
