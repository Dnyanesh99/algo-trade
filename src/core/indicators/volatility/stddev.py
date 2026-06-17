"""
STDDEV (Standard Deviation) Indicator Implementation.
Exact replication of the logic from feature_calculator.py.
"""

from typing import Any

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class StandardDeviationIndicator(BaseIndicator):
    """Standard Deviation indicator implementation."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate STDDEV using TA-Lib."""
        length = params.get("length", 20)
        source = params.get("source", "close")

        if self.talib is None:
            raise RuntimeError("TA-Lib not available for STDDEV calculation")

        try:
            stddev = self.talib.STDDEV(df[source].values, timeperiod=length, nbdev=1.0)
        except Exception as e:
            logger.error(f"STDDEV calculation failed for {len(df)} data points with params {params}: {e}")
            raise
        return pd.Series(stddev, index=df.index)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate STDDEV using pandas."""
        length = params.get("length", 20)
        source = params.get("source", "close")
        return df[source].rolling(window=length).std()
