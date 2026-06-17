"""
OBV (On-Balance Volume) Indicator Implementation.
Exact replication of the logic from feature_calculator.py.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class OBVIndicator(BaseIndicator):
    """On-Balance Volume (OBV) indicator implementation with a configurable signal line."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate OBV and its signal line using TA-Lib and pandas."""
        if self.talib is None:
            raise RuntimeError("TA-Lib not available for OBV calculation")

        source = params.get("source", "close")
        signal_type = params.get("signal_smoothing_type", "SMA").upper()
        signal_length = params.get("signal_smoothing_length", 21)

        try:
            obv = self.talib.OBV(df[source].values, df["volume"].values)
        except Exception as e:
            logger.error(f"OBV calculation failed for {len(df)} data points with params {params}: {e}")
            raise
        obv_series = pd.Series(obv, index=df.index)

        signal = self._calculate_signal_line(obv_series, signal_type, signal_length)

        return pd.DataFrame({"obv": obv_series, "obv_signal": signal})

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate OBV and its signal line using pandas."""
        source = params.get("source", "close")
        signal_type = params.get("signal_smoothing_type", "SMA").upper()
        signal_length = params.get("signal_smoothing_length", 21)

        price_diff = df[source].diff()
        signed_volume = (np.sign(price_diff) * df["volume"]).fillna(0)
        obv_series = signed_volume.cumsum()

        signal = self._calculate_signal_line(obv_series, signal_type, signal_length)

        return pd.DataFrame({"obv": obv_series, "obv_signal": signal})

    def _calculate_signal_line(self, obv_series: pd.Series, signal_type: str, signal_length: int) -> pd.Series:
        """Calculates the signal line for the OBV."""
        if signal_type == "EMA":
            return obv_series.ewm(span=signal_length, adjust=False).mean()
        # Default to SMA
        return obv_series.rolling(window=signal_length).mean()
