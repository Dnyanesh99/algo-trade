"""
ADOSC (Chaikin A/D Oscillator) Indicator Implementation.
Exact replication of the logic from feature_calculator.py.
"""

from typing import Any

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class ADOSCIndicator(BaseIndicator):
    """Chaikin A/D Oscillator (ADOSC) indicator implementation."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate ADOSC using TA-Lib."""
        fastperiod = params.get("fastperiod", 3)
        slowperiod = params.get("slowperiod", 10)

        if self.talib is None:
            raise RuntimeError("TA-Lib not available for ADOSC calculation")

        try:
            adosc = self.talib.ADOSC(
                df["high"].values,
                df["low"].values,
                df["close"].values,
                df["volume"].values,
                fastperiod=fastperiod,
                slowperiod=slowperiod,
            )
        except Exception as e:
            logger.error(f"ADOSC calculation failed for {len(df)} data points with params {params}: {e}")
            raise
        return pd.Series(adosc, index=df.index)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate ADOSC using pandas."""
        fastperiod = params.get("fastperiod", 3)
        slowperiod = params.get("slowperiod", 10)

        # Calculate Money Flow Multiplier
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
        mfm = mfm.fillna(0)

        # Calculate Money Flow Volume
        mfv = mfm * df["volume"]

        # Calculate A/D Line
        ad_line = mfv.cumsum()

        # Calculate ADOSC as difference of two EMAs
        fast_ema = ad_line.ewm(span=fastperiod, adjust=False).mean()
        slow_ema = ad_line.ewm(span=slowperiod, adjust=False).mean()

        return fast_ema - slow_ema
