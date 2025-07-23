"""
MACD (Moving Average Convergence Divergence) Indicator Implementation.
"""

from typing import Any

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class MACDIndicator(BaseIndicator):
    """
    MACD trend-following momentum indicator implementation.
    Note: The pandas implementation uses EMA, which may have minor differences
    from the TA-Lib implementation.
    """

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate MACD using TA-Lib."""
        if self.talib is None:
            raise RuntimeError("TA-Lib not available for MACD calculation")

        fastperiod = max(1, int(params.get("fastperiod", 12)))
        slowperiod = max(1, int(params.get("slowperiod", 26)))
        signalperiod = max(1, int(params.get("signalperiod", 9)))
        source = params.get("source", "close")

        min_required = max(slowperiod, fastperiod) + signalperiod
        if len(df) < min_required:
            logger.warning(f"Insufficient data for MACD calculation: {len(df)} < {min_required}")
            return pd.DataFrame(
                {
                    "macd": [float("nan")] * len(df),
                    "macdsignal": [float("nan")] * len(df),
                    "macdhist": [float("nan")] * len(df),
                },
                index=df.index,
            )

        try:
            macd, macdsignal, macdhist = self.talib.MACD(
                df[source].values, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod
            )
            return pd.DataFrame({"macd": macd, "macdsignal": macdsignal, "macdhist": macdhist}, index=df.index)
        except Exception as e:
            logger.error(f"MACD calculation failed for {len(df)} data points with params {params}: {e}")
            raise

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate MACD using pandas."""
        fastperiod = max(1, int(params.get("fastperiod", 12)))
        slowperiod = max(1, int(params.get("slowperiod", 26)))
        signalperiod = max(1, int(params.get("signalperiod", 9)))
        source = params.get("source", "close")
        oscillator_ma_type = params.get("oscillator_ma_type", "EMA").upper()
        signal_ma_type = params.get("signal_ma_type", "EMA").upper()

        min_required = max(slowperiod, fastperiod) + signalperiod
        if len(df) < min_required:
            logger.warning(f"Insufficient data for MACD calculation: {len(df)} < {min_required}")
            return pd.DataFrame(
                {
                    "macd": [float("nan")] * len(df),
                    "macdsignal": [float("nan")] * len(df),
                    "macdhist": [float("nan")] * len(df),
                },
                index=df.index,
            )

        try:
            if oscillator_ma_type == "SMA":
                fast_ma = df[source].rolling(window=fastperiod).mean()
                slow_ma = df[source].rolling(window=slowperiod).mean()
            else:  # Default to EMA
                fast_ma = df[source].ewm(span=fastperiod, adjust=False).mean()
                slow_ma = df[source].ewm(span=slowperiod, adjust=False).mean()

            macd = fast_ma - slow_ma

            if signal_ma_type == "SMA":
                macdsignal = macd.rolling(window=signalperiod).mean()
            else:  # Default to EMA
                macdsignal = macd.ewm(span=signalperiod, adjust=False).mean()

            macdhist = macd - macdsignal
            return pd.DataFrame({"macd": macd, "macdsignal": macdsignal, "macdhist": macdhist}, index=df.index)
        except Exception as e:
            logger.error(f"Pandas MACD calculation failed: {e}")
            raise
