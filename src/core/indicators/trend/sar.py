"""
SAR (Parabolic Stop and Reverse) Indicator Implementation.
"""

from typing import Any

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class SARIndicator(BaseIndicator):
    """Parabolic SAR trend indicator implementation."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate SAR using TA-Lib."""
        if self.talib is None:
            raise RuntimeError("TA-Lib not available for SAR calculation")

        start = float(params.get("start", 0.02))
        increment = float(params.get("increment", 0.02))
        maximum = float(params.get("maximum", 0.2))

        if start != increment:
            logger.warning(
                f"TA-Lib SAR uses a single acceleration factor. Using increment={increment} and ignoring start={start}."
            )

        if len(df) < 2:
            logger.warning(f"Insufficient data for SAR calculation: {len(df)} < 2")
            return pd.Series([float("nan")] * len(df), index=df.index)

        try:
            return self.talib.SAR(df["high"].values, df["low"].values, acceleration=increment, maximum=maximum)
        except Exception as e:
            logger.error(f"SAR calculation failed for {len(df)} data points with params {params}: {e}")
            return pd.Series([float("nan")] * len(df), index=df.index)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate Parabolic SAR using a correct, standard pandas implementation."""
        start = float(params.get("start", 0.02))
        increment = float(params.get("increment", 0.02))
        maximum = float(params.get("maximum", 0.2))

        if len(df) < 2:
            logger.warning(f"Insufficient data for SAR calculation: {len(df)} < 2")
            return pd.Series([float("nan")] * len(df), index=df.index)

        try:
            high = df["high"].values
            low = df["low"].values
            sar_series = pd.Series(index=df.index, dtype=float)

            # Initial values
            rising = True
            af = start
            ep = high[0]
            sar_val = low[0]
            sar_series.iloc[0] = sar_val

            for i in range(1, len(df)):
                # Calculate current SAR
                sar_val = sar_val + af * (ep - sar_val)

                # Process trend switch
                if rising:
                    if low[i] < sar_val:
                        rising = False
                        sar_val = ep  # Switch to old EP
                        ep = low[i]  # New EP is the new low
                        af = start
                else:  # Falling
                    if high[i] > sar_val:
                        rising = True
                        sar_val = ep  # Switch to old EP
                        ep = high[i]  # New EP is the new high
                        af = start

                # Ensure SAR is not placed inside the previous or current bar
                if rising:
                    sar_val = min(sar_val, low[i - 1], low[i])
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + increment, maximum)
                else:  # Falling
                    sar_val = max(sar_val, high[i - 1], high[i])
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + increment, maximum)

                sar_series.iloc[i] = sar_val

            return sar_series
        except Exception as e:
            logger.error(f"Pandas SAR calculation failed: {e}")
            return pd.Series([float("nan")] * len(df), index=df.index)
