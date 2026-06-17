"""
Aroon Indicator Implementation.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class AroonIndicator(BaseIndicator):
    """Aroon trend indicator implementation."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate Aroon using TA-Lib."""
        if self.talib is None:
            raise RuntimeError("TA-Lib not available for AROON calculation")

        timeperiod = max(1, int(params.get("timeperiod", 14)))

        if len(df) < timeperiod:
            logger.warning(f"Insufficient data for AROON calculation: {len(df)} < {timeperiod}")
            return pd.DataFrame(
                {"aroondown": [float("nan")] * len(df), "aroonup": [float("nan")] * len(df)}, index=df.index
            )

        try:
            aroondown, aroonup = self.talib.AROON(df["high"].values, df["low"].values, timeperiod=timeperiod)
            return pd.DataFrame(
                {"aroondown": pd.Series(aroondown, index=df.index), "aroonup": pd.Series(aroonup, index=df.index)},
                index=df.index,
            )
        except Exception as e:
            logger.error(f"AROON calculation failed for {len(df)} data points with params {params}: {e}")
            raise

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate the standard Aroon indicator using a vectorized pandas implementation."""
        timeperiod = max(1, int(params.get("timeperiod", 14)))

        if len(df) < timeperiod:
            logger.warning(f"Insufficient data for AROON calculation: {len(df)} < {timeperiod}")
            return pd.DataFrame(
                {"aroondown": [float("nan")] * len(df), "aroonup": [float("nan")] * len(df)}, index=df.index
            )

        try:
            # Get the index of the highest/lowest value in the rolling window
            rolling_high_idx = df["high"].rolling(window=timeperiod, min_periods=0).apply(np.argmax, raw=True)
            rolling_low_idx = df["low"].rolling(window=timeperiod, min_periods=0).apply(np.argmin, raw=True)

            # The index from argmax/argmin is the offset from the start of the window.
            # We need the number of periods from the end of the window.
            periods_since_high = (timeperiod - 1) - rolling_high_idx
            periods_since_low = (timeperiod - 1) - rolling_low_idx

            # Standard Aroon formula
            aroon_up = ((timeperiod - periods_since_high) / timeperiod) * 100
            aroon_down = ((timeperiod - periods_since_low) / timeperiod) * 100

            return pd.DataFrame({"aroonup": aroon_up, "aroondown": aroon_down}, index=df.index)
        except Exception as e:
            logger.error(f"Pandas AROON calculation failed for {len(df)} data points with params {params}: {e}")
            raise
