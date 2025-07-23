from typing import Any

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class UltimateOscillatorIndicator(BaseIndicator):
    """Ultimate Oscillator (ULTOSC) indicator implementation."""

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate ULTOSC using TA-Lib."""
        timeperiod1 = params.get("timeperiod1", 7)
        timeperiod2 = params.get("timeperiod2", 14)
        timeperiod3 = params.get("timeperiod3", 28)

        if self.talib is None:
            raise RuntimeError("TA-Lib not available for ULTOSC calculation")

        try:
            ultosc = self.talib.ULTOSC(
                df["high"].values,
                df["low"].values,
                df["close"].values,
                timeperiod1=timeperiod1,
                timeperiod2=timeperiod2,
                timeperiod3=timeperiod3,
            )
        except Exception as e:
            logger.error(f"ULTOSC calculation failed for {len(df)} data points with params {params}: {e}")
            raise
        return pd.Series(ultosc, index=df.index)

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Calculate ULTOSC using pandas."""
        timeperiod1 = params.get("timeperiod1", 7)
        timeperiod2 = params.get("timeperiod2", 14)
        timeperiod3 = params.get("timeperiod3", 28)

        try:
            low = df["low"]
            high = df["high"]
            close = df["close"]
            close_prev = close.shift(1)

            true_low = pd.concat([low, close_prev], axis=1).min(axis=1)
            true_range = pd.concat([high, close_prev], axis=1).max(axis=1) - true_low

            buying_pressure = close - true_low

            bp_sum1 = buying_pressure.rolling(window=timeperiod1).sum()
            tr_sum1 = true_range.rolling(window=timeperiod1).sum()

            bp_sum2 = buying_pressure.rolling(window=timeperiod2).sum()
            tr_sum2 = true_range.rolling(window=timeperiod2).sum()

            bp_sum3 = buying_pressure.rolling(window=timeperiod3).sum()
            tr_sum3 = true_range.rolling(window=timeperiod3).sum()

            avg1 = bp_sum1 / tr_sum1.replace(0, float("nan"))
            avg2 = bp_sum2 / tr_sum2.replace(0, float("nan"))
            avg3 = bp_sum3 / tr_sum3.replace(0, float("nan"))

            return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
        except Exception as e:
            logger.error(f"Pandas ULTOSC calculation failed for {len(df)} data points with params {params}: {e}")
            raise
