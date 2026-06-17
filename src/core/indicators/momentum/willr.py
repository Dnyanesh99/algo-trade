"""
Williams %R Indicator Implementation.
"""

from typing import Any, Union

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class WilliamsRIndicator(BaseIndicator):
    """
    Williams %R momentum oscillator indicator implementation.
    Matches TradingView Pine Script logic exactly.
    """

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Validate Williams %R-specific parameters."""
        timeperiod = params.get("timeperiod", 21)
        if not isinstance(timeperiod, (int, float)) or timeperiod <= 0:
            raise RuntimeError(f"{self.name}: timeperiod must be a positive number - trading system compromised")

        # Validate EMA smoothing parameters
        ema_smoothing = params.get("ema_smoothing", False)
        if ema_smoothing:
            ema_length = params.get("ema_length", 13)
            if not isinstance(ema_length, (int, float)) or ema_length <= 0:
                raise RuntimeError(f"{self.name}: ema_length must be a positive number - trading system compromised")

        # Validate trigger levels
        upper_band = params.get("upper_band", -20)
        lower_band = params.get("lower_band", -80)
        if not (-100 <= lower_band < upper_band <= 0):
            raise RuntimeError(
                f"{self.name}: Invalid band levels - lower_band={lower_band}, upper_band={upper_band} - trading system compromised"
            )

        return True

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> Union[pd.Series, pd.DataFrame]:
        """Calculate Williams %R using TA-Lib with optional EMA smoothing."""
        if self.talib is None:
            raise RuntimeError("TA-Lib not available for WILLR calculation")

        timeperiod = max(1, int(params.get("timeperiod", 21)))

        if len(df) < timeperiod:
            logger.warning(f"Insufficient data for WILLR calculation: {len(df)} < {timeperiod}")
            return pd.Series([float("nan")] * len(df), index=df.index)

        try:
            willr = self.talib.WILLR(df["high"].values, df["low"].values, df["close"].values, timeperiod=timeperiod)
            willr_series = pd.Series(willr, index=df.index)

            # Apply EMA smoothing if enabled
            ema_smoothing = params.get("ema_smoothing", False)
            if ema_smoothing:
                ema_length = int(params.get("ema_length", 13))
                willr_ema = willr_series.ewm(span=ema_length, adjust=False).mean()
                return pd.DataFrame({"willr": willr_series, "willr_ema": willr_ema})

            return willr_series
        except Exception as e:
            raise RuntimeError(
                f"WILLR calculation failed for {len(df)} data points with params {params}: {e} - trading system compromised"
            ) from e

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> Union[pd.Series, pd.DataFrame]:
        """Calculate Williams %R using pandas - matches TradingView Pine Script exactly."""
        timeperiod = max(1, int(params.get("timeperiod", 21)))

        if len(df) < timeperiod:
            logger.warning(f"Insufficient data for WILLR calculation: {len(df)} < {timeperiod}")
            return pd.Series([float("nan")] * len(df), index=df.index)

        try:
            # Pine Script: upper = highest(length), lower = lowest(length)
            upper = df["high"].rolling(window=timeperiod).max()  # highest(high, length)
            lower = df["low"].rolling(window=timeperiod).min()  # lowest(low, length)

            # Pine Script: output = 100 * (close - upper) / (upper - lower)
            # Note: Pine Script uses (close - upper), not (upper - close) like traditional Williams %R
            range_diff = upper - lower
            willr = 100 * (df["close"] - upper) / range_diff.replace(0, float("nan"))
            willr = willr.clip(-100, 0)  # Williams %R is always negative

            # Apply EMA smoothing if enabled
            ema_smoothing = params.get("ema_smoothing", False)
            if ema_smoothing:
                ema_length = int(params.get("ema_length", 13))
                willr_ema = willr.ewm(span=ema_length, adjust=False).mean()
                return pd.DataFrame({"willr": willr, "willr_ema": willr_ema})

            return willr
        except Exception as e:
            raise RuntimeError(
                f"Pandas WILLR calculation failed for {len(df)} data points with params {params}: {e} - trading system compromised"
            ) from e
