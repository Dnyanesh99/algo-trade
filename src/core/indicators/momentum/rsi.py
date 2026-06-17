"""
RSI (Relative Strength Index) Indicator Implementation.
"""

from typing import Any, Union

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class RSIIndicator(BaseIndicator):
    """
    RSI (Relative Strength Index) Implementation.
    The pandas implementation uses an Exponential Moving Average (EMA) to better align with TA-Lib.
    """

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Validate RSI-specific parameters."""
        timeperiod = params.get("timeperiod", 14)
        if not isinstance(timeperiod, (int, float)) or timeperiod <= 0:
            logger.error(f"{self.name}: timeperiod must be a positive number")
            return False

        # Validate source column
        source = params.get("source", "close")
        if not isinstance(source, str):
            logger.error(f"{self.name}: source must be a string")
            return False

        # Validate smoothing parameters
        smoothing = params.get("smoothing", {})
        if smoothing:
            smoothing_type = smoothing.get("type", "None")
            valid_types = ["None", "SMA", "SMA + Bollinger Bands", "EMA", "SMMA (RMA)", "WMA", "VWMA"]
            if smoothing_type not in valid_types:
                logger.error(f"{self.name}: Invalid smoothing type: {smoothing_type}")
                return False

            if smoothing_type != "None":
                length = smoothing.get("length", 14)
                if not isinstance(length, (int, float)) or length <= 0:
                    logger.error(f"{self.name}: smoothing length must be a positive number")
                    return False

        return True

    def validate_result(self, result: Union[pd.Series, pd.DataFrame]) -> bool:
        """Validate RSI calculation result."""
        if not super().validate_result(result):
            return False
        if isinstance(result, pd.Series):
            valid_range = result.dropna()
            if len(valid_range) > 0 and ((valid_range < 0).any() or (valid_range > 100).any()):
                logger.warning(f"{self.name}: RSI values outside valid range [0, 100]")
                return False
        return True

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> Union[pd.Series, pd.DataFrame]:
        """Calculate RSI using TA-Lib with optional smoothing."""
        if self.talib is None:
            raise RuntimeError("TA-Lib not available for RSI calculation")

        timeperiod = max(1, int(params.get("timeperiod", 14)))
        source = params.get("source", "close")

        if len(df) < timeperiod:
            logger.warning(f"{self.name}: Insufficient data for calculation")
            return pd.Series([float("nan")] * len(df), index=df.index)

        try:
            # Calculate base RSI
            rsi = self.talib.RSI(df[source].values, timeperiod=timeperiod)
            rsi_series = pd.Series(rsi, index=df.index)

            # Apply smoothing if configured
            smoothing = params.get("smoothing", {})
            if smoothing and smoothing.get("type", "None") != "None":
                return self._apply_smoothing(rsi_series, smoothing)

            return rsi_series
        except Exception as e:
            logger.error(f"{self.name}: TA-Lib calculation failed for {len(df)} data points with params {params}: {e}")
            raise

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> Union[pd.Series, pd.DataFrame]:
        """Calculate RSI using pandas with RMA (as per TradingView) and optional smoothing."""
        timeperiod = max(1, int(params.get("timeperiod", 14)))
        source = params.get("source", "close")

        if len(df) < timeperiod:
            logger.warning(f"{self.name}: Insufficient data for calculation")
            return pd.Series([float("nan")] * len(df), index=df.index)

        try:
            # Calculate using RMA (ta.rma) as per Pine Script - equivalent to Wilder's smoothing
            delta = df[source].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # RMA calculation (Wilder's smoothing) - more accurate to TradingView
            alpha = 1.0 / timeperiod
            avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
            avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

            # Handle division by zero as per Pine Script logic
            rs = avg_gain / avg_loss.replace(0, float("nan"))
            rsi = 100 - (100 / (1 + rs))

            # Handle edge cases as per Pine Script
            rsi = rsi.where(avg_loss != 0, 100)  # if down == 0 ? 100
            rsi = rsi.where(avg_gain != 0, 0)  # if up == 0 ? 0

            # Apply smoothing if configured
            smoothing = params.get("smoothing", {})
            if smoothing and smoothing.get("type", "None") != "None":
                return self._apply_smoothing(rsi, smoothing)

            return rsi
        except Exception as e:
            logger.error(f"{self.name}: Pandas calculation failed: {e}")
            raise

    def _apply_smoothing(self, rsi: pd.Series, smoothing: dict[str, Any]) -> pd.DataFrame:
        """Apply smoothing moving average to RSI as per Pine Script."""
        smoothing_type = smoothing.get("type", "None")
        length = int(smoothing.get("length", 14))

        try:
            if smoothing_type == "SMA":
                smoothed = rsi.rolling(window=length).mean()
            elif smoothing_type == "EMA":
                smoothed = rsi.ewm(span=length, adjust=False).mean()
            elif smoothing_type == "SMMA (RMA)":
                alpha = 1.0 / length
                smoothed = rsi.ewm(alpha=alpha, adjust=False).mean()
            elif smoothing_type == "WMA":
                weights = pd.Series(range(1, length + 1))
                smoothed = rsi.rolling(window=length).apply(lambda x: (x * weights).sum() / weights.sum(), raw=False)
            elif smoothing_type == "VWMA":
                # For VWMA, we'd need volume data - fallback to SMA for now
                logger.warning(f"{self.name}: VWMA not implemented for RSI smoothing, using SMA")
                smoothed = rsi.rolling(window=length).mean()
            elif smoothing_type == "SMA + Bollinger Bands":
                smoothed = rsi.rolling(window=length).mean()
                bb_std_dev = smoothing.get("bb_std_dev", 2.0)
                std = rsi.rolling(window=length).std()
                upper_band = smoothed + (std * bb_std_dev)
                lower_band = smoothed - (std * bb_std_dev)

                return pd.DataFrame(
                    {"rsi": rsi, "rsi_smoothed": smoothed, "rsi_bb_upper": upper_band, "rsi_bb_lower": lower_band}
                )
            else:
                return pd.DataFrame({"rsi": rsi})

            return pd.DataFrame({"rsi": rsi, "rsi_smoothed": smoothed})

        except Exception as e:
            logger.error(f"{self.name}: Smoothing calculation failed: {e}")
            return pd.DataFrame({"rsi": rsi})
