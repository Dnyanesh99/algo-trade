"""
Stochastic %K and %D Oscillator Implementation.
Enhanced version with comprehensive parameter validation and fallback mechanisms.
"""

from typing import Any, Union

import pandas as pd

from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.utils.logger import LOGGER as logger


class StochasticIndicator(BaseIndicator):
    """
    Stochastic %K and %D Oscillator Implementation.
    The pandas implementation only supports SMA, while TA-Lib supports multiple MA types.
    """

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)
        self.output_columns = ["slowk", "slowd"]

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Validate Stochastic-specific parameters."""
        # TradingView-style parameters (primary)
        length = params.get("length", 14)
        d_length = params.get("d_length", 3)

        # Legacy TA-Lib parameters (fallback)
        fastk_period = params.get("fastk_period", length)
        slowk_period = params.get("slowk_period", 1)
        slowd_period = params.get("slowd_period", d_length)
        slowk_matype = params.get("slowk_matype", 0)
        slowd_matype = params.get("slowd_matype", 0)

        if not all(
            isinstance(p, (int, float)) and p > 0 for p in [length, d_length, fastk_period, slowk_period, slowd_period]
        ):
            logger.error(f"{self.name}: All periods must be positive numbers")
            return False
        if not all(0 <= ma_type <= 8 for ma_type in [slowk_matype, slowd_matype]):
            logger.error(f"{self.name}: MA types must be between 0 and 8")
            return False

        # Validate trigger levels
        upper_trigger = params.get("upper_trigger", 80)
        lower_trigger = params.get("lower_trigger", 20)
        if not (0 <= lower_trigger < upper_trigger <= 100):
            logger.error(
                f"{self.name}: Invalid trigger levels - lower_trigger={lower_trigger}, upper_trigger={upper_trigger}"
            )
            return False

        return True

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate input data for Stochastic calculation."""
        if not super().validate_data(df):
            return False
        fastk_period = self.config.params.get("fastk_period", 14)
        min_required = fastk_period
        if len(df) < min_required:
            logger.warning(f"{self.name}: Insufficient data points: {len(df)} < {min_required}")
            return False
        invalid_hlc = (df["high"] < df["low"]) | (df["high"] < df["close"]) | (df["low"] > df["close"])
        if invalid_hlc.any():
            logger.warning(f"{self.name}: Found {invalid_hlc.sum()} invalid high/low/close relationships")
        return True

    def validate_result(self, result: Union[pd.Series, pd.DataFrame]) -> bool:
        """Validate Stochastic calculation result."""
        if not super().validate_result(result):
            return False
        if isinstance(result, pd.DataFrame):
            if not all(col in result.columns for col in self.output_columns):
                logger.error(f"{self.name}: Missing expected columns {self.output_columns}")
                return False
            for col in self.output_columns:
                if col in result.columns:
                    valid_range = result[col].dropna()
                    if len(valid_range) > 0 and ((valid_range < 0).any() or (valid_range > 100).any()):
                        logger.warning(f"{self.name}: {col} values outside valid range [0, 100]")
                        return False
        return True

    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate Stochastic using TA-Lib with parameter mapping."""
        if self.talib is None:
            raise RuntimeError("TA-Lib not available for STOCH calculation")

        # Map TradingView parameters to TA-Lib
        length = int(params.get("length", 14))
        d_length = int(params.get("d_length", 3))

        fastk_period = max(1, int(params.get("fastk_period", length)))
        slowk_period = max(1, int(params.get("slowk_period", 1)))  # Fast %K (no smoothing)
        slowd_period = max(1, int(params.get("slowd_period", d_length)))
        slowk_matype = max(0, min(8, int(params.get("slowk_matype", 0))))
        slowd_matype = max(0, min(8, int(params.get("slowd_matype", 0))))

        min_required = fastk_period + slowk_period + slowd_period - 2
        if len(df) < min_required:
            logger.warning(f"{self.name}: Insufficient data for calculation: {len(df)} < {min_required}")
            return pd.DataFrame({"slowk": [float("nan")] * len(df), "slowd": [float("nan")] * len(df)}, index=df.index)

        try:
            slowk, slowd = self.talib.STOCH(
                df["high"].values,
                df["low"].values,
                df["close"].values,
                fastk_period=fastk_period,
                slowk_period=slowk_period,
                slowd_period=slowd_period,
                slowk_matype=slowk_matype,
                slowd_matype=slowd_matype,
            )
            slowk = pd.Series(slowk, index=df.index).clip(0, 100)
            slowd = pd.Series(slowd, index=df.index).clip(0, 100)
            return pd.DataFrame({"slowk": slowk, "slowd": slowd}, index=df.index)
        except Exception as e:
            logger.error(f"{self.name}: TA-Lib calculation failed for {len(df)} data points with params {params}: {e}")
            raise

    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate Stochastic using pandas (SMA only)."""
        fastk_period = max(1, int(params.get("fastk_period", 14)))
        slowk_period = max(1, int(params.get("slowk_period", 3)))
        slowd_period = max(1, int(params.get("slowd_period", 3)))

        if params.get("slowk_matype", 0) != 0 or params.get("slowd_matype", 0) != 0:
            logger.warning(f"{self.name}: MA types other than SMA (0) are not supported in pandas fallback")

        min_required = fastk_period + slowk_period + slowd_period - 2
        if len(df) < min_required:
            logger.warning(f"{self.name}: Insufficient data for calculation: {len(df)} < {min_required}")
            return pd.DataFrame({"slowk": [float("nan")] * len(df), "slowd": [float("nan")] * len(df)}, index=df.index)

        try:
            lowest_low = df["low"].rolling(window=fastk_period).min()
            highest_high = df["high"].rolling(window=fastk_period).max()
            range_diff = highest_high - lowest_low
            fastk = 100 * (df["close"] - lowest_low) / range_diff.replace(0, float("nan"))
            slowk = fastk.rolling(window=slowk_period).mean()
            slowd = slowk.rolling(window=slowd_period).mean()
            return pd.DataFrame({"slowk": slowk.clip(0, 100), "slowd": slowd.clip(0, 100)}, index=df.index)
        except Exception as e:
            logger.error(f"{self.name}: Pandas calculation failed: {e}")
            raise
