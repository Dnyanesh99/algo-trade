from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

from src.utils.logger import LOGGER as logger


class IndicatorConfig(BaseModel):
    """Configuration model for indicator parameters."""

    name: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    timeframe_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    fallback_enabled: bool = True
    validation_enabled: bool = True


class IndicatorResult(BaseModel):
    """Model for indicator calculation results."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    success: bool
    data: Optional[Union[pd.Series, pd.DataFrame]] = None
    error_message: Optional[str] = None
    calculation_time_ms: float = 0.0
    validation_passed: bool = True
    method_used: str = "talib"  # "talib" or "pandas"


class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators.

    Provides:
    - Common interface for TA-Lib and pandas implementations
    - Enable/disable functionality
    - Parameter validation
    - Error handling and logging
    - Timeframe-specific parameter support
    """

    def __init__(self, config: IndicatorConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.fallback_enabled = config.fallback_enabled
        self.validation_enabled = config.validation_enabled

        # Import talib with fallback
        self.talib: Optional[Any] = None
        try:
            import talib

            self.talib = talib
        except ImportError:
            logger.warning(f"TA-Lib not available for {self.name}. Using pandas fallback.")

    @abstractmethod
    def calculate_talib(self, df: pd.DataFrame, params: dict[str, Any]) -> Union[pd.Series, pd.DataFrame]:
        """Calculate indicator using TA-Lib implementation."""

    @abstractmethod
    def calculate_pandas(self, df: pd.DataFrame, params: dict[str, Any]) -> Union[pd.Series, pd.DataFrame]:
        """Calculate indicator using pandas implementation."""

    def get_timeframe_params(self, timeframe_minutes: int) -> dict[str, Any]:
        """Get parameters for specific timeframe with fallback to defaults."""
        tf_key = f"{timeframe_minutes}m"
        default_params = self.config.params.copy()
        tf_params = self.config.timeframe_params.get(tf_key, {})

        # Merge timeframe-specific params with defaults
        final_params = default_params.copy()
        final_params.update(tf_params)

        return final_params

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Validate indicator parameters. Override in subclasses for specific validation."""
        return True

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate input data. Override in subclasses for specific validation."""
        if df.empty:
            return False

        # Get required columns from config or use default
        required_columns = self.config.params.get("required_columns", ["open", "high", "low", "close", "volume"])
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise RuntimeError(
                f"CRITICAL TRADING SYSTEM FAILURE: Indicator {self.name} cannot proceed - "
                f"missing required data columns: {missing_columns}. Trading signal integrity compromised."
            )

        return True

    def validate_result(self, result: Union[pd.Series, pd.DataFrame]) -> bool:
        """Validate calculation result. Override in subclasses for specific validation."""
        if result is None:
            return False

        if isinstance(result, pd.Series):
            return not result.empty
        if isinstance(result, pd.DataFrame):
            return not result.empty and len(result.columns) > 0

        return False

    def calculate(self, df: pd.DataFrame, timeframe_minutes: int, force_pandas: bool = False) -> IndicatorResult:
        """
        Main calculation method with comprehensive error handling.

        Args:
            df: OHLCV DataFrame
            timeframe_minutes: Timeframe for parameter selection
            force_pandas: Force use of pandas implementation

        Returns:
            IndicatorResult with calculation outcome
        """
        start_time = pd.Timestamp.now()

        # Early return if indicator is disabled
        if not self.enabled:
            return IndicatorResult(
                name=self.name,
                success=False,
                error_message=f"Indicator {self.name} is disabled",
                method_used="disabled",
            )

        # Validate input data
        if self.validation_enabled and not self.validate_data(df):
            return IndicatorResult(
                name=self.name,
                success=False,
                error_message=f"Input data validation failed for {self.name}",
                validation_passed=False,
            )

        # Get timeframe-specific parameters
        params = self.get_timeframe_params(timeframe_minutes)
        logger.debug(f"{self.name}: Calculating with params: {params}")

        # Validate parameters
        if self.validation_enabled and not self.validate_params(params):
            return IndicatorResult(
                name=self.name,
                success=False,
                error_message=f"Parameter validation failed for {self.name}",
                validation_passed=False,
            )

        # Attempt calculation
        result = None
        method_used = "talib"
        error_message = None

        # Try TA-Lib first (unless forced to use pandas)
        if not force_pandas and self.talib is not None:
            try:
                logger.debug(f"{self.name}: Attempting calculation with TA-Lib.")
                result = self.calculate_talib(df, params)
                method_used = "talib"
            except Exception as e:
                error_message = f"TA-Lib calculation failed for {self.name}: {str(e)}"
                logger.warning(error_message)
                result = None

        # Fall back to pandas if TA-Lib failed or was not available
        if result is None and self.fallback_enabled:
            if self.talib is None:
                logger.debug(f"{self.name}: TA-Lib not available, attempting pandas calculation.")
            else:
                logger.warning(f"{self.name}: TA-Lib failed, attempting fallback to pandas.")
            try:
                result = self.calculate_pandas(df, params)
                method_used = "pandas"
                if error_message:
                    logger.info(f"{self.name}: Fallback to pandas calculation successful.")
            except Exception as e:
                error_message = f"Pandas calculation failed for {self.name}: {str(e)}"
                raise RuntimeError(
                    f"CRITICAL TRADING SYSTEM FAILURE: Both TA-Lib and pandas calculations failed for {self.name}. "
                    f"Cannot generate reliable trading signals. Error: {str(e)}"
                ) from e

        # Validate result
        validation_passed = True
        if self.validation_enabled and result is not None:
            validation_passed = self.validate_result(result)
            if not validation_passed:
                raise RuntimeError(
                    f"CRITICAL TRADING SYSTEM FAILURE: Result validation failed for indicator {self.name}. "
                    f"Calculated values are invalid and could lead to erroneous trading signals. "
                    f"Trading system integrity compromised."
                )

        # Calculate processing time
        processing_time = (pd.Timestamp.now() - start_time).total_seconds() * 1000

        return IndicatorResult(
            name=self.name,
            success=result is not None and validation_passed,
            data=result,
            error_message=error_message,
            calculation_time_ms=processing_time,
            validation_passed=validation_passed,
            method_used=method_used,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})"
