# src/validation/candle_validator.py

from __future__ import annotations

from datetime import timedelta
from enum import Enum, auto
from typing import Any, Final

import pandas as pd

from src.utils.config_loader import ConfigLoader, DataQualityConfig
from src.utils.logger import LOGGER as logger
from src.validation.interfaces import IValidator
from src.validation.models import DataQualityReport
from src.validation.outlier_detector import OutlierDetector

# Constants
TIMESTAMP_COLUMNS: Final[list[str]] = ["ts", "timestamp", "datetime", "time"]
PRICE_COLUMNS: Final[list[str]] = ["open", "high", "low", "close"]


# Custom Exceptions
class ValidationError(Exception):
    """Raised when validation fails with unrecoverable errors."""


class ConfigurationError(ValidationError):
    """Raised when configuration is invalid or missing."""


class InstrumentType(Enum):
    INDEX = auto()
    EQUITY = auto()
    FNO = auto()
    UNKNOWN = auto()

    @classmethod
    def from_string(cls, value: str, config: DataQualityConfig) -> InstrumentType:
        if not value:
            return cls.UNKNOWN
        value_upper = value.upper().strip()
        # Logic driven by config.yaml
        index_segments = config.validation.instrument_segments.index_segments
        if value_upper in index_segments:
            return cls.INDEX
        equity_segments = config.validation.instrument_segments.equity_segments
        if value_upper in equity_segments:
            return cls.EQUITY
        fno_segments = config.validation.instrument_segments.fno_segments
        if any(fno_seg in value_upper for fno_seg in fno_segments):
            return cls.FNO
        return cls.UNKNOWN

    @property
    def allows_zero_volume(self) -> bool:
        return self == InstrumentType.INDEX


class ValidationMetrics:
    def __init__(self) -> None:
        self.initial_rows: int = 0
        self.nan_counts: dict[str, int] = {}
        self.ohlc_violations: int = 0
        self.duplicates: int = 0
        self.time_gaps: int = 0
        self.outliers: dict[str, int] = {}
        self.issues: list[str] = []


class ValidationContext:
    def __init__(self, df: pd.DataFrame, config: DataQualityConfig, **kwargs: Any) -> None:
        self.df = df.copy()
        self.config = config
        self.symbol = kwargs.get("symbol", "UNKNOWN")
        self.instrument_type = InstrumentType.from_string(kwargs.get("instrument_type", "UNKNOWN"), config)
        self.timeframe = kwargs.get("timeframe")
        self.metrics = ValidationMetrics()
        self.metrics.initial_rows = len(df)

    @property
    def timeframe_minutes(self) -> int | None:
        if not self.timeframe:
            return None
        try:
            return int("".join(filter(str.isdigit, self.timeframe)))
        except (ValueError, TypeError):
            return None


class CandleValidator(IValidator):
    """
    Production-grade OHLCV candle validator with configurable column validation.

    Key Features:
    - Fully configurable validation based on required_columns setting
    - Segment-aware validation (INDEX allows zero volume, EQUITY/FNO requires positive volume)
    - Conditional OHLC relationship validation (only when all OHLC columns required)
    - Flexible outlier detection on any subset of [open, high, low, close, volume, oi]
    - Optional OI validation for futures instruments
    - Comprehensive data quality scoring and reporting

    Configuration Examples:
    - Index data: required_columns: ["ts", "open", "high", "low", "close"]
    - Stock data: required_columns: ["ts", "open", "high", "low", "close", "volume"]
    - Futures data: required_columns: ["ts", "open", "high", "low", "close", "volume", "oi"]
    """

    def __init__(
        self, config: DataQualityConfig | None = None, outlier_detector: OutlierDetector | None = None
    ) -> None:
        if config is None:
            app_config = ConfigLoader().get_config()
            self.config = app_config.data_quality
        else:
            self.config = config
        self.outlier_detector = outlier_detector or OutlierDetector()
        self._validate_config()
        logger.info("CandleValidator initialized with project-specific, config-driven logic.")

    def _validate_config(self) -> None:
        if not self.config.validation.required_columns:
            raise ConfigurationError("Required columns must be specified in config")
        if not (0 <= self.config.validation.quality_score_threshold <= 100):
            raise ConfigurationError("Quality score threshold must be between 0 and 100")

    def validate(self, df: pd.DataFrame, **kwargs: Any) -> tuple[bool, pd.DataFrame, DataQualityReport]:
        context = ValidationContext(df, self.config, **kwargs)
        try:
            if context.df.empty:
                context.metrics.issues.append("No data to validate")
                return False, context.df, self._generate_report(context)

            self._run_validation_pipeline(context)
            is_valid = self._is_data_valid(context)
            self._log_validation_results(context, is_valid)
            return is_valid, context.df, self._generate_report(context)
        except Exception as e:
            raise RuntimeError(
                f"🚨 CRITICAL VALIDATION FAILURE: Validation failed for {context.symbol}: {e} - Data validation system completely broken"
            ) from e

    def _run_validation_pipeline(self, context: ValidationContext) -> None:
        self._prepare_and_clean_data(context)
        self._correct_ohlc_violations(context)
        self._remove_duplicates(context)
        self._detect_time_gaps(context)
        self._handle_outliers(context)

    def _prepare_and_clean_data(self, context: ValidationContext) -> None:
        ts_col = next((c for c in TIMESTAMP_COLUMNS if c in context.df.columns), None)
        if not ts_col:
            raise ValidationError(f"Missing timestamp column. Expected one of: {TIMESTAMP_COLUMNS}")
        if ts_col != "ts":
            context.df.rename(columns={ts_col: "ts"}, inplace=True)
        context.df["ts"] = pd.to_datetime(context.df["ts"], errors="coerce")
        context.df.sort_values("ts", inplace=True, ignore_index=True)

        required_cols = self.config.validation.required_columns
        context.metrics.nan_counts = context.df[required_cols].isnull().sum().to_dict()
        rows_before = len(context.df)
        context.df.dropna(subset=required_cols, inplace=True)
        if (rows_before - len(context.df)) > 0:
            context.metrics.issues.append(f"Removed {rows_before - len(context.df)} rows with NaN values.")

        # Price validation - only for required price columns
        required_price_cols = [col for col in PRICE_COLUMNS if col in required_cols]
        if required_price_cols:
            context.df = context.df[(context.df[required_price_cols] > 0).all(axis=1)]

        # Volume validation - only if volume is required
        if "volume" in required_cols:
            if context.instrument_type.allows_zero_volume:
                context.df = context.df[context.df["volume"] >= 0]
            else:
                context.df = context.df[context.df["volume"] > 0]

        # OI validation - only if oi is required
        if "oi" in required_cols:
            context.df = context.df[context.df["oi"] >= 0]
        context.df.reset_index(drop=True, inplace=True)

    def _correct_ohlc_violations(self, context: ValidationContext) -> None:
        df = context.df
        violations = 0
        required_cols = self.config.validation.required_columns

        # Only perform OHLC validation if all OHLC columns are required
        if all(col in required_cols for col in PRICE_COLUMNS):
            high_violations_mask = df["high"] < df[["open", "close"]].max(axis=1)
            low_violations_mask = df["low"] > df[["open", "close"]].min(axis=1)

            if high_violations_mask.any():
                violations += int(high_violations_mask.sum())
                df.loc[high_violations_mask, "high"] = df.loc[high_violations_mask, ["open", "high", "close"]].max(
                    axis=1
                )
                context.metrics.issues.append(f"Corrected {high_violations_mask.sum()} high price violations.")

            if low_violations_mask.any():
                violations += int(low_violations_mask.sum())
                df.loc[low_violations_mask, "low"] = df.loc[low_violations_mask, ["open", "low", "close"]].min(axis=1)
                context.metrics.issues.append(f"Corrected {low_violations_mask.sum()} low price violations.")
        else:
            missing_ohlc = [col for col in PRICE_COLUMNS if col not in required_cols]
            if missing_ohlc:
                context.metrics.issues.append(f"Skipping OHLC validation - missing required columns: {missing_ohlc}")

        context.metrics.ohlc_violations = violations

    def _remove_duplicates(self, context: ValidationContext) -> None:
        duplicates = int(context.df["ts"].duplicated().sum())
        if duplicates > 0:
            context.df.drop_duplicates(subset=["ts"], keep="last", inplace=True, ignore_index=True)
            context.metrics.issues.append(f"Removed {duplicates} duplicate timestamps.")
        context.metrics.duplicates = duplicates

    def _detect_time_gaps(self, context: ValidationContext) -> None:
        if len(context.df) <= 1 or context.timeframe_minutes is None:
            return
        expected_interval = timedelta(minutes=context.timeframe_minutes)
        time_diffs = context.df["ts"].diff().dropna()
        gap_threshold = expected_interval * self.config.time_series.gap_multiplier
        gaps = int((time_diffs > gap_threshold).sum())
        if gaps > 0:
            context.metrics.issues.append(f"Detected {gaps} significant time gaps.")
        context.metrics.time_gaps = gaps

    def _handle_outliers(self, context: ValidationContext) -> None:
        # Only check for outliers in columns that are both required and suitable for outlier detection
        potential_outlier_cols = PRICE_COLUMNS + ["volume", "oi"]
        required_cols = self.config.validation.required_columns
        outlier_cols = [col for col in potential_outlier_cols if col in required_cols]

        if outlier_cols:
            context.df, context.metrics.outliers = self.outlier_detector.detect_and_handle(
                context.df, outlier_cols, self.config, context.metrics.issues
            )
        else:
            context.metrics.outliers = {}
            context.metrics.issues.append("No suitable columns found for outlier detection")

    def _calculate_quality_score(self, context: ValidationContext) -> float:
        m = context.metrics
        if m.initial_rows == 0:
            return 0.0
        penalties = self.config.penalties
        base_score = len(context.df) / m.initial_rows
        penalty_factor = (
            (1 - (sum(m.outliers.values()) / len(context.df)) * penalties.outlier_penalty if len(context.df) > 0 else 1)
            * (1 - (m.time_gaps / len(context.df)) * penalties.gap_penalty if len(context.df) > 0 else 1)
            * (1 - (m.ohlc_violations / m.initial_rows) * penalties.ohlc_violation_penalty)
            * (1 - (m.duplicates / m.initial_rows) * penalties.duplicate_penalty)
        )
        return max(0.0, base_score * penalty_factor * 100)

    def _is_data_valid(self, context: ValidationContext) -> bool:
        return (
            len(context.df) >= self.config.validation.min_valid_rows
            and self._calculate_quality_score(context) >= self.config.validation.quality_score_threshold
        )

    def _generate_report(self, context: ValidationContext) -> DataQualityReport:
        m = context.metrics
        return DataQualityReport(
            m.initial_rows,
            len(context.df),
            m.nan_counts,
            m.ohlc_violations,
            m.duplicates,
            m.time_gaps,
            m.outliers,
            self._calculate_quality_score(context),
            m.issues,
        )

    def _log_validation_results(self, context: ValidationContext, is_valid: bool) -> None:
        if not is_valid:
            raise RuntimeError(
                f"🚨 CRITICAL QUALITY FAILURE: Data quality for {context.symbol} below threshold ({len(context.metrics.issues)} total issues): {context.metrics.issues[:3]} - Data integrity compromised, cannot proceed with trading"
            )
