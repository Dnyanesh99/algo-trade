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
        """
        Validate data with comprehensive logging of the validation process.
        """
        context = ValidationContext(df, self.config, **kwargs)
        try:
            if context.df.empty:
                context.metrics.issues.append("No data to validate")
                return False, context.df, self._generate_report(context)

            logger.info(f"Starting validation for {context.symbol} ({context.instrument_type})")
            logger.info(f"Initial data shape: {context.df.shape}")
            logger.info(f"Required columns: {self.config.validation.required_columns}")

            # Log data sample before validation
            logger.debug(f"Data sample before validation:\n{context.df.head(2)}")

            # Run validation pipeline with detailed logging
            self._run_validation_pipeline(context)

            # Calculate and log quality score
            is_valid = self._is_data_valid(context)
            quality_score = self._calculate_quality_score(context)

            logger.info(
                f"Validation complete for {context.symbol}:\n"
                f"- Final data shape: {context.df.shape}\n"
                f"- Quality Score: {quality_score:.2f}\n"
                f"- Validation Result: {'✅ PASS' if is_valid else '❌ FAIL'}\n"
                f"- Issues Found: {len(context.metrics.issues)}"
            )

            if context.metrics.issues:
                logger.debug("Validation issues:\n" + "\n".join(f"- {issue}" for issue in context.metrics.issues))

            self._log_validation_results(context, is_valid)
            return is_valid, context.df, self._generate_report(context)

        except Exception as e:
            logger.error(f"🚨 Validation failed for {context.symbol} with error: {str(e)}", exc_info=True)
            raise RuntimeError(f"🚨 CRITICAL VALIDATION FAILURE: Validation failed for {context.symbol}: {e}") from e

    def _run_validation_pipeline(self, context: ValidationContext) -> None:
        """
        Run the validation pipeline with detailed logging of each step.
        """
        logger.info(f"Starting validation pipeline for {context.symbol}")

        # Step 1: Prepare and clean data
        logger.info("Step 1: Data preparation and cleaning")
        rows_before = len(context.df)
        self._prepare_and_clean_data(context)
        rows_after = len(context.df)
        logger.info(
            f"Data cleaning results:\n"
            f"- Initial rows: {rows_before}\n"
            f"- Rows after cleaning: {rows_after}\n"
            f"- Rows removed: {rows_before - rows_after}\n"
            f"- Retention rate: {(rows_after / rows_before * 100):.2f}%"
        )

        # Step 2: OHLC violations check
        logger.info("Step 2: Checking OHLC violations")
        violations_before = context.metrics.ohlc_violations
        self._correct_ohlc_violations(context)
        logger.info(
            f"OHLC validation results:\n"
            f"- Violations found: {context.metrics.ohlc_violations}\n"
            f"- Violations corrected: {context.metrics.ohlc_violations - violations_before}"
        )

        # Step 3: Remove duplicates
        logger.info("Step 3: Checking for duplicates")
        self._remove_duplicates(context)
        logger.info(f"Duplicates found and removed: {context.metrics.duplicates}")

        # Step 4: Time gap detection
        logger.info("Step 4: Detecting time gaps")
        self._detect_time_gaps(context)
        logger.info(f"Time gaps detected: {context.metrics.time_gaps}")

        # Step 5: Outlier detection
        logger.info("Step 5: Outlier detection")
        self._handle_outliers(context)
        if context.metrics.outliers:
            logger.info(
                "Outliers detected:\n"
                + "\n".join(f"- {col}: {count} outliers" for col, count in context.metrics.outliers.items())
            )
        else:
            logger.info("No outliers detected")

        # Log final validation metrics
        logger.info(
            f"Validation pipeline complete for {context.symbol}:\n"
            f"Final Metrics:\n"
            f"- Rows processed: {context.metrics.initial_rows}\n"
            f"- Rows retained: {len(context.df)}\n"
            f"- Data retention: {(len(context.df) / context.metrics.initial_rows * 100):.2f}%\n"
            f"- OHLC violations: {context.metrics.ohlc_violations}\n"
            f"- Duplicates: {context.metrics.duplicates}\n"
            f"- Time gaps: {context.metrics.time_gaps}\n"
            f"- Total issues: {len(context.metrics.issues)}"
        )

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
        """
        Handle outliers only for required columns that are suitable for outlier detection.
        For indices, this typically means only OHLC columns.
        """
        required_cols = self.config.validation.required_columns

        # Only check outliers in required columns that are numeric
        numeric_cols = context.df.select_dtypes(include=["float64", "int64"]).columns
        outlier_cols = [col for col in required_cols if col in numeric_cols]

        if outlier_cols:
            # Log which columns we're checking for outliers
            logger.debug(f"Checking outliers for {context.symbol} in columns: {outlier_cols}")

            context.df, detected_outliers = self.outlier_detector.detect_and_handle(
                context.df, outlier_cols, self.config, context.metrics.issues
            )

            # Only store outliers for required columns
            context.metrics.outliers = {col: count for col, count in detected_outliers.items() if col in required_cols}

            if context.metrics.outliers:
                logger.debug(
                    f"Found outliers in {context.symbol}: "
                    f"{', '.join(f'{k}: {v}' for k, v in context.metrics.outliers.items())}"
                )
        else:
            context.metrics.outliers = {}
            context.metrics.issues.append(
                f"No suitable columns found for outlier detection among required columns: {required_cols}"
            )

    def _calculate_quality_score(self, context: ValidationContext) -> float:
        """
        Calculate quality score based only on required columns and their validations.

        The score is calculated as:
        1. Base score: Percentage of rows retained after cleaning
        2. Penalties applied only for required columns:
           - Outliers in required columns
           - Time gaps (always checked as it's timestamp-based)
           - OHLC violations (only for required price columns)
           - Duplicates (always checked as it's timestamp-based)
        """
        m = context.metrics
        if m.initial_rows == 0:
            return 0.0

        required_cols = self.config.validation.required_columns
        penalties = self.config.penalties

        # Base score - percentage of rows retained after cleaning
        base_score = len(context.df) / m.initial_rows

        # Calculate outlier penalty only for required columns
        required_outliers = {col: count for col, count in m.outliers.items() if col in required_cols}

        # Calculate individual penalty components
        if required_outliers and len(context.df) > 0:
            outlier_penalty = 1 - ((sum(required_outliers.values()) / len(context.df)) * penalties.outlier_penalty)
        else:
            outlier_penalty = 1.0

        gap_penalty = 1 - m.time_gaps / len(context.df) * penalties.gap_penalty if len(context.df) > 0 else 1.0

        # OHLC violations penalty - only for required price columns
        required_price_cols = [col for col in PRICE_COLUMNS if col in required_cols]
        if required_price_cols:
            ohlc_penalty = 1 - ((m.ohlc_violations / m.initial_rows) * penalties.ohlc_violation_penalty)
        else:
            ohlc_penalty = 1.0

        # Duplicate penalty (always applied as it's timestamp-based)
        duplicate_penalty = 1 - ((m.duplicates / m.initial_rows) * penalties.duplicate_penalty)

        # Calculate final penalty factor
        penalty_factor = outlier_penalty * gap_penalty * ohlc_penalty * duplicate_penalty

        quality_score = max(0.0, base_score * penalty_factor * 100)

        # Log detailed score components for debugging
        logger.debug(
            f"Quality score components for {context.symbol}:\n"
            f"Base score: {base_score:.3f}\n"
            f"Outlier penalty (required cols only): {outlier_penalty:.3f}\n"
            f"Gap penalty: {gap_penalty:.3f}\n"
            f"OHLC penalty (required cols only): {ohlc_penalty:.3f}\n"
            f"Duplicate penalty: {duplicate_penalty:.3f}\n"
            f"Final score: {quality_score:.2f}"
        )

        return quality_score

    def _is_data_valid(self, context: ValidationContext) -> bool:
        return (
            len(context.df) >= self.config.validation.min_valid_rows
            and self._calculate_quality_score(context) >= self.config.validation.quality_score_threshold
        )

    def _generate_report(self, context: ValidationContext) -> DataQualityReport:
        """
        Generate a detailed data quality report with comprehensive logging.
        """
        m = context.metrics
        quality_score = self._calculate_quality_score(context)

        report = DataQualityReport(
            m.initial_rows,
            len(context.df),
            m.nan_counts,
            m.ohlc_violations,
            m.duplicates,
            m.time_gaps,
            m.outliers,
            quality_score,
            m.issues,
        )

        # Log detailed quality report
        logger.info(
            f"\n{'=' * 50}\n"
            f"DATA QUALITY REPORT: {context.symbol}\n"
            f"{'=' * 50}\n"
            f"Instrument Type: {context.instrument_type}\n"
            f"Quality Score: {quality_score:.2f}\n"
            f"\nData Volume Metrics:\n"
            f"- Initial rows: {m.initial_rows}\n"
            f"- Final rows: {len(context.df)}\n"
            f"- Data retention: {(len(context.df) / m.initial_rows * 100 if m.initial_rows > 0 else 0):.2f}%\n"
            f"\nData Quality Metrics:\n"
            f"- OHLC violations: {m.ohlc_violations}\n"
            f"- Duplicate timestamps: {m.duplicates}\n"
            f"- Time gaps detected: {m.time_gaps}\n"
            f"- Total outliers: {sum(m.outliers.values())}\n"
            f"\nMissing Data (NaN) by Column:\n"
            + "\n".join(f"- {col}: {count}" for col, count in m.nan_counts.items())
            + "\n\nOutliers by Column:\n"
            + "\n".join(f"- {col}: {count}" for col, count in m.outliers.items())
            + "\n\nValidation Issues:\n"
            + "\n".join(f"- {issue}" for issue in m.issues)
            + f"\n{'=' * 50}"
        )

        return report

    def _log_validation_results(self, context: ValidationContext, is_valid: bool) -> None:
        if not is_valid:
            logger.error(
                f"🚨 QUALITY CHECK FAILED: Data quality for {context.symbol} below threshold ({len(context.metrics.issues)} total issues): {context.metrics.issues[:3]}"
            )
        else:
            logger.info(f"✅ Data quality check passed for {context.symbol} with {len(context.df)} valid rows")
