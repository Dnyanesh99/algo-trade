"""
Production-grade OHLCV candle validation with comprehensive checks.
Validates OHLC relationships, detects gaps, and ensures data quality.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field, validator

from src.utils.config_loader import config_loader
from src.utils.data_quality import DataQualityReport
from src.utils.logger import LOGGER as logger

# Load configuration
config = config_loader.get_config()


class CandleData(BaseModel):
    """Pydantic model for strict candle data validation."""

    timestamp: datetime = Field(..., description="Candle timestamp")
    open: float = Field(..., gt=0, description="Open price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Close price")
    volume: float = Field(..., ge=0, description="Volume")
    oi: Optional[float] = Field(None, ge=0, description="Open interest")

    @validator("high")
    def validate_high(cls, v, values) -> float:
        """Validate high >= max(open, close)."""
        if "open" in values and "close" in values and v < max(values["open"], values["close"]):
            raise ValueError(f"High {v} must be >= max(open={values['open']}, close={values['close']})")
        return v

    @validator("low")
    def validate_low(cls, v, values) -> float:
        """Validate low <= min(open, close)."""
        if "open" in values and "close" in values and v > min(values["open"], values["close"]):
            raise ValueError(f"Low {v} must be <= min(open={values['open']}, close={values['close']})")
        return v

    @validator("high")
    def validate_high_low_relationship(cls, v, values) -> float:
        """Validate high >= low."""
        if "low" in values and v < values["low"]:
            raise ValueError(f"High {v} must be >= low {values['low']}")
        return v


@dataclass
class ValidationResult:
    """Result of candle validation."""

    is_valid: bool
    cleaned_data: pd.DataFrame
    quality_report: DataQualityReport
    validation_errors: list[str]
    warnings: list[str]


class CandleValidator:
    """
    Production-grade OHLCV candle validator with comprehensive checks.

    Validates:
    - OHLC relationships (H >= max(O,C), L <= min(O,C))
    - Volume constraints (Volume >= 0)
    - Price positivity constraints
    - Gap detection and analysis
    - Outlier detection using statistical methods
    - Duplicate timestamp detection
    - Data completeness validation
    """

    def __init__(self) -> None:
        self.validation_config = config.data_quality.validation
        self.outlier_config = config.data_quality.outlier_detection
        self.time_series_config = config.data_quality.time_series
        self.penalties_config = config.data_quality.penalties

        logger.info("CandleValidator initialized with production-grade validation rules")

    async def validate_candles(self, df: pd.DataFrame, instrument_id: int, timeframe: str = "1min") -> ValidationResult:
        """
        Comprehensive candle validation with quality assessment.

        Args:
            df: DataFrame with OHLCV data
            instrument_id: Instrument identifier for logging
            timeframe: Timeframe for gap detection

        Returns:
            ValidationResult with validation outcome and quality report
        """
        start_time = datetime.now()

        try:
            logger.info(f"Starting candle validation for instrument {instrument_id}, timeframe {timeframe}")

            if df.empty:
                return ValidationResult(
                    is_valid=False,
                    cleaned_data=df,
                    quality_report=DataQualityReport(
                        total_rows=0,
                        valid_rows=0,
                        nan_counts={},
                        ohlc_violations=0,
                        duplicate_timestamps=0,
                        time_gaps=0,
                        outliers={},
                        quality_score=0.0,
                        issues=["Empty dataset provided"],
                    ),
                    validation_errors=["Empty dataset"],
                    warnings=[],
                )

            # Prepare working copy
            df_work = df.copy()
            initial_rows = len(df_work)
            issues = []
            warnings: list[str] = []
            validation_errors = []

            # Ensure required columns exist
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df_work.columns]
            if missing_columns:
                error_msg = f"Missing required columns: {missing_columns}"
                validation_errors.append(error_msg)
                logger.error(error_msg)
                return ValidationResult(
                    is_valid=False,
                    cleaned_data=df_work,
                    quality_report=DataQualityReport(
                        total_rows=initial_rows,
                        valid_rows=0,
                        nan_counts={},
                        ohlc_violations=0,
                        duplicate_timestamps=0,
                        time_gaps=0,
                        outliers={},
                        quality_score=0.0,
                        issues=[error_msg],
                    ),
                    validation_errors=validation_errors,
                    warnings=warnings,
                )

            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_work["timestamp"]):
                try:
                    df_work["timestamp"] = pd.to_datetime(df_work["timestamp"])
                except Exception as e:
                    error_msg = f"Failed to convert timestamp column to datetime: {e}"
                    validation_errors.append(error_msg)
                    logger.error(error_msg)

            # Sort by timestamp
            df_work = df_work.sort_values("timestamp").reset_index(drop=True)

            # 1. Check for NaN values
            nan_counts = df_work.isnull().sum().to_dict()
            total_nan = sum(nan_counts.values())
            if total_nan > 0:
                issues.append(f"Found {total_nan} NaN values: {nan_counts}")
                # Remove rows with NaN in critical columns
                df_work = df_work.dropna(subset=["open", "high", "low", "close", "volume"])
                logger.warning(f"Removed {initial_rows - len(df_work)} rows with NaN values")

            # 2. Check for non-positive prices
            price_columns = ["open", "high", "low", "close"]
            for col in price_columns:
                non_positive = (df_work[col] <= 0).sum()
                if non_positive > 0:
                    issues.append(f"Found {non_positive} non-positive values in {col}")
                    df_work = df_work[df_work[col] > 0]

            # 3. Check for negative volume
            negative_volume = (df_work["volume"] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Found {negative_volume} negative volume values")
                df_work = df_work[df_work["volume"] >= 0]

            # 4. OHLC relationship validation
            ohlc_violations = 0

            # High >= max(Open, Close)
            high_violations = (df_work["high"] < df_work[["open", "close"]].max(axis=1)).sum()
            if high_violations > 0:
                ohlc_violations += high_violations
                issues.append(f"Found {high_violations} high < max(open, close) violations")
                # Fix violations by setting high = max(open, close, high)
                df_work["high"] = df_work[["open", "high", "close"]].max(axis=1)
                warnings.append(f"Corrected {high_violations} high price violations")

            # Low <= min(Open, Close)
            low_violations = (df_work["low"] > df_work[["open", "close"]].min(axis=1)).sum()
            if low_violations > 0:
                ohlc_violations += low_violations
                issues.append(f"Found {low_violations} low > min(open, close) violations")
                # Fix violations by setting low = min(open, low, close)
                df_work["low"] = df_work[["open", "low", "close"]].min(axis=1)
                warnings.append(f"Corrected {low_violations} low price violations")

            # High >= Low
            hl_violations = (df_work["high"] < df_work["low"]).sum()
            if hl_violations > 0:
                ohlc_violations += hl_violations
                issues.append(f"Found {hl_violations} high < low violations")
                # Fix by swapping
                mask = df_work["high"] < df_work["low"]
                df_work.loc[mask, ["high", "low"]] = df_work.loc[mask, ["low", "high"]].values
                warnings.append(f"Corrected {hl_violations} high < low violations")

            # 5. Check for duplicate timestamps
            duplicate_timestamps = df_work["timestamp"].duplicated().sum()
            if duplicate_timestamps > 0:
                issues.append(f"Found {duplicate_timestamps} duplicate timestamps")
                df_work = df_work.drop_duplicates(subset=["timestamp"], keep="first")
                warnings.append(f"Removed {duplicate_timestamps} duplicate timestamps")

            # 6. Gap detection
            time_gaps = await self._detect_time_gaps(df_work, timeframe)
            if time_gaps > 0:
                issues.append(f"Found {time_gaps} time gaps in data")
                warnings.append(f"Detected {time_gaps} time gaps - may need backfill")

            # 7. Outlier detection
            outliers = await self._detect_outliers(df_work)
            total_outliers = sum(outliers.values())
            if total_outliers > 0:
                issues.append(f"Found {total_outliers} outliers: {outliers}")
                warnings.append("Detected outliers - review data quality")

            # 8. Calculate quality score
            valid_rows = len(df_work)
            quality_score = self._calculate_quality_score(
                initial_rows, valid_rows, ohlc_violations, duplicate_timestamps, time_gaps, total_outliers
            )

            # 9. Determine if data is valid
            is_valid = (
                valid_rows >= self.validation_config.min_valid_rows
                and quality_score >= self.validation_config.quality_score_threshold
            )

            if not is_valid:
                validation_errors.append(
                    f"Data quality below threshold: {quality_score:.2f}% "
                    f"(min: {self.validation_config.quality_score_threshold}%)"
                )

            # Create quality report
            quality_report = DataQualityReport(
                total_rows=initial_rows,
                valid_rows=valid_rows,
                nan_counts=nan_counts,
                ohlc_violations=ohlc_violations,
                duplicate_timestamps=duplicate_timestamps,
                time_gaps=time_gaps,
                outliers=outliers,
                quality_score=quality_score,
                issues=issues,
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(
                f"Candle validation completed for instrument {instrument_id} "
                f"in {processing_time:.2f}ms. Quality score: {quality_score:.2f}%"
            )

            return ValidationResult(
                is_valid=is_valid,
                cleaned_data=df_work,
                quality_report=quality_report,
                validation_errors=validation_errors,
                warnings=warnings,
            )

        except Exception as e:
            error_msg = f"Candle validation failed for instrument {instrument_id}: {e}"
            logger.error(error_msg, exc_info=True)
            return ValidationResult(
                is_valid=False,
                cleaned_data=df,
                quality_report=DataQualityReport(
                    total_rows=len(df),
                    valid_rows=0,
                    nan_counts={},
                    ohlc_violations=0,
                    duplicate_timestamps=0,
                    time_gaps=0,
                    outliers={},
                    quality_score=0.0,
                    issues=[error_msg],
                ),
                validation_errors=[error_msg],
                warnings=[],
            )

    async def _detect_time_gaps(self, df: pd.DataFrame, timeframe: str) -> int:
        """
        Detect time gaps in the data series.

        Args:
            df: DataFrame with timestamp column
            timeframe: Expected timeframe (1min, 5min, 15min, 60min)

        Returns:
            Number of gaps detected
        """
        try:
            if len(df) < 2:
                return 0

            # Parse timeframe to get expected interval
            timeframe_mapping = {"1min": 1, "5min": 5, "15min": 15, "60min": 60}

            expected_interval_minutes = timeframe_mapping.get(timeframe, 1)
            expected_interval = timedelta(minutes=expected_interval_minutes)

            # Calculate actual intervals
            time_diffs = df["timestamp"].diff().dropna()

            # Find gaps (intervals significantly larger than expected)
            gap_threshold = expected_interval * self.time_series_config.gap_multiplier
            gaps = time_diffs > gap_threshold

            return int(gaps.sum())

        except Exception as e:
            logger.error(f"Error in gap detection: {e}")
            return 0

    async def _detect_outliers(self, df: pd.DataFrame) -> dict[str, int]:
        """
        Detect outliers using IQR method.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with outlier counts per column
        """
        try:
            outliers = {}
            price_columns = ["open", "high", "low", "close"]

            for col in price_columns + ["volume"]:
                if col not in df.columns:
                    continue

                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                if IQR == 0:  # No variation
                    outliers[col] = 0
                    continue

                lower_bound = Q1 - self.outlier_config.iqr_multiplier * IQR
                upper_bound = Q3 + self.outlier_config.iqr_multiplier * IQR

                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

                outliers[col] = outlier_count

            return outliers

        except Exception as e:
            logger.error(f"Error in outlier detection: {e}")
            return {}

    def _calculate_quality_score(
        self,
        initial_rows: int,
        valid_rows: int,
        ohlc_violations: int,
        duplicate_timestamps: int,
        time_gaps: int,
        total_outliers: int,
    ) -> float:
        """
        Calculate overall data quality score.

        Args:
            initial_rows: Initial number of rows
            valid_rows: Number of valid rows after cleaning
            ohlc_violations: Number of OHLC violations
            duplicate_timestamps: Number of duplicate timestamps
            time_gaps: Number of time gaps
            total_outliers: Total number of outliers

        Returns:
            Quality score (0-100)
        """
        try:
            if initial_rows == 0:
                return 0.0

            # Base score from data retention
            base_score = (valid_rows / initial_rows) * 100

            # Apply penalties
            outlier_penalty = min(
                (total_outliers / initial_rows) * 100 * self.penalties_config.outlier_penalty,
                self.penalties_config.outlier_penalty * 100,
            )

            gap_penalty = min(
                (time_gaps / initial_rows) * 100 * self.penalties_config.gap_penalty,
                self.penalties_config.gap_penalty * 100,
            )

            # OHLC violations penalty (more severe)
            ohlc_penalty = min(
                (ohlc_violations / initial_rows) * 100 * self.penalties_config.ohlc_violation_penalty,
                self.penalties_config.ohlc_violation_penalty * 100,
            )

            # Duplicate penalty
            duplicate_penalty = min(
                (duplicate_timestamps / initial_rows) * 100 * self.penalties_config.duplicate_penalty,
                self.penalties_config.duplicate_penalty * 100,
            )

            final_score = base_score - outlier_penalty - gap_penalty - ohlc_penalty - duplicate_penalty

            return max(0.0, min(100.0, final_score))

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0

    def validate_single_candle(self, candle_data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate a single candle using Pydantic model.

        Args:
            candle_data: Dictionary with candle data

        Returns:
            Tuple of (is_valid, validation_errors)
        """
        try:
            CandleData(**candle_data)
            return True, []
        except ValueError as e:
            return False, [str(e)]
        except Exception as e:
            return False, [f"Unexpected validation error: {e}"]



