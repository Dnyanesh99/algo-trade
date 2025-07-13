"""
Production-grade OHLCV candle validation with comprehensive checks.
Validates OHLC relationships, detects gaps, and ensures data quality.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from src.database.models import OHLCVData
from src.utils.config_loader import ConfigLoader
from src.utils.data_quality import DataQualityReport, DataValidator
from src.utils.logger import LOGGER as logger

config_loader = ConfigLoader()

# Load configuration
config = ConfigLoader().get_config()


class CandleData(BaseModel):
    """Pydantic model for strict candle data validation."""

    timestamp: datetime = Field(..., description="Candle timestamp")
    open: float = Field(..., gt=0, description="Open price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Close price")
    volume: int = Field(..., ge=0, description="Volume")
    oi: Optional[int] = Field(None, ge=0, description="Open interest")

    @field_validator("high")
    def validate_high(cls, v: float, info: ValidationInfo) -> float:
        if "low" in info.data and v < info.data["low"]:
            raise ValueError(f"High price {v} cannot be less than low price {info.data['low']}")
        if all(k in info.data for k in ["open", "close"]) and v < max(info.data["open"], info.data["close"]):
            raise ValueError(f"High price {v} must be >= max(open={info.data['open']}), close={info.data['close']})")
        return v

    @field_validator("low")
    def validate_low(cls, v: float, info: ValidationInfo) -> float:
        if all(k in info.data for k in ["open", "close"]) and v > min(info.data["open"], info.data["close"]):
            raise ValueError(f"Low price {v} must be <= min(open={info.data['open']}), close={info.data['close']})")
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
    """

    def __init__(self) -> None:
        if config.data_quality is None:
            raise ValueError("Data quality configuration is required")
        self.validation_config = config.data_quality.validation
        self.outlier_config = config.data_quality.outlier_detection
        self.time_series_config = config.data_quality.time_series
        self.penalties_config = config.data_quality.penalties

        logger.info("CandleValidator initialized with production-grade validation rules")

    async def validate_candles(self, df: pd.DataFrame, instrument_id: int, timeframe: str = "1min") -> ValidationResult:
        """
        Comprehensive candle validation with quality assessment.
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

            df_work = df.copy()
            initial_rows = len(df_work)
            issues, warnings, validation_errors = [], [], []

            required_columns = self.validation_config.required_columns
            missing_columns = [col for col in required_columns if col not in df_work.columns]
            if missing_columns:
                error_msg = f"Missing required columns: {missing_columns}"
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
                    validation_errors=[error_msg],
                    warnings=[],
                )

            if not pd.api.types.is_datetime64_any_dtype(df_work["timestamp"]):
                df_work["timestamp"] = pd.to_datetime(df_work["timestamp"], errors="coerce")

            df_work = df_work.sort_values("timestamp").reset_index(drop=True)
            if not isinstance(df_work, pd.DataFrame):
                raise TypeError(f"Expected a pandas DataFrame, but got {type(df_work)}")

            nan_rows_before = int(df_work.isnull().any(axis=1).sum())
            if nan_rows_before > 0:
                df_work.dropna(subset=required_columns, inplace=True)
                rows_removed = nan_rows_before - int(df_work.isnull().any(axis=1).sum())
                issues.append(f"Removed {rows_removed} rows with NaN values in critical columns.")

            # Apply validation for price columns (must be > 0) and volume (>= 0 for indices, > 0 for stocks)
            price_columns = ["open", "high", "low", "close"]
            price_valid = (df_work[price_columns] > 0).all(axis=1)

            # For volume, allow >= 0 (to accommodate indices with 0 volume)
            volume_valid = df_work["volume"] >= 0

            df_work = df_work[price_valid & volume_valid]

            ohlc_violations = 0
            high_violations = (df_work["high"] < df_work[["open", "close"]].max(axis=1)).sum()
            if high_violations > 0:
                ohlc_violations += high_violations
                df_work["high"] = df_work[["open", "high", "close"]].max(axis=1)
                warnings.append(f"Corrected {high_violations} high price violations.")

            low_violations = (df_work["low"] > df_work[["open", "close"]].min(axis=1)).sum()
            if low_violations > 0:
                ohlc_violations += low_violations
                df_work["low"] = df_work[["open", "low", "close"]].min(axis=1)
                warnings.append(f"Corrected {low_violations} low price violations.")

            hl_violations = (df_work["high"] < df_work["low"]).sum()
            if hl_violations > 0:
                ohlc_violations += hl_violations
                mask = df_work["high"] < df_work["low"]
                df_work.loc[mask, ["high", "low"]] = df_work.loc[mask, ["low", "high"]].values
                warnings.append(f"Corrected {hl_violations} high < low violations by swapping.")

            if ohlc_violations > 0:
                issues.append(f"Detected and corrected {ohlc_violations} total OHLC violations.")

            duplicate_timestamps = int(df_work["timestamp"].duplicated().sum())
            if duplicate_timestamps > 0:
                df_work.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
                issues.append(f"Removed {duplicate_timestamps} duplicate timestamps, keeping the first entry.")

            try:
                timeframe_minutes = int("".join(filter(str.isdigit, timeframe)))
                time_gaps = await self._detect_time_gaps(df_work, timeframe_minutes)
            except (ValueError, TypeError):
                logger.error(f"Could not parse integer timeframe from string: '{timeframe}'")
                time_gaps = 0

            if time_gaps > 0:
                issues.append(f"Detected {time_gaps} significant time gaps in the data series.")

            outliers = await self._detect_outliers(df_work)
            total_outliers = sum(outliers.values())
            if total_outliers > 0:
                issues.append(f"Detected {total_outliers} statistical outliers across columns.")

            valid_rows = len(df_work)
            config = config_loader.get_config()
            if not config.data_quality:
                raise ValueError("Data quality configuration is required")
            quality_score = DataValidator.calculate_quality_score(
                initial_rows,
                valid_rows,
                ohlc_violations,
                duplicate_timestamps,
                time_gaps,
                total_outliers,
                config.data_quality,
            )

            is_valid = (
                valid_rows >= self.validation_config.min_valid_rows
                and quality_score >= self.validation_config.quality_score_threshold
            )
            if not is_valid:
                validation_errors.append(
                    f"Data quality score {quality_score:.2f}% is below threshold of {self.validation_config.quality_score_threshold}%."
                )

            quality_report = DataQualityReport(
                total_rows=initial_rows,
                valid_rows=valid_rows,
                nan_counts={},
                ohlc_violations=ohlc_violations,
                duplicate_timestamps=duplicate_timestamps,
                time_gaps=time_gaps,
                outliers=outliers,
                quality_score=quality_score,
                issues=issues,
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(
                f"Candle validation for {instrument_id} completed in {processing_time:.2f}ms. "
                f"Quality score: {quality_score:.2f}%"
            )

            return ValidationResult(
                is_valid=is_valid,
                cleaned_data=df_work,
                quality_report=quality_report,
                validation_errors=validation_errors,
                warnings=warnings,
            )

        except Exception as e:
            error_msg = f"Unexpected error during candle validation for {instrument_id}: {e}"
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

    async def _detect_time_gaps(self, df: pd.DataFrame, timeframe_minutes: int) -> int:
        if len(df) < 2:
            return 0
        expected_interval = timedelta(minutes=timeframe_minutes)
        time_diffs = df["timestamp"].diff().dropna()
        gap_threshold = expected_interval * self.time_series_config.gap_multiplier
        return int((time_diffs > gap_threshold).sum())

    async def _detect_outliers(self, df: pd.DataFrame) -> dict[str, int]:
        outliers = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns or df[col].empty:
                continue
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - self.outlier_config.iqr_multiplier * IQR
                upper_bound = Q3 + self.outlier_config.iqr_multiplier * IQR
                outliers[col] = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
            else:
                outliers[col] = 0
        return outliers

    def validate_single_candle(self, candle_data: dict[str, Any]) -> tuple[bool, list[str]]:
        try:
            OHLCVData(**candle_data)
            return True, []
        except ValueError as e:
            return False, [str(e)]
