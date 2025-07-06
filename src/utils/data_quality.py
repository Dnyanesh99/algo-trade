from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

config = config_loader.get_config()


@dataclass
class DataQualityReport:
    """
    Data quality assessment results.
    """

    total_rows: int
    valid_rows: int
    nan_counts: dict[str, int]
    ohlc_violations: int
    duplicate_timestamps: int
    time_gaps: int
    outliers: dict[str, int]
    quality_score: float
    issues: list[str]

    def to_dict(self) -> dict:
        return self.__dict__


class DataValidator:
    """Comprehensive data validation and cleaning"""

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame, symbol: str = "UNKNOWN") -> tuple[bool, pd.DataFrame, DataQualityReport]:
        """
        Validate and clean OHLCV data with detailed reporting
        """
        df_clean = df.copy()
        issues = []

        # Required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = set(required_columns) - set(df_clean.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        initial_rows = len(df_clean)

        # Track data quality metrics
        nan_counts = df_clean[required_columns].isnull().sum().to_dict()

        # Remove NaN values
        if sum(nan_counts.values()) > 0:
            df_clean = df_clean.dropna(subset=required_columns)
            issues.append(f"Removed {initial_rows - len(df_clean)} rows with NaN values due to NaNs in required columns.")

        # Check OHLC relationships and remove invalid rows
        initial_ohlc_violations = (
            (df_clean["high"] < df_clean["low"])
            | (df_clean["high"] < df_clean["open"])
            | (df_clean["high"] < df_clean["close"])
            | (df_clean["low"] > df_clean["open"])
            | (df_clean["low"] > df_clean["close"])
            | (df_clean["close"] <= 0)
            | (df_clean["volume"] < 0)
        )
        ohlc_violations_count = initial_ohlc_violations.sum()

        if ohlc_violations_count > 0:
            df_clean = df_clean[~initial_ohlc_violations]
            issues.append(f"Removed {ohlc_violations_count} rows due to OHLC violations.")

        # Handle duplicates
        duplicates = df_clean["timestamp"].duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates(subset=["timestamp"], keep="last")
            issues.append(f"Removed {duplicates} duplicate timestamps.")

        # Sort by timestamp
        df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)

        # Detect time gaps for fixed-interval data
        expected_interval = timedelta(minutes=config.time_synchronizer.candle_interval_minutes)
        time_diff = df_clean["timestamp"].diff()
        time_gaps_count = (time_diff > expected_interval * config.data_quality.time_series.gap_multiplier).sum()

        if time_gaps_count > 0:
            issues.append(f"Detected {time_gaps_count} significant time gaps.")

        # Detect outliers using IQR method
        outliers = {}
        for col in ["open", "high", "low", "close", "volume"]:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            outlier_multiplier = config.data_quality.outlier_detection.iqr_multiplier
            outliers[col] = (
                (df_clean[col] < q1 - outlier_multiplier * iqr) | (df_clean[col] > q3 + outlier_multiplier * iqr)
            ).sum()

        # Calculate quality score
        # Start with a base score based on valid rows vs initial rows
        quality_score = (len(df_clean) / initial_rows) * 100 if initial_rows > 0 else 0

        # Apply penalties for detected issues
        if sum(outliers.values()) > 0:
            quality_score *= (1 - config.data_quality.penalties.outlier_penalty)
            issues.append(f"Applied outlier penalty: {config.data_quality.penalties.outlier_penalty * 100:.1f}%")
        if time_gaps_count > 0:
            quality_score *= (1 - config.data_quality.penalties.gap_penalty)
            issues.append(f"Applied time gap penalty: {config.data_quality.penalties.gap_penalty * 100:.1f}%")
        if ohlc_violations_count > 0:
            quality_score *= (1 - config.data_quality.penalties.ohlc_violation_penalty)
            issues.append(f"Applied OHLC violation penalty: {config.data_quality.penalties.ohlc_violation_penalty * 100:.1f}%")
        if duplicates > 0:
            quality_score *= (1 - config.data_quality.penalties.duplicate_penalty)
            issues.append(f"Applied duplicate timestamp penalty: {config.data_quality.penalties.duplicate_penalty * 100:.1f}%")

        report = DataQualityReport(
            total_rows=initial_rows,
            valid_rows=len(df_clean),
            nan_counts=nan_counts,
            ohlc_violations=ohlc_violations_count,
            duplicate_timestamps=duplicates,
            time_gaps=time_gaps_count,
            outliers=outliers,
            quality_score=quality_score,
            issues=issues,
        )

        is_valid = (
            len(df_clean) >= config.data_quality.validation.min_valid_rows
            and quality_score >= config.data_quality.validation.quality_score_threshold
        )

        if not is_valid:
            logger.warning(f"Data quality below threshold for {symbol}: {quality_score:.2f}%")

        return is_valid, df_clean, report