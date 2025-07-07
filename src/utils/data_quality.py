from dataclasses import dataclass

import pandas as pd

from src.utils.config_loader import DataQualityPenaltiesConfig, config_loader
from src.utils.logger import LOGGER as logger

config = config_loader.get_config()


@dataclass
class DataQualityReport:
    """Data quality assessment results."""

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
    """Comprehensive data validation and cleaning."""

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame, symbol: str = "UNKNOWN", instrument_type: str = "UNKNOWN") -> tuple[bool, pd.DataFrame, DataQualityReport]:
        """Validate and clean OHLCV data with detailed reporting."""
        if df.empty:
            report = DataQualityReport(0, 0, {}, 0, 0, 0, {}, 0.0, ["No data to validate"])
            return False, df, report

        df_clean = df.copy()
        issues = []
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = set(required_columns) - set(df_clean.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        initial_rows = len(df_clean)
        nan_counts = df_clean[required_columns].isnull().sum().to_dict()

        if sum(nan_counts.values()) > 0:
            rows_before_drop = len(df_clean)
            df_clean = df_clean.dropna(subset=required_columns)
            issues.append(f"Removed {rows_before_drop - len(df_clean)} rows with NaN values.")

        # Apply different validation rules based on instrument type
        is_index = instrument_type.upper() in ["INDEX", "INDICES", "EQ"] and any(word in symbol.upper() for word in ["NIFTY", "SENSEX", "INDEX"])
        
        if is_index:
            # For indices: volume can be 0, only check for negative volume and invalid prices
            invalid_condition = (df_clean["close"] <= 0) | (df_clean["volume"] < 0)
            validation_msg = "invalid price (<=0) or negative volume"
        else:
            # For stocks/other instruments: volume should be positive for valid trading
            invalid_condition = (df_clean["close"] <= 0) | (df_clean["volume"] <= 0)
            validation_msg = "invalid price (<=0) or non-positive volume"
            
        invalid_price_volume_count = invalid_condition.sum()
        if invalid_price_volume_count > 0:
            df_clean = df_clean[~invalid_condition]
            issues.append(f"Removed {invalid_price_volume_count} rows with {validation_msg}.")

        high_low_violation = df_clean["high"] < df_clean["low"]
        high_low_violation_count = high_low_violation.sum()
        if high_low_violation_count > 0:
            df_clean = df_clean[~high_low_violation]
            issues.append(f"Removed {high_low_violation_count} rows where high < low.")

        df_clean["high"] = df_clean[["open", "close", "high"]].max(axis=1)
        df_clean["low"] = df_clean[["open", "close", "low"]].min(axis=1)

        ohlc_violations_count = invalid_price_volume_count + high_low_violation_count
        if ohlc_violations_count > 0:
            issues.append(f"Total OHLC violations resulting in row removal: {ohlc_violations_count}.")

        duplicates = df_clean["timestamp"].duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates(subset=["timestamp"], keep="last")
            issues.append(f"Removed {duplicates} duplicate timestamps, keeping last entry.")

        df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)

        # Gap detection logic requires a timeframe, which is not available here.
        # This should be handled by a higher-level validator like CandleValidator.
        time_gaps_count = 0  # Assuming this is a basic OHLCV check, not a time-series continuity check.

        outliers = {}
        # Skip volume outlier detection for indices since volume is always 0
        outlier_columns = ["open", "high", "low", "close"]
        if not is_index:
            outlier_columns.append("volume")
            
        for col in outlier_columns:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_multiplier = config.data_quality.outlier_detection.iqr_multiplier if config.data_quality and config.data_quality.outlier_detection else 1.5 # Default value
                lower_bound = q1 - outlier_multiplier * iqr
                upper_bound = q3 + outlier_multiplier * iqr
                outliers[col] = int(((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum())
            else:
                outliers[col] = 0

        total_outliers = sum(outliers.values())
        if total_outliers > 0:
            issues.append(f"Detected a total of {total_outliers} outliers across OHLCV columns.")

        quality_score = DataValidator.calculate_quality_score(
            initial_rows, len(df_clean), ohlc_violations_count, duplicates, time_gaps_count, total_outliers
        )

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
            len(df_clean) >= (config.data_quality.validation.min_valid_rows if config.data_quality and config.data_quality.validation else 0)
            and quality_score >= (config.data_quality.validation.quality_score_threshold if config.data_quality and config.data_quality.validation else 0.0)
        )
        if not is_valid:
            logger.warning(f"Data quality below threshold for {symbol}: {quality_score:.2f}%. Issues: {issues}")

        return is_valid, df_clean, report

    @staticmethod
    def calculate_quality_score(
        initial_rows: int,
        valid_rows: int,
        ohlc_violations: int,
        duplicate_timestamps: int,
        time_gaps: int,
        total_outliers: int,
    ) -> float:
        """
        Calculate a robust overall data quality score using multiplicative penalties.
        This is the single source of truth for quality scoring across the system.
        """
        if initial_rows == 0:
            return 0.0

        try:
            if config.data_quality and config.data_quality.penalties:
                penalties = config.data_quality.penalties
            else:
                # Default penalties if config is not available
                penalties = DataQualityPenaltiesConfig(
                    outlier_penalty=0.1,
                    gap_penalty=0.1,
                    ohlc_violation_penalty=0.1,
                    duplicate_penalty=0.1
                )

            base_score_factor = valid_rows / initial_rows

            # Calculate penalty factors as a proportion of issues
            outlier_penalty = (total_outliers / valid_rows) * penalties.outlier_penalty if valid_rows > 0 else 0
            gap_penalty = (time_gaps / valid_rows) * penalties.gap_penalty if valid_rows > 0 else 0
            # Penalties for issues that cause row removal are based on initial rows
            ohlc_penalty = (ohlc_violations / initial_rows) * penalties.ohlc_violation_penalty
            duplicate_penalty = (duplicate_timestamps / initial_rows) * penalties.duplicate_penalty

            # Apply penalties multiplicatively for a more realistic score decay
            final_score_factor = (
                base_score_factor
                * (1 - min(outlier_penalty, 1.0))
                * (1 - min(gap_penalty, 1.0))
                * (1 - min(ohlc_penalty, 1.0))
                * (1 - min(duplicate_penalty, 1.0))
            )

            return max(0.0, final_score_factor * 100)

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
