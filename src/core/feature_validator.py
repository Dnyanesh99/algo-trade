import numpy as np
import pandas as pd

from src.utils.config_loader import config_loader
from src.utils.data_quality import DataQualityReport, DataValidator
from src.utils.logger import LOGGER as logger

config = config_loader.get_config()


class FeatureValidator:
    """Validates calculated features for correctness and quality."""

    def __init__(self) -> None:
        self.validation_config = config.data_quality.validation
        self.outlier_config = config.data_quality.outlier_detection
        logger.info("FeatureValidator initialized.")

    def validate_features(
        self, df: pd.DataFrame, instrument_id: int = 0
    ) -> tuple[bool, pd.DataFrame, DataQualityReport]:
        """
        Validates a DataFrame of features.

        Args:
            df (pd.DataFrame): DataFrame with features and a 'timestamp' column.
            instrument_id (int): Instrument ID for logging purposes.

        Returns:
            Tuple[bool, pd.DataFrame, DataQualityReport]: (is_valid, cleaned_df, data_quality_report).
        """
        df_clean = df.copy()
        issues = []
        initial_rows = len(df_clean)

        if initial_rows == 0:
            report = DataQualityReport(0, 0, {}, 0, 0, 0, {}, 0.0, ["No feature data to validate"])
            return False, df_clean, report

        # Ensure timestamp is the index for validation operations
        if "timestamp" in df_clean.columns:
            df_clean = df_clean.set_index("timestamp")

        # 1. Check for NaN and Inf values
        nan_counts = df_clean.isnull().sum().to_dict()
        # Check for infinite values - df_clean.isin([np.inf, -np.inf]).sum().to_dict()
        total_nan_inf_rows_before = df_clean.isnull().any(axis=1).sum()

        if total_nan_inf_rows_before > 0:
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
            rows_removed = total_nan_inf_rows_before - (df_clean.isnull().any(axis=1).sum())
            issues.append(f"Removed {rows_removed} rows with NaN/Inf values.")

        # 2. Outlier detection (using IQR for each feature column)
        outliers = {}
        total_outliers = 0
        for col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:  # Avoid division by zero or constant columns
                outlier_multiplier = self.outlier_config.iqr_multiplier
                lower_bound = Q1 - outlier_multiplier * IQR
                upper_bound = Q3 + outlier_multiplier * IQR
                col_outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                col_outliers_count = col_outliers_mask.sum()
                if col_outliers_count > 0:
                    outliers[col] = col_outliers_count
                    total_outliers += col_outliers_count
                    # Optional: Clip outliers instead of just reporting
                    # df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        if total_outliers > 0:
            issues.append(f"Detected a total of {total_outliers} outliers across all features.")

        valid_rows_after_cleaning = len(df_clean)

        # Reset index to return a consistent DataFrame format
        df_clean = df_clean.reset_index()

        quality_score = DataValidator.calculate_quality_score(
            initial_rows,
            valid_rows_after_cleaning,
            ohlc_violations=0,
            duplicate_timestamps=0,
            time_gaps=0,
            total_outliers=total_outliers,
        )

        report = DataQualityReport(
            total_rows=initial_rows,
            valid_rows=valid_rows_after_cleaning,
            nan_counts=nan_counts,
            ohlc_violations=0,
            duplicate_timestamps=0,
            time_gaps=0,
            outliers=outliers,
            quality_score=quality_score,
            issues=issues,
        )

        is_valid = (
            valid_rows_after_cleaning >= self.validation_config.min_valid_rows
            and quality_score >= self.validation_config.quality_score_threshold
        )

        if not is_valid:
            logger.warning(
                f"Feature data quality below threshold for instrument {instrument_id}: "
                f"{quality_score:.2f}%. Issues: {report.issues}"
            )

        return is_valid, df_clean, report
