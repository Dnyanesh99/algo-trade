from datetime import datetime

import numpy as np
import pandas as pd

from src.utils.config_loader import config_loader
from src.utils.data_quality import DataQualityReport
from src.utils.logger import LOGGER as logger

config = config_loader.get_config()


class FeatureValidator:
    """
    Validates calculated features for correctness and quality.
    Checks for NaN/Inf values, performs range validation, and assesses feature completeness.
    """

    def __init__(self) -> None:
        logger.info("FeatureValidator initialized.")

    def validate_features(
        self, df: pd.DataFrame, instrument_id: int = 0
    ) -> tuple[bool, pd.DataFrame, DataQualityReport]:
        """
        Validates a DataFrame of features.

        Args:
            df (pd.DataFrame): DataFrame with features. Expected to have a 'ts' index.
            instrument_id (int): Instrument ID for logging purposes.

        Returns:
            Tuple[bool, pd.DataFrame, DataQualityReport]: (is_valid, cleaned_df, data_quality_report).
        """
        df_clean = df.copy()
        issues = []

        initial_rows = len(df_clean)
        if initial_rows == 0:
            report = DataQualityReport(0, 0, {}, 0, 0, 0, {}, 0.0, ["No data to validate"])
            return False, df_clean, report

        # Ensure index is datetime
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            try:
                df_clean.index = pd.to_datetime(df_clean.index)
            except Exception:
                raise ValueError("DataFrame index must be convertible to DatetimeIndex.") from None

        # 1. Check for NaN and Inf values
        nan_counts = df_clean.isnull().sum().to_dict()
        inf_counts = df_clean.isin([np.inf, -np.inf]).sum().to_dict()

        total_nan_inf = sum(nan_counts.values()) + sum(inf_counts.values())
        if total_nan_inf > 0:
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
            issues.append(f"Removed {total_nan_inf} rows with NaN/Inf values. Remaining rows: {len(df_clean)}")

        # 2. Range validation (example: normalized features should be between -1 and 1)
        # This assumes certain features are normalized. Adjust as per actual features.
        range_violations = 0
        for col in df_clean.columns:
            # Example: if feature name suggests it should be normalized
            if "normalized" in col or "pct" in col or "oscillator" in col:
                col_violations = df_clean[(df_clean[col] < -1.0) | (df_clean[col] > 1.0)].shape[0]
                if col_violations > 0:
                    range_violations += col_violations
                    issues.append(f"Detected {col_violations} out-of-range values in normalized feature '{col}'.")
                    # Optionally, clip or remove these rows
                    df_clean[col] = df_clean[col].clip(-1.0, 1.0)

        # 3. Feature completeness (check if all expected features are present for each timestamp)
        # This requires knowing the expected features beforehand. For now, we check if any row is empty.
        incomplete_rows = df_clean.apply(lambda x: x.isnull().any(), axis=1).sum()
        if incomplete_rows > 0:
            issues.append(f"Detected {incomplete_rows} rows with incomplete feature sets after cleaning.")
            # Optionally, remove incomplete rows
            df_clean = df_clean.dropna()

        # 4. Outlier detection (using IQR for each feature column)
        outliers = {}
        for col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_multiplier = config.data_quality.outlier_detection.iqr_multiplier
            lower_bound = Q1 - outlier_multiplier * IQR
            upper_bound = Q3 + outlier_multiplier * IQR
            col_outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)].shape[0]
            if col_outliers > 0:
                outliers[col] = col_outliers
                issues.append(f"Detected {col_outliers} outliers in feature '{col}'.")
                # Optionally, cap outliers or remove rows
                # df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        valid_rows_after_cleaning = len(df_clean)
        quality_score = (valid_rows_after_cleaning / initial_rows) * 100 if initial_rows > 0 else 0
        # Apply penalties for issues
        quality_score *= 1 - min(
            total_nan_inf / initial_rows, config.data_quality.penalties.outlier_penalty
        )  # Reusing outlier penalty for NaN/Inf
        quality_score *= 1 - min(
            range_violations / initial_rows, config.data_quality.penalties.gap_penalty
        )  # Reusing gap penalty for range violations
        quality_score *= 1 - min(sum(outliers.values()) / initial_rows, config.data_quality.penalties.outlier_penalty)

        report = DataQualityReport(
            total_rows=initial_rows,
            valid_rows=valid_rows_after_cleaning,
            nan_counts=nan_counts,
            ohlc_violations=0,  # Not applicable for features directly
            duplicate_timestamps=0,  # Assumed handled by previous steps
            time_gaps=0,  # Assumed handled by previous steps
            outliers=outliers,
            quality_score=quality_score,
            issues=issues,
        )

        is_valid = (
            valid_rows_after_cleaning >= config.data_quality.validation.min_valid_rows
            and quality_score >= config.data_quality.validation.quality_score_threshold
        )

        if not is_valid:
            logger.warning(
                f"Feature data quality below threshold for instrument {instrument_id}: {quality_score:.2f}%. Issues: {report.issues}"
            )

        return is_valid, df_clean, report



