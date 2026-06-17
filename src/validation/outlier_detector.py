# src/validation/outlier_detector.py


import pandas as pd

from src.utils.config_loader import DataQualityConfig
from src.utils.logger import LOGGER as logger


class OutlierDetector:
    """
    A specialized class for detecting and handling outliers in DataFrames.
    This class adheres to the Single Responsibility Principle.
    """

    @staticmethod
    def detect_and_handle(
        df: pd.DataFrame,
        columns: list[str],
        config: DataQualityConfig,
        issues_log: list[str],
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """
        Detects and handles outliers using the IQR method based on the provided configuration.

        Args:
            df: DataFrame to process.
            columns: List of columns to check for outliers. Only these columns will be processed.
                     Commonly used: ["open", "high", "low", "close", "volume", "oi"]
                     But can be any subset based on required_columns configuration.
            config: The data quality configuration object.
            issues_log: A list to which issue descriptions will be appended.

        Returns:
            A tuple containing the processed DataFrame and a dictionary of outlier counts.
        """
        df_processed = df.copy()
        outliers: dict[str, int] = {}
        rows_to_remove = pd.Series(False, index=df_processed.index)
        strategy = config.outlier_detection.handling_strategy

        for col in columns:
            if col not in df_processed.columns or pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                outliers[col] = 0
                continue

            # Skip columns with all NaN or empty values
            if df_processed[col].isna().all() or len(df_processed[col].dropna()) == 0:
                outliers[col] = 0
                continue

            # Use only non-NaN values for quantile calculation
            col_data = df_processed[col].dropna()
            if (
                len(col_data) < config.outlier_detection.min_iqr_data_points
            ):  # Need at least 4 points for meaningful IQR
                outliers[col] = 0
                continue

            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1

            if iqr <= 0:
                logger.warning(
                    f"OutlierDetector: IQR is zero or negative for column '{col}'. Skipping outlier detection for this column."
                )
                outliers[col] = 0
                continue

            multiplier = config.outlier_detection.iqr_multiplier
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            # Apply outlier detection only to non-NaN values
            valid_mask = df_processed[col].notna()
            outlier_mask = valid_mask & ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound))
            outlier_count = int(outlier_mask.sum())
            outliers[col] = outlier_count

            if outlier_count > 0:
                if strategy == "clip":
                    df_processed.loc[outlier_mask, col] = df_processed[col].clip(lower_bound, upper_bound)
                    issues_log.append(
                        f"Clipped {outlier_count} outliers in '{col}' to range [{lower_bound:.2f}, {upper_bound:.2f}]."
                    )
                elif strategy == "remove":
                    rows_to_remove |= outlier_mask
                elif strategy == "flag":
                    flag_col = f"{col}_outlier_flag"
                    df_processed[flag_col] = outlier_mask
                    issues_log.append(f"Flagged {outlier_count} outliers in '{col}' with column '{flag_col}'.")

        if strategy == "remove" and rows_to_remove.any():
            rows_removed_count = int(rows_to_remove.sum())
            df_processed = df_processed[~rows_to_remove].reset_index(drop=True)
            issues_log.append(f"Removed {rows_removed_count} rows containing outliers.")
            # After removal, outlier counts for the removed rows are effectively zeroed out.
            for col in columns:
                if col in outliers:
                    outliers[col] = 0

        return df_processed, outliers
