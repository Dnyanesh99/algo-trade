# src/validation/feature_validator.py

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.config_loader import AppConfig, ConfigLoader
from src.utils.logger import LOGGER as logger
from src.validation.consistency_validator import ConsistencyValidator
from src.validation.interfaces import IValidator
from src.validation.models import DataQualityReport, FeatureValidationResult
from src.validation.outlier_detector import OutlierDetector
from src.validation.rules.parsers import CustomRuleParser, IndicatorConfigParser, ModelTrainingRangeParser
from src.validation.rules.registry import FeatureValidationRegistry


class FeatureValidator(IValidator):
    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or ConfigLoader().get_config()
        self.registry = self._initialize_registry()
        # Initialize consistency validator with proper configuration path
        consistency_rules: list[dict[str, Any]] = []
        if (
            self.config.data_quality
            and self.config.data_quality.feature_validation
            and self.config.data_quality.feature_validation.consistency_rules
        ):
            consistency_rules = self.config.data_quality.feature_validation.consistency_rules
        self.consistency_validator = ConsistencyValidator(consistency_rules)
        self.outlier_detector = OutlierDetector()
        logger.info("FeatureValidator initialized.")

    def _initialize_registry(self) -> FeatureValidationRegistry:
        """
        Initialize the feature validation registry with all available parsers.
        This includes automatic rule discovery from indicator configurations.
        """
        registry = FeatureValidationRegistry()

        # Initialize all parsers in priority order
        parsers = [
            ModelTrainingRangeParser(),  # Highest priority - explicit ranges
            IndicatorConfigParser(),  # High priority - automatic discovery
            CustomRuleParser(),  # Medium priority - custom rules
        ]

        total_rules = 0
        for parser in parsers:
            try:
                rules = parser.parse(self.config)
                for rule in rules:
                    registry.add_rule(rule)
                total_rules += len(rules)
                logger.debug(f"{parser.__class__.__name__} contributed {len(rules)} validation rules")
            except Exception as e:
                logger.error(f"CRITICAL PARSER FAILURE: Failed to parse rules from {parser.__class__.__name__}: {e}")
                raise RuntimeError(
                    f"CRITICAL PARSER FAILURE: Failed to parse rules from {parser.__class__.__name__}: {e} - Validation rule system broken"
                ) from e

        logger.info(f"FeatureValidationRegistry initialized with {total_rules} total rules")
        return registry

    def validate(self, df: pd.DataFrame, **kwargs: Any) -> tuple[bool, pd.DataFrame, DataQualityReport]:
        instrument_id = kwargs.get("instrument_id", 0)
        logger.info(f"Starting feature validation for instrument {instrument_id}")

        if df.empty:
            return False, df, DataQualityReport(0, 0, {}, 0, 0, 0, {}, 0.0, ["No feature data to validate"])

        df_clean = df.copy()
        issues: list[str] = []
        initial_rows = len(df_clean)
        handling_strategy = self.config.data_quality.outlier_detection.handling_strategy

        # 1. Range Validation
        range_validation_results = self._validate_feature_ranges(df_clean, handling_strategy)
        df_clean, range_issues = self._apply_range_validation_results(
            df_clean, range_validation_results, handling_strategy
        )
        issues.extend(range_issues)

        # 2. Consistency Validation
        consistency_issues = self.consistency_validator.validate(df_clean)
        issues.extend(consistency_issues)

        # 3. Outlier Detection for features without specific rules
        df_clean, outlier_issues, outlier_counts = self._apply_generic_outlier_detection(df_clean, handling_strategy)
        issues.extend(outlier_issues)

        # 4. Final Report Generation
        nan_counts = df_clean.isnull().sum().to_dict()
        total_nans = sum(nan_counts.values())
        if total_nans > 0:
            issues.append(f"Found {total_nans} NaN values across all features after validation")

        valid_rows = len(df_clean.dropna())
        total_violations = sum(result.violations_count for result in range_validation_results)
        total_outliers = sum(outlier_counts.values())

        quality_score = self._calculate_quality_score(initial_rows, valid_rows, total_violations, total_outliers)
        report = DataQualityReport(
            initial_rows, valid_rows, nan_counts, total_violations, 0, 0, outlier_counts, quality_score, issues
        )

        is_valid = (
            valid_rows >= self.config.data_quality.validation.min_valid_rows
            and quality_score >= self.config.data_quality.validation.quality_score_threshold
        )

        if not is_valid:
            error_msg = f"CRITICAL FEATURE QUALITY FAILURE: Feature data quality for instrument {instrument_id} below threshold: {quality_score:.2f}%. Issues: {report.issues}"
            logger.error(error_msg + " - Feature validation compromised, cannot proceed with trading")
            raise RuntimeError(error_msg + " - Feature validation compromised, cannot proceed with trading")

        return is_valid, df_clean, report

    def _validate_feature_ranges(self, df: pd.DataFrame, handling_strategy: str) -> list[FeatureValidationResult]:
        results = []
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        for feature_name in numeric_features:
            if feature_name == "timestamp":
                continue
            rule = self.registry.find_validation_rule(feature_name)
            if rule:
                violations_mask = pd.Series(False, index=df.index)
                if rule.min_value is not None:
                    violations_mask |= df[feature_name] < rule.min_value
                if rule.max_value is not None:
                    violations_mask |= df[feature_name] > rule.max_value
                violations_count = int(violations_mask.sum())
                results.append(
                    FeatureValidationResult(
                        feature_name, violations_count == 0, violations_count, handling_strategy, rule
                    )
                )
        return results

    def _apply_range_validation_results(
        self, df: pd.DataFrame, results: list[FeatureValidationResult], strategy: str
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Apply range validation results to DataFrame based on handling strategy.

        Args:
            df: DataFrame to process
            results: list of validation results
            strategy: How to handle violations ("clip", "remove", "flag")

        Returns:
            tuple of (processed_df, issues_list)
        """
        issues: list[str] = []
        df_processed = df.copy()
        rows_to_remove = pd.Series(False, index=df_processed.index)

        for result in results:
            if result.violations_count > 0 and result.validation_rule is not None:
                feature_name = result.feature_name
                rule = result.validation_rule

                # Create violation mask
                violation_mask = pd.Series(False, index=df_processed.index)

                # Check min value constraint
                if rule.min_value is not None:
                    violation_mask |= df_processed[feature_name] < rule.min_value

                # Check max value constraint
                if rule.max_value is not None:
                    violation_mask |= df_processed[feature_name] > rule.max_value

                # Apply handling strategy
                if strategy == "clip":
                    # Clip to valid range
                    if rule.min_value is not None:
                        df_processed.loc[df_processed[feature_name] < rule.min_value, feature_name] = rule.min_value
                    if rule.max_value is not None:
                        df_processed.loc[df_processed[feature_name] > rule.max_value, feature_name] = rule.max_value

                    issues.append(
                        f"Clipped {result.violations_count} values in '{feature_name}' to range "
                        f"[{rule.min_value}, {rule.max_value}]"
                    )

                elif strategy == "remove":
                    # Mark rows for removal
                    rows_to_remove |= violation_mask
                    issues.append(
                        f"Marked {result.violations_count} rows for removal due to '{feature_name}' "
                        f"values outside range [{rule.min_value}, {rule.max_value}]"
                    )

                elif strategy == "flag":
                    # Add flag column
                    flag_col = f"{feature_name}_range_violation_flag"
                    df_processed[flag_col] = violation_mask
                    issues.append(
                        f"Flagged {result.violations_count} violations in '{feature_name}' "
                        f"outside range [{rule.min_value}, {rule.max_value}] with column '{flag_col}'"
                    )

        # Apply row removal if using 'remove' strategy
        if strategy == "remove" and rows_to_remove.sum() > 0:
            rows_removed = int(rows_to_remove.sum())
            df_processed = df_processed[~rows_to_remove].reset_index(drop=True)
            issues.append(f"Removed {rows_removed} rows due to feature range violations")

        return df_processed, issues

    def _apply_generic_outlier_detection(
        self, df: pd.DataFrame, strategy: str
    ) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
        issues: list[str] = []
        all_rules = self.registry.get_all_rules()
        validated_features = {rule.feature_pattern.strip("^$") for rule in all_rules}
        features_to_check = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col not in validated_features and col != "timestamp"
        ]

        df_processed, outlier_counts = self.outlier_detector.detect_and_handle(
            df, features_to_check, self.config.data_quality, issues
        )
        return df_processed, issues, outlier_counts

    def _calculate_quality_score(
        self, initial_rows: int, valid_rows: int, total_violations: int, total_outliers: int
    ) -> float:
        if initial_rows == 0:
            return 0.0
        penalties = self.config.data_quality.penalties
        base_score = valid_rows / initial_rows
        violation_penalty = (
            total_violations / initial_rows
        ) * penalties.ohlc_violation_penalty  # Re-using ohlc_violation_penalty for general violations
        outlier_penalty = (total_outliers / valid_rows) * penalties.outlier_penalty if valid_rows > 0 else 0
        penalty_factor = (1 - violation_penalty) * (1 - outlier_penalty)
        return max(0.0, base_score * penalty_factor * 100)
