# src/validation/consistency_validator.py

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import LOGGER as logger


class ConsistencyValidator:
    def __init__(self, rules: list[dict[str, Any]]):
        self.rules = rules

    def validate(self, df: pd.DataFrame) -> list[str]:
        issues = []
        for rule in self.rules:
            try:
                rule_type = rule.get("type")
                if rule_type == "volatility_comparison":
                    issues.extend(self._validate_volatility_comparison(df, rule))
                elif rule_type == "band_ordering":
                    issues.extend(self._validate_band_ordering(df, rule))
                elif rule_type == "calculation_check":
                    issues.extend(self._validate_calculation_check(df, rule))
            except Exception as e:
                logger.error(f"Consistency check '{rule.get('name', 'Unnamed')}' failed critically: {e}")
                issues.append(f"Consistency check '{rule.get('name', 'Unnamed')}' failed: {e}")
        return issues

    def _validate_volatility_comparison(self, df: pd.DataFrame, rule: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        feature1, feature2 = rule["features"]
        ratio_threshold = rule["params"].get("ratio_threshold")
        window = rule["params"].get("window")

        if ratio_threshold is None or window is None:
            logger.warning(f"Missing required parameters for volatility_comparison rule: {rule.get('name', 'Unnamed')}")
            return issues

        if feature1 in df.columns and feature2 in df.columns:
            vol1 = df[feature1].rolling(window=window).std()
            vol2 = df[feature2].rolling(window=window).std()
            inconsistent_mask = (vol2 > vol1 * ratio_threshold) & ~(vol1.isna() | vol2.isna())
            inconsistent_count = inconsistent_mask.sum()
            if inconsistent_count > 0:
                issues.append(
                    f"Found {inconsistent_count} periods where {feature2} is significantly more volatile than {feature1}."
                )
        return issues

    def _validate_band_ordering(self, df: pd.DataFrame, rule: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        upper, middle, lower = rule["features"]
        tolerance = rule["params"].get("tolerance")

        if tolerance is None:
            logger.warning(
                f"Missing required parameter 'tolerance' for band_ordering rule: {rule.get('name', 'Unnamed')}"
            )
            return issues

        if all(f in df.columns for f in [upper, middle, lower]):
            violations = ((df[upper] < df[middle] - tolerance) | (df[middle] < df[lower] - tolerance)).sum()
            if violations > 0:
                issues.append(f"Found {violations} ordering violations for bands: {upper}, {middle}, {lower}.")
        return issues

    def _validate_calculation_check(self, df: pd.DataFrame, rule: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        feature1, feature2, target_feature = rule["features"]
        formula = rule["params"].get("formula")
        tolerance = rule["params"].get("tolerance")

        if tolerance is None:
            logger.warning(
                f"Missing required parameter 'tolerance' for calculation_check rule: {rule.get('name', 'Unnamed')}"
            )
            return issues

        if all(f in df.columns for f in [feature1, feature2, target_feature]) and formula == "macd - macd_signal":
            # This is a simple example, a more robust implementation would parse the formula
            calculated_target = df[feature1] - df[feature2]
            diff = np.abs(df[target_feature] - calculated_target)
            violations = (diff > tolerance).sum()
            if violations > 0:
                issues.append(f"Found {violations} inconsistencies for calculated feature {target_feature}.")
        return issues
