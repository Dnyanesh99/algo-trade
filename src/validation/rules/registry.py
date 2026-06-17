# src/validation/rules/registry.py

import re
from typing import Any, Optional

import pandas as pd

from src.utils.logger import LOGGER as logger
from src.validation.models import FeatureValidationResult, ValidationRule


class FeatureValidationRegistry:
    """
    Production-grade feature validation registry with comprehensive rule management.
    Provides efficient lookup and validation capabilities.
    """

    def __init__(self) -> None:
        self.rules: list[ValidationRule] = []
        self._pattern_cache: dict[str, ValidationRule] = {}
        self._exact_match_cache: dict[str, ValidationRule] = {}

    def add_rule(self, rule: ValidationRule) -> None:
        """
        Add a validation rule with intelligent caching for performance.

        Args:
            rule: ValidationRule to add
        """
        self.rules.append(rule)

        # Cache exact matches for O(1) lookup
        if self._is_exact_pattern(rule.feature_pattern):
            feature_name = rule.feature_pattern.strip("^$")
            self._exact_match_cache[feature_name] = rule

        # Clear pattern cache to force rebuild
        self._pattern_cache.clear()

        # Sort rules by priority for consistent application
        self.rules.sort(key=lambda x: x.priority, reverse=True)

    def get_all_rules(self) -> list[ValidationRule]:
        """Get all validation rules."""
        return self.rules.copy()

    def find_validation_rule(self, feature_name: str) -> Optional[ValidationRule]:
        """
        Find the best matching validation rule for a feature.
        Uses intelligent caching for high-performance lookups.

        Args:
            feature_name: Name of the feature to validate

        Returns:
            ValidationRule if found, None otherwise
        """
        # Check exact match cache first (O(1) lookup)
        if feature_name in self._exact_match_cache:
            return self._exact_match_cache[feature_name]

        # Check pattern cache
        if feature_name in self._pattern_cache:
            return self._pattern_cache[feature_name]

        # Search through regex patterns (sorted by priority)
        for rule in self.rules:
            try:
                if re.match(rule.feature_pattern, feature_name):
                    # Cache the result for future lookups
                    self._pattern_cache[feature_name] = rule
                    return rule
            except re.error as e:
                logger.error(
                    f"CRITICAL REGEX ERROR: Invalid regex pattern '{rule.feature_pattern}': {e}. Validation rule system corrupted."
                )
                raise RuntimeError(
                    f"CRITICAL REGEX ERROR: Invalid regex pattern '{rule.feature_pattern}': {e} - Validation rule system corrupted"
                ) from e

        # No matching rule found
        return None

    def validate_feature(
        self, feature_name: str, feature_series: pd.Series, handling_strategy: str
    ) -> FeatureValidationResult:
        """
        Validate a feature using the appropriate rule.

        Args:
            feature_name: Name of the feature
            feature_series: Pandas Series containing feature values
            handling_strategy: How to handle violations ("clip", "remove", "flag")

        Returns:
            FeatureValidationResult containing validation results
        """
        rule = self.find_validation_rule(feature_name)

        if rule is None:
            return FeatureValidationResult(
                feature_name=feature_name,
                is_valid=True,
                violations_count=0,
                applied_strategy="none",
                validation_rule=None,
                error_message=None,
            )

        try:
            # Calculate violations
            violations_mask = self._calculate_violations_mask(feature_series, rule)
            violations_count = int(violations_mask.sum())

            return FeatureValidationResult(
                feature_name=feature_name,
                is_valid=violations_count == 0,
                violations_count=violations_count,
                applied_strategy=handling_strategy,
                validation_rule=rule,
                error_message=None,
            )

        except Exception as e:
            error_msg = f"Validation failed for feature '{feature_name}': {str(e)}"
            logger.error(f"CRITICAL FEATURE VALIDATION ERROR: {error_msg} - Feature validation system broken")
            raise RuntimeError(
                f"CRITICAL FEATURE VALIDATION ERROR: {error_msg} - Feature validation system broken"
            ) from e

    def get_rules_summary(self) -> dict[str, Any]:
        """
        Get a summary of all validation rules for debugging and monitoring.

        Returns:
            dictionary containing rule statistics and details
        """
        rule_count_by_type: dict[str, int] = {}
        priority_distribution: dict[int, int] = {}

        for rule in self.rules:
            rule_count_by_type[rule.validator_type] = rule_count_by_type.get(rule.validator_type, 0) + 1
            priority_distribution[rule.priority] = priority_distribution.get(rule.priority, 0) + 1

        return {
            "total_rules": len(self.rules),
            "rule_types": rule_count_by_type,
            "priority_distribution": priority_distribution,
            "exact_match_cache_size": len(self._exact_match_cache),
            "pattern_cache_size": len(self._pattern_cache),
            "rules_by_priority": [
                {
                    "pattern": rule.feature_pattern,
                    "priority": rule.priority,
                    "type": rule.validator_type,
                    "description": rule.description,
                }
                for rule in self.rules
            ],
        }

    def clear_cache(self) -> None:
        """Clear all caches. Useful for testing or when rules change."""
        self._pattern_cache.clear()
        self._exact_match_cache.clear()

    def _is_exact_pattern(self, pattern: str) -> bool:
        """Check if a pattern is an exact match (not a regex)."""
        # Simple heuristic: if pattern starts with ^ and ends with $ and has no regex special chars
        if pattern.startswith("^") and pattern.endswith("$"):
            middle = pattern[1:-1]
            # Check if it's just a literal string (no regex metacharacters)
            return not any(char in middle for char in r".+*?[]{}()|\^$")
        return False

    def _calculate_violations_mask(self, feature_series: pd.Series, rule: ValidationRule) -> pd.Series:
        """Calculate which values violate the validation rule."""
        violations_mask = pd.Series(False, index=feature_series.index)

        # Check minimum value constraint
        if rule.min_value is not None:
            violations_mask |= feature_series < rule.min_value

        # Check maximum value constraint
        if rule.max_value is not None:
            violations_mask |= feature_series > rule.max_value

        return violations_mask
