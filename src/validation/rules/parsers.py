# src/validation/rules/parsers.py

import re
from typing import Any, Optional

from src.utils.config_loader import AppConfig
from src.utils.logger import LOGGER as logger
from src.validation.models import ValidationRule


class BaseRuleParser:
    def parse(self, config: AppConfig) -> list[ValidationRule]:
        raise NotImplementedError


class ModelTrainingRangeParser(BaseRuleParser):
    def parse(self, config: AppConfig) -> list[ValidationRule]:
        rules: list[ValidationRule] = []
        if not (config.model_training and config.model_training.feature_ranges):
            return rules
        for name, range_vals in config.model_training.feature_ranges.items():
            if isinstance(range_vals, list) and len(range_vals) == 2:
                min_val, max_val = range_vals
                rule = ValidationRule(
                    feature_pattern=f"^{re.escape(name)}$",
                    min_value=min_val if min_val != float("-inf") else None,
                    max_value=max_val if max_val != float("inf") else None,
                    priority=config.data_quality.feature_validation.rule_priorities.model_training_range_parser,
                    description=f"Explicit range for {name} from model_training config.",
                )
                rules.append(rule)
        return rules


class CustomRuleParser(BaseRuleParser):
    def parse(self, config: AppConfig) -> list[ValidationRule]:
        rules: list[ValidationRule] = []
        if not (
            config.data_quality
            and config.data_quality.feature_validation
            and config.data_quality.feature_validation.custom_rules
        ):
            return rules
        for rule_config in config.data_quality.feature_validation.custom_rules.range_rules:
            rule = ValidationRule(
                feature_pattern=rule_config.pattern,
                min_value=rule_config.min_value,
                max_value=rule_config.max_value,
                priority=config.data_quality.feature_validation.rule_priorities.custom_rule_parser,
                description=rule_config.description or "Custom validation rule.",
            )
            rules.append(rule)
        return rules


class IndicatorConfigParser(BaseRuleParser):
    """
    Production-grade parser for automatic validation rule discovery from indicator configurations.
    This enables zero-code-change validation for new indicators.
    """

    def parse(self, config: AppConfig) -> list[ValidationRule]:
        """
        Parse indicator configurations to automatically generate validation rules.

        This method scans all indicator configurations in trading.features.configurations
        and creates validation rules based on:
        1. validation_range parameter in indicator configs
        2. Known indicator types and their standard ranges
        3. Indicator-specific parameter constraints

        Args:
            config: Full application configuration

        Returns:
            list of ValidationRule objects for indicators
        """
        rules: list[ValidationRule] = []

        if not (config.trading and config.trading.features and config.trading.features.configurations):
            logger.error(
                "CRITICAL CONFIG FAILURE: No trading feature configurations found for indicator validation. Validation system cannot operate."
            )
            raise RuntimeError(
                "CRITICAL CONFIG FAILURE: No trading feature configurations found for indicator validation - Validation system cannot operate"
            )

        # Process each indicator configuration
        for indicator_config in config.trading.features.configurations:
            indicator_rules = self._process_indicator_config(indicator_config.model_dump(), config)
            rules.extend(indicator_rules)

        logger.info(f"IndicatorConfigParser generated {len(rules)} validation rules from indicator configurations")
        return rules

    def _process_indicator_config(self, indicator_config: dict[str, Any], config: AppConfig) -> list[ValidationRule]:
        """Process a single indicator configuration to generate validation rules."""
        rules = []

        indicator_name = indicator_config.get("name", "unknown")
        indicator_function = indicator_config.get("function", "")
        params = indicator_config.get("params", {})

        # 1. Check for explicit validation_range in any timeframe
        for timeframe, timeframe_params in params.items():
            if isinstance(timeframe_params, dict) and "validation_range" in timeframe_params:
                validation_range = timeframe_params["validation_range"]
                rule = self._create_validation_rule_from_range(
                    indicator_name, validation_range, f"Explicit range for {indicator_name} ({timeframe})", config
                )
                if rule:
                    rules.append(rule)

        # 2. Generate rules based on known indicator types
        standard_rules = self._generate_standard_indicator_rules(indicator_name, indicator_function, config)
        rules.extend(standard_rules)

        # 3. Generate rules based on indicator parameters
        param_rules = self._generate_parameter_based_rules(indicator_name, params, config)
        rules.extend(param_rules)

        return rules

    def _create_validation_rule_from_range(
        self, indicator_name: str, validation_range: list[float], description: str, config: AppConfig
    ) -> Optional[ValidationRule]:
        """Create a validation rule from a range specification."""
        if not isinstance(validation_range, list) or len(validation_range) != 2:
            logger.error(
                f"CRITICAL VALIDATION CONFIG ERROR: Invalid validation_range format for {indicator_name}: {validation_range}. Validation rules corrupted."
            )
            raise RuntimeError(
                f"CRITICAL VALIDATION CONFIG ERROR: Invalid validation_range format for {indicator_name}: {validation_range} - Validation rules corrupted"
            )

        min_val, max_val = validation_range

        # Create rule for the main pattern (most specific)
        main_pattern = f"^{re.escape(indicator_name)}"

        return ValidationRule(
            feature_pattern=main_pattern,
            min_value=min_val if min_val != float("-inf") else None,
            max_value=max_val if max_val != float("inf") else None,
            priority=config.data_quality.feature_validation.rule_priorities.indicator_config_parser,
            description=description,
        )

    def _generate_standard_indicator_rules(
        self, indicator_name: str, function: str, config: AppConfig
    ) -> list[ValidationRule]:
        """Generate standard validation rules based on well-known indicator types."""
        rules = []

        # Standard indicator ranges - based on technical analysis knowledge
        standard_ranges = config.data_quality.feature_validation.standard_indicator_ranges or {}

        # Check if this indicator matches any standard type
        for standard_type, (min_val, max_val) in standard_ranges.items():
            if function.upper() == standard_type:
                patterns = self._generate_indicator_patterns(indicator_name)

                for pattern in patterns:
                    rule = ValidationRule(
                        feature_pattern=pattern,
                        min_value=min_val if min_val != float("-inf") else None,
                        max_value=max_val if max_val != float("inf") else None,
                        priority=config.data_quality.feature_validation.rule_priorities.indicator_config_parser,
                        description=f"Standard range for {standard_type} indicator",
                    )
                    rules.append(rule)
                break

        return rules

    def _generate_parameter_based_rules(
        self, indicator_name: str, params: dict[str, Any], config: AppConfig
    ) -> list[ValidationRule]:
        """Generate rules based on indicator parameters."""
        rules: list[ValidationRule] = []

        # For Bollinger Bands, we can set reasonable bounds based on multiplier
        if indicator_name.lower() == "bbands":
            default_params = params.get("default", {})
            mult = default_params.get("mult", 2.0)

            # Bollinger Bands position should be roughly bounded
            rules.append(
                ValidationRule(
                    feature_pattern=f"^{re.escape(indicator_name)}_position$",
                    min_value=-2.0 * mult,
                    max_value=2.0 * mult,
                    priority=config.data_quality.feature_validation.rule_priorities.indicator_config_parser,
                    description=f"Bollinger Bands position with multiplier {mult}",
                )
            )

        # For RSI with smoothing, adjust bounds
        if indicator_name.lower() == "rsi":
            default_params = params.get("default", {})
            smoothing = default_params.get("smoothing", {})

            if smoothing.get("type") == "SMA + Bollinger Bands":
                # RSI with Bollinger Bands might have extended range
                rules.append(
                    ValidationRule(
                        feature_pattern=f"^{re.escape(indicator_name)}_.*$",
                        min_value=-50,  # Extended range for smoothed RSI
                        max_value=150,
                        priority=config.data_quality.feature_validation.rule_priorities.indicator_config_parser,
                        description="Extended range for smoothed RSI with Bollinger Bands",
                    )
                )

        return rules

    def _generate_indicator_patterns(self, indicator_name: str) -> list[str]:
        """Generate regex patterns for all possible output names from an indicator."""
        patterns: list[str] = []

        # Main indicator pattern
        patterns.append(f"^{re.escape(indicator_name)}$")

        # Common variations
        patterns.append(f"^{re.escape(indicator_name)}_\\d+$")  # With period (e.g., rsi_14)
        patterns.append(f"^{re.escape(indicator_name)}_.*$")  # Any suffix

        # Handle specific multi-output indicators
        if indicator_name.lower() == "stoch":
            patterns.extend(["^stoch_k$", "^stoch_d$", "^slowk$", "^slowd$"])
        elif indicator_name.lower() == "macd":
            patterns.extend(["^macd$", "^macd_signal$", "^macd_hist$", "^macdsignal$", "^macdhist$"])
        elif indicator_name.lower() == "bbands":
            patterns.extend(["^bb_upper$", "^bb_middle$", "^bb_lower$", "^upperband$", "^middleband$", "^lowerband$"])
        elif indicator_name.lower() == "aroon":
            patterns.extend(["^aroon_up$", "^aroon_down$", "^aroonup$", "^aroondown$"])

        return patterns
