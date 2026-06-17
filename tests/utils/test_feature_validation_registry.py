"""
Test suite for FeatureValidationRegistry - Production-grade feature validation system.
"""

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.utils.feature_validation_registry import (
    FeatureValidationRegistry,
    FeatureValidationResult,
    ValidationRule,
)


class TestFeatureValidationRegistry:
    """Test suite for FeatureValidationRegistry."""

    @pytest.fixture
    def mock_data_quality_config(self):
        """Create a mock data quality configuration."""
        config = MagicMock()
        
        # Mock model_training.feature_ranges
        config.model_training.feature_ranges = {
            "rsi_14": [0, 100],
            "bb_position": [-3, 3],
            "macd_signal": [-2, 2],
            "stoch_k": [0, 100],
            "stoch_d": [0, 100],
        }
        
        # Mock trading.features.configurations
        config.trading.features.configurations = [
            {"name": "rsi", "function": "RSI"},
            {"name": "stoch", "function": "STOCH"},
            {"name": "willr", "function": "WILLR"},
            {"name": "macd", "function": "MACD"},
            {"name": "atr", "function": "ATR"},
        ]
        
        return config

    def test_initialization(self, mock_data_quality_config):
        """Test registry initialization."""
        registry = FeatureValidationRegistry(mock_data_quality_config)
        
        assert len(registry.validation_rules) > 0
        assert all(isinstance(rule, ValidationRule) for rule in registry.validation_rules)
        
        # Check that rules are sorted by priority (descending)
        priorities = [rule.priority for rule in registry.validation_rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_config_ranges_loading(self, mock_data_quality_config):
        """Test loading validation rules from config feature ranges."""
        registry = FeatureValidationRegistry(mock_data_quality_config)
        
        # Find rules loaded from config
        config_rules = [
            rule for rule in registry.validation_rules 
            if rule.priority == 10  # Config rules have priority 10
        ]
        
        assert len(config_rules) == 5  # 5 feature ranges in mock config
        
        # Check specific rule
        rsi_rule = next((rule for rule in config_rules if "rsi_14" in rule.feature_pattern), None)
        assert rsi_rule is not None
        assert rsi_rule.min_value == 0
        assert rsi_rule.max_value == 100

    def test_indicator_discovery(self, mock_data_quality_config):
        """Test auto-discovery of validation rules from indicators."""
        registry = FeatureValidationRegistry(mock_data_quality_config)
        
        # Find rules from indicator discovery
        indicator_rules = [
            rule for rule in registry.validation_rules 
            if rule.priority == 8  # Indicator rules have priority 8
        ]
        
        assert len(indicator_rules) > 0
        
        # Check for RSI rule
        rsi_rules = [rule for rule in indicator_rules if "rsi" in rule.feature_pattern]
        assert len(rsi_rules) > 0
        
        # Check for STOCH rules
        stoch_rules = [rule for rule in indicator_rules if "stoch" in rule.feature_pattern]
        assert len(stoch_rules) >= 2  # Should have rules for both _k and _d

    def test_find_validation_rule(self, mock_data_quality_config):
        """Test finding validation rules for specific features."""
        registry = FeatureValidationRegistry(mock_data_quality_config)
        
        # Test exact match
        rule = registry.find_validation_rule("rsi_14")
        assert rule is not None
        assert rule.min_value == 0
        assert rule.max_value == 100
        
        # Test pattern match
        rule = registry.find_validation_rule("stoch_k")
        assert rule is not None
        assert rule.min_value == 0
        assert rule.max_value == 100
        
        # Test no match
        rule = registry.find_validation_rule("unknown_feature")
        assert rule is None

    def test_validate_feature_with_rule(self, mock_data_quality_config):
        """Test feature validation with specific rules."""
        registry = FeatureValidationRegistry(mock_data_quality_config)
        
        # Create test data with violations
        test_data = pd.Series([50, 120, -10, 80, 90])  # RSI data with violations
        
        result = registry.validate_feature("rsi_14", test_data, "clip")
        
        assert isinstance(result, FeatureValidationResult)
        assert result.feature_name == "rsi_14"
        assert not result.is_valid  # Should have violations
        assert result.violations_count == 2  # 120 and -10 are violations
        assert result.applied_strategy == "clip"
        assert result.validation_rule is not None

    def test_validate_feature_no_rule(self, mock_data_quality_config):
        """Test feature validation when no rule exists."""
        registry = FeatureValidationRegistry(mock_data_quality_config)
        
        test_data = pd.Series([1, 2, 3, 4, 5])
        
        result = registry.validate_feature("unknown_feature", test_data, "clip")
        
        assert result.is_valid
        assert result.violations_count == 0
        assert result.applied_strategy == "none"
        assert result.validation_rule is None

    def test_validate_feature_no_violations(self, mock_data_quality_config):
        """Test feature validation with no violations."""
        registry = FeatureValidationRegistry(mock_data_quality_config)
        
        # Create valid RSI data
        test_data = pd.Series([30, 50, 70, 40, 60])
        
        result = registry.validate_feature("rsi_14", test_data, "clip")
        
        assert result.is_valid
        assert result.violations_count == 0
        assert result.applied_strategy == "none"

    def test_indicator_rules_generation(self, mock_data_quality_config):
        """Test specific indicator rule generation."""
        registry = FeatureValidationRegistry(mock_data_quality_config)
        
        # Test RSI rules
        rsi_rules = registry._get_indicator_rules("rsi", "RSI")
        assert len(rsi_rules) == 1
        assert rsi_rules[0].min_value == 0.0
        assert rsi_rules[0].max_value == 100.0
        
        # Test STOCH rules
        stoch_rules = registry._get_indicator_rules("stoch", "STOCH")
        assert len(stoch_rules) == 2  # _k and _d
        
        # Test WILLR rules
        willr_rules = registry._get_indicator_rules("willr", "WILLR")
        assert len(willr_rules) == 1
        assert willr_rules[0].min_value == -100.0
        assert willr_rules[0].max_value == 0.0
        
        # Test MACD rules (should be empty - no fixed range)
        macd_rules = registry._get_indicator_rules("macd", "MACD")
        assert len(macd_rules) == 0

    def test_custom_rules_loading(self):
        """Test loading custom validation rules."""
        # Create mock config with custom rules
        config = MagicMock()
        config.model_training.feature_ranges = {}
        config.trading.features.configurations = []
        
        # Mock custom rules
        config.data_quality.feature_validation.custom_rules = {
            'range_rules': [
                {
                    'pattern': '^test_feature$',
                    'min_value': 0.0,
                    'max_value': 1.0,
                    'priority': 9,
                    'description': 'Test rule'
                }
            ]
        }
        
        registry = FeatureValidationRegistry(config)
        
        # Find custom rule
        custom_rules = [rule for rule in registry.validation_rules if rule.priority == 9]
        assert len(custom_rules) == 1
        assert custom_rules[0].feature_pattern == '^test_feature$'
        assert custom_rules[0].min_value == 0.0
        assert custom_rules[0].max_value == 1.0

    def test_rules_summary(self, mock_data_quality_config):
        """Test getting rules summary."""
        registry = FeatureValidationRegistry(mock_data_quality_config)
        
        summary = registry.get_rules_summary()
        
        assert "total_rules" in summary
        assert "by_validator_type" in summary
        assert "by_priority" in summary
        assert "rules" in summary
        
        assert summary["total_rules"] == len(registry.validation_rules)
        assert isinstance(summary["rules"], list)
        assert len(summary["rules"]) == summary["total_rules"]

    def test_priority_ordering(self, mock_data_quality_config):
        """Test that rules are properly ordered by priority."""
        registry = FeatureValidationRegistry(mock_data_quality_config)
        
        # Add a feature that could match multiple rules
        test_data = pd.Series([50, 60, 70])
        
        # The highest priority rule should be found
        rule = registry.find_validation_rule("rsi_14")
        assert rule is not None
        # Should be the config rule (priority 10) not the indicator rule (priority 8)
        assert rule.priority == 10

    def test_range_validation_edge_cases(self, mock_data_quality_config):
        """Test range validation with edge cases."""
        registry = FeatureValidationRegistry(mock_data_quality_config)
        
        # Test with NaN values
        test_data = pd.Series([50, np.nan, 70, 40])
        result = registry.validate_feature("rsi_14", test_data, "clip")
        assert result.violations_count == 0  # NaN should not be counted as violation
        
        # Test with infinite values
        test_data = pd.Series([50, np.inf, 70, -np.inf])
        result = registry.validate_feature("rsi_14", test_data, "clip")
        assert result.violations_count == 2  # Both inf values should be violations

    def test_pattern_matching(self, mock_data_quality_config):
        """Test regex pattern matching for features."""
        registry = FeatureValidationRegistry(mock_data_quality_config)
        
        # Add a custom pattern rule
        custom_rule = ValidationRule(
            feature_pattern="^.*_normalized$",
            min_value=-1.0,
            max_value=1.0,
            priority=7,
            description="Normalized features"
        )
        registry.validation_rules.append(custom_rule)
        registry.validation_rules.sort(key=lambda x: x.priority, reverse=True)
        
        # Test pattern matching
        rule = registry.find_validation_rule("price_normalized")
        assert rule is not None
        assert rule.min_value == -1.0
        assert rule.max_value == 1.0
        
        rule = registry.find_validation_rule("momentum_normalized")
        assert rule is not None
        assert rule.min_value == -1.0
        assert rule.max_value == 1.0
        
        # Test non-matching
        rule = registry.find_validation_rule("price_raw")
        assert rule is None or rule != custom_rule

    def test_error_handling(self, mock_data_quality_config):
        """Test error handling in registry."""
        # Test with malformed config
        config = MagicMock()
        config.model_training.feature_ranges = None  # This should cause an error
        config.trading.features.configurations = []
        
        # Should not crash, just log warning
        registry = FeatureValidationRegistry(config)
        assert len(registry.validation_rules) >= 0  # Should handle gracefully

    def test_infinite_bounds_handling(self, mock_data_quality_config):
        """Test handling of infinite bounds in feature ranges."""
        config = MagicMock()
        config.model_training.feature_ranges = {
            "unbounded_feature": [float('-inf'), float('inf')],
            "lower_bounded": [0, float('inf')],
            "upper_bounded": [float('-inf'), 100]
        }
        config.trading.features.configurations = []
        
        registry = FeatureValidationRegistry(config)
        
        # Find unbounded feature rule
        rule = registry.find_validation_rule("unbounded_feature")
        assert rule is not None
        assert rule.min_value is None
        assert rule.max_value is None
        
        # Find lower bounded rule
        rule = registry.find_validation_rule("lower_bounded")
        assert rule is not None
        assert rule.min_value == 0
        assert rule.max_value is None
        
        # Find upper bounded rule
        rule = registry.find_validation_rule("upper_bounded")
        assert rule is not None
        assert rule.min_value is None
        assert rule.max_value == 100