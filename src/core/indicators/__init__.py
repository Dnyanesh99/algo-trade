"""
Modular indicators architecture for the trading system.
Provides plug-and-play indicator functionality with enable/disable capabilities.
"""

from .base.base_indicator import BaseIndicator, IndicatorConfig
from .base.indicator_registry import IndicatorRegistry
from .main import IndicatorCalculator

__all__ = ["IndicatorCalculator", "BaseIndicator", "IndicatorConfig", "IndicatorRegistry"]
