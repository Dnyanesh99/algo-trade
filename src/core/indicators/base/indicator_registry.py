from typing import Any, Optional, Union

from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

from .base_indicator import BaseIndicator, IndicatorConfig


class IndicatorRegistry:
    """
    Registry for managing indicator instances and enable/disable functionality.

    Provides:
    - Centralized indicator management
    - Enable/disable control
    - Category-based organization
    - Dynamic loading and configuration
    """

    def __init__(self) -> None:
        self._indicators: dict[str, BaseIndicator] = {}
        self._categories: dict[str, list[str]] = {
            "trend": [],
            "momentum": [],
            "volatility": [],
            "volume": [],
            "enhanced": [],
        }
        self._config: Optional[Any] = None
        self._config_loaded = False
        self._indicator_control: dict[str, Any] = {
            "enabled": True,
            "disabled_indicators": [],
            "categories": {
                "trend": True,
                "momentum": True,
                "volatility": True,
                "volume": True,
                "enhanced": True,
            },
        }
        # Don't load configuration immediately - use lazy loading

    def _ensure_config_loaded(self) -> bool:
        """Ensure configuration is loaded (lazy loading)."""
        if self._config_loaded:
            return True

        try:
            self._config = ConfigLoader().get_config()
            self._config_loaded = True
            self._load_configuration()
            return True
        except Exception as e:
            raise RuntimeError(
                f"CRITICAL TRADING SYSTEM FAILURE: Cannot load indicator configuration. "
                f"Trading system cannot initialize properly without valid configuration. Error: {e}"
            ) from e

    def _load_configuration(self) -> None:
        """Load indicator configuration from config.yaml."""
        try:
            if not self._config or not hasattr(self._config, "trading") or not self._config.trading:
                logger.info("No trading configuration found in config")
                return

            if not hasattr(self._config.trading, "features") or not self._config.trading.features:
                logger.info("No trading feature configurations found in config")
                return

            # Load indicator control settings
            loaded_control = getattr(self._config.trading.features, "indicator_control", None)
            if loaded_control is not None:
                if isinstance(loaded_control, dict):
                    self._indicator_control.update(loaded_control)
                else:
                    raise RuntimeError(
                        f"CRITICAL TRADING SYSTEM FAILURE: indicator_control configuration must be a dictionary, "
                        f"got {type(loaded_control)}. Invalid configuration format compromises trading system initialization."
                    )
            else:
                logger.debug("No indicator_control section found, using defaults")

        except Exception as e:
            raise RuntimeError(
                f"CRITICAL TRADING SYSTEM FAILURE: Error loading indicator configuration. "
                f"Trading indicators cannot be properly configured, compromising signal reliability. Error: {e}"
            ) from e

    def register_indicator(
        self, indicator_class: type[BaseIndicator], config: IndicatorConfig, category: str = "base"
    ) -> None:
        """
        Register an indicator with the registry.

        Args:
            indicator_class: The indicator class to register
            config: Configuration for the indicator
            category: Category for organization (trend, momentum, etc.)
        """
        # Ensure config is loaded before registering
        self._ensure_config_loaded()

        # Check if indicator is globally disabled
        if not self._indicator_control.get("enabled", True):
            config.enabled = False
            logger.info(f"Indicator system disabled, {config.name} will be disabled")

        # Check if specific indicator is disabled
        if config.name in self._indicator_control.get("disabled_indicators", []):
            config.enabled = False
            logger.info(f"Indicator {config.name} is specifically disabled")

        # Check if category is disabled
        if not self._indicator_control.get("categories", {}).get(category, True):
            config.enabled = False
            logger.info(f"Category {category} is disabled, {config.name} will be disabled")

        # Create indicator instance
        try:
            indicator_instance = indicator_class(config)
            self._indicators[config.name] = indicator_instance

            # Add to category
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(config.name)

            logger.info(f"Registered indicator: {config.name} (category: {category}, enabled: {config.enabled})")

        except Exception as e:
            raise RuntimeError(
                f"CRITICAL TRADING SYSTEM FAILURE: Failed to register indicator {config.name}. "
                f"Indicator cannot be initialized for trading signal generation. Error: {e}"
            ) from e

    def get_indicator(self, name: str) -> Optional[BaseIndicator]:
        """Get indicator by name."""
        return self._indicators.get(name)

    def get_indicators_by_category(self, category: str) -> list[BaseIndicator]:
        """Get all indicators in a category."""
        indicator_names = self._categories.get(category, [])
        return [self._indicators[name] for name in indicator_names if name in self._indicators]

    def get_all_indicators(self) -> dict[str, BaseIndicator]:
        """Get all registered indicators."""
        # Ensure config is loaded
        self._ensure_config_loaded()
        return self._indicators.copy()

    def get_enabled_indicators(self) -> dict[str, BaseIndicator]:
        """Get only enabled indicators."""
        return {name: indicator for name, indicator in self._indicators.items() if indicator.enabled}

    def enable_indicator(self, name: str) -> bool:
        """Enable an indicator by name."""
        if name in self._indicators:
            self._indicators[name].enabled = True
            logger.info(f"Enabled indicator: {name}")
            return True
        return False

    def disable_indicator(self, name: str) -> bool:
        """Disable an indicator by name."""
        if name in self._indicators:
            self._indicators[name].enabled = False
            logger.info(f"Disabled indicator: {name}")
            return True
        return False

    def enable_category(self, category: str) -> None:
        """Enable all indicators in a category."""
        for indicator_name in self._categories.get(category, []):
            self.enable_indicator(indicator_name)

    def disable_category(self, category: str) -> None:
        """Disable all indicators in a category."""
        for indicator_name in self._categories.get(category, []):
            self.disable_indicator(indicator_name)

    def get_registry_stats(self) -> dict[str, Union[int, dict[str, int]]]:
        """Get registry statistics."""
        total_indicators = len(self._indicators)
        enabled_indicators = len(self.get_enabled_indicators())

        category_stats: dict[str, int] = {}
        for category, indicator_names in self._categories.items():
            category_stats[category] = len(indicator_names)

        return {
            "total_indicators": total_indicators,
            "enabled_indicators": enabled_indicators,
            "disabled_indicators": total_indicators - enabled_indicators,
            "categories": category_stats,
        }

    def list_indicators(self) -> dict[str, dict[str, Union[str, bool]]]:
        """list all indicators with their status."""
        result: dict[str, dict[str, Union[str, bool]]] = {}
        for name, indicator in self._indicators.items():
            # Find category
            category = "unknown"
            for cat, indicators in self._categories.items():
                if name in indicators:
                    category = cat
                    break

            result[name] = {
                "enabled": indicator.enabled,
                "category": category,
                "fallback_enabled": indicator.fallback_enabled,
                "validation_enabled": indicator.validation_enabled,
            }

        return result

    def clear_registry(self) -> None:
        """Clear all registered indicators."""
        self._indicators.clear()
        for category in self._categories:
            self._categories[category].clear()
        logger.info("Cleared all indicators from registry")


# Global registry instance - created lazily to avoid config dependency
indicator_registry = None


def get_indicator_registry() -> IndicatorRegistry:
    """Get the global indicator registry instance."""
    global indicator_registry
    if indicator_registry is None:
        indicator_registry = IndicatorRegistry()
    return indicator_registry
