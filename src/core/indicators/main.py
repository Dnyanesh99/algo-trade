import importlib
import inspect
import pkgutil
from typing import Any, Optional

import pandas as pd

import src.core.indicators
from src.core.indicators.base.base_indicator import BaseIndicator, IndicatorConfig
from src.core.indicators.base.indicator_registry import IndicatorRegistry
from src.utils.config_loader import AppConfig, ConfigLoader
from src.utils.logger import LOGGER as logger


class IndicatorCalculator:
    """Main indicator calculator that orchestrates all indicators."""

    def __init__(self, require_full_config: bool = True) -> None:
        """
        Initialize indicator calculator.

        Args:
            require_full_config: If True, require full config validation (production mode).
                                If False, allow minimal config for testing.
        """
        self.registry = IndicatorRegistry()
        self.config_loader = ConfigLoader()
        self.config: Optional[AppConfig] = None
        self.require_full_config = require_full_config

        if require_full_config:
            # PRODUCTION MODE: Strict validation, fail fast
            try:
                self.config = self.config_loader.get_config()
                if not self.config:
                    raise RuntimeError("❌ CRITICAL: Configuration is None - system cannot proceed")

                if not hasattr(self.config, "trading") or not self.config.trading:
                    raise RuntimeError("❌ CRITICAL: Missing 'trading' configuration section")

                if not hasattr(self.config.trading, "features") or not self.config.trading.features:
                    raise RuntimeError("❌ CRITICAL: Missing 'trading.features' configuration section")

                if not self.config.trading.features.configurations:
                    raise RuntimeError(
                        "❌ CRITICAL: Missing 'trading.features.configurations' - no indicators configured"
                    )

                self._load_indicators()

            except Exception as e:
                raise RuntimeError(
                    f"🚨 CRITICAL SYSTEM HALT: IndicatorCalculator initialization failed: {e}. Fix configuration in config/config.yaml. Required sections: trading.features.configurations - Indicator system completely broken"
                ) from e
        else:
            # TESTING MODE: Try to load config, but allow minimal fallback for testing only
            try:
                self.config = self.config_loader.get_config()
                if (
                    self.config
                    and hasattr(self.config, "trading")
                    and self.config.trading
                    and hasattr(self.config.trading, "features")
                    and self.config.trading.features
                    and self.config.trading.features.configurations
                ):
                    logger.info("📁 Using full configuration for testing")
                    self._load_indicators()
                else:
                    raise ValueError("Config incomplete - using test mode")
            except Exception as e:
                raise RuntimeError(
                    f"🚨 CRITICAL CONFIG FAILURE: Full config unavailable ({e}), cannot use minimal indicator set in production - Configuration system broken"
                ) from e

    def _load_indicators(self) -> None:
        """Dynamically load all indicators from the indicators package."""
        if self.config is None:
            raise RuntimeError("Config must be validated before calling _load_indicators")

        indicator_configs = {cfg.name: cfg for cfg in self.config.trading.features.configurations}
        indicator_control = getattr(self.config.trading.features, "indicator_control", {})
        disabled_indicators = indicator_control.get("disabled_indicators", [])

        if not indicator_configs:
            raise RuntimeError("❌ CRITICAL: No indicator configurations found in trading.features.configurations")

        for module_info in pkgutil.walk_packages(src.core.indicators.__path__, src.core.indicators.__name__ + "."):
            module = importlib.import_module(module_info.name)
            for _, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseIndicator) and obj is not BaseIndicator:
                    indicator_name = obj.__name__.replace("Indicator", "").lower()

                    # Determine category from module path
                    category = "base"
                    if ".trend." in module_info.name:
                        category = "trend"
                    elif ".momentum." in module_info.name:
                        category = "momentum"
                    elif ".volatility." in module_info.name:
                        category = "volatility"
                    elif ".volume." in module_info.name:
                        category = "volume"
                    elif ".enhanced." in module_info.name:
                        category = "enhanced"
                    elif ".cycle." in module_info.name:
                        category = "cycle"
                    elif ".other." in module_info.name:
                        category = "other"

                    if indicator_name in indicator_configs:
                        config_data = indicator_configs[indicator_name]
                        config = IndicatorConfig(
                            name=indicator_name,
                            enabled=indicator_name not in disabled_indicators,
                            params=config_data.params.get("default", {}),
                            timeframe_params={tf: p for tf, p in config_data.params.items() if tf != "default"},
                        )
                        self.registry.register_indicator(obj, config, category)
                        logger.info(
                            f"Registered indicator: {indicator_name} (category: {category}, enabled: {config.enabled})"
                        )
                    else:
                        logger.debug(f"Skipping indicator {indicator_name} - not in configuration")

        # Validate that at least some indicators were loaded
        total_indicators = len(self.registry.get_all_indicators())
        enabled_indicators = len(self.registry.get_enabled_indicators())

        if total_indicators == 0:
            raise RuntimeError(
                "❌ CRITICAL: No indicators were loaded! Check indicator module imports and configuration."
            )

        if enabled_indicators == 0:
            raise RuntimeError(
                "❌ CRITICAL: All indicators are disabled! Check indicator_control.disabled_indicators configuration."
            )

        logger.info(f"✅ Successfully loaded {total_indicators} indicators ({enabled_indicators} enabled)")
        return

    def _load_minimal_test_indicators(self) -> None:
        """Load minimal set of indicators for testing when full config is unavailable."""
        # Define essential indicators for testing
        test_indicators: dict[str, dict[str, Any]] = {
            "rsi": {"timeperiod": 14},
            "macd": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
            "atr": {"timeperiod": 14},
            "bbands": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
        }

        for module_info in pkgutil.walk_packages(src.core.indicators.__path__, src.core.indicators.__name__ + "."):
            try:
                module = importlib.import_module(module_info.name)
                for _, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, BaseIndicator) and obj is not BaseIndicator:
                        indicator_name = obj.__name__.replace("Indicator", "").lower()

                        # Only load essential indicators for testing
                        if indicator_name in test_indicators:
                            # Determine category from module path
                            category = "base"
                            if ".trend." in module_info.name:
                                category = "trend"
                            elif ".momentum." in module_info.name:
                                category = "momentum"
                            elif ".volatility." in module_info.name:
                                category = "volatility"
                            elif ".volume." in module_info.name:
                                category = "volume"

                            config = IndicatorConfig(
                                name=indicator_name,
                                enabled=True,
                                params=test_indicators[indicator_name],
                                timeframe_params={},
                            )
                            self.registry.register_indicator(obj, config, category)
                            logger.info(f"🧪 TEST: Loaded {indicator_name} (category: {category})")
            except Exception as e:
                logger.debug(f"Failed to load test indicator from {module_info.name}: {e}")

        # Validate minimal indicators loaded
        total_indicators = len(self.registry.get_all_indicators())
        if total_indicators == 0:
            raise RuntimeError("❌ CRITICAL: Failed to load even minimal test indicators!")

        raise RuntimeError(
            f"🚨 CRITICAL PRODUCTION ERROR: Loaded {total_indicators} minimal indicators for testing. This is NOT suitable for production - fix the configuration! Trading system compromised"
        )

    def calculate_timeframe_features(self, df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
        """Calculate features for a specific timeframe using registered indicators with dependency support."""
        if df.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=df.index)

        # Get all indicators grouped by category for dependency-aware calculation
        all_indicators = self.registry.get_all_indicators()
        enhanced_indicators = {
            name: ind
            for name, ind in all_indicators.items()
            if name
            in [
                "momentum_ratios",
                "volatility_adjustments",
                "trend_confirmations",
                "mean_reversion_signals",
                "breakout_indicators",
            ]
        }
        base_indicators = {name: ind for name, ind in all_indicators.items() if name not in enhanced_indicators}

        # Step 1: Calculate base indicators first
        logger.debug(f"Calculating {len(base_indicators)} base indicators")
        for name, indicator in base_indicators.items():
            if not indicator.enabled:
                continue

            try:
                result = indicator.calculate(df, timeframe_minutes)
                if result.success and result.data is not None:
                    if isinstance(result.data, pd.Series):
                        features[result.name] = result.data
                    elif isinstance(result.data, pd.DataFrame):
                        for col in result.data.columns:
                            features[f"{result.name}_{col}"] = result.data[col]
                else:
                    raise RuntimeError(
                        f"🚨 CRITICAL INDICATOR FAILURE: Indicator {name} calculation failed: {result.error_message} - Trading signal compromised"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"🚨 CRITICAL CALCULATION ERROR: Error calculating {name}: {e} - Indicator system broken"
                ) from e

        # Step 2: Calculate enhanced indicators with base feature dependencies
        if enhanced_indicators:
            logger.debug(f"Calculating {len(enhanced_indicators)} enhanced indicators with base feature dependencies")
            enhanced_features = self._calculate_enhanced_indicators(
                df, timeframe_minutes, enhanced_indicators, features
            )

            # Merge enhanced features
            for col_name in enhanced_features.columns:
                features[col_name] = enhanced_features[col_name]

        return features

    def _calculate_enhanced_indicators(
        self,
        df: pd.DataFrame,
        timeframe_minutes: int,
        enhanced_indicators: dict[str, Any],
        base_features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate enhanced indicators that depend on base indicator results."""
        enhanced_features = pd.DataFrame(index=df.index)

        # Convert base features DataFrame to latest values dictionary for enhanced indicators
        if not base_features_df.empty:
            # Get the latest row of base features
            latest_base_features = base_features_df.iloc[-1].to_dict()

            # Apply feature name normalization for enhanced indicator compatibility
            normalized_base_features = self._normalize_base_feature_names(latest_base_features)
        else:
            normalized_base_features = {}

        logger.debug(f"Base features available for enhanced indicators: {list(normalized_base_features.keys())}")

        for name, indicator in enhanced_indicators.items():
            if not indicator.enabled:
                continue

            try:
                # Get timeframe-specific parameters and inject base features
                params = indicator.get_timeframe_params(timeframe_minutes)
                params["base_features"] = normalized_base_features

                # For breakout indicators, we need segment information - get from config or default to "EQUITY"
                if name == "breakout_indicators":
                    params["segment"] = getattr(self.config, "segment", "EQUITY") if self.config else "EQUITY"

                # Use direct calculation methods since we're providing custom parameters
                result_data = None
                try:
                    # Try TA-Lib implementation first
                    if indicator.talib is not None:
                        result_data = indicator.calculate_talib(df, params)
                    else:
                        result_data = indicator.calculate_pandas(df, params)
                except Exception as calc_error:
                    logger.warning(f"Direct calculation failed for {name}: {calc_error}")
                    # Fallback to the base calculate method without custom params
                    result = indicator.calculate(df, timeframe_minutes)
                    if result.success:
                        result_data = result.data

                if result_data is not None:
                    if isinstance(result_data, pd.Series):
                        enhanced_features[name] = result_data
                    elif isinstance(result_data, pd.DataFrame):
                        for col in result_data.columns:
                            # Enhanced indicators already include descriptive names
                            enhanced_features[col] = result_data[col]
                else:
                    logger.warning(f"Enhanced indicator {name} returned no data")

            except Exception as e:
                logger.error(f"Error calculating enhanced indicator {name}: {e}")
                # Don't fail the entire calculation for enhanced indicator errors
                # Enhanced indicators are supplementary features
                continue

        logger.debug(f"Generated {len(enhanced_features.columns)} enhanced indicator features")
        return enhanced_features

    def _normalize_base_feature_names(self, base_features: dict[str, float]) -> dict[str, float]:
        """
        Normalize base feature names for enhanced indicator compatibility.
        Maps modular system output names to expected enhanced indicator input names.
        """
        normalized_features = {}

        # Load internal mappings from config
        internal_name_mappings = {}
        if self.config and hasattr(self.config.trading.features, "internal_feature_mapping"):
            internal_name_mappings = self.config.trading.features.internal_feature_mapping

        for original_name, value in base_features.items():
            # Apply internal mappings
            mapped_name = internal_name_mappings.get(original_name, original_name)
            normalized_features[mapped_name] = value

            # Also keep original names for backward compatibility
            normalized_features[original_name] = value

        logger.debug(f"Normalized {len(base_features)} base features to {len(normalized_features)} mapped features")
        return normalized_features

    def apply_feature_mappings(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature name mappings from config.yaml."""
        if not self.config:
            return features_df
        name_mapping = self.config.trading.features.name_mapping
        if not name_mapping:
            return features_df
        return features_df.rename(columns=name_mapping)

    def calculate_all_features(self, df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
        """Calculate all features and apply mappings - main entry point."""
        try:
            raw_features = self.calculate_timeframe_features(df, timeframe_minutes)
            mapped_features = self.apply_feature_mappings(raw_features)
            logger.info(f"Calculated {len(mapped_features.columns)} features for timeframe {timeframe_minutes}m")
            return mapped_features
        except Exception as e:
            raise RuntimeError(
                f"🚨 CRITICAL FEATURE ERROR: Error calculating features: {e} - Feature calculation system completely broken"
            ) from e
