"""
Comprehensive configuration mock that matches the actual config.yaml structure.
This avoids Pydantic validation issues while maintaining realistic test conditions.
"""
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MockValidationConfig:
    enabled: bool = True
    min_valid_rows: int = 50
    quality_score_threshold: float = 80.0
    required_columns: list[str] = None
    expected_columns: dict[str, str] = None
    # Additional fields that exist in config.yaml but not in Pydantic model
    indicator_validation: dict[str, Any] = None
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = ["ts", "open", "high", "low", "close", "volume"]
        if self.expected_columns is None:
            self.expected_columns = {
                "date": "datetime64[ns]",
                "open": "float64",
                "high": "float64", 
                "low": "float64",
                "close": "float64",
                "volume": "int64"
            }
        if self.indicator_validation is None:
            self.indicator_validation = {
                "min_data_length": 20,
                "rsi_upper_bound": 100,
                "rsi_lower_bound": 0,
                "rsi_data_buffer": 10
            }


@dataclass
class MockTimeSeriesConfig:
    gap_multiplier: float = 2.0


@dataclass
class MockOutlierDetectionConfig:
    iqr_multiplier: float = 1.5
    handling_strategy: str = "clip"


@dataclass
class MockPenaltiesConfig:
    outlier_penalty: float = 0.1
    gap_penalty: float = 0.2
    ohlc_violation_penalty: float = 0.2
    duplicate_penalty: float = 0.1


@dataclass
class MockFeatureValidationConfig:
    enabled: bool = True
    auto_discovery: bool = True
    # Additional fields that exist in config.yaml but not in Pydantic model
    consistency_rules: list[dict[str, Any]] = None
    
    def __post_init__(self):
        if self.consistency_rules is None:
            self.consistency_rules = [
                {
                    "name": "stochastic_consistency",
                    "pattern": "stoch_.*",
                    "validation_type": "range",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "tolerance": 0.001
                }
            ]


@dataclass
class MockDataQualityConfig:
    time_series: MockTimeSeriesConfig
    outlier_detection: MockOutlierDetectionConfig
    penalties: MockPenaltiesConfig
    validation: MockValidationConfig
    feature_validation: MockFeatureValidationConfig


@dataclass
class MockLabelingConfig:
    atr_period: int = 14
    tp_atr_multiplier: float = 2.0
    sl_atr_multiplier: float = 1.0
    max_holding_periods: int = 100
    min_bars_required: int = 50
    # Additional fields that exist in config.yaml but not in Pydantic model
    dynamic_epsilon: dict[str, Any] = None
    path_dependent_timeout: dict[str, Any] = None
    
    def __post_init__(self):
        if self.dynamic_epsilon is None:
            self.dynamic_epsilon = {
                "market_session_times": {
                    "opening": {"start": "09:15", "end": "10:00"},
                    "midday": {"start": "12:00", "end": "14:00"}, 
                    "closing": {"start": "15:00", "end": "15:30"}
                }
            }
        if self.path_dependent_timeout is None:
            self.path_dependent_timeout = {
                "enabled": True,
                "upper_threshold": 0.5,
                "lower_threshold": -0.5
            }


@dataclass
class MockFeatureConfig:
    enabled: bool = True
    period: Optional[int] = None


@dataclass 
class MockTradingFeaturesConfig:
    lookback_period: int = 50
    rsi: Optional[MockFeatureConfig] = None
    macd: Optional[MockFeatureConfig] = None
    bollinger_bands: Optional[MockFeatureConfig] = None
    atr: Optional[MockFeatureConfig] = None
    # Additional fields that exist in config.yaml but not in Pydantic model
    indicator_control: dict[str, Any] = None
    
    def __post_init__(self):
        if self.rsi is None:
            self.rsi = MockFeatureConfig(period=14)
        if self.macd is None:
            self.macd = MockFeatureConfig(period=26)
        if self.bollinger_bands is None:
            self.bollinger_bands = MockFeatureConfig(period=20)
        if self.atr is None:
            self.atr = MockFeatureConfig(period=14)
        if self.indicator_control is None:
            self.indicator_control = {
                "enabled": True,
                "disabled_indicators": [],
                "experimental": True,
                "enhanced": True
            }


@dataclass
class MockTradingConfig:
    labeling: MockLabelingConfig
    features: MockTradingFeaturesConfig
    aggregation_timeframes: list[int] = None
    feature_timeframes: list[int] = None
    labeling_timeframes: list[int] = None
    
    def __post_init__(self):
        if self.aggregation_timeframes is None:
            self.aggregation_timeframes = [5, 15, 60]
        if self.feature_timeframes is None:
            self.feature_timeframes = [5, 15, 60]
        if self.labeling_timeframes is None:
            self.labeling_timeframes = [15]


@dataclass
class MockSystemConfig:
    version: str = "2.0.0"
    mode: str = "HISTORICAL_MODE"
    timezone: str = "Asia/Kolkata"
    openmp_threads: int = 1
    mkl_threads: int = 1
    numexpr_threads: int = 1


@dataclass
class MockBrokerConfig:
    api_key: str = "test_api_key"
    api_secret: str = "test_api_secret"
    redirect_url: str = "http://localhost:8000/callback"
    websocket_mode: str = "QUOTE"
    historical_interval: str = "minute"


@dataclass
class MockLoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/app.log"


@dataclass
class MockAppConfig:
    system: MockSystemConfig
    broker: MockBrokerConfig  
    logging: MockLoggingConfig
    trading: MockTradingConfig
    data_quality: MockDataQualityConfig


def create_comprehensive_mock_config(required_columns: Optional[list[str]] = None) -> MockDataQualityConfig:
    """Create a comprehensive mock configuration that matches config.yaml structure."""
    
    # Create the base config
    validation_config = MockValidationConfig()
    if required_columns is not None:
        validation_config.required_columns = required_columns
        
    return MockDataQualityConfig(
        time_series=MockTimeSeriesConfig(),
        outlier_detection=MockOutlierDetectionConfig(), 
        penalties=MockPenaltiesConfig(),
        validation=validation_config,
        feature_validation=MockFeatureValidationConfig()
    )


def create_full_app_config() -> MockAppConfig:
    """Create a full app config mock for tests that need the complete structure."""
    return MockAppConfig(
        system=MockSystemConfig(),
        broker=MockBrokerConfig(),
        logging=MockLoggingConfig(),
        trading=MockTradingConfig(
            labeling=MockLabelingConfig(),
            features=MockTradingFeaturesConfig()
        ),
        data_quality=create_comprehensive_mock_config()
    )