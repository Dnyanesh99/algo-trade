from datetime import date, time
from typing import Any, Dict, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.logger import LOGGER as logger


# Base model for all configuration classes to enforce strict validation
class StrictBaseModel(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")


class MarketSessionTime(StrictBaseModel):
    start: str = Field(pattern=r'^\d{2}:\d{2}$')  # HH:MM format
    end: str = Field(pattern=r'^\d{2}:\d{2}$')    # HH:MM format


class PathDependentTimeoutConfig(StrictBaseModel):
    enabled: bool = True
    upper_threshold: float = Field(default=0.5, ge=-1, le=1)
    lower_threshold: float = Field(default=-0.5, ge=-1, le=1)


class SystemConfig(StrictBaseModel):
    version: str
    mode: Literal["HISTORICAL_MODE", "LIVE_MODE"]
    timezone: str
    openmp_threads: int = 1
    mkl_threads: int = 1
    numexpr_threads: int = 1


class ConnectionManagerConfig(StrictBaseModel):
    max_reconnect_attempts: int = Field(gt=0)
    initial_reconnect_delay: int = Field(gt=0)
    heartbeat_timeout: int = Field(gt=0)
    monitor_interval: int = Field(gt=0)


class BrokerConfig(StrictBaseModel):
    api_key: str
    api_secret: str
    redirect_url: str
    websocket_mode: Literal["LTP", "QUOTE", "FULL"]
    historical_interval: Literal["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute", "day"]
    should_fetch_instruments: bool = False
    # 🔧 MIGRATION: historical_data_lookback_days moved to model_training section for consistency
    segment_types: list[str] = Field(default_factory=list)
    connection_manager: ConnectionManagerConfig


class LoggingConfig(StrictBaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    format: str
    file: str
    rotation: str = "10 MB"
    retention: str = "30 days"
    compression: str = "zip"


class DynamicEpsilonConfig(StrictBaseModel):
    low_volatility_multiplier: float = Field(default=1.5, gt=0)
    normal_volatility_multiplier: float = Field(default=1.0, gt=0)
    high_volatility_multiplier: float = Field(default=0.7, gt=0)
    extreme_volatility_multiplier: float = Field(default=2.0, gt=0)
    low_volatility_threshold: float = Field(default=0.5, gt=0)
    high_volatility_threshold: float = Field(default=1.5, gt=0)
    extreme_volatility_threshold: float = Field(default=2.5, gt=0)
    market_open_multiplier: float = Field(default=0.8, gt=0)
    pre_lunch_multiplier: float = Field(default=1.3, gt=0)
    market_close_multiplier: float = Field(default=0.9, gt=0)
    monday_multiplier: float = Field(default=1.1, gt=0)
    friday_multiplier: float = Field(default=0.85, gt=0)
    extreme_zscore_threshold: float = Field(default=2.5, gt=0)
    moderate_zscore_threshold: float = Field(default=1.5, gt=0)
    stable_zscore_threshold: float = Field(default=0.5, gt=0)
    extreme_zscore_multiplier: float = Field(default=1.5, gt=0)
    moderate_zscore_multiplier: float = Field(default=1.2, gt=0)
    stable_zscore_multiplier: float = Field(default=0.9, gt=0)
    min_epsilon_multiplier: float = Field(default=0.3, gt=0)
    max_epsilon_multiplier: float = Field(default=3.0, gt=0)
    weekly_expiry_day: int = Field(default=4, ge=0, le=6)  # 0=Monday, 6=Sunday
    monthly_expiry_week: int = Field(default=-1, ge=-1, le=4)  # -1=last week
    expiry_volatility_multiplier: float = Field(default=0.75, gt=0)
    market_session_times: Optional[Dict[str, MarketSessionTime]] = None


class LabelingConfig(StrictBaseModel):
    atr_period: int = Field(gt=0)
    tp_atr_multiplier: float = Field(gt=0)
    sl_atr_multiplier: float = Field(gt=0)
    max_holding_periods: int = Field(gt=0)
    min_bars_required: int = Field(gt=0)
    epsilon: float = Field(gt=0)
    atr_smoothing: Literal["ema", "sma"]
    use_dynamic_barriers: bool = True
    use_dynamic_epsilon: bool = Field(default=False)
    volatility_lookback: int = Field(default=20, gt=0)
    atr_cache_size: int = Field(default=100, gt=0)
    dynamic_barrier_tp_sensitivity: float = Field(default=0.2, gt=0)
    dynamic_barrier_sl_sensitivity: float = Field(default=0.1, gt=0)
    sample_weight_decay_factor: float = Field(default=0.5, gt=0)
    dynamic_epsilon: DynamicEpsilonConfig = Field(default_factory=DynamicEpsilonConfig)
    path_dependent_timeout: Optional[PathDependentTimeoutConfig] = None
    # Data fetching limits for main.py operations (not used by labeler algorithm)
    ohlcv_data_limit_for_labeling: int = Field(default=5000, gt=0)  # Reasonable limit for DB query performance
    minimum_ohlcv_data_for_labeling: int = Field(default=200, gt=0)  # Must have enough data before labeling

    @model_validator(mode="after")
    def check_atr_period_and_min_bars(self) -> "LabelingConfig":
        if self.min_bars_required < self.atr_period:
            raise ValueError("min_bars_required cannot be less than atr_period")
        return self


class FeatureConfig(StrictBaseModel):
    enabled: bool = True
    period: Optional[int] = Field(default=None, gt=0)
    fast_period: Optional[int] = Field(default=None, gt=0)
    slow_period: Optional[int] = Field(default=None, gt=0)
    signal_period: Optional[int] = Field(default=None, gt=0)
    std_dev: Optional[float] = Field(default=None, gt=0)


class FeatureConfiguration(StrictBaseModel):
    name: str
    function: str
    params: dict[str, Any] = Field(default_factory=dict)


class TradingFeaturesConfig(StrictBaseModel):
    lookback_period: int = Field(gt=0)
    rsi: Optional[FeatureConfig] = None
    macd: Optional[FeatureConfig] = None
    bollinger_bands: Optional[FeatureConfig] = None
    atr: Optional[FeatureConfig] = None
    ema_fast: Optional[FeatureConfig] = None
    ema_slow: Optional[FeatureConfig] = None
    sma: Optional[FeatureConfig] = None
    volume_sma: Optional[FeatureConfig] = None
    configurations: list[FeatureConfiguration] = Field(default_factory=list)
    name_mapping: dict[str, str] = Field(default_factory=dict)
    internal_feature_mapping: dict[str, str] = Field(default_factory=dict)
    indicator_control: dict[str, Any] = Field(
        default_factory=lambda: {
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
    )


class TradingInstrumentConfig(StrictBaseModel):
    label: str
    tradingsymbol: str
    exchange: str
    instrument_type: str
    segment: str


class TradingConfig(StrictBaseModel):
    labeling: LabelingConfig
    aggregation_timeframes: list[int] = Field(default_factory=lambda: [5, 15, 60])
    feature_timeframes: list[int] = Field(default_factory=lambda: [5, 15, 60])
    labeling_timeframes: list[int] = Field(default_factory=lambda: [15])  # Default to 15min only
    features: TradingFeaturesConfig
    instruments: list[TradingInstrumentConfig]


class ProcessingConfig(StrictBaseModel):
    chunk_size: int = Field(gt=0)
    max_workers: int = Field(gt=0)
    parallel_timeout_seconds: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    feature_batch_size: int = Field(gt=0)
    tick_queue_max_size: int = Field(gt=0)


class MemoryConfig(StrictBaseModel):
    limit_percentage: float = Field(gt=0, le=1)
    safety_factor: float = Field(gt=0, le=1)
    low_memory_threshold_gb: float = Field(ge=0)


class CacheConfig(StrictBaseModel):
    max_size: int = Field(gt=0)
    cleanup_batch_size: int = Field(gt=0)


class PerformanceConfig(StrictBaseModel):
    processing: ProcessingConfig
    memory: MemoryConfig
    cache: CacheConfig
    max_retries: int = Field(gt=0)


class ParquetConfig(StrictBaseModel):
    compression: str
    use_dictionary: bool
    version: str


class OutputConfig(StrictBaseModel):
    save_statistics: bool
    statistics_format: Literal["json", "yaml"]


class MetadataConfig(StrictBaseModel):
    include_config: bool
    include_checksum: bool
    checksum_length: int = Field(gt=0)


class StorageConfig(StrictBaseModel):
    parquet: ParquetConfig
    output: OutputConfig
    metadata: MetadataConfig


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_", extra="forbid")
    host: str
    port: int
    dbname: str
    user: str
    password: str
    min_connections: int = Field(gt=0)
    max_connections: int = Field(gt=0)
    timeout: int = Field(gt=0)


class DataGenerationConfig(StrictBaseModel):
    random_seed: int
    start_date: str
    end_date: str
    frequency: str
    initial_price: float = Field(gt=0)
    return_mean: float
    return_std: float = Field(ge=0)
    price_noise_std: float = Field(ge=0)
    high_noise_std: float = Field(ge=0)
    low_noise_std: float = Field(ge=0)
    volume_log_mean: float
    volume_log_std: float = Field(ge=0)


class SampleOutputConfig(StrictBaseModel):
    file_name: str
    rows_to_save: int = Field(gt=0)


class WalkForwardValidationConfig(StrictBaseModel):
    initial_train_ratio: float = Field(gt=0, le=1)
    validation_ratio: float = Field(gt=0, le=1)
    step_ratio: float = Field(gt=0, le=1)
    n_splits: int = Field(default=5, gt=0)


class OptimizationConfig(StrictBaseModel):
    enabled: bool = True
    n_trials: int = Field(default=50, gt=0)
    n_splits: int = Field(default=5, gt=0)
    test_size: int = Field(default=2000, gt=0)
    n_estimators_range: list[int] = Field(default_factory=lambda: [100, 500])
    learning_rate_range: list[float] = Field(default_factory=lambda: [0.01, 0.2])
    num_leaves_range: list[int] = Field(default_factory=lambda: [31, 100])
    max_depth_range: list[int] = Field(default_factory=lambda: [3, 10])
    min_child_samples_range: list[int] = Field(default_factory=lambda: [20, 100])
    feature_fraction_range: list[float] = Field(default_factory=lambda: [0.8, 1.0])
    bagging_fraction_range: list[float] = Field(default_factory=lambda: [0.8, 1.0])
    bagging_freq_range: list[int] = Field(default_factory=lambda: [1, 5])
    lambda_l1_range: list[float] = Field(default_factory=lambda: [0.0, 0.1])
    lambda_l2_range: list[float] = Field(default_factory=lambda: [0.0, 0.1])
    roc_auc_multi_class: str = Field(default="ovr")
    roc_auc_average: str = Field(default="macro")
    early_stopping_rounds: int = Field(default=100, gt=0)


class FinalModelConfig(StrictBaseModel):
    num_boost_round: int = Field(default=1000, gt=0)


class RetentionConfig(StrictBaseModel):
    max_model_versions: int = Field(default=5, gt=0)


class PredictorConfig(StrictBaseModel):
    required_artifacts: dict[str, str] = Field(default_factory=dict)
    prediction_history_window_small: int = Field(default=100, gt=0)
    prediction_history_window_large: int = Field(default=1000, gt=0)
    model_staleness_max_days: int = Field(default=7, gt=0)
    real_time_accuracy_windows: list[int] = Field(default_factory=lambda: [50, 100, 200])
    confidence_degradation_threshold: float = Field(default=0.1, ge=0, le=1)
    error_rate_threshold: float = Field(default=0.2, ge=0, le=1)


class FeatureGenerationConfig(StrictBaseModel):
    enabled: bool = True
    patterns: list[str] = Field(
        default_factory=lambda: [
            "momentum_ratios",
            "volatility_adjustments",
            "trend_confirmations",
            "mean_reversion_signals",
            "breakout_indicators",
        ]
    )
    lookback_periods: list[int] = Field(default_factory=lambda: [5, 10, 20, 50])
    min_quality_score: float = Field(default=0.6, ge=0.0, le=1.0)


class FeatureSelectionConfig(StrictBaseModel):
    enabled: bool = True
    target_feature_count: int = Field(default=50, gt=0)
    importance_history_length: int = Field(default=10, gt=0)
    correlation_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    stability_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    consistency_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    importance_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    min_selection_frequency: float = Field(default=0.2, ge=0.0, le=1.0)
    correlation_data_lookback_multiplier: int = Field(default=7, gt=0)
    min_history_for_selection: int = Field(default=3, gt=0)


class CrossAssetConfig(StrictBaseModel):
    enabled: bool = True
    correlation_lookback_days: int = Field(default=30, gt=0)
    minimum_correlation_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    update_frequency_minutes: int = Field(default=15, gt=0)
    max_related_instruments: int = Field(default=5, gt=0)
    cache_duration_minutes: int = Field(default=5, gt=0)
    instruments: dict[int, str] = Field(default_factory=dict)


class FeatureEngineeringConfig(StrictBaseModel):
    enabled: bool = True
    feature_generation: FeatureGenerationConfig = Field(default_factory=FeatureGenerationConfig)
    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)
    cross_asset: CrossAssetConfig = Field(default_factory=CrossAssetConfig)
    max_features_per_instrument: int = Field(default=200, gt=0)
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class ModelTrainingConfig(StrictBaseModel):
    artifacts_path: str
    min_data_for_training: int = Field(gt=0)
    historical_data_lookback_days: int = Field(gt=0)
    walk_forward_validation: WalkForwardValidationConfig
    lgbm_params: dict[str, Any]
    optimization: OptimizationConfig = Field(default_factory=lambda: OptimizationConfig())
    final_model: FinalModelConfig = Field(default_factory=FinalModelConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    drift_threshold_z_score: float = Field(default=3.0, gt=0.0)
    model_staleness_days: int = Field(default=7, gt=0)
    predictor: PredictorConfig = Field(default_factory=lambda: PredictorConfig())
    feature_engineering: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    class_label_mapping: dict[int, str]
    top_n_features: int = Field(gt=0)
    optimize_hyperparams: bool = False
    feature_ranges: Optional[dict[str, list[float]]] = None


class ExampleConfig(StrictBaseModel):
    data_generation: DataGenerationConfig
    sample_output: SampleOutputConfig


class SharpeRatioConfig(StrictBaseModel):
    enabled: bool


class WinRateConfig(StrictBaseModel):
    enabled: bool


class TradingCalendarConfig(StrictBaseModel):
    bars_per_year_15min: int = Field(gt=0)


class SpecialSession(StrictBaseModel):
    date: date
    open: time
    close: time


class MarketCalendarConfig(StrictBaseModel):
    holidays_cache_path: str
    holidays: list[str]
    market_open_time: str
    market_close_time: str
    muhurat_trading_sessions: list[SpecialSession] = Field(default_factory=list)
    half_day_sessions: list[SpecialSession] = Field(default_factory=list)


class AuthServerConfig(StrictBaseModel):
    max_retries: int = Field(gt=0)
    server_host: str
    timeout_seconds: int = Field(gt=0)
    open_browser: bool
    auto_close_seconds: int = Field(gt=0)


class StatisticsConfig(StrictBaseModel):
    sharpe_ratio: SharpeRatioConfig
    win_rate: WinRateConfig
    trading_calendar: TradingCalendarConfig


class DataQualityTimeSeriesConfig(StrictBaseModel):
    gap_multiplier: float = Field(gt=0)


class DataQualityOutlierDetectionConfig(StrictBaseModel):
    iqr_multiplier: float = Field(gt=0)
    handling_strategy: str = Field(default="clip", pattern="^(clip|remove|flag)$")
    min_iqr_data_points: int = Field(gt=0, default=4)


class DataQualityPenaltiesConfig(StrictBaseModel):
    outlier_penalty: float = Field(ge=0, le=1)
    gap_penalty: float = Field(ge=0, le=1)
    ohlc_violation_penalty: float = Field(ge=0, le=1)
    duplicate_penalty: float = Field(ge=0, le=1)


class InstrumentSegmentsConfig(StrictBaseModel):
    index_segments: list[str] = Field(default_factory=lambda: ["INDEX", "INDICES"])
    equity_segments: list[str] = Field(default_factory=lambda: ["EQ", "EQUITY"])
    fno_segments: list[str] = Field(default_factory=lambda: ["FUT", "OPT", "NFO-FUT", "NFO-OPT", "BFO-FUT", "BFO-OPT"])


class DataQualityValidationConfig(StrictBaseModel):
    enabled: bool
    min_valid_rows: int = Field(ge=0)
    quality_score_threshold: float = Field(ge=0, le=100)
    expected_columns: dict[str, str] = Field(default_factory=dict)
    required_columns: list[str] = Field(default_factory=list)
    instrument_segments: InstrumentSegmentsConfig = Field(default_factory=InstrumentSegmentsConfig)
    indicator_validation: Optional[dict[str, Any]] = None


class DataQualityFeatureValidationCustomRuleConfig(StrictBaseModel):
    pattern: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    priority: int = Field(default=5, ge=1, le=10)
    description: str = ""


class DataQualityFeatureValidationCustomRulesConfig(StrictBaseModel):
    range_rules: list[DataQualityFeatureValidationCustomRuleConfig] = Field(default_factory=list)


class DataQualityFeatureValidationOscillatorConsistencyConfig(StrictBaseModel):
    stochastic_volatility_ratio: float = Field(default=1.5, gt=0)
    rsi_correlation_threshold: float = Field(default=0.8, ge=0, le=1)


class DataQualityFeatureValidationBollingerConfig(StrictBaseModel):
    enabled: bool = True
    tolerance: float = Field(default=0.001, ge=0)


class DataQualityFeatureValidationMacdConfig(StrictBaseModel):
    enabled: bool = True
    histogram_tolerance: float = Field(default=0.001, ge=0)


class DataQualityFeatureValidationCrossValidationConfig(StrictBaseModel):
    enabled: bool = True
    oscillator_consistency: DataQualityFeatureValidationOscillatorConsistencyConfig = Field(
        default_factory=DataQualityFeatureValidationOscillatorConsistencyConfig
    )
    bollinger_validation: DataQualityFeatureValidationBollingerConfig = Field(
        default_factory=DataQualityFeatureValidationBollingerConfig
    )
    macd_validation: DataQualityFeatureValidationMacdConfig = Field(
        default_factory=DataQualityFeatureValidationMacdConfig
    )


class RulePrioritiesConfig(StrictBaseModel):
    model_training_range_parser: int = Field(default=10, gt=0)
    indicator_config_parser: int = Field(default=8, gt=0)
    custom_rule_parser: int = Field(default=7, gt=0)


class DataQualityFeatureValidationConfig(StrictBaseModel):
    enabled: bool = True
    auto_discovery: bool = True
    custom_rules: DataQualityFeatureValidationCustomRulesConfig = Field(
        default_factory=DataQualityFeatureValidationCustomRulesConfig
    )
    cross_validation: DataQualityFeatureValidationCrossValidationConfig = Field(
        default_factory=DataQualityFeatureValidationCrossValidationConfig
    )
    rule_priorities: RulePrioritiesConfig = Field(default_factory=RulePrioritiesConfig)
    consistency_rules: Optional[list[dict[str, Any]]] = None
    consistency_rule_defaults: Optional[dict[str, dict[str, Any]]] = None


class DataQualityConfig(StrictBaseModel):
    time_series: DataQualityTimeSeriesConfig
    outlier_detection: DataQualityOutlierDetectionConfig
    penalties: DataQualityPenaltiesConfig
    validation: DataQualityValidationConfig
    feature_validation: DataQualityFeatureValidationConfig = Field(default_factory=DataQualityFeatureValidationConfig)
    live_data_lpt_threshold: int = Field(gt=0)


class SchedulerConfig(StrictBaseModel):
    prediction_interval_minutes: int = Field(gt=0)
    readiness_check_time: str
    live_mode_sleep_duration: int = Field(default=3600, gt=0)
    prediction_timeframe: str = "15min"


class HealthMonitorConfig(StrictBaseModel):
    data_freshness_threshold_minutes: int = Field(gt=0)
    monitor_interval: int = Field(default=300, gt=0)


class CircuitBreakerConfig(StrictBaseModel):
    failure_threshold: int = Field(gt=0)
    recovery_timeout: int = Field(gt=0)
    half_open_attempts: int = Field(gt=0)


class TimeSynchronizerConfig(StrictBaseModel):
    candle_interval_minutes: int = Field(gt=0)
    latency_buffer_seconds: int = Field(ge=0)
    max_sync_attempts: int = Field(default=3, gt=0)
    sync_tolerance_seconds: float = Field(default=2.0, gt=0)


class CandleBufferConfig(StrictBaseModel):
    timeframes: list[int]
    persistence_path: str
    persistence_interval_seconds: int = Field(gt=0)


class SignalGenerationConfig(StrictBaseModel):
    buy_threshold: float = Field(ge=0, le=1)
    sell_threshold: float = Field(ge=0, le=1)


class DataPipelineConfig(StrictBaseModel):
    historical_data_max_days_per_request: int = Field(gt=0, le=60)

    @model_validator(mode="after")
    def check_historical_data_limit(self) -> "DataPipelineConfig":
        if self.historical_data_max_days_per_request > 60:
            raise ValueError(
                "historical_data_max_days_per_request cannot exceed 60 days due to KiteConnect API limits."
            )
        return self


class LiveAggregatorConfig(StrictBaseModel):
    max_partial_candles: int = Field(gt=0)
    partial_candle_cleanup_hours: int = Field(gt=0)
    health_check_success_rate_threshold: float = Field(ge=0, le=1)
    health_check_avg_processing_time_ms_threshold: float = Field(ge=0)
    health_check_validation_failures_threshold: float = Field(ge=0, le=1)
    required_tick_fields: list[str] = Field(default_factory=list)
    required_candle_fields: list[str] = Field(default_factory=list)


class ApiRateLimit(StrictBaseModel):
    limit: int = Field(gt=0)
    interval: int = Field(gt=0)


class ApiRateLimits(StrictBaseModel):
    historical_data: ApiRateLimit
    websocket_subscriptions: ApiRateLimit
    order_placement: ApiRateLimit
    general_api: ApiRateLimit


class MetricDefinition(StrictBaseModel):
    name: str
    type: Literal["counter", "gauge", "histogram"]
    description: str
    labels: list[str] = Field(default_factory=list)
    buckets: Optional[list[float]] = None
    dashboard_panel: str


class DashboardPanel(StrictBaseModel):
    name: str
    position: dict[str, Union[int, float]]


class DashboardConfig(StrictBaseModel):
    title: str
    refresh_interval: str = "5s"
    time_range: str = "6h"
    panels: list[DashboardPanel]


class MetricsConfig(StrictBaseModel):
    enabled: bool = True
    endpoint_port: int = Field(default=8001, gt=0, le=65535)
    endpoint_path: str = "/metrics"
    default_histogram_buckets: list[float] = Field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 5.0, 10.0])
    definitions: dict[str, list[MetricDefinition]]
    dashboard: DashboardConfig


class APIConfig(StrictBaseModel):
    """Configuration for API settings."""
    admin_token: str
    host: str = "127.0.0.1"


class ProcessingControlConfig(StrictBaseModel):
    """Configuration for processing control and state management."""

    enabled: bool = True
    force_reprocess: bool = False
    reset_state_on_startup: bool = False
    processing_mode: Literal["smart", "force", "skip"] = "smart"

    # Component-specific controls
    historical_processing_enabled: bool = True
    historical_max_gap_hours: int = Field(default=24, gt=0)
    historical_parallel_instruments: int = Field(default=3, gt=0)

    aggregation_processing_enabled: bool = True
    aggregation_reprocess_on_source_change: bool = True

    feature_processing_enabled: bool = True
    feature_incremental_calculation: bool = True
    feature_recompute_on_config_change: bool = True

    labeling_processing_enabled: bool = True
    labeling_batch_size: int = Field(default=1000, gt=0)
    
    # Data sufficiency thresholds for smart processing
    min_candles_for_historical: int = Field(default=100, gt=0)
    min_candles_for_aggregation: int = Field(default=100, gt=0)
    min_candles_for_features: int = Field(default=50, gt=0)
    min_candles_for_labeling: int = Field(default=200, gt=0)
    min_features_for_labeling: int = Field(default=100, gt=0)
    min_existing_features_threshold: int = Field(default=10, gt=0)
    min_existing_labels_threshold: int = Field(default=50, gt=0)


class AppConfig(StrictBaseModel):
    # Core required configurations
    system: SystemConfig
    broker: BrokerConfig
    logging: LoggingConfig
    trading: TradingConfig
    performance: PerformanceConfig
    data_quality: DataQualityConfig
    health_monitor: HealthMonitorConfig
    error_handler: CircuitBreakerConfig
    model_training: ModelTrainingConfig
    candle_buffer: CandleBufferConfig
    data_pipeline: DataPipelineConfig
    database: DatabaseConfig

    # Optional sections
    api_rate_limits: Optional[ApiRateLimits] = None
    storage: Optional[StorageConfig] = None
    example: Optional[ExampleConfig] = None
    statistics: Optional[StatisticsConfig] = None
    market_calendar: Optional[MarketCalendarConfig] = None
    scheduler: Optional[SchedulerConfig] = None
    time_synchronizer: Optional[TimeSynchronizerConfig] = None
    live_aggregator: Optional[LiveAggregatorConfig] = None
    signal_generation: Optional[SignalGenerationConfig] = None
    metrics: Optional[MetricsConfig] = None
    auth_server: Optional[AuthServerConfig] = None
    processing_control: Optional[ProcessingControlConfig] = None
    api: Optional[APIConfig] = None


class QueriesConfig(StrictBaseModel):
    chart_repo: dict[str, str]
    feature_repo: dict[str, str]
    instrument_repo: dict[str, str]
    label_repo: dict[str, str]
    migrate: dict[str, str]
    model_registry: dict[str, str]
    ohlcv_repo: dict[str, str]
    signal_repo: dict[str, str]
    label_stats_repo: dict[str, str]
    processing_state_repo: dict[str, str]


class ConfigLoader:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        metrics_path: Optional[str] = "config/metrics.yaml",
        queries_path: Optional[str] = "config/queries.yaml",
    ) -> None:
        self.config_path = config_path
        self.metrics_path = metrics_path
        self.queries_path = queries_path
        self._config: Optional[AppConfig] = None
        self._queries: Optional[QueriesConfig] = None
        self._config_lock = None
        self._backup_suffix = ".backup"
        self._temp_suffix = ".tmp"

    def get_config(self) -> AppConfig:
        if self._config is None:
            try:
                # Load main config
                with open(self.config_path, encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
                if config_data is None:
                    logger.error(f"ConfigurationError: Config file '{self.config_path}' is empty or invalid.")
                    raise ValueError("Config file is empty or invalid")

                # Load metrics config if it exists and path is provided
                if self.metrics_path is not None:
                    try:
                        with open(self.metrics_path, encoding="utf-8") as f:
                            metrics_data = yaml.safe_load(f)
                        if metrics_data and "metrics" in metrics_data:
                            config_data["metrics"] = metrics_data["metrics"]
                    except FileNotFoundError:
                        logger.warning(f"Metrics config file '{self.metrics_path}' not found. Proceeding without it.")
                        # Metrics config is optional
                    except yaml.YAMLError as e:
                        logger.error(f"Error parsing metrics config file '{self.metrics_path}': {e}")
                        raise ValueError(f"Error parsing metrics config file '{self.metrics_path}': {e}") from e

                # Load database configuration from environment variables using Pydantic BaseSettings
                db_config = DatabaseConfig()  # type: ignore[call-arg]  # nosec
                config_data["database"] = db_config.model_dump()

                # Load environment-based configurations
                # Broker configuration: credentials from environment, other settings from YAML
                if "broker" in config_data:
                    yaml_broker = config_data["broker"].copy()
                    # Load credentials from environment variables
                    import os

                    api_key = os.getenv("KITE_API_KEY")
                    api_secret = os.getenv("KITE_API_SECRET")

                    if not api_key:
                        logger.error("ConfigurationError: KITE_API_KEY environment variable is required but not set.")
                        raise ValueError("KITE_API_KEY environment variable is required but not set")
                    if not api_secret:
                        logger.error(
                            "ConfigurationError: KITE_API_SECRET environment variable is required but not set."
                        )
                        raise ValueError("KITE_API_SECRET environment variable is required but not set")

                    yaml_broker["api_key"] = api_key
                    yaml_broker["api_secret"] = api_secret
                    config_data["broker"] = yaml_broker
                else:
                    # No broker section in YAML - this should not happen in our case
                    logger.error(
                        "ConfigurationError: Broker configuration section is required in config.yaml but not found."
                    )
                    raise ValueError("Broker configuration section is required in config.yaml")

                self._config = AppConfig(**config_data)
            except FileNotFoundError:
                logger.error(f"ConfigurationError: Config file '{self.config_path}' not found.")
                raise FileNotFoundError(f"Configuration file '{self.config_path}' not found.") from None
            except ValidationError as e:
                # Provide a more user-friendly error message
                error_messages = []
                for error in e.errors():
                    loc = " -> ".join(map(str, error["loc"]))
                    msg = error["msg"]
                    error_messages.append(f"  - In section '{loc}': {msg}")
                error_str = "\n".join(error_messages)
                logger.error(
                    f"ConfigurationValidationError: Configuration validation failed for '{self.config_path}':\n{error_str}"
                )
                raise ValueError(f"Configuration validation failed:\n{error_str}") from e
            except Exception as e:
                logger.error(
                    f"ConfigurationError: Unexpected error loading config file '{self.config_path}': {e}", exc_info=True
                )
                raise Exception(f"Error loading config file '{self.config_path}': {e}") from e
        return self._config

    def save_config(self, config: AppConfig) -> None:
        """Save configuration to YAML file with atomic write and backup"""
        import shutil
        from pathlib import Path

        config_path = Path(self.config_path)
        backup_path = config_path.with_suffix(config_path.suffix + self._backup_suffix)
        temp_path = config_path.with_suffix(config_path.suffix + self._temp_suffix)

        try:
            # Create backup of current config
            if config_path.exists():
                shutil.copy2(config_path, backup_path)

            # Prepare config data for YAML serialization
            config_dict = self._prepare_config_for_yaml(config)

            # Write to temporary file first (atomic operation)
            with open(temp_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)

            # Atomic move to final location
            shutil.move(temp_path, config_path)

            # Update cached config
            self._config = config

        except Exception as e:
            # Cleanup temp file if it exists
            if temp_path.exists():
                temp_path.unlink()

            # Restore backup if save failed
            if backup_path.exists() and not config_path.exists():
                shutil.move(backup_path, config_path)

            raise Exception(f"Failed to save configuration: {e}") from e
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    def _prepare_config_for_yaml(self, config: AppConfig) -> dict[str, Any]:
        """Prepare config object for YAML serialization"""

        # Convert to dict and handle special cases
        config_dict = config.model_dump()

        # Remove sensitive data that should come from environment
        if "broker" in config_dict:
            broker_dict = config_dict["broker"].copy()
            # Remove credentials - they should always come from environment
            broker_dict.pop("api_key", None)
            broker_dict.pop("api_secret", None)
            config_dict["broker"] = broker_dict

        # Remove database config - it comes from environment
        config_dict.pop("database", None)

        # Handle date/time objects
        serialized = self._serialize_datetime_objects(config_dict)
        # We know this will be a dict since we started with config_dict (dict)
        return serialized if isinstance(serialized, dict) else {}

    def _serialize_datetime_objects(self, obj: Any) -> Union[dict[str, Any], list[Any], str, Any]:
        """Recursively serialize datetime objects to ISO format"""
        if isinstance(obj, dict):
            return {k: self._serialize_datetime_objects(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._serialize_datetime_objects(item) for item in obj]
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return obj

    def validate_config_section(self, section: str, data: dict[str, Any]) -> bool:
        """Validate configuration section data"""
        try:
            config = self.get_config()

            # Get the section model class
            section_field = config.model_fields.get(section)
            if not section_field:
                raise ValueError(f"Unknown configuration section: {section}")

            # Get the annotation (type) for this section
            section_type = section_field.annotation

            # Handle Optional types
            if section_type and hasattr(section_type, "__origin__") and section_type.__origin__ is Union:
                # Extract the non-None type from Optional[T]
                args = getattr(section_type, "__args__", None)
                if args:
                    section_type = next((arg for arg in args if arg is not type(None)), None)

            # Validate the data against the section model
            if section_type is not None:
                section_type(**data)
            return True

        except Exception as e:
            raise ValueError(f"Configuration validation failed for section '{section}': {e}") from e

    def update_config_section(self, section: str, data: dict[str, Any]) -> None:
        """Update a specific configuration section"""
        try:
            # Validate the section data first
            self.validate_config_section(section, data)

            # Load current config
            config = self.get_config()

            # Update the section
            current_section = getattr(config, section)

            # If current section is None (optional section), create new instance
            if current_section is None:
                section_field = config.model_fields.get(section)
                if section_field is not None:
                    section_type = section_field.annotation

                    # Handle Optional types
                    if section_type and hasattr(section_type, "__origin__") and section_type.__origin__ is Union:
                        args = getattr(section_type, "__args__", None)
                        if args:
                            section_type = next((arg for arg in args if arg is not type(None)), None)

                    if section_type is not None:
                        updated_section = section_type(**data)
                    else:
                        raise ValueError(f"Could not determine type for section '{section}'")
                else:
                    raise ValueError(f"Section '{section}' not found in config model")
            else:
                # Update existing section
                updated_section = current_section.model_copy(update=data)

            # Create new config with updated section
            updated_config = config.model_copy(update={section: updated_section})

            # Save the updated configuration
            self.save_config(updated_config)

        except Exception as e:
            raise Exception(f"Failed to update configuration section '{section}': {e}") from e

    def get_config_section(self, section: str) -> dict[str, Any]:
        """Get a specific configuration section as dict"""
        config = self.get_config()

        if not hasattr(config, section):
            raise ValueError(f"Unknown configuration section: {section}")

        section_obj = getattr(config, section)

        if section_obj is None:
            return {}

        if hasattr(section_obj, "model_dump"):
            result = section_obj.model_dump()
            return result if isinstance(result, dict) else {}
        if isinstance(section_obj, dict):
            return section_obj
        # Convert other types to dict representation
        return {"value": section_obj}

    def reload_config(self) -> AppConfig:
        """Reload configuration from file"""
        self._config = None
        return self.get_config()

    def backup_config(self) -> str:
        """Create a timestamped backup of current config"""
        import shutil
        from datetime import datetime
        from pathlib import Path

        config_path = Path(self.config_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config_path.with_suffix(f".{timestamp}.backup")

        if config_path.exists():
            shutil.copy2(config_path, backup_path)
            return str(backup_path)

        raise FileNotFoundError(f"Config file {config_path} not found")

    def restore_config(self, backup_path: str) -> None:
        """Restore configuration from backup"""
        import shutil
        from pathlib import Path

        backup_file = Path(backup_path)
        config_path = Path(self.config_path)

        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file {backup_path} not found")

        # Validate backup file before restoring
        try:
            with open(backup_file, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # Basic validation
            if not config_data or "system" not in config_data:
                raise ValueError("Invalid backup file format")

            # Copy backup to current config
            shutil.copy2(backup_file, config_path)

            # Reload configuration
            self.reload_config()

        except Exception as e:
            raise Exception(f"Failed to restore configuration from backup: {e}") from e

    def get_queries(self) -> QueriesConfig:
        if self._queries is None:
            try:
                with open(self.queries_path or "config/queries.yaml", encoding="utf-8") as f:
                    queries_data = yaml.safe_load(f)
                if queries_data is None:
                    raise ValueError("Queries file is empty or invalid")
                self._queries = QueriesConfig(**queries_data)
            except FileNotFoundError:
                raise FileNotFoundError(f"Queries file '{self.queries_path}' not found.") from None
            except ValidationError as e:
                error_messages = []
                for error in e.errors():
                    loc = " -> ".join(map(str, error["loc"]))
                    msg = error["msg"]
                    error_messages.append(f"  - In section '{loc}': {msg}")
                error_str = "\n".join(error_messages)
                raise ValueError(f"Queries validation failed:\n{error_str}") from e
            except Exception as e:
                raise Exception(f"Error loading queries file '{self.queries_path}': {e}") from e
        return self._queries


# Global instance for backward compatibility
config_loader = ConfigLoader()
