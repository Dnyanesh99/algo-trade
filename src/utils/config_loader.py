from datetime import date, time
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Base model for all configuration classes to enforce strict validation
class StrictBaseModel(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")


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
    exchange_types: list[str] = Field(default_factory=list)
    connection_manager: ConnectionManagerConfig


class LoggingConfig(StrictBaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    format: str
    file: str
    rotation: str = "10 MB"
    retention: str = "30 days"
    compression: str = "zip"


class LabelingConfig(StrictBaseModel):
    atr_period: int = Field(gt=0)
    tp_atr_multiplier: float = Field(gt=0)
    sl_atr_multiplier: float = Field(gt=0)
    max_holding_periods: int = Field(gt=0)
    close_open_threshold: float = Field(ge=0, le=1)
    min_bars_required: int = Field(gt=0)
    epsilon: float = Field(gt=0)
    atr_smoothing: Literal["ema", "sma"]
    use_dynamic_barriers: bool = True
    volatility_lookback: int = Field(default=20, gt=0)
    max_position_size: float = Field(default=1.0, gt=0)
    min_price_change: float = Field(default=0.001, gt=0)
    correlation_threshold: float = Field(default=0.7, ge=0, le=1)
    min_return_threshold: float = Field(default=0.001, gt=0)
    barrier_adjustment_factor: float = Field(default=1.0, gt=0)
    ohlcv_data_limit_for_labeling: int = Field(default=1000, gt=0)
    minimum_ohlcv_data_for_labeling: int = Field(default=100, gt=0)
    label_threshold: float = Field(default=0.5, ge=0, le=1)

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
    lgbm_params: dict
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


class DataQualityPenaltiesConfig(StrictBaseModel):
    outlier_penalty: float = Field(ge=0, le=1)
    gap_penalty: float = Field(ge=0, le=1)
    ohlc_violation_penalty: float = Field(ge=0, le=1)
    duplicate_penalty: float = Field(ge=0, le=1)


class DataQualityValidationConfig(StrictBaseModel):
    enabled: bool
    min_valid_rows: int = Field(ge=0)
    quality_score_threshold: float = Field(ge=0, le=100)
    expected_columns: dict[str, str] = Field(default_factory=dict)
    required_columns: list[str] = Field(default_factory=list)


class DataQualityConfig(StrictBaseModel):
    time_series: DataQualityTimeSeriesConfig
    outlier_detection: DataQualityOutlierDetectionConfig
    penalties: DataQualityPenaltiesConfig
    validation: DataQualityValidationConfig
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


class QueriesConfig(StrictBaseModel):
    feature_repo: dict[str, str]
    instrument_repo: dict[str, str]
    label_repo: dict[str, str]
    migrate: dict[str, str]
    ohlcv_repo: dict[str, str]
    signal_repo: dict[str, str]


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

    def get_config(self) -> AppConfig:
        if self._config is None:
            try:
                # Load main config
                with open(self.config_path, encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
                if config_data is None:
                    raise ValueError("Config file is empty or invalid")

                # Load metrics config if it exists and path is provided
                if self.metrics_path is not None:
                    try:
                        with open(self.metrics_path, encoding="utf-8") as f:
                            metrics_data = yaml.safe_load(f)
                        if metrics_data and "metrics" in metrics_data:
                            config_data["metrics"] = metrics_data["metrics"]
                    except FileNotFoundError:
                        # Metrics config is optional
                        pass

                # Load environment-based configurations
                # Broker configuration: credentials from environment, other settings from YAML
                if "broker" in config_data:
                    yaml_broker = config_data["broker"].copy()
                    # Load credentials from environment variables
                    import os

                    api_key = os.getenv("KITE_API_KEY")
                    api_secret = os.getenv("KITE_API_SECRET")

                    if not api_key:
                        raise ValueError("KITE_API_KEY environment variable is required but not set")
                    if not api_secret:
                        raise ValueError("KITE_API_SECRET environment variable is required but not set")

                    yaml_broker["api_key"] = api_key
                    yaml_broker["api_secret"] = api_secret
                    config_data["broker"] = yaml_broker
                else:
                    # No broker section in YAML - this should not happen in our case
                    raise ValueError("Broker configuration section is required in config.yaml")

                self._config = AppConfig(**config_data)

                self._config = AppConfig(**config_data)
            except FileNotFoundError:
                raise FileNotFoundError(f"Configuration file '{self.config_path}' not found.") from None
            except ValidationError as e:
                # Provide a more user-friendly error message
                error_messages = []
                for error in e.errors():
                    loc = " -> ".join(map(str, error["loc"]))
                    msg = error["msg"]
                    error_messages.append(f"  - In section '{loc}': {msg}")
                error_str = "\n".join(error_messages)
                raise ValueError(f"Configuration validation failed:\n{error_str}") from e
            except Exception as e:
                raise Exception(f"Error loading config file '{self.config_path}': {e}") from e
        return self._config

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
