from datetime import date, time
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SystemConfig(BaseModel):
    version: str
    mode: Literal["HISTORICAL_MODE", "LIVE_MODE"]
    timezone: str


class ConnectionManagerConfig(BaseModel):
    max_reconnect_attempts: int = Field(gt=0)
    initial_reconnect_delay: int = Field(gt=0)
    heartbeat_timeout: int = Field(gt=0)
    monitor_interval: int = Field(gt=0)


class BrokerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KITE_", extra="ignore")  # KITE_API_KEY, KITE_API_SECRET
    api_key: str
    api_secret: str
    redirect_url: str
    websocket_mode: Literal["LTP", "QUOTE", "FULL"]
    historical_interval: Literal["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute", "day"]
    connection_manager: ConnectionManagerConfig


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    format: str
    file: str
    rotation: str = "10 MB"
    retention: str = "30 days"
    compression: str = "zip"


class LabelingConfig(BaseModel):
    atr_period: int = Field(..., gt=0)
    tp_atr_multiplier: float = Field(..., gt=0)
    sl_atr_multiplier: float = Field(..., gt=0)
    max_holding_periods: int = Field(..., gt=0)
    close_open_threshold: float = Field(..., ge=0, le=1)
    min_bars_required: int = Field(..., gt=0)
    epsilon: float = Field(..., gt=0)
    atr_smoothing: Literal["ema", "sma"]

    @model_validator(mode="after")
    def check_atr_period_and_min_bars(self) -> "LabelingConfig":
        if self.min_bars_required < self.atr_period:
            raise ValueError("min_bars_required cannot be less than atr_period")
        return self


class FeatureConfig(BaseModel):
    enabled: bool = True
    period: Optional[int] = Field(None, gt=0)
    fast_period: Optional[int] = Field(None, gt=0)
    slow_period: Optional[int] = Field(None, gt=0)
    signal_period: Optional[int] = Field(None, gt=0)
    std_dev: Optional[float] = Field(None, gt=0)


class TradingFeaturesConfig(BaseModel):
    lookback_period: int = Field(..., gt=0)
    rsi: Optional[FeatureConfig] = None
    macd: Optional[FeatureConfig] = None
    bollinger_bands: Optional[FeatureConfig] = None
    atr: Optional[FeatureConfig] = None
    ema_fast: Optional[FeatureConfig] = None
    ema_slow: Optional[FeatureConfig] = None
    sma: Optional[FeatureConfig] = None
    volume_sma: Optional[FeatureConfig] = None


class TradingInstrumentConfig(BaseModel):
    label: str
    tradingsymbol: str
    exchange: str
    instrument_type: str


class TradingConfig(BaseModel):
    labeling: LabelingConfig
    aggregation_timeframes: list[int] = Field(default_factory=lambda: [5, 15, 60])
    feature_timeframes: list[int] = Field(default_factory=lambda: [5, 15, 60])
    features: TradingFeaturesConfig
    instruments: list[TradingInstrumentConfig]


class ProcessingConfig(BaseModel):
    chunk_size: int = Field(..., gt=0)
    max_workers: int = Field(..., gt=0)
    parallel_timeout_seconds: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)  # For historical aggregator
    feature_batch_size: int = Field(..., gt=0)  # For feature calculator
    tick_queue_max_size: int = Field(..., gt=0)


class MemoryConfig(BaseModel):
    limit_percentage: float = Field(..., gt=0, le=1)
    safety_factor: float = Field(..., gt=0, le=1)
    low_memory_threshold_gb: float = Field(..., ge=0)


class CacheConfig(BaseModel):
    max_size: int = Field(..., gt=0)
    cleanup_batch_size: int = Field(..., gt=0)


class MetricsConfig(BaseModel):
    enabled: bool


class PerformanceConfig(BaseModel):
    processing: ProcessingConfig
    memory: MemoryConfig
    cache: CacheConfig
    metrics: MetricsConfig
    max_retries: int = Field(..., gt=0)


class ParquetConfig(BaseModel):
    compression: str
    use_dictionary: bool
    version: str


class OutputConfig(BaseModel):
    save_statistics: bool
    statistics_format: Literal["json", "yaml"]


class MetadataConfig(BaseModel):
    include_config: bool
    include_checksum: bool
    checksum_length: int = Field(..., gt=0)


class StorageConfig(BaseModel):
    parquet: ParquetConfig
    output: OutputConfig
    metadata: MetadataConfig


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")
    host: str
    port: int
    dbname: str
    user: str
    password: str
    min_connections: int = Field(..., gt=0)
    max_connections: int = Field(..., gt=0)
    timeout: int = Field(..., gt=0)


class DataGenerationConfig(BaseModel):
    random_seed: int
    start_date: str
    end_date: str
    frequency: str
    initial_price: float = Field(..., gt=0)
    return_mean: float
    return_std: float = Field(..., ge=0)
    price_noise_std: float = Field(..., ge=0)
    high_noise_std: float = Field(..., ge=0)
    low_noise_std: float = Field(..., ge=0)
    volume_log_mean: float
    volume_log_std: float = Field(..., ge=0)


class SampleOutputConfig(BaseModel):
    file_name: str
    rows_to_save: int = Field(..., gt=0)


class WalkForwardValidationConfig(BaseModel):
    initial_train_size: int = Field(..., gt=0)
    validation_size: int = Field(..., gt=0)
    step_size: int = Field(..., gt=0)


class ModelTrainingConfig(BaseModel):
    artifacts_path: str
    min_data_for_training: int = Field(..., gt=0)
    historical_data_lookback_days: int = Field(..., gt=0)
    walk_forward_validation: WalkForwardValidationConfig
    lgbm_params: dict


class ExampleConfig(BaseModel):
    data_generation: DataGenerationConfig
    sample_output: SampleOutputConfig


class SharpeRatioConfig(BaseModel):
    enabled: bool


class WinRateConfig(BaseModel):
    enabled: bool


class TradingCalendarConfig(BaseModel):
    bars_per_year_15min: int = Field(..., gt=0)


class SpecialSession(BaseModel):
    date: date
    open: time
    close: time


class MarketCalendarConfig(BaseModel):
    holidays: list[str]
    market_open_time: str
    market_close_time: str
    muhurat_trading_sessions: list[SpecialSession] = Field(default_factory=list)
    half_day_sessions: list[SpecialSession] = Field(default_factory=list)


class StatisticsConfig(BaseModel):
    sharpe_ratio: SharpeRatioConfig
    win_rate: WinRateConfig
    trading_calendar: TradingCalendarConfig


class DataQualityTimeSeriesConfig(BaseModel):
    gap_multiplier: float = Field(..., gt=0)


class DataQualityOutlierDetectionConfig(BaseModel):
    iqr_multiplier: float = Field(..., gt=0)


class DataQualityPenaltiesConfig(BaseModel):
    outlier_penalty: float = Field(..., ge=0, le=1)
    gap_penalty: float = Field(..., ge=0, le=1)
    ohlc_violation_penalty: float = Field(..., ge=0, le=1)
    duplicate_penalty: float = Field(..., ge=0, le=1)


class DataQualityValidationConfig(BaseModel):
    enabled: bool
    min_valid_rows: int = Field(..., ge=0)
    quality_score_threshold: float = Field(..., ge=0, le=100)


class DataQualityConfig(BaseModel):
    time_series: DataQualityTimeSeriesConfig
    outlier_detection: DataQualityOutlierDetectionConfig
    penalties: DataQualityPenaltiesConfig
    validation: DataQualityValidationConfig
    live_data_lpt_threshold: int = Field(..., gt=0)


class SchedulerConfig(BaseModel):
    prediction_interval_minutes: int = Field(..., gt=0)
    readiness_check_time: str


class HealthMonitorConfig(BaseModel):
    data_freshness_threshold_minutes: int = Field(..., gt=0)
    monitor_interval: int = Field(..., gt=0)


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = Field(..., gt=0)
    recovery_timeout: int = Field(..., gt=0)
    half_open_attempts: int = Field(..., gt=0)


class TimeSynchronizerConfig(BaseModel):
    candle_interval_minutes: int = Field(..., gt=0)
    latency_buffer_seconds: int = Field(..., ge=0)
    max_sync_attempts: int = Field(3, gt=0)
    sync_tolerance_seconds: float = Field(2.0, gt=0)


class CandleBufferConfig(BaseModel):
    timeframes: list[int]
    persistence_path: str
    persistence_interval_seconds: int = Field(..., gt=0)


class SignalGenerationConfig(BaseModel):
    buy_threshold: float = Field(..., ge=0, le=1)
    sell_threshold: float = Field(..., ge=0, le=1)


class DataPipelineConfig(BaseModel):
    historical_data_max_days_per_request: int = Field(..., gt=0, le=60)

    @model_validator(mode="after")
    def check_historical_data_limit(self) -> "DataPipelineConfig":
        if self.historical_data_max_days_per_request > 60:
            raise ValueError(
                "historical_data_max_days_per_request cannot exceed 60 days due to KiteConnect API limits."
            )
        return self


class LiveAggregatorConfig(BaseModel):
    max_partial_candles: int = Field(..., gt=0)
    partial_candle_cleanup_hours: int = Field(..., gt=0)
    health_check_success_rate_threshold: float = Field(..., ge=0, le=1)  # Changed to 0-1
    health_check_avg_processing_time_ms_threshold: float = Field(..., ge=0)
    health_check_validation_failures_threshold: float = Field(..., ge=0, le=1)


class ApiRateLimit(BaseModel):
    limit: int = Field(gt=0)
    interval: int = Field(gt=0)


class ApiRateLimits(BaseModel):
    historical_data: ApiRateLimit
    websocket_subscriptions: ApiRateLimit
    order_placement: ApiRateLimit
    general_api: ApiRateLimit


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(yaml_file="config/config.yaml", yaml_file_encoding="utf-8")

    system: SystemConfig
    broker: BrokerConfig
    logging: LoggingConfig
    trading: TradingConfig
    performance: PerformanceConfig
    api_rate_limits: ApiRateLimits
    storage: StorageConfig
    database: DatabaseConfig
    example: ExampleConfig
    statistics: StatisticsConfig
    data_quality: DataQualityConfig
    market_calendar: MarketCalendarConfig
    scheduler: SchedulerConfig
    time_synchronizer: TimeSynchronizerConfig
    data_pipeline: DataPipelineConfig
    health_monitor: HealthMonitorConfig
    error_handler: CircuitBreakerConfig
    model_training: ModelTrainingConfig
    live_aggregator: LiveAggregatorConfig
    signal_generation: SignalGenerationConfig
    candle_buffer: CandleBufferConfig


class ConfigLoader:
    _instance = None
    _config: Optional[AppConfig] = None

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_config(self) -> AppConfig:
        if self._config is None:
            try:
                self._config = AppConfig()
            except ValidationError as e:
                raise ValueError(f"Configuration validation error: {e}") from e
            except Exception as e:
                raise Exception(f"Error loading config file: {e}") from e
        return self._config


config_loader = ConfigLoader()
