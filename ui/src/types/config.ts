export interface SystemConfig {
  version: string
  mode: 'HISTORICAL_MODE' | 'LIVE_MODE'
  timezone: string
  openmp_threads: number
  mkl_threads: number
  numexpr_threads: number
}

export interface BrokerConfig {
  redirect_url: string
  websocket_mode: 'LTP' | 'QUOTE' | 'FULL'
  historical_interval: string
  should_fetch_instruments: boolean
  historical_data_lookback_days: number
  segment_types: string[]
  connection_manager: {
    max_reconnect_attempts: number
    initial_reconnect_delay: number
    heartbeat_timeout: number
    monitor_interval: number
  }
}

export interface TradingConfig {
  aggregation_timeframes: number[]
  feature_timeframes: number[]
  labeling_timeframes: number[]
  labeling: LabelingConfig
  features: FeaturesConfig
  instruments: InstrumentConfig[]
}

export interface LabelingConfig {
  atr_period: number
  tp_atr_multiplier: number
  sl_atr_multiplier: number
  atr_smoothing: 'ema' | 'sma'
  max_holding_periods: number
  min_bars_required: number
  epsilon: number
  use_dynamic_barriers: boolean
  use_dynamic_epsilon: boolean
  volatility_lookback: number
  dynamic_epsilon: DynamicEpsilonConfig
  atr_cache_size: number
  dynamic_barrier_tp_sensitivity: number
  dynamic_barrier_sl_sensitivity: number
  sample_weight_decay_factor: number
  ohlcv_data_limit_for_labeling: number
  minimum_ohlcv_data_for_labeling: number
}

export interface DynamicEpsilonConfig {
  low_volatility_multiplier: number
  normal_volatility_multiplier: number
  high_volatility_multiplier: number
  extreme_volatility_multiplier: number
  low_volatility_threshold: number
  high_volatility_threshold: number
  extreme_volatility_threshold: number
  market_open_multiplier: number
  pre_lunch_multiplier: number
  market_close_multiplier: number
  monday_multiplier: number
  friday_multiplier: number
  extreme_zscore_threshold: number
  moderate_zscore_threshold: number
  stable_zscore_threshold: number
  extreme_zscore_multiplier: number
  moderate_zscore_multiplier: number
  stable_zscore_multiplier: number
  min_epsilon_multiplier: number
  max_epsilon_multiplier: number
  weekly_expiry_day: number
  monthly_expiry_week: number
  expiry_volatility_multiplier: number
}

export interface FeaturesConfig {
  lookback_period: number
  configurations: IndicatorConfig[]
  name_mapping: Record<string, string>
}

export interface IndicatorConfig {
  name: string
  function: string
  params: {
    default: Record<string, number | string | boolean | number[]>
    [timeframe: string]: Record<string, number | string | boolean | number[]>
  }
}

export interface InstrumentConfig {
  label: string
  tradingsymbol: string
  exchange: string
  instrument_type: string
  segment: string
}

export interface ModelTrainingConfig {
  artifacts_path: string
  historical_data_lookback_days: number
  min_data_for_training: number
  confidence_threshold: number
  drift_threshold_z_score: number
  model_staleness_days: number
  optimize_hyperparams: boolean
  top_n_features: number
  predictor: PredictorConfig
  lgbm_params: Record<string, number | string | boolean | number[]>
  optimization: OptimizationConfig
  walk_forward_validation: WalkForwardValidationConfig
  final_model: FinalModelConfig
  retention: RetentionConfig
  class_label_mapping: Record<string, string>
  feature_ranges: Record<string, number[]>
  feature_engineering: FeatureEngineeringConfig
}

export interface PredictorConfig {
  required_artifacts: Record<string, string>
  prediction_history_window_small: number
  prediction_history_window_large: number
  model_staleness_max_days: number
  real_time_accuracy_windows: number[]
  confidence_degradation_threshold: number
  error_rate_threshold: number
}

export interface OptimizationConfig {
  enabled: boolean
  n_trials: number
  test_size: number
  n_estimators_range: number[]
  learning_rate_range: number[]
  num_leaves_range: number[]
  max_depth_range: number[]
  min_child_samples_range: number[]
  feature_fraction_range: number[]
  bagging_fraction_range: number[]
  bagging_freq_range: number[]
  lambda_l1_range: number[]
  lambda_l2_range: number[]
  roc_auc_multi_class: string
  roc_auc_average: string
  early_stopping_rounds: number
}

export interface WalkForwardValidationConfig {
  initial_train_ratio: number
  validation_ratio: number
  step_ratio: number
  n_splits: number
}

export interface FinalModelConfig {
  num_boost_round: number
}

export interface RetentionConfig {
  max_model_versions: number
}

export interface FeatureEngineeringConfig {
  enabled: boolean
  max_features_per_instrument: number
  quality_threshold: number
  feature_generation: FeatureGenerationConfig
  feature_selection: FeatureSelectionConfig
  cross_asset: CrossAssetConfig
}

export interface FeatureGenerationConfig {
  enabled: boolean
  patterns: string[]
  lookback_periods: number[]
  min_quality_score: number
}

export interface FeatureSelectionConfig {
  enabled: boolean
  target_feature_count: number
  importance_history_length: number
  correlation_threshold: number
  correlation_data_lookback_multiplier: number
  stability_weight: number
  consistency_weight: number
  importance_weight: number
  min_selection_frequency: number
}

export interface CrossAssetConfig {
  enabled: boolean
  correlation_lookback_days: number
  minimum_correlation_threshold: number
  update_frequency_minutes: number
  max_related_instruments: number
  cache_duration_minutes: number
  instruments: Record<string, string>
}

export interface SignalGenerationConfig {
  buy_threshold: number
  sell_threshold: number
}

export interface SchedulerConfig {
  prediction_interval_minutes: number
  readiness_check_time: string
  live_mode_sleep_duration: number
  prediction_timeframe: string
}

export interface MarketCalendarConfig {
  holidays_cache_path: string
  holidays: string[]
  market_open_time: string
  market_close_time: string
  muhurat_trading_sessions: TradingSession[]
  half_day_sessions: TradingSession[]
}

export interface TradingSession {
  date: string
  open: string
  close: string
}

export interface AppConfig {
  system: SystemConfig
  broker: BrokerConfig
  trading: TradingConfig
  model_training: ModelTrainingConfig
  signal_generation: SignalGenerationConfig
  scheduler: SchedulerConfig
  market_calendar: MarketCalendarConfig
  // Add other config sections as needed
}