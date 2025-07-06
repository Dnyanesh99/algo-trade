-- =================================================================
--  ADDITIONAL INDEXES FOR PERFORMANCE OPTIMIZATION
--  Applied after initial schema creation
-- =================================================================

-- Additional instrument indexes
CREATE INDEX IF NOT EXISTS idx_instruments_exchange_type ON instruments (exchange, instrument_type);
CREATE INDEX IF NOT EXISTS idx_instruments_last_updated ON instruments (last_updated);

-- OHLCV performance indexes
-- 1-min indexes are now in v1_initial.sql
CREATE INDEX IF NOT EXISTS idx_ohlcv_5min_ts_instrument ON ohlcv_5min (ts, instrument_id);
CREATE INDEX IF NOT EXISTS idx_ohlcv_15min_ts_instrument ON ohlcv_15min (ts, instrument_id);
CREATE INDEX IF NOT EXISTS idx_ohlcv_60min_ts_instrument ON ohlcv_60min (ts, instrument_id);

-- Features performance indexes
CREATE INDEX IF NOT EXISTS idx_features_ts_instrument ON features (ts, instrument_id);
CREATE INDEX IF NOT EXISTS idx_features_timeframe_name ON features (timeframe, feature_name);

-- Signals performance indexes
CREATE INDEX IF NOT EXISTS idx_signals_ts_instrument ON signals (ts, instrument_id);
CREATE INDEX IF NOT EXISTS idx_signals_confidence ON signals (confidence_score);

-- Labels performance indexes
CREATE INDEX IF NOT EXISTS idx_labels_ts_instrument ON labels (ts, instrument_id);
CREATE INDEX IF NOT EXISTS idx_labels_barrier_return ON labels (barrier_return);

-- Model registry indexes
CREATE INDEX IF NOT EXISTS idx_model_registry_created_at ON model_registry (created_at);
CREATE INDEX IF NOT EXISTS idx_model_registry_training_date ON model_registry (training_end_date);