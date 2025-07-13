-- Migration V2: Feature Engineering Tables
-- Adds support for automated feature selection, engineering, and cross-asset features

-- 1. Engineered Features Table
CREATE TABLE IF NOT EXISTS engineered_features (
    id BIGSERIAL PRIMARY KEY,
    instrument_id INTEGER NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    feature_name VARCHAR(200) NOT NULL,
    feature_value DOUBLE PRECISION NOT NULL,
    generation_method VARCHAR(100) NOT NULL,
    source_features TEXT[] NOT NULL,
    quality_score DOUBLE PRECISION DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT engineered_features_timestamp_instrument_timeframe_feature_key 
    UNIQUE (timestamp, instrument_id, timeframe, feature_name)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('engineered_features', 'timestamp', 
    chunk_time_interval => INTERVAL '7 days');

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_engineered_features_instrument_timeframe_time 
ON engineered_features (instrument_id, timeframe, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_engineered_features_name_time 
ON engineered_features (feature_name, timestamp DESC);

-- 2. Feature Selection Scores Table
CREATE TABLE IF NOT EXISTS feature_scores (
    id BIGSERIAL PRIMARY KEY,
    instrument_id INTEGER NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    feature_name VARCHAR(200) NOT NULL,
    importance_score DOUBLE PRECISION NOT NULL,
    stability_score DOUBLE PRECISION NOT NULL,
    consistency_score DOUBLE PRECISION NOT NULL,
    composite_score DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    training_timestamp TIMESTAMPTZ NOT NULL,
    is_selected BOOLEAN DEFAULT FALSE,
    selection_rank INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT feature_scores_instrument_timeframe_feature_training_key 
    UNIQUE (instrument_id, timeframe, feature_name, training_timestamp)
);

-- Create hypertable for feature score history
SELECT create_hypertable('feature_scores', 'training_timestamp', 
    chunk_time_interval => INTERVAL '30 days');

-- Indexes for feature selection queries
CREATE INDEX IF NOT EXISTS idx_feature_scores_instrument_timeframe_selected 
ON feature_scores (instrument_id, timeframe, is_selected, composite_score DESC);

CREATE INDEX IF NOT EXISTS idx_feature_scores_feature_time 
ON feature_scores (feature_name, training_timestamp DESC);

-- 3. Cross-Asset Features Table
CREATE TABLE IF NOT EXISTS cross_asset_features (
    id BIGSERIAL PRIMARY KEY,
    primary_instrument_id INTEGER NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    feature_name VARCHAR(200) NOT NULL,
    feature_value DOUBLE PRECISION NOT NULL,
    calculation_method VARCHAR(100) NOT NULL,
    related_instruments INTEGER[] NOT NULL,
    correlation_window_days INTEGER NOT NULL,
    quality_score DOUBLE PRECISION DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT cross_asset_features_timestamp_instrument_timeframe_feature_key 
    UNIQUE (timestamp, primary_instrument_id, timeframe, feature_name)
);

-- Create hypertable for cross-asset features
SELECT create_hypertable('cross_asset_features', 'timestamp', 
    chunk_time_interval => INTERVAL '7 days');

-- Indexes for cross-asset feature queries
CREATE INDEX IF NOT EXISTS idx_cross_asset_features_primary_instrument_time 
ON cross_asset_features (primary_instrument_id, timeframe, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_cross_asset_features_related_instruments 
ON cross_asset_features USING GIN (related_instruments);

-- 4. Feature Lineage Table (Track dependencies)
CREATE TABLE IF NOT EXISTS feature_lineage (
    id BIGSERIAL PRIMARY KEY,
    feature_name VARCHAR(200) NOT NULL,
    feature_type VARCHAR(50) NOT NULL, -- 'base', 'engineered', 'cross_asset'
    source_features TEXT[] NOT NULL,
    generation_config JSONB,
    dependencies TEXT[] NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT feature_lineage_feature_name_key UNIQUE (feature_name)
);

-- Index for lineage queries
CREATE INDEX IF NOT EXISTS idx_feature_lineage_type ON feature_lineage (feature_type);
CREATE INDEX IF NOT EXISTS idx_feature_lineage_dependencies 
ON feature_lineage USING GIN (dependencies);

-- 5. Feature Selection History Table
CREATE TABLE IF NOT EXISTS feature_selection_history (
    id BIGSERIAL PRIMARY KEY,
    instrument_id INTEGER NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    selection_timestamp TIMESTAMPTZ NOT NULL,
    selected_features TEXT[] NOT NULL,
    selection_criteria JSONB NOT NULL,
    total_features_available INTEGER NOT NULL,
    features_selected INTEGER NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT feature_selection_history_instrument_timeframe_timestamp_key 
    UNIQUE (instrument_id, timeframe, selection_timestamp)
);

-- Create hypertable for selection history
SELECT create_hypertable('feature_selection_history', 'selection_timestamp', 
    chunk_time_interval => INTERVAL '30 days');

-- Index for selection history queries
CREATE INDEX IF NOT EXISTS idx_feature_selection_history_instrument_timeframe_time 
ON feature_selection_history (instrument_id, timeframe, selection_timestamp DESC);

-- 6. Continuous Aggregates for Performance
CREATE MATERIALIZED VIEW IF NOT EXISTS feature_importance_summary
WITH (timescaledb.continuous) AS
SELECT 
    instrument_id,
    timeframe,
    feature_name,
    time_bucket('1 day', training_timestamp) AS day,
    AVG(composite_score) AS avg_score,
    STDDEV(composite_score) AS score_stability,
    COUNT(*) AS training_sessions,
    MAX(composite_score) AS best_composite_score
FROM feature_scores
GROUP BY instrument_id, timeframe, feature_name, day;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('feature_importance_summary',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- 7. Performance Views for Quick Access
CREATE OR REPLACE VIEW current_selected_features AS
SELECT DISTINCT ON (fs.instrument_id, fs.timeframe, fs.feature_name)
    fs.instrument_id,
    fs.timeframe,
    fs.feature_name,
    fs.composite_score,
    fs.is_selected,
    fs.selection_rank,
    fs.training_timestamp
FROM feature_scores fs
WHERE fs.is_selected = TRUE
ORDER BY fs.instrument_id, fs.timeframe, fs.feature_name, fs.training_timestamp DESC;

CREATE OR REPLACE VIEW feature_performance_metrics AS
SELECT 
    instrument_id,
    timeframe,
    feature_name,
    AVG(composite_score) as avg_importance,
    STDDEV(composite_score) as importance_stability,
    AVG(composite_score) as avg_model_performance,
    COUNT(*) as selection_frequency,
    MAX(training_timestamp) as last_updated
FROM feature_scores
WHERE training_timestamp > NOW() - INTERVAL '30 days'
GROUP BY instrument_id, timeframe, feature_name
ORDER BY avg_importance DESC;

-- 8. Enable compression settings for feature engineering tables
-- Policies will be applied in v4_policies.sql
ALTER TABLE engineered_features SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id', timescaledb.compress_orderby = 'timestamp DESC');
ALTER TABLE cross_asset_features SET (timescaledb.compress, timescaledb.compress_segmentby = 'primary_instrument_id', timescaledb.compress_orderby = 'timestamp DESC');
ALTER TABLE feature_scores SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id', timescaledb.compress_orderby = 'training_timestamp DESC');
ALTER TABLE feature_selection_history SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id', timescaledb.compress_orderby = 'selection_timestamp DESC');