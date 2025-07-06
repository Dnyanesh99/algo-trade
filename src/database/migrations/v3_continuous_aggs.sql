-- =================================================================
--  CONTINUOUS AGGREGATES FOR TIMESERIES OPTIMIZATION
--  Pre-calculated views for faster querying
-- =================================================================

-- Daily OHLCV aggregates from 15-minute data
CREATE MATERIALIZED VIEW daily_ohlcv_agg
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', ts) AS bucket,
    instrument_id,
    FIRST(open, ts) as daily_open,
    MAX(high) as daily_high,
    MIN(low) as daily_low,
    LAST(close, ts) as daily_close,
    SUM(volume) as daily_volume,
    LAST(oi, ts) as daily_oi
FROM ohlcv_15min
GROUP BY bucket, instrument_id;

-- Add policy to refresh continuous aggregate
SELECT add_continuous_aggregate_policy('daily_ohlcv_agg',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- Hourly feature aggregates
CREATE MATERIALIZED VIEW hourly_feature_agg
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', ts) AS bucket,
    instrument_id,
    timeframe,
    feature_name,
    AVG(feature_value) as avg_value,
    MIN(feature_value) as min_value,
    MAX(feature_value) as max_value,
    STDDEV(feature_value) as stddev_value,
    COUNT(*) as count_values
FROM features
WHERE timeframe IN ('5min', '15min')
GROUP BY bucket, instrument_id, timeframe, feature_name;

-- Add policy to refresh feature aggregates
SELECT add_continuous_aggregate_policy('hourly_feature_agg',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '30 minutes',
    schedule_interval => INTERVAL '30 minutes');

-- Daily signal performance aggregates
CREATE MATERIALIZED VIEW daily_signal_performance
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', ts) AS bucket,
    instrument_id,
    signal_type,
    direction,
    COUNT(*) as signal_count,
    AVG(confidence_score) as avg_confidence,
    MIN(confidence_score) as min_confidence,
    MAX(confidence_score) as max_confidence
FROM signals
GROUP BY bucket, instrument_id, signal_type, direction;

-- Add policy to refresh signal aggregates
SELECT add_continuous_aggregate_policy('daily_signal_performance',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '2 hours');