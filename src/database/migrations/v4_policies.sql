-- =================================================================
--  V4: DATA MANAGEMENT POLICIES
--  Applies compression and retention policies to all hypertables.
-- =================================================================

-- This migration should be run after all hypertables are created.
-- It uses proper checks to ensure idempotency.

-- Helper function to safely add compression policies.
CREATE OR REPLACE FUNCTION apply_compression_policy(hypertable_name TEXT, compress_after INTERVAL) 
RETURNS VOID AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_schema = 'public' AND hypertable_name = $1) THEN
        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.compression_settings WHERE hypertable_name = $1) THEN
            RAISE NOTICE 'Enabling compression on %...', $1;
            EXECUTE format('ALTER TABLE public.%I SET (
                timescaledb.compress,
                timescaledb.compress_orderby = ''ts DESC'',
                timescaledb.compress_segmentby = ''instrument_id''
            );', $1);
        END IF;

        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.jobs j JOIN timescaledb_information.hypertables h ON j.hypertable_id = h.id WHERE h.hypertable_name = $1 AND j.proc_name = 'policy_compression') THEN
            RAISE NOTICE 'Adding compression policy to %...', $1;
            EXECUTE format('SELECT add_compression_policy(%L, %L);', $1, compress_after);
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Helper function to safely add retention policies.
CREATE OR REPLACE FUNCTION apply_retention_policy(hypertable_name TEXT, drop_after INTERVAL) 
RETURNS VOID AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_schema = 'public' AND hypertable_name = $1) THEN
        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.jobs j JOIN timescaledb_information.hypertables h ON j.hypertable_id = h.id WHERE h.hypertable_name = $1 AND j.proc_name = 'policy_retention') THEN
            RAISE NOTICE 'Adding retention policy to %...', $1;
            EXECUTE format('SELECT add_retention_policy(%L, %L);', $1, drop_after);
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- =================================================================
-- 1. COMPRESSION POLICIES
-- =================================================================

-- OHLCV Tables: Compress older data to save space.
SELECT apply_compression_policy('ohlcv_1min', INTERVAL '1 day');
SELECT apply_compression_policy('ohlcv_5min', INTERVAL '7 days');
SELECT apply_compression_policy('ohlcv_15min', INTERVAL '15 days');
SELECT apply_compression_policy('ohlcv_60min', INTERVAL '30 days');

-- Feature Tables: Compress after a week.
SELECT apply_compression_policy('features', INTERVAL '7 days');
SELECT apply_compression_policy('engineered_features', INTERVAL '7 days');
SELECT apply_compression_policy('cross_asset_features', INTERVAL '7 days');
SELECT apply_compression_policy('feature_scores', INTERVAL '30 days');
SELECT apply_compression_policy('feature_selection_history', INTERVAL '30 days');

-- Label/Signal Tables: Compress after specified intervals.
SELECT apply_compression_policy('labels', INTERVAL '60 days');
SELECT apply_compression_policy('signals', INTERVAL '30 days');

-- =================================================================
-- 2. RETENTION POLICIES
-- =================================================================

-- OHLCV Tables: Keep high-frequency data for shorter periods.
SELECT apply_retention_policy('ohlcv_1min', INTERVAL '30 days');
SELECT apply_retention_policy('ohlcv_5min', INTERVAL '90 days');
SELECT apply_retention_policy('ohlcv_15min', INTERVAL '180 days');
SELECT apply_retention_policy('ohlcv_60min', INTERVAL '365 days');

-- Feature Tables: Retain for specified periods.
SELECT apply_retention_policy('features', INTERVAL '90 days');
SELECT apply_retention_policy('engineered_features', INTERVAL '90 days');
SELECT apply_retention_policy('cross_asset_features', INTERVAL '90 days');
SELECT apply_retention_policy('feature_scores', INTERVAL '180 days');
SELECT apply_retention_policy('feature_selection_history', INTERVAL '365 days');

-- Label/Signal Tables: Retain for a full year for long-term analysis.
SELECT apply_retention_policy('labels', INTERVAL '365 days');
SELECT apply_retention_policy('signals', INTERVAL '365 days');

-- Clean up helper functions
DROP FUNCTION apply_compression_policy(TEXT, INTERVAL);
DROP FUNCTION apply_retention_policy(TEXT, INTERVAL);
