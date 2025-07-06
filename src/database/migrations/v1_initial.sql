-- =================================================================
--  1. INSTRUMENTS TABLE
--  Stores metadata for all financial instruments.
-- =================================================================

CREATE TABLE instruments (
    instrument_id SERIAL PRIMARY KEY,
    instrument_token BIGINT NOT NULL,
    exchange_token TEXT,
    tradingsymbol TEXT NOT NULL,
    name TEXT,
    last_price DOUBLE PRECISION,
    expiry DATE,
    strike DOUBLE PRECISION,
    tick_size DOUBLE PRECISION,
    lot_size INTEGER,
    instrument_type TEXT,
    segment TEXT,
    exchange TEXT NOT NULL,
    last_updated TIMESTAMPTZ NOT NULL,
    -- Unique constraint to prevent duplicate instrument definitions
    UNIQUE (exchange, tradingsymbol, expiry, strike, instrument_type)
);

-- Indexes for faster lookups
CREATE INDEX idx_instruments_token ON instruments (instrument_token);
CREATE INDEX idx_instruments_tradingsymbol ON instruments (tradingsymbol);
CREATE INDEX idx_instruments_expiry ON instruments (expiry);
CREATE INDEX idx_instruments_type ON instruments (instrument_type);

-- =================================================================
--  2. OHLCV DATA TABLES (HYPERTABLES)
--  Separate tables for each candle interval.
-- =================================================================

-- 1-Minute OHLCV Table (Live data from WebSocket)
CREATE TABLE ohlcv_1min (
    ts TIMESTAMPTZ NOT NULL,
    instrument_id INTEGER NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    oi BIGINT,
    CONSTRAINT fk_instrument_1min FOREIGN KEY (instrument_id) REFERENCES instruments (instrument_id),
    PRIMARY KEY (instrument_id, ts)
);
SELECT create_hypertable('ohlcv_1min', 'ts', chunk_time_interval => INTERVAL '1 hour');

-- 5-Minute OHLCV Table
CREATE TABLE ohlcv_5min (
    ts TIMESTAMPTZ NOT NULL,
    instrument_id INTEGER NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    oi BIGINT,
    CONSTRAINT fk_instrument_5min FOREIGN KEY (instrument_id) REFERENCES instruments (instrument_id),
    PRIMARY KEY (instrument_id, ts)
);
SELECT create_hypertable('ohlcv_5min', 'ts', chunk_time_interval => INTERVAL '1 day');

-- 15-Minute OHLCV Table
CREATE TABLE ohlcv_15min (
    ts TIMESTAMPTZ NOT NULL,
    instrument_id INTEGER NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    oi BIGINT,
    CONSTRAINT fk_instrument_15min FOREIGN KEY (instrument_id) REFERENCES instruments (instrument_id),
    PRIMARY KEY (instrument_id, ts)
);
SELECT create_hypertable('ohlcv_15min', 'ts', chunk_time_interval => INTERVAL '1 day');

-- 60-Minute OHLCV Table
CREATE TABLE ohlcv_60min (
    ts TIMESTAMPTZ NOT NULL,
    instrument_id INTEGER NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    oi BIGINT,
    CONSTRAINT fk_instrument_60min FOREIGN KEY (instrument_id) REFERENCES instruments (instrument_id),
    PRIMARY KEY (instrument_id, ts)
);
SELECT create_hypertable('ohlcv_60min', 'ts', chunk_time_interval => INTERVAL '7 days');

-- =================================================================
--  3. COMPRESSION POLICIES FOR OHLCV TABLES
--  To optimize storage and query speed on historical data.
-- =================================================================

-- Compression for 1-min data after 1 day (most recent data)
ALTER TABLE ohlcv_1min SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id', timescaledb.compress_orderby = 'ts DESC');
SELECT add_compression_policy('ohlcv_1min', INTERVAL '1 day');

-- Compression for 5-min data after 7 days
ALTER TABLE ohlcv_5min SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id', timescaledb.compress_orderby = 'ts DESC');
SELECT add_compression_policy('ohlcv_5min', INTERVAL '7 days');

-- Compression for 15-min data after 15 days
ALTER TABLE ohlcv_15min SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id', timescaledb.compress_orderby = 'ts DESC');
SELECT add_compression_policy('ohlcv_15min', INTERVAL '15 days');

-- Compression for 60-min data after 30 days
ALTER TABLE ohlcv_60min SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id', timescaledb.compress_orderby = 'ts DESC');
SELECT add_compression_policy('ohlcv_60min', INTERVAL '30 days');


-- =================================================================
--  4. CALCULATED FEATURES TABLE (HYPERTABLE)
--  Stores derived metrics from OHLCV data.
-- =================================================================

CREATE TABLE features (
    ts TIMESTAMPTZ NOT NULL,
    instrument_id INTEGER NOT NULL,
    timeframe TEXT NOT NULL, -- e.g., '5min', '15min', '60min', '1day'
    feature_name TEXT NOT NULL,
    feature_value DOUBLE PRECISION NOT NULL,
    CONSTRAINT fk_instrument_feature FOREIGN KEY (instrument_id) REFERENCES instruments (instrument_id),
    PRIMARY KEY (instrument_id, timeframe, feature_name, ts)
);

SELECT create_hypertable('features', 'ts', chunk_time_interval => INTERVAL '7 days');
CREATE INDEX idx_features_name_timeframe ON features (feature_name, timeframe);


-- =================================================================
--  5. TRADING SIGNALS TABLE (HYPERTABLE)
--  Stores generated entry/exit signals.
-- =================================================================

CREATE TABLE signals (
    ts TIMESTAMPTZ NOT NULL,
    instrument_id INTEGER NOT NULL,
    signal_type TEXT NOT NULL, -- e.g., 'Entry', 'Exit', 'Trend_Change'
    direction TEXT NOT NULL, -- e.g., 'Long', 'Short', 'Neutral'
    confidence_score DOUBLE PRECISION, -- Score from 0.0 to 1.0
    source_feature_name TEXT,
    price_at_signal DECIMAL(10,2) NOT NULL,
    source_feature_value DOUBLE PRECISION,
    details JSONB, -- For extra context or parameters
    CONSTRAINT fk_instrument_signal FOREIGN KEY (instrument_id) REFERENCES instruments (instrument_id),
    PRIMARY KEY (instrument_id, signal_type, ts)
);

-- Create the hypertable and indexes
SELECT create_hypertable('signals', 'ts', chunk_time_interval => INTERVAL '7 days');
CREATE INDEX idx_signals_type ON signals (signal_type);
CREATE INDEX idx_signals_direction ON signals (direction);
CREATE INDEX idx_signals_source_feature ON signals (source_feature_name);

-- Add compression policy for signals older than 30 days
ALTER TABLE signals SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id', timescaledb.compress_orderby = 'ts DESC');
SELECT add_compression_policy('signals', INTERVAL '30 days');

-- =================================================================
--  6. LABELS TABLE (HYPERTABLE)
--  Stores training labels for machine learning models.
-- =================================================================

CREATE TABLE labels (
    ts TIMESTAMPTZ NOT NULL,
    instrument_id INTEGER NOT NULL,
    timeframe TEXT NOT NULL, -- e.g., '15min'
    label INTEGER NOT NULL, -- -1, 0, 1 for SELL, NEUTRAL, BUY
    tp_price DOUBLE PRECISION,
    sl_price DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    exit_reason TEXT,
    exit_bar_offset INTEGER,
    barrier_return DOUBLE PRECISION,
    max_favorable_excursion DOUBLE PRECISION,
    max_adverse_excursion DOUBLE PRECISION,
    risk_reward_ratio DOUBLE PRECISION,
    volatility_at_entry DOUBLE PRECISION, -- Added missing column
    CONSTRAINT fk_instrument_label FOREIGN KEY (instrument_id) REFERENCES instruments (instrument_id),
    PRIMARY KEY (instrument_id, timeframe, ts)
);

SELECT create_hypertable('labels', 'ts', chunk_time_interval => INTERVAL '7 days');
CREATE INDEX idx_labels_label ON labels (label);
CREATE INDEX idx_labels_timeframe ON labels (timeframe);
CREATE INDEX idx_labels_exit_reason ON labels (exit_reason);

-- Add compression policy for labels older than 60 days
ALTER TABLE labels SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id', timescaledb.compress_orderby = 'ts DESC');
SELECT add_compression_policy('labels', INTERVAL '60 days');

-- =================================================================
--  7. MODEL REGISTRY TABLE
--  Tracks model versions and performance metrics.
-- =================================================================

CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    training_end_date DATE NOT NULL,
    accuracy DECIMAL(5,2) NOT NULL,
    f1_score DECIMAL(5,2) NOT NULL,
    sharpe_ratio DECIMAL(5,2) NOT NULL,
    max_drawdown DECIMAL(5,2) NOT NULL,
    total_signals INTEGER NOT NULL,
    profitable_signals INTEGER NOT NULL,
    config JSONB,
    is_active BOOLEAN NOT NULL DEFAULT FALSE
);

-- Ensure only one active model at a time
CREATE UNIQUE INDEX idx_model_registry_active ON model_registry (is_active) WHERE is_active = TRUE;