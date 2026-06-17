-- Migration to create the labeling_statistics table

CREATE TABLE IF NOT EXISTS labeling_statistics (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    total_bars INTEGER NOT NULL,
    labeled_bars INTEGER NOT NULL,
    label_distribution JSONB NOT NULL,
    avg_return_by_label JSONB NOT NULL,
    exit_reasons JSONB NOT NULL,
    avg_holding_period DOUBLE PRECISION NOT NULL,
    processing_time_ms DOUBLE PRECISION NOT NULL,
    data_quality_score DOUBLE PRECISION NOT NULL,
    sharpe_ratio DOUBLE PRECISION NOT NULL,
    win_rate DOUBLE PRECISION NOT NULL,
    profit_factor DOUBLE PRECISION NOT NULL,
    run_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_labeling_statistics_symbol_timestamp ON labeling_statistics (symbol, run_timestamp DESC);

-- Grant usage to the application user if it exists
DO $$
BEGIN
   IF EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'user') THEN
      GRANT ALL ON TABLE labeling_statistics TO "user";
      GRANT ALL ON SEQUENCE labeling_statistics_id_seq TO "user";
   END IF;
END
$$;
