-- ============================================================================
-- MIGRATION v6: Processing State Management Tables
-- Description: Add tables for tracking processing completion state and data ranges
--              to prevent data duplication and enable idempotent operations
-- ============================================================================

-- Create processing_state table for tracking completion of processing steps
CREATE TABLE IF NOT EXISTS processing_state (
    instrument_id INTEGER NOT NULL,
    process_type VARCHAR(50) NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'completed',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (instrument_id, process_type)
);

-- Create index for faster queries by instrument_id
CREATE INDEX IF NOT EXISTS idx_processing_state_instrument 
ON processing_state (instrument_id);

-- Create index for faster queries by process_type
CREATE INDEX IF NOT EXISTS idx_processing_state_process_type 
ON processing_state (process_type);

-- Create index for faster queries by completion time
CREATE INDEX IF NOT EXISTS idx_processing_state_completed_at 
ON processing_state (completed_at DESC);

-- Create data_ranges table for tracking data coverage per instrument/timeframe
CREATE TABLE IF NOT EXISTS data_ranges (
    instrument_id INTEGER NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    earliest_ts TIMESTAMPTZ NOT NULL,
    latest_ts TIMESTAMPTZ NOT NULL,
    record_count INTEGER DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (instrument_id, timeframe)
);

-- Create index for faster queries by instrument_id
CREATE INDEX IF NOT EXISTS idx_data_ranges_instrument 
ON data_ranges (instrument_id);

-- Create index for faster queries by timeframe
CREATE INDEX IF NOT EXISTS idx_data_ranges_timeframe 
ON data_ranges (timeframe);

-- Create index for faster queries by time ranges
CREATE INDEX IF NOT EXISTS idx_data_ranges_time_coverage 
ON data_ranges (earliest_ts, latest_ts);

-- Add foreign key constraints if instruments table exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'instruments') THEN
        -- Add foreign key constraint for processing_state
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints 
            WHERE constraint_name = 'fk_processing_state_instrument'
        ) THEN
            ALTER TABLE processing_state 
            ADD CONSTRAINT fk_processing_state_instrument 
            FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id) 
            ON DELETE CASCADE;
        END IF;

        -- Add foreign key constraint for data_ranges
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints 
            WHERE constraint_name = 'fk_data_ranges_instrument'
        ) THEN
            ALTER TABLE data_ranges 
            ADD CONSTRAINT fk_data_ranges_instrument 
            FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id) 
            ON DELETE CASCADE;
        END IF;
    END IF;
END $$;

-- Add check constraints for data quality
DO $$
BEGIN
    -- Add check constraint for process_type if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'chk_process_type' AND table_name = 'processing_state'
    ) THEN
        ALTER TABLE processing_state 
        ADD CONSTRAINT chk_process_type 
        CHECK (process_type IN ('historical_fetch', 'aggregation', 'features', 'labeling', 'training', 'instrument_sync'));
    END IF;

    -- Add check constraint for status if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'chk_status' AND table_name = 'processing_state'
    ) THEN
        ALTER TABLE processing_state 
        ADD CONSTRAINT chk_status 
        CHECK (status IN ('completed', 'failed', 'in_progress'));
    END IF;

    -- Add check constraint for timeframe if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'chk_timeframe' AND table_name = 'data_ranges'
    ) THEN
        ALTER TABLE data_ranges 
        ADD CONSTRAINT chk_timeframe 
        CHECK (timeframe IN ('1min', '5min', '15min', '60min'));
    END IF;

    -- Add check constraint for time_order if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'chk_time_order' AND table_name = 'data_ranges'
    ) THEN
        ALTER TABLE data_ranges 
        ADD CONSTRAINT chk_time_order 
        CHECK (earliest_ts <= latest_ts);
    END IF;

    -- Add check constraint for record_count if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'chk_record_count' AND table_name = 'data_ranges'
    ) THEN
        ALTER TABLE data_ranges 
        ADD CONSTRAINT chk_record_count 
        CHECK (record_count >= 0);
    END IF;
END $$;

-- Create function to automatically update last_updated timestamp
CREATE OR REPLACE FUNCTION update_data_ranges_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update last_updated on data_ranges updates
DROP TRIGGER IF EXISTS trg_data_ranges_update_timestamp ON data_ranges;
CREATE TRIGGER trg_data_ranges_update_timestamp
    BEFORE UPDATE ON data_ranges
    FOR EACH ROW
    EXECUTE FUNCTION update_data_ranges_timestamp();

-- Grant appropriate permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON processing_state TO trading_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON data_ranges TO trading_app;

-- Add comments for documentation
COMMENT ON TABLE processing_state IS 'Tracks completion status of processing steps per instrument to enable idempotent operations';
COMMENT ON COLUMN processing_state.instrument_id IS 'Reference to the instrument being processed';
COMMENT ON COLUMN processing_state.process_type IS 'Type of processing: historical_fetch, aggregation, features, labeling';
COMMENT ON COLUMN processing_state.completed_at IS 'Timestamp when processing was completed';
COMMENT ON COLUMN processing_state.metadata IS 'Additional metadata about the processing (JSON format)';
COMMENT ON COLUMN processing_state.status IS 'Current status: completed, failed, in_progress';

COMMENT ON TABLE data_ranges IS 'Tracks data coverage ranges per instrument and timeframe for gap detection';
COMMENT ON COLUMN data_ranges.instrument_id IS 'Reference to the instrument';
COMMENT ON COLUMN data_ranges.timeframe IS 'Timeframe of the data (1min, 5min, 15min, 60min)';
COMMENT ON COLUMN data_ranges.earliest_ts IS 'Earliest timestamp of available data';
COMMENT ON COLUMN data_ranges.latest_ts IS 'Latest timestamp of available data';
COMMENT ON COLUMN data_ranges.record_count IS 'Approximate number of records in this range';
COMMENT ON COLUMN data_ranges.last_updated IS 'When this range was last updated';

-- ============================================================================
-- Migration complete: v6_processing_state.sql
-- ============================================================================