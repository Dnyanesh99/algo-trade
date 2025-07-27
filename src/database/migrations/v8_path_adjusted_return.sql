-- V8: Add path_adjusted_return column to labels table
-- This column stores the path-dependent timeout calculation result for transparency

-- Add path_adjusted_return column to labels table
ALTER TABLE labels ADD COLUMN IF NOT EXISTS path_adjusted_return DOUBLE PRECISION;

-- Add comment explaining the purpose
COMMENT ON COLUMN labels.path_adjusted_return IS 'Path-dependent timeout calculation result per De Prado AFML. Null for TP_HIT/SL_HIT cases.';

-- Create index for analysis queries
CREATE INDEX IF NOT EXISTS idx_labels_path_adjusted_return ON labels (path_adjusted_return) WHERE path_adjusted_return IS NOT NULL;

-- Grant permissions (assuming same as other tables)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading_user') THEN
        GRANT SELECT ON labels TO trading_user;
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        -- Ignore errors if role doesn't exist
        NULL;
END $$;