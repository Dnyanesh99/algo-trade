-- ============================================================================
-- MIGRATION v7: System Operations Table
-- Description: Create a separate table for tracking system-level operations
--              to avoid polluting the instruments table with non-trading data
-- ============================================================================

-- Remove the incorrectly inserted system instrument if it exists
DELETE FROM instruments WHERE instrument_id = -1;

-- Create separate table for system-level operations tracking
CREATE TABLE IF NOT EXISTS system_operations_state (
    operation_type VARCHAR(50) PRIMARY KEY,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'completed',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for faster queries by operation_type
CREATE INDEX IF NOT EXISTS idx_system_operations_type 
ON system_operations_state (operation_type);

-- Create index for faster queries by completion time
CREATE INDEX IF NOT EXISTS idx_system_operations_completed_at 
ON system_operations_state (completed_at DESC);

-- Add check constraint for valid operation types
ALTER TABLE system_operations_state 
ADD CONSTRAINT chk_system_operation_type 
CHECK (operation_type IN ('instrument_sync', 'system_health', 'global_cleanup', 'database_maintenance'));

-- Add check constraint for valid status values
ALTER TABLE system_operations_state 
ADD CONSTRAINT chk_system_operation_status 
CHECK (status IN ('completed', 'failed', 'in_progress'));

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_system_operations_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic timestamp updates
DROP TRIGGER IF EXISTS trg_system_operations_update_timestamp ON system_operations_state;
CREATE TRIGGER trg_system_operations_update_timestamp
    BEFORE UPDATE ON system_operations_state
    FOR EACH ROW
    EXECUTE FUNCTION update_system_operations_timestamp();

-- Add comments for documentation
COMMENT ON TABLE system_operations_state IS 'Tracks completion status of system-level operations (not tied to specific instruments)';
COMMENT ON COLUMN system_operations_state.operation_type IS 'Type of system operation: instrument_sync, system_health, etc.';
COMMENT ON COLUMN system_operations_state.completed_at IS 'Timestamp when operation was completed';
COMMENT ON COLUMN system_operations_state.metadata IS 'Additional metadata about the operation (JSON format)';
COMMENT ON COLUMN system_operations_state.status IS 'Current status: completed, failed, in_progress';

-- ============================================================================
-- Migration complete: v7_system_instrument.sql
-- ============================================================================