#!/usr/bin/env python3
"""
Database Migration Runner for Algo Trading System

Runs database migrations in sequence to ensure proper schema setup.
"""

import asyncio
import time
from pathlib import Path

from src.database.db_utils import db_manager
from src.utils.logger import LOGGER as logger


class DatabaseMigrator:
    """Handles database schema migrations for the trading system."""

    def __init__(self):
        self.db_manager = db_manager
        self.migrations_dir = Path(__file__).parent / "migrations"

    async def initialize_migration_tracking(self):
        """Create migration tracking table if it doesn't exist."""
        create_migrations_table = """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(50) PRIMARY KEY,
                executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                execution_time_ms INTEGER NOT NULL
            );
        """

        async with self.db_manager.get_connection() as conn:
            await conn.execute(create_migrations_table)
            logger.info("Migration tracking table initialized")

    async def get_applied_migrations(self) -> set[str]:
        """Get list of already applied migrations."""
        query = "SELECT version FROM schema_migrations ORDER BY executed_at"

        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(query)
            applied = {row["version"] for row in rows}
            logger.info(f"Found {len(applied)} applied migrations: {applied}")
            return applied

    async def run_migration(self, migration_file: Path) -> bool:
        """Run a single migration file."""
        version = migration_file.stem
        logger.info(f"Executing migration: {version}")

        start_time = asyncio.get_event_loop().time()

        try:
            # Read migration SQL
            with open(migration_file) as f:
                sql_content = f.read()

            # Execute migration in transaction
            async with self.db_manager.transaction() as conn:
                # Split by semicolon and execute each statement
                statements = [stmt.strip() for stmt in sql_content.split(";") if stmt.strip()]

                for statement in statements:
                    if statement:
                        try:
                            await conn.execute(statement)
                        except Exception as e:
                            # Some statements like SELECT create_hypertable might fail if already exists
                            if "already exists" in str(e).lower() or "already a hypertable" in str(e).lower():
                                logger.warning(f"Skipping existing object in {version}: {e}")
                                continue
                            raise

                # Record successful migration
                execution_time = int((time.monotonic() - start_time) * 1000)
                await conn.execute(
                    "INSERT INTO schema_migrations (version, execution_time_ms) VALUES ($1, $2)",
                    version,
                    execution_time,
                )

                logger.info(f"✅ Migration {version} completed in {execution_time}ms")
                return True

        except Exception as e:
            logger.error(f"❌ Migration {version} failed: {e}")
            return False

    async def run_all_migrations(self):
        """Run all pending migrations in order."""
        logger.info("🚀 Starting database migration process")

        # Initialize migration tracking
        await self.initialize_migration_tracking()

        # Get applied migrations
        applied_migrations = await self.get_applied_migrations()

        # Find all migration files
        migration_files = sorted([f for f in self.migrations_dir.glob("*.sql") if f.is_file()])

        if not migration_files:
            logger.warning("No migration files found")
            return

        logger.info(f"Found {len(migration_files)} migration files")

        # Run pending migrations
        pending_count = 0
        successful_count = 0

        for migration_file in migration_files:
            version = migration_file.stem

            if version in applied_migrations:
                logger.info(f"⏭️  Skipping already applied migration: {version}")
                continue

            pending_count += 1

            if await self.run_migration(migration_file):
                successful_count += 1
            else:
                logger.error(f"Migration {version} failed - stopping migration process")
                break

        # Summary
        if pending_count == 0:
            logger.info("✅ All migrations are up to date")
        elif successful_count == pending_count:
            logger.info(f"✅ Successfully applied {successful_count} migrations")
        else:
            logger.error(f"❌ Applied {successful_count}/{pending_count} migrations")

        logger.info("🏁 Migration process completed")

    async def get_migration_status(self):
        """Get current migration status."""
        applied_migrations = await self.get_applied_migrations()

        migration_files = sorted([f for f in self.migrations_dir.glob("*.sql") if f.is_file()])

        logger.info("📊 Migration Status:")
        for migration_file in migration_files:
            version = migration_file.stem
            status = "✅ Applied" if version in applied_migrations else "⏳ Pending"
            logger.info(f"  {version}: {status}")
