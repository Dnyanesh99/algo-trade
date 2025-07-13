#!/usr/bin/env python3
"""
Database Migration Runner for Algo Trading System

Runs database migrations in sequence to ensure proper schema setup.
"""

import asyncio
import time
from pathlib import Path

from src.database.db_utils import db_manager
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

queries = ConfigLoader().get_queries()


class DatabaseMigrator:
    """Handles database schema migrations for the trading system."""

    def __init__(self) -> None:
        self.db_manager = db_manager
        self.migrations_dir = Path(__file__).parent / "migrations"

    async def initialize_migration_tracking(self) -> None:
        """Create migration tracking table if it doesn't exist."""
        create_migrations_table = queries.migrate["create_migrations_table"]

        async with self.db_manager.get_connection() as conn:
            await conn.execute(create_migrations_table)
            logger.info("Migration tracking table initialized")

    async def get_applied_migrations(self) -> set[str]:
        """Get list of already applied migrations."""
        query = queries.migrate["get_applied_migrations"]

        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(query)
            applied = {row["version"] for row in rows}
            logger.info(f"Found {len(applied)} applied migrations: {applied}")
            return applied

    async def run_migration(self, migration_file: Path) -> bool:
        """
        Run a single migration file, handling transactions robustly.
        """
        version = migration_file.stem
        logger.info(f"Executing migration: {version}")
        start_time = time.monotonic()

        try:
            with open(migration_file) as f:
                sql_content = f.read()

            if not sql_content.strip():
                logger.warning(f"Migration file {version} is empty, skipping.")
                return True

            # First, attempt to run the entire script in a single transaction.
            # This is the safest method and works for most migrations.
            try:
                async with self.db_manager.get_connection() as conn, conn.transaction():
                    await conn.execute(sql_content)
                logger.info(f"Migration {version} successfully executed within a transaction.")

            except Exception as e:
                # If the error indicates a statement cannot run in a transaction
                # (e.g., creating continuous aggregates), retry outside a transaction.
                if "cannot run inside a transaction block" in str(e).lower():
                    logger.warning(
                        f"Migration {version} contains non-transactional DDL. Retrying outside of a transaction."
                    )
                    async with self.db_manager.get_connection() as conn:
                        await conn.execute(sql_content)
                    logger.info(f"Migration {version} successfully executed outside a transaction.")
                else:
                    # For all other errors, re-raise to be caught by the outer block.
                    raise

            # Record the successful migration in the tracking table
            execution_time = int((time.monotonic() - start_time) * 1000)
            async with self.db_manager.get_connection() as conn:
                await conn.execute(queries.migrate["insert_migration"], version, execution_time)

            logger.info(f"✅ Migration {version} completed in {execution_time}ms")
            return True

        except Exception as e:
            logger.error(f"❌ Migration {version} failed: {e}")
            return False

    async def run_all_migrations(self) -> None:
        """Run all pending migrations in order."""
        logger.info("🚀 Starting database migration process")

        await self.db_manager.initialize()
        await self.initialize_migration_tracking()
        applied_migrations = await self.get_applied_migrations()

        migration_files = sorted(self.migrations_dir.glob("v*.sql"))

        if not migration_files:
            logger.warning("No migration files found")
            return

        logger.info(f"Found {len(migration_files)} migration files")

        pending_migrations = [m for m in migration_files if m.stem not in applied_migrations]
        if not pending_migrations:
            logger.info("✅ All migrations are up to date")
            return

        logger.info(f"Found {len(pending_migrations)} pending migrations to apply.")
        successful_count = 0
        for migration_file in pending_migrations:
            if await self.run_migration(migration_file):
                successful_count += 1
            else:
                logger.error(f"Stopping migration process due to failure in {migration_file.stem}")
                break
        if successful_count == len(pending_migrations):
            logger.info(f"✅ Successfully applied {successful_count} new migrations.")
        else:
            logger.error(f"❌ Applied {successful_count}/{len(pending_migrations)} pending migrations.")

        logger.info("🏁 Migration process completed")

    async def get_migration_status(self) -> None:
        """Get current migration status."""
        await self.db_manager.initialize()
        applied_migrations = await self.get_applied_migrations()
        migration_files = sorted(self.migrations_dir.glob("v*.sql"))

        logger.info("📊 Migration Status:")
        for migration_file in migration_files:
            version = migration_file.stem
            status = "✅ Applied" if version in applied_migrations else "⏳ Pending"
            logger.info(f"  {version}: {status}")


if __name__ == "__main__":
    migrator = DatabaseMigrator()
    asyncio.run(migrator.run_all_migrations())
