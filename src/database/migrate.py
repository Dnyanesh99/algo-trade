#!/usr/bin/env python3
"""
Database Migration Runner for Algo Trading System

Runs database migrations in sequence to ensure proper schema setup.
"""

import asyncio
import time
from pathlib import Path

import asyncpg

from src.database.db_utils import db_manager
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

queries = ConfigLoader().get_queries()


class DatabaseMigrator:
    """Handles database schema migrations for the trading system."""

    def __init__(self) -> None:
        self.db_manager = db_manager
        self.migrations_dir = Path(__file__).parent / "migrations"
        if not self.migrations_dir.is_dir():
            logger.error(f"Migrations directory not found at: {self.migrations_dir}")
            raise FileNotFoundError(f"Migrations directory not found: {self.migrations_dir}")
        logger.info(f"DatabaseMigrator initialized. Using migrations from: {self.migrations_dir}")

    async def initialize_migration_tracking(self) -> None:
        """Create migration tracking table if it doesn't exist."""
        logger.info("Initializing migration tracking...")
        create_migrations_table_query = queries.migrate.get("create_migrations_table")
        if not create_migrations_table_query:
            raise RuntimeError("'create_migrations_table' query not found in queries.yaml")

        try:
            await self.db_manager.execute(create_migrations_table_query)
            logger.info("Migration tracking table is ready.")
        except Exception as e:
            logger.critical(f"Failed to create or verify migration tracking table: {e}", exc_info=True)
            raise RuntimeError("Could not initialize migration tracking.") from e

    async def get_applied_migrations(self) -> set[str]:
        """Get list of already applied migrations."""
        query = queries.migrate.get("get_applied_migrations")
        if not query:
            raise RuntimeError("'get_applied_migrations' query not found in queries.yaml")

        try:
            rows = await self.db_manager.fetch_rows(query)
            applied = {row["version"] for row in rows}
            logger.info(f"Found {len(applied)} applied migrations in the database.")
            logger.debug(f"Applied versions: {sorted(applied)}")
            return applied
        except Exception as e:
            logger.critical(f"Failed to get applied migrations from the database: {e}", exc_info=True)
            raise RuntimeError("Could not query applied migrations.") from e

    async def run_migration(self, migration_file: Path) -> bool:
        """
        Run a single migration file, handling transactions robustly.
        """
        version = migration_file.stem
        logger.info(f"--- Starting migration: {version} ---")
        start_time = time.monotonic()

        try:
            with open(migration_file) as f:
                sql_content = f.read()

            if not sql_content.strip():
                logger.warning(f"Migration file {version} is empty, skipping.")
                return True

            # Use the db_manager's transaction context for robustness
            try:
                async with self.db_manager.transaction() as conn:
                    logger.debug(f"Executing migration {version} within a transaction.")
                    await conn.execute(sql_content)
                logger.info(f"Migration {version} successfully executed within a single transaction.")
            except asyncpg.exceptions.InternalClientError as e:
                if "cannot run inside a transaction block" in str(e).lower():
                    logger.warning(
                        f"Migration {version} contains non-transactional statements. Retrying outside of a transaction."
                    )
                    await self.db_manager.execute(sql_content)
                    logger.info(f"Migration {version} successfully executed outside a transaction.")
                else:
                    raise

            # Record the successful migration in the tracking table
            execution_time = int((time.monotonic() - start_time) * 1000)
            insert_query = queries.migrate.get("insert_migration")
            if not insert_query:
                raise RuntimeError("'insert_migration' query not found in queries.yaml")
            await self.db_manager.execute(insert_query, version, execution_time)

            logger.info(f"✅ Migration {version} completed and recorded in {execution_time}ms.")
            return True

        except (OSError, FileNotFoundError) as e:
            logger.critical(f"CRITICAL: Could not read migration file {migration_file}: {e}", exc_info=True)
        except Exception as e:
            logger.critical(f"❌ CRITICAL: Migration {version} failed with an unexpected error: {e}", exc_info=True)

        return False

    async def run_all_migrations(self) -> bool:
        """Run all pending migrations in order. Returns True if all migrations were successful."""
        logger.info("🚀 Starting database migration process...")
        try:
            await self.db_manager.initialize()
            await self.initialize_migration_tracking()
            applied_migrations = await self.get_applied_migrations()

            migration_files = sorted(self.migrations_dir.glob("v*.sql"))

            if not migration_files:
                logger.warning("No migration files found in the migrations directory.")
                return True

            logger.info(f"Found {len(migration_files)} total migration files.")

            pending_migrations = [m for m in migration_files if m.stem not in applied_migrations]
            if not pending_migrations:
                logger.info("✅ Database schema is already up to date.")
                return True

            logger.info(
                f"Found {len(pending_migrations)} pending migrations to apply: {[p.name for p in pending_migrations]}"
            )
            successful_count = 0
            for migration_file in pending_migrations:
                if await self.run_migration(migration_file):
                    successful_count += 1
                else:
                    logger.critical(f"CRITICAL: Stopping migration process due to failure in {migration_file.stem}.")
                    return False

            logger.info(f"✅ Successfully applied {successful_count} new migrations.")
            logger.info("🏁 Migration process completed successfully.")
            return True

        except Exception as e:
            logger.critical(f"CRITICAL: The migration process failed with an unhandled exception: {e}", exc_info=True)
            return False

    async def get_migration_status(self) -> None:
        """Get current migration status."""
        logger.info("--- MIGRATION STATUS ---")
        try:
            await self.db_manager.initialize()
            applied_migrations = await self.get_applied_migrations()
            migration_files = sorted(self.migrations_dir.glob("v*.sql"))

            if not migration_files:
                logger.warning("No migration files found to check status against.")
                return

            for migration_file in migration_files:
                version = migration_file.stem
                status = "✅ Applied" if version in applied_migrations else "⏳ Pending"
                logger.info(f"  - {version}: {status}")
        except Exception as e:
            logger.error(f"Could not retrieve migration status due to an error: {e}", exc_info=True)
        logger.info("------------------------")


if __name__ == "__main__":
    logger.info("Running Database Migrator directly.")
    migrator = DatabaseMigrator()
    try:
        # In a real application, you might parse command-line arguments here
        # to choose between running migrations or checking status.
        # For this example, we will run migrations by default.
        success = asyncio.run(migrator.run_all_migrations())
        if success:
            logger.info("Migration process finished successfully.")
            # Optionally check status after running
            # asyncio.run(migrator.get_migration_status())
        else:
            logger.critical("Migration process failed. See logs for details.")
            exit(1)  # Exit with a non-zero code to indicate failure
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.critical(f"A critical error prevented the migrator from running: {e}", exc_info=True)
        exit(1)
    except KeyboardInterrupt:
        logger.info("Migration process interrupted by user.")
        exit(1)
