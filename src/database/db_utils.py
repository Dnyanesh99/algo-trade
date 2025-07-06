import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Optional

import asyncpg

from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger  # Centralized logger

# Load configuration
config = config_loader.get_config()


class DatabaseManager:
    """
    Manages database connections and transactions using asyncpg connection pooling.
    Provides methods for acquiring connections and executing queries.
    """

    _instance: Optional["DatabaseManager"] = None
    _pool: Optional[asyncpg.Pool] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pool = None  # Ensure pool is None initially
            cls._instance._initialized = False  # Initialize flag
        return cls._instance

    async def initialize(self) -> None:
        """
        Initializes the database connection pool.
        This method should be called once at application startup.
        """
        async with self._lock:
            if self._pool is None:
                logger.info("Initializing database connection pool...")
                db_config = config.database
                try:
                    self._pool = await asyncpg.create_pool(
                        user=db_config.user,
                        password=db_config.password,
                        host=db_config.host,
                        port=db_config.port,
                        database=db_config.dbname,
                        min_size=db_config.min_connections,
                        max_size=db_config.max_connections,
                        timeout=db_config.timeout,
                    )
                    self._initialized = True
                    logger.info("Database connection pool initialized successfully.")
                except Exception as e:
                    logger.critical(f"Failed to initialize database connection pool: {e}")
                    # Re-raise to ensure application startup fails if DB connection cannot be established
                    raise RuntimeError(f"Database initialization failed: {e}") from e

    async def close(self) -> None:
        """
        Closes the database connection pool.
        """
        async with self._lock:
            if self._pool and self._initialized:
                logger.info("Closing database connection pool...")
                await self._pool.close()
                self._pool = None
                self._initialized = False
                logger.info("Database connection pool closed.")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Provides an asynchronous context manager for acquiring a database connection
        from the pool. The connection is released back to the pool automatically.
        """
        if self._pool is None:
            logger.error("DatabaseManager not initialized. Call .initialize() first.")
            raise RuntimeError("DatabaseManager not initialized.")

        conn: Optional[asyncpg.Connection] = None
        try:
            conn = await self._pool.acquire()
            yield conn
        except Exception as e:
            logger.error(f"Error acquiring database connection: {e}")
            raise
        finally:
            if conn:
                await self._pool.release(conn)

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Provides an asynchronous context manager for managing database transactions.
        Commits on success, rolls back on error.
        """
        async with self.get_connection() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                yield conn
                await tr.commit()
                logger.debug("Transaction committed successfully.")
            except Exception as e:
                await tr.rollback()
                logger.error(f"Transaction rolled back due to error: {e}")
                raise

    async def fetch_rows(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """
        Executes a query and fetches all matching rows.
        """
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)

    async def fetch_row(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        """
        Executes a query and fetches a single matching row.
        """
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        """
        Executes a query that does not return rows (e.g., INSERT, UPDATE, DELETE).
        Returns the command status.
        """
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)


db_manager = DatabaseManager()
