import asyncio
import functools
import re
import time
from collections.abc import AsyncGenerator, Awaitable
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional, TypeVar, cast

import asyncpg

from src.metrics import metrics_registry
from src.utils.config_loader import DatabaseConfig
from src.utils.logger import LOGGER as logger

_F = TypeVar("_F", bound=Callable[..., Awaitable[Any]])


def _extract_table_name(query: str) -> str:
    """Extract table name from SQL query."""
    # This is a simplified parser, might need to be more robust
    match = re.search(r"\s(?:FROM|INTO|UPDATE)\s+([\w.]+)", query, re.IGNORECASE)
    if match:
        return match.group(1)
    return "unknown"


def measure_db_operation(operation: str) -> Callable[[_F], _F]:
    """Decorator to measure database operation metrics."""

    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.monotonic()
            success = True
            query = args[1] if len(args) > 1 else ""
            table = _extract_table_name(query)
            rows = 0  # Initialize rows
            try:
                result = await func(*args, **kwargs)
                # For fetch_rows, rows is len(result)
                if func.__name__ == "fetch_rows":
                    rows = len(result) if result else 0
                # For fetch_row, rows is 1 if result else 0
                elif func.__name__ == "fetch_row":
                    rows = 1 if result else 0
                # For execute, we can't easily get row count, so we'll use 0
                # The original code had an else here, but it was not assigning rows
                return result
            except Exception as e:
                success = False
                raise e
            finally:
                duration = time.monotonic() - start_time
                metrics_registry.record_db_query(operation, table, success, duration, rows)

        return cast("_F", wrapper)

    return decorator


class DatabaseManager:
    """
    Manages database connections and transactions using asyncpg connection pooling.
    Provides methods for acquiring connections and executing queries.
    """

    _instance: Optional["DatabaseManager"] = None
    _pool: Optional[asyncpg.Pool] = None
    _lock = asyncio.Lock()
    _initialized: bool = False

    def __new__(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pool = None
            cls._instance._initialized = False
        return cls._instance

    async def initialize(self) -> None:
        """
        Initializes the database connection pool.
        This method is critical and must be called once at application startup.
        """
        async with self._lock:
            if self._pool is not None and self._initialized:
                logger.warning("DatabaseManager.initialize called, but pool is already initialized.")
                return

            logger.info("Initializing database connection pool...")
            try:
                db_config = DatabaseConfig()  # type: ignore[call-arg]
            except Exception as e:
                logger.critical(
                    f"CRITICAL: Failed to load database configuration. Check environment variables. Error: {e}",
                    exc_info=True,
                )
                raise ValueError("Database configuration is invalid or missing essential values.") from e

            logger.info(
                f"Attempting to create connection pool for database '{db_config.dbname}' at {db_config.host}:{db_config.port}"
            )
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
                # Verify connection by acquiring and releasing a connection
                async with self._pool.acquire() as conn:
                    server_version = conn.get_server_version()
                    logger.info(
                        f"Database connection pool initialized successfully. Connected to PostgreSQL server version {server_version.major}.{server_version.minor}"
                    )

                self._initialized = True

            except (
                asyncpg.exceptions.InvalidPasswordError,
                asyncpg.exceptions.InvalidAuthorizationSpecificationError,
            ) as e:
                logger.critical(
                    f"CRITICAL: Database authentication failed for user '{db_config.user}'. Check credentials. Error: {e}"
                )
                raise RuntimeError("Database authentication failed.") from e
            except ConnectionRefusedError as e:
                logger.critical(
                    f"CRITICAL: Database connection refused at {db_config.host}:{db_config.port}. Is the database running and accessible? Error: {e}"
                )
                raise RuntimeError("Database connection was refused.") from e
            except Exception as e:
                logger.critical(f"CRITICAL: Failed to create database connection pool. Error: {e}", exc_info=True)
                raise RuntimeError("Database initialization failed due to an unexpected error.") from e

    async def close(self) -> None:
        """
        Closes the database connection pool gracefully.
        """
        async with self._lock:
            if self._pool and self._initialized:
                logger.info("Closing database connection pool...")
                try:
                    await self._pool.close()
                    logger.info("Database connection pool closed successfully.")
                except Exception as e:
                    logger.error(f"An error occurred while closing the database pool: {e}", exc_info=True)
                finally:
                    self._pool = None
                    self._initialized = False
            else:
                logger.info("Database connection pool was not initialized or already closed.")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Provides an asynchronous context manager for acquiring a database connection
        from the pool. The connection is released back to the pool automatically.
        """
        if not self._pool or not self._initialized:
            logger.critical("CRITICAL: Attempted to get a DB connection from an uninitialized pool.")
            raise RuntimeError("DatabaseManager is not initialized. Call initialize() at application startup.")

        conn: Optional[asyncpg.Connection] = None
        try:
            conn = await self._pool.acquire()
            yield conn
        except Exception as e:
            logger.error(f"Failed to acquire a database connection from the pool: {e}", exc_info=True)
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
            logger.debug("Starting a new database transaction.")
            tr = conn.transaction()
            await tr.start()
            try:
                yield conn
                await tr.commit()
                logger.debug("Transaction committed successfully.")
            except Exception as e:
                logger.error(f"Transaction rolled back due to an error: {e}", exc_info=True)
                if not tr.is_completed():
                    await tr.rollback()
                    logger.info("Transaction has been rolled back.")
                raise

    @measure_db_operation(operation="fetch")
    async def fetch_rows(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """
        Executes a query and fetches all matching rows.
        """
        async with self.get_connection() as conn:
            result = await conn.fetch(query, *args)
            return list(result)

    @measure_db_operation(operation="fetch")
    async def fetch_row(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        """
        Executes a query and fetches a single matching row.
        """
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)

    @measure_db_operation(operation="fetch")
    async def fetchval(self, query: str, *args: Any) -> Any:
        """
        Executes a query and fetches a single value.
        """
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)

    @measure_db_operation(operation="execute")
    async def execute(self, query: str, *args: Any) -> str:
        """
        Executes a query that does not return rows (e.g., INSERT, UPDATE, DELETE).
        Returns the command status.
        """
        async with self.get_connection() as conn:
            result = await conn.execute(query, *args)
            return str(result)


db_manager = DatabaseManager()
