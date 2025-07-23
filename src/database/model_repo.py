from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

from src.database.db_utils import db_manager
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

config = ConfigLoader().get_config()
queries = ConfigLoader().get_queries()


class ModelMetadata(BaseModel):
    """Model metadata for model registry"""

    id: Optional[int] = None
    version: str
    created_at: Optional[datetime] = None
    training_end_date: datetime
    accuracy: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    total_signals: int
    profitable_signals: int
    config: dict[str, Any]
    is_active: bool = False


class ModelRepository:
    """
    Repository for interacting with model registry in TimescaleDB.
    Manages model versions, metadata, and performance tracking.
    """

    def __init__(self) -> None:
        self.db_manager = db_manager
        logger.info("ModelRepository initialized.")

    async def register_model(
        self,
        version: str,
        training_end_date: datetime,
        accuracy: float,
        f1_score: float,
        sharpe_ratio: float,
        max_drawdown: float,
        total_signals: int,
        profitable_signals: int,
        model_config: dict[str, Any],
        activate: bool = False,
    ) -> int:
        """
        Register a new trained model in the registry.

        Args:
            version: Unique version identifier for the model.
            training_end_date: Date when training data ends.
            accuracy: Model accuracy metric.
            f1_score: F1 score metric.
            sharpe_ratio: Sharpe ratio for trading performance.
            max_drawdown: Maximum drawdown percentage.
            total_signals: Total number of signals generated during validation.
            profitable_signals: Number of profitable signals.
            model_config: Model configuration and hyperparameters.
            activate: Whether to activate this model (deactivates others).

        Returns:
            The ID of the registered model.

        Raises:
            RuntimeError: If the database operation fails.
        """
        logger.info(f"Attempting to register model version '{version}'...")
        try:
            async with self.db_manager.transaction() as conn:
                if activate:
                    logger.info("Activation flag is set. Deactivating all other models first.")
                    await conn.execute(queries.model_registry["deactivate_all_models"])

                model_id = await conn.fetchval(
                    queries.model_registry["insert_model"],
                    version,
                    training_end_date,
                    accuracy,
                    f1_score,
                    sharpe_ratio,
                    max_drawdown,
                    total_signals,
                    profitable_signals,
                    model_config,
                    activate,
                )

                if model_id is None:
                    raise RuntimeError("Database did not return a model ID after insertion.")

                logger.info(
                    f"Successfully registered model '{version}' with ID {model_id} (Active: {activate}, Accuracy: {accuracy:.2%})"
                )
                return int(model_id)

        except Exception as e:
            logger.error(f"Database error registering model version '{version}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to register model version '{version}'.") from e

    async def get_active_model(self) -> Optional[ModelMetadata]:
        """
        Get the currently active model.

        Returns:
            Active model metadata or None if no active model is found.

        Raises:
            RuntimeError: If the database query fails.
        """
        logger.debug("Fetching the currently active model from the registry.")
        try:
            row = await self.db_manager.fetch_row(queries.model_registry["get_active_model"])
            if row:
                logger.info(f"Found active model: Version '{row['version']}' (ID: {row['id']}).")
                return ModelMetadata.model_validate(dict(row))

            logger.warning("No active model found in the model registry.")
            return None

        except Exception as e:
            logger.error(f"Database error fetching the active model: {e}", exc_info=True)
            raise RuntimeError("Could not retrieve the active model.") from e

    async def get_model_by_version(self, version: str) -> Optional[ModelMetadata]:
        """
        Get model by version string.

        Args:
            version: Model version identifier.

        Returns:
            Model metadata or None if not found.

        Raises:
            RuntimeError: If the database query fails.
        """
        logger.debug(f"Fetching model from registry by version: '{version}'.")
        try:
            row = await self.db_manager.fetch_row(queries.model_registry["get_model_by_version"], version)
            if row:
                logger.debug(f"Found model version '{version}' with ID {row['id']}.")
                return ModelMetadata.model_validate(dict(row))

            logger.debug(f"Model version '{version}' not found in the registry.")
            return None

        except Exception as e:
            logger.error(f"Database error fetching model version '{version}': {e}", exc_info=True)
            raise RuntimeError(f"Could not retrieve model version '{version}'.") from e

    async def get_model_history(self, limit: int = 10) -> list[ModelMetadata]:
        """
        Get model history ordered by creation date.

        Args:
            limit: Maximum number of models to return.

        Returns:
            A list of model metadata.

        Raises:
            RuntimeError: If the database query fails.
        """
        if limit <= 0:
            logger.warning(f"get_model_history called with non-positive limit: {limit}. Returning empty list.")
            return []

        logger.debug(f"Fetching latest {limit} models from the registry history.")
        try:
            rows = await self.db_manager.fetch_rows(queries.model_registry["get_model_history"], limit)
            logger.info(f"Successfully fetched {len(rows)} model history records.")
            return [ModelMetadata.model_validate(dict(row)) for row in rows]

        except Exception as e:
            logger.error(f"Database error fetching model history: {e}", exc_info=True)
            raise RuntimeError("Could not retrieve model history.") from e

    async def activate_model(self, version: str) -> bool:
        """
        Activate a model by version, deactivating all others atomically.

        Args:
            version: Model version to activate.

        Returns:
            True if successful, False if model not found.

        Raises:
            RuntimeError: If the database operation fails.
        """
        logger.info(f"Attempting to activate model version '{version}'.")
        try:
            async with self.db_manager.transaction() as conn:
                model_exists = await conn.fetchval("SELECT 1 FROM model_registry WHERE version = $1", version)
                if not model_exists:
                    logger.error(f"Cannot activate model: Version '{version}' not found in the registry.")
                    return False

                logger.info("Deactivating all currently active models...")
                await conn.execute(queries.model_registry["deactivate_all_models"])

                logger.info(f"Activating new model: '{version}'...")
                await conn.execute(queries.model_registry["activate_model"], version)

                logger.info(f"Successfully activated model version '{version}'.")
                return True

        except Exception as e:
            logger.error(f"Database error activating model version '{version}': {e}", exc_info=True)
            raise RuntimeError(f"Could not activate model version '{version}'.") from e

    async def deactivate_all_models(self) -> None:
        """
        Deactivate all models in the registry.

        Raises:
            RuntimeError: If the database operation fails.
        """
        logger.info("Deactivating all models in the registry.")
        try:
            await self.db_manager.execute(queries.model_registry["deactivate_all_models"])
            logger.info("Successfully deactivated all models.")
        except Exception as e:
            logger.error(f"Database error deactivating all models: {e}", exc_info=True)
            raise RuntimeError("Could not deactivate all models.") from e
