from datetime import datetime
from typing import Any, Optional

import pandas as pd

from src.database.db_utils import db_manager
from src.database.models import FeatureData
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger
from src.utils.time_helper import APP_TIMEZONE

config = ConfigLoader().get_config()
queries = ConfigLoader().get_queries()


class FeatureRepository:
    """
    Repository for interacting with calculated features in the TimescaleDB.
    Provides methods for inserting and fetching feature data.
    """

    def __init__(self) -> None:
        self.db_manager = db_manager
        logger.info("FeatureRepository initialized.")

    async def insert_features(self, instrument_id: int, timeframe: str, features_data: list[FeatureData]) -> None:
        """
        Inserts calculated features data into the features table using a robust transaction.

        Args:
            instrument_id: The ID of the instrument.
            timeframe: The timeframe (e.g., '5min', '15min', '60min').
            features_data: A list of FeatureData objects to insert.

        Raises:
            ValueError: If input arguments are invalid.
            RuntimeError: If the database insertion fails.
        """
        if not features_data:
            logger.warning(
                f"insert_features called with no data for instrument {instrument_id} ({timeframe}). Nothing to do."
            )
            return
        if not isinstance(instrument_id, int) or not isinstance(timeframe, str):
            raise ValueError("Invalid instrument_id or timeframe type.")

        query = queries.feature_repo["insert_features"]
        records = [
            (
                APP_TIMEZONE.localize(d.ts) if d.ts.tzinfo is None else d.ts,
                instrument_id,
                timeframe,
                d.feature_name,
                d.feature_value,
            )
            for d in features_data
        ]

        try:
            async with self.db_manager.transaction() as conn:
                await conn.executemany(query, records)
            logger.info(
                f"Successfully inserted/updated {len(records)} feature records for instrument {instrument_id} ({timeframe})."
            )
        except Exception as e:
            logger.error(
                f"Database error inserting features for instrument {instrument_id} ({timeframe}): {e}", exc_info=True
            )
            raise RuntimeError("Failed to insert feature data.") from e

    async def get_features(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> list[FeatureData]:
        """
        Fetches features data for a given instrument and timeframe within a time range.

        Returns:
            A list of FeatureData objects.

        Raises:
            RuntimeError: If the database query fails.
        """
        query = queries.feature_repo["get_features"]
        logger.debug(
            f"Fetching features for instrument {instrument_id} ({timeframe}) between {start_time} and {end_time}."
        )
        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe, start_time, end_time)
            logger.info(f"Fetched {len(rows)} feature records for instrument {instrument_id} ({timeframe}).")
            return [FeatureData.model_validate(dict(row)) for row in rows]
        except Exception as e:
            logger.error(
                f"Database error fetching features for instrument {instrument_id} ({timeframe}): {e}", exc_info=True
            )
            raise RuntimeError("Failed to fetch feature data.") from e

    async def get_latest_features(
        self, instrument_id: int, timeframe: str, num_features: Optional[int] = None
    ) -> list[FeatureData]:
        """
        Fetches the latest features for a given instrument and timeframe.
        If num_features is specified, returns that many latest feature sets.
        Otherwise, returns all features for the latest timestamp.

        Raises:
            RuntimeError: If the database query fails.
        """
        try:
            if num_features:
                if num_features <= 0:
                    raise ValueError("num_features must be a positive integer.")
                logger.debug(f"Fetching latest {num_features} features for instrument {instrument_id} ({timeframe}).")
                query = queries.feature_repo["get_latest_n_features"]
                rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe, num_features)
                rows.reverse()  # Return in chronological order
            else:
                logger.debug(
                    f"Fetching all features for the latest timestamp for instrument {instrument_id} ({timeframe})."
                )
                query = queries.feature_repo["get_latest_features"]
                rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe)

            logger.info(f"Fetched {len(rows)} latest feature records for instrument {instrument_id} ({timeframe}).")
            return [FeatureData.model_validate(dict(row)) for row in rows]
        except ValueError as e:
            logger.error(f"Invalid argument for get_latest_features: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Database error fetching latest features for {instrument_id} ({timeframe}): {e}", exc_info=True
            )
            raise RuntimeError("Failed to fetch latest feature data.") from e

    async def get_training_data(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetches and pivots features and labels into a wide DataFrame suitable for model training.

        Raises:
            RuntimeError: If the database query or data processing fails.
        """
        query = queries.feature_repo["get_training_data"]
        logger.info(f"Fetching training data for instrument {instrument_id} ({timeframe}).")

        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe, start_time, end_time)
            if not rows:
                logger.warning(
                    f"No training data found for instrument {instrument_id} ({timeframe}) in the specified range."
                )
                return pd.DataFrame()

            logger.debug(f"Fetched {len(rows)} raw records for training data. Now pivoting.")
            df = pd.DataFrame([dict(row) for row in rows])

            # Efficiently pivot the data
            pivoted_df = df.pivot_table(
                index=["ts", "label"],
                columns="feature_name",
                values="feature_value",
                aggfunc="first",  # Use first to handle potential duplicates if any
            ).reset_index()

            pivoted_df.columns.name = None
            pivoted_df = pivoted_df.set_index("ts").sort_index()

            logger.info(
                f"Successfully prepared {len(pivoted_df)} training samples for instrument {instrument_id} ({timeframe})."
            )
            return pivoted_df
        except Exception as e:
            logger.error(
                f"Failed to get or process training data for instrument {instrument_id} ({timeframe}): {e}",
                exc_info=True,
            )
            raise RuntimeError("Could not retrieve or process training data.") from e

    async def get_features_for_correlation(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetches and pivots features into a wide DataFrame suitable for correlation analysis.

        Raises:
            RuntimeError: If the database query or data processing fails.
        """
        query = queries.feature_repo["get_features"]
        logger.info(f"Fetching features for correlation analysis for instrument {instrument_id} ({timeframe}).")

        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe, start_time, end_time)
            if not rows:
                logger.warning(f"No feature data found for correlation for instrument {instrument_id} ({timeframe}).")
                return pd.DataFrame()

            logger.debug(f"Fetched {len(rows)} raw feature records for correlation. Now pivoting.")
            df = pd.DataFrame([dict(row) for row in rows])

            pivoted_df = df.pivot_table(
                index="ts", columns="feature_name", values="feature_value", aggfunc="first"
            ).reset_index()

            pivoted_df.columns.name = None
            pivoted_df = pivoted_df.set_index("ts").sort_index()

            logger.info(f"Successfully prepared {len(pivoted_df)} samples for correlation analysis.")
            return pivoted_df
        except Exception as e:
            logger.error(
                f"Failed to get or process features for correlation for instrument {instrument_id} ({timeframe}): {e}",
                exc_info=True,
            )
            raise RuntimeError("Could not retrieve or process features for correlation.") from e

    # Feature Engineering Repository Methods
    async def insert_engineered_features(
        self, instrument_id: int, timeframe: str, engineered_features: list[dict[str, Any]]
    ) -> None:
        """
        Insert engineered features into the engineered_features table.
        """
        if not engineered_features:
            logger.info("No engineered features to insert.")
            return

        insert_query = queries.feature_repo["insert_engineered_features"]
        records = [
            (
                instrument_id,
                timeframe,
                feat["timestamp"],
                feat["feature_name"],
                feat["feature_value"],
                feat["generation_method"],
                feat["source_features"],
                feat.get("quality_score", 1.0),
            )
            for feat in engineered_features
        ]

        try:
            async with self.db_manager.transaction() as conn:
                await conn.executemany(insert_query, records)
            logger.info(
                f"Successfully inserted {len(records)} engineered features for instrument {instrument_id} ({timeframe})."
            )
        except Exception as e:
            logger.error(f"Database error inserting engineered features: {e}", exc_info=True)
            raise RuntimeError("Failed to insert engineered features.") from e

    async def insert_feature_scores(
        self, instrument_id: int, timeframe: str, feature_scores: list[dict[str, Any]]
    ) -> None:
        """
        Insert feature importance scores into the feature_scores table.
        """
        if not feature_scores:
            logger.info("No feature scores to insert.")
            return

        insert_query = queries.feature_repo["insert_feature_scores"]
        records = [
            (
                instrument_id,
                timeframe,
                score["training_timestamp"],
                score["feature_name"],
                score["importance_score"],
                score["stability_score"],
                score["consistency_score"],
                score["composite_score"],
                score["model_version"],
            )
            for score in feature_scores
        ]

        try:
            async with self.db_manager.transaction() as conn:
                await conn.executemany(insert_query, records)
            logger.info(
                f"Successfully inserted {len(records)} feature scores for instrument {instrument_id} ({timeframe})."
            )
        except Exception as e:
            logger.error(f"Database error inserting feature scores: {e}", exc_info=True)
            raise RuntimeError("Failed to insert feature scores.") from e

    async def get_latest_feature_selection(self, instrument_id: int, timeframe: str) -> Optional[list[str]]:
        """
        Get the latest feature selection for an instrument/timeframe.
        """
        query = queries.feature_repo["get_latest_feature_selection"]
        logger.debug(f"Fetching latest feature selection for instrument {instrument_id} ({timeframe}).")
        try:
            row = await self.db_manager.fetch_row(query, instrument_id, timeframe)
            if row and row["selected_features"]:
                logger.info(
                    f"Found {len(row['selected_features'])} selected features for instrument {instrument_id} ({timeframe})."
                )
                return list(row["selected_features"])
            logger.info(f"No feature selection history found for instrument {instrument_id} ({timeframe}).")
            return None
        except Exception as e:
            logger.error(f"Database error fetching latest feature selection: {e}", exc_info=True)
            raise RuntimeError("Failed to fetch latest feature selection.") from e

    async def insert_feature_selection_history(
        self,
        instrument_id: int,
        timeframe: str,
        selected_features: list[str],
        selection_criteria: dict[str, Any],
        total_features_available: int,
        selection_method: str,
        model_version: str,
    ) -> None:
        """
        Insert a feature selection history record.
        """
        insert_query = queries.feature_repo["insert_feature_selection_history"]
        timestamp = datetime.now(APP_TIMEZONE)
        logger.info(f"Inserting feature selection history for instrument {instrument_id} ({timeframe}) at {timestamp}.")

        try:
            await self.db_manager.execute(
                insert_query,
                instrument_id,
                timeframe,
                timestamp,
                selected_features,
                selection_criteria,
                total_features_available,
                len(selected_features),
                selection_method,
                model_version,
            )
            logger.info("Successfully inserted feature selection history.")
        except Exception as e:
            logger.error(f"Database error inserting feature selection history: {e}", exc_info=True)
            raise RuntimeError("Failed to insert feature selection history.") from e
