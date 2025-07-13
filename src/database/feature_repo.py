from datetime import datetime
from typing import Optional

import asyncpg
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

    async def insert_features(self, instrument_id: int, timeframe: str, features_data: list[FeatureData]) -> str:
        """
        Inserts calculated features data into the features table.

        Args:
            instrument_id (int): The ID of the instrument.
            timeframe (str): The timeframe (e.g., '5min', '15min', '60min').
            features_data (List[FeatureData]): List of FeatureData objects.

        Returns:
            str: Command status from the database.
        """
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
                # Using executemany for batch insert/update
                await conn.executemany(queries.feature_repo["insert_features"], records)
                logger.info(f"Inserted/Updated {len(features_data)} feature records for {instrument_id} ({timeframe}).")
                return "INSERT OK"
        except Exception as e:
            logger.error(f"Error inserting features data for {instrument_id} ({timeframe}): {e}")
            raise

    async def get_features(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> list[asyncpg.Record]:
        """
        Fetches features data for a given instrument and timeframe within a time range.
        """
        try:
            rows = await self.db_manager.fetch_rows(
                queries.feature_repo["get_features"], instrument_id, timeframe, start_time, end_time
            )
            logger.debug(
                f"Fetched {len(rows)} feature records for {instrument_id} ({timeframe}) from {start_time} to {end_time}."
            )
            return rows
        except Exception as e:
            logger.error(f"Error fetching features data for {instrument_id} ({timeframe}): {e}")
            raise

    async def get_latest_features(
        self, instrument_id: int, timeframe: str, num_features: Optional[int] = None
    ) -> list[FeatureData]:
        """
        Fetches the latest features for a given instrument and timeframe.
        If num_features is specified, returns that many latest feature sets.
        Otherwise, returns all features for the latest timestamp.
        """
        if num_features:
            # This query fetches the latest 'num_features' entries, assuming each entry is a feature value
            # If you need a complete set of features for the *latest timestamp*, a different query is needed.
            try:
                rows = await self.db_manager.fetch_rows(
                    queries.feature_repo["get_latest_n_features"], instrument_id, timeframe, num_features
                )
                rows.reverse()  # Return in chronological order
                logger.debug(f"Fetched {len(rows)} latest feature records for {instrument_id} ({timeframe}).")
                return [FeatureData.model_validate(dict(row)) for row in rows]
            except Exception as e:
                logger.error(f"Error fetching latest features for {instrument_id} ({timeframe}): {e}")
                raise
        else:
            # Fetch all features for the single latest timestamp
            try:
                rows = await self.db_manager.fetch_rows(
                    queries.feature_repo["get_latest_features"], instrument_id, timeframe
                )
                logger.debug(f"Fetched all features for the latest timestamp for {instrument_id} ({timeframe}).")
                return [FeatureData.model_validate(dict(row)) for row in rows]
            except Exception as e:
                logger.error(f"Error fetching latest features for {instrument_id} ({timeframe}): {e}")
                raise

    async def get_features_by_instrument(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> list[FeatureData]:
        """
        Fetches features data for a given instrument and timeframe within a time range.
        Returns properly typed FeatureData objects.
        """
        try:
            rows = await self.db_manager.fetch_rows(
                queries.feature_repo["get_features"], instrument_id, timeframe, start_time, end_time
            )
            logger.debug(
                f"Fetched {len(rows)} feature records for {instrument_id} ({timeframe}) from {start_time} to {end_time}."
            )
            return [FeatureData.model_validate(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching features data for {instrument_id} ({timeframe}): {e}")
            raise

    async def get_training_data(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetches features and labels for a given instrument and timeframe within a time range,
        and returns them as a single pivoted Pandas DataFrame suitable for training.
        """
        # SQL query to fetch features and labels
        query = queries.feature_repo["get_training_data"]

        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe, start_time, end_time)
            if not rows:
                logger.info(
                    f"No training data found for {instrument_id} ({timeframe}) from {start_time} to {end_time}."
                )
                return pd.DataFrame()

            # Convert asyncpg.Record objects to dictionaries and then to DataFrame
            df = pd.DataFrame([dict(row) for row in rows])

            # Pivot features to wide format and merge with labels
            # Create a unique identifier for each feature row before pivoting
            df["__temp_id"] = df.groupby(["ts", "feature_name"]).cumcount()

            pivoted_df = df.pivot_table(
                index=["ts", "label", "__temp_id"], columns="feature_name", values="feature_value"
            ).reset_index()

            # Drop the temporary ID column
            pivoted_df = pivoted_df.drop(columns=["__temp_id"])

            # Clean up column names from pivot_table
            pivoted_df.columns.name = None

            # Set 'ts' as index and sort
            pivoted_df = pivoted_df.set_index("ts").sort_index()

            logger.info(f"Fetched and prepared {len(pivoted_df)} training samples for {instrument_id} ({timeframe}).")
            return pivoted_df
        except Exception as e:
            logger.error(f"Error fetching training data for {instrument_id} ({timeframe}): {e}")
            raise

    async def get_features_for_correlation(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetches features data for a given instrument and timeframe within a time range,
        and returns them as a single pivoted Pandas DataFrame suitable for correlation calculation.
        """
        query = queries.feature_repo["get_features"]

        try:
            rows = await self.db_manager.fetch_rows(query, instrument_id, timeframe, start_time, end_time)
            if not rows:
                logger.info(
                    f"No feature data found for correlation for {instrument_id} ({timeframe}) from {start_time} to {end_time}."
                )
                return pd.DataFrame()

            df = pd.DataFrame([dict(row) for row in rows])

            # Pivot features to wide format
            df["__temp_id"] = df.groupby(["ts", "feature_name"]).cumcount()
            pivoted_df = df.pivot_table(
                index=["ts", "__temp_id"], columns="feature_name", values="feature_value"
            ).reset_index()
            pivoted_df = pivoted_df.drop(columns=["__temp_id"])
            pivoted_df.columns.name = None
            pivoted_df = pivoted_df.set_index("ts").sort_index()

            logger.debug(
                f"Fetched and prepared {len(pivoted_df)} feature samples for correlation for {instrument_id} ({timeframe})."
            )
            return pivoted_df
        except Exception as e:
            logger.error(f"Error fetching features for correlation for {instrument_id} ({timeframe}): {e}")
            raise

    # Feature Engineering Repository Methods
    async def insert_engineered_features(
        self, instrument_id: int, timeframe: str, engineered_features: list[dict]
    ) -> None:
        """
        Insert engineered features into the engineered_features table.
        """
        if not engineered_features:
            return

        insert_query = queries.feature_repo["insert_engineered_features"]

        try:
            async with self.db_manager.get_connection() as conn:
                await conn.executemany(
                    insert_query,
                    [
                        (
                            instrument_id,
                            timeframe,
                            feat["timestamp"],
                            feat["feature_name"],
                            feat["feature_value"],
                            feat["generation_method"],
                            feat["source_features"],
                            feat["quality_score"],
                        )
                        for feat in engineered_features
                    ],
                )
            logger.info(f"Inserted {len(engineered_features)} engineered features for {instrument_id} ({timeframe})")
        except Exception as e:
            logger.error(f"Error inserting engineered features: {e}")
            raise

    async def insert_feature_scores(self, instrument_id: int, timeframe: str, feature_scores: list[dict]) -> None:
        """
        Insert feature importance scores into the feature_scores table.
        """
        if not feature_scores:
            return

        insert_query = queries.feature_repo["insert_feature_scores"]

        try:
            async with self.db_manager.get_connection() as conn:
                await conn.executemany(
                    insert_query,
                    [
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
                    ],
                )
            logger.info(f"Inserted {len(feature_scores)} feature scores for {instrument_id} ({timeframe})")
        except Exception as e:
            logger.error(f"Error inserting feature scores: {e}")
            raise

    async def get_latest_feature_selection(self, instrument_id: int, timeframe: str) -> Optional[list[str]]:
        """
        Get the latest feature selection for an instrument/timeframe.
        """
        query = queries.feature_repo["get_latest_feature_selection"]

        try:
            async with self.db_manager.get_connection() as conn:
                row = await conn.fetchrow(query, instrument_id, timeframe)
                if row:
                    return list(row["selected_features"])
                return None
        except Exception as e:
            logger.error(f"Error fetching latest feature selection: {e}")
            raise

    async def insert_feature_selection_history(
        self,
        instrument_id: int,
        timeframe: str,
        selected_features: list[str],
        selection_criteria: dict,
        total_features_available: int,
        selection_method: str,
        model_version: str,
    ) -> None:
        """
        Insert feature selection history record.
        """
        insert_query = queries.feature_repo["insert_feature_selection_history"]

        try:
            timestamp = datetime.now(APP_TIMEZONE)
            async with self.db_manager.get_connection() as conn:
                await conn.execute(
                    insert_query,
                    instrument_id,
                    timeframe,
                    timestamp,
                    selected_features,
                    selection_criteria,
                    total_features_available,
                    len(selected_features),  # features_selected
                    selection_method,
                    model_version,
                )
            logger.info(f"Inserted feature selection history for {instrument_id} ({timeframe})")
        except Exception as e:
            logger.error(f"Error inserting feature selection history: {e}")
            raise
