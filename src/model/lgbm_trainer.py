import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.database.feature_repo import FeatureRepository
from src.database.label_repo import LabelRepository
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger
from src.utils.performance_metrics import PerformanceMetrics

config = config_loader.get_config()


class LGBMTrainer:
    """
    Trains a LightGBM model using historical features and labels with proper
    time-series validation (walk-forward).
    """

    def __init__(
        self,
        feature_repo: FeatureRepository,
        label_repo: LabelRepository,
        error_handler: ErrorHandler,
        health_monitor: HealthMonitor,
        performance_metrics: PerformanceMetrics,
    ):
        self.feature_repo = feature_repo
        self.label_repo = label_repo
        self.error_handler = error_handler
        self.health_monitor = health_monitor
        self.performance_metrics = performance_metrics

        # TODO: Add model_training config section
        # TODO: Add model_artifacts_path to StorageConfig
        self.artifacts_path = Path(config.model_training.artifacts_path)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)

        logger.info("LGBMTrainer initialized with walk-forward validation.")

    async def train_model(self, instrument_id: int, timeframe: str) -> Optional[Path]:
        """
        Orchestrates the model training process using walk-forward validation.
        """

        try:
            logger.info(f"Starting LightGBM model training for {instrument_id} ({timeframe})...")

            # 1. Fetch all available training data
            full_df = await self._fetch_training_data(instrument_id, timeframe)
            if full_df.empty or len(full_df) < config.model_training.min_data_for_training:
                logger.warning(f"Insufficient data for training for {instrument_id} ({timeframe}).")
                return None

            # 2. Perform walk-forward validation and get final model
            model, metrics, features = await self._walk_forward_validation(full_df)

            if model is None:
                logger.error(f"Model training failed during walk-forward validation for {instrument_id}.")
                return None

            # 3. Log final evaluation metrics
            logger.info(f"Final Walk-Forward Evaluation for {instrument_id} ({timeframe}):")
            for key, value in metrics.items():
                logger.info(f"  {key.replace('_', ' ').title()}: {value:.4f}")

            # 4. Save model and metadata
            model_filename = f"lgbm_model_{instrument_id}_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            model_path = self.artifacts_path / model_filename
            model.save_model(str(model_path))

            metadata = {
                "instrument_id": instrument_id,
                "timeframe": timeframe,
                "training_date": datetime.now().isoformat(),
                "metrics": metrics,
                "lgbm_params": config.model_training.lgbm_params,
                "features": features,
                "walk_forward_config": {
                    "initial_train_size": config.model_training.walk_forward_validation.initial_train_size,
                    "validation_size": config.model_training.walk_forward_validation.validation_size,
                    "step_size": config.model_training.walk_forward_validation.step_size,
                },
            }
            metadata_path = model_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            logger.info(f"Model saved to {model_path}")
            logger.info(f"Model metadata saved to {metadata_path}")

            # TODO: Add record_successful_operation method to HealthMonitor
            self.performance_metrics.stop_timer(
                "lgbm_trainer", self.performance_metrics.start_timer("lgbm_trainer"), True
            )

            return model_path

        except Exception as e:
            await self.error_handler.handle_error(
                "lgbm_trainer",
                f"Error during model training for {instrument_id} ({timeframe}): {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe, "error": str(e)},
            )
            self.performance_metrics.stop_timer(
                "lgbm_trainer", self.performance_metrics.start_timer("lgbm_trainer"), False
            )
            return None

    async def _fetch_training_data(self, instrument_id: int, timeframe: str) -> pd.DataFrame:
        """Fetches and prepares features and labels for training."""
        end_time = datetime.now()
        start_time = end_time - pd.DateOffset(days=config.model_training.historical_data_lookback_days)

        features_records = await self.feature_repo.get_features(instrument_id, timeframe, start_time, end_time)
        labels_records = await self.label_repo.get_labels(instrument_id, start_time, end_time)

        if not features_records or not labels_records:
            return pd.DataFrame()

        features_df = pd.DataFrame(features_records)
        labels_df = pd.DataFrame(labels_records)

        features_pivot_df = features_df.pivot_table(index="ts", columns="feature_name", values="feature_value")

        merged_df = pd.merge(features_pivot_df, labels_df[["ts", "label"]], left_index=True, right_on="ts", how="inner")
        merged_df = merged_df.set_index("ts").sort_index()
        merged_df = merged_df.dropna()
        merged_df["label"] = merged_df["label"].astype(int)

        logger.info(f"Fetched and prepared {len(merged_df)} rows for training for {instrument_id} ({timeframe}).")
        return merged_df

    async def _walk_forward_validation(
        self, df: pd.DataFrame
    ) -> tuple[Optional[lgb.Booster], dict[str, float], list[str]]:
        """
        Performs walk-forward validation and returns the final trained model and aggregated metrics.
        """
        initial_train_size = config.model_training.walk_forward_validation.initial_train_size
        validation_size = config.model_training.walk_forward_validation.validation_size
        step_size = config.model_training.walk_forward_validation.step_size

        X = df.drop(columns=["label"])
        y = df["label"]

        all_preds, all_true = [], []
        model = None

        loop = asyncio.get_running_loop()

        def _train_and_predict(train_X, train_y, valid_X) -> tuple:
            lgbm = lgb.LGBMClassifier()  # TODO: Add lgbm_params to config
            lgbm.fit(train_X, train_y)
            return lgbm.predict(valid_X), lgbm.booster_

        start_index = 0
        while start_index + initial_train_size + validation_size <= len(df):
            train_end = start_index + initial_train_size
            validation_end = train_end + validation_size

            train_X, train_y = X.iloc[start_index:train_end], y.iloc[start_index:train_end]
            valid_X, valid_y = X.iloc[train_end:validation_end], y.iloc[train_end:validation_end]

            logger.info(f"Training window: {train_X.index.min()} to {train_X.index.max()} ({len(train_X)} samples)")
            logger.info(f"Validation window: {valid_X.index.min()} to {valid_X.index.max()} ({len(valid_X)} samples)")

            # Offload the CPU-bound training to a thread
            preds, model = await loop.run_in_executor(None, _train_and_predict, train_X, train_y, valid_X)

            all_preds.extend(preds)
            all_true.extend(valid_y.values)

            start_index += step_size

        if not all_true:
            logger.warning("Not enough data to perform a single walk-forward step.")
            return None, {}, []

        metrics = {
            "accuracy": accuracy_score(all_true, all_preds),
            "precision": precision_score(all_true, all_preds, average="weighted", zero_division=0),
            "recall": recall_score(all_true, all_preds, average="weighted", zero_division=0),
            "f1_score": f1_score(all_true, all_preds, average="weighted", zero_division=0),
        }

        return model, metrics, list(X.columns)
