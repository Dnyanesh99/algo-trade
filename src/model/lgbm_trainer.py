import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.core.feature_calculator import FeatureCalculator
from src.database.feature_repo import FeatureRepository
from src.database.label_repo import LabelRepository
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import config
from src.utils.logger import LoggerSetup
from src.utils.performance_metrics import PerformanceMetrics

logger = LoggerSetup.get_logger()


class LGBMTrainer:
    """
    Production-grade LightGBM trainer with advanced features:
    - Walk-forward validation with proper time series splits
    - Hyperparameter optimization
    - Feature importance tracking
    - Model versioning and artifact management
    - Performance monitoring and alerting
    """

    def __init__(
        self,
        feature_repo: FeatureRepository,
        label_repo: LabelRepository,
        error_handler: ErrorHandler,
        health_monitor: HealthMonitor,
        performance_metrics: PerformanceMetrics,
        feature_calculator: FeatureCalculator,
    ):
        self.feature_repo = feature_repo
        self.label_repo = label_repo
        self.error_handler = error_handler
        self.health_monitor = health_monitor
        self.performance_metrics = performance_metrics
        self.feature_calculator = feature_calculator

        # Load configuration
        self.model_config = config.model
        self.artifacts_path = Path(config.paths.artifact_dir)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)

        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()

        # Track training history
        self.training_history: list[dict[str, Any]] = []

        logger.info("LGBMTrainer initialized with production configuration.")

    async def train_model(
        self, instrument_id: int, timeframe: str, optimize_hyperparams: bool = False
    ) -> Optional[Path]:
        """
        Main training orchestration with optional hyperparameter optimization.
        """
        timer_id = self.performance_metrics.start_timer("model_training")

        try:
            logger.info(
                f"Starting LightGBM training for {instrument_id} ({timeframe}) "
                f"with hyperparameter optimization: {optimize_hyperparams}"
            )

            # 1. Fetch and prepare data
            train_df = await self._fetch_training_data(instrument_id, timeframe)
            min_data_for_training = self.model_config.walk_forward_splits * 50  # A reasonable minimum
            if train_df.empty or len(train_df) < min_data_for_training:
                logger.warning(
                    f"Insufficient data for {instrument_id} ({timeframe}): {len(train_df)} < {min_data_for_training}"
                )
                return None

            # 2. Clean and validate data
            train_df = self._clean_data(train_df)

            # 3. Feature engineering (assuming this is part of your calculator)
            # train_df = self.feature_calculator.add_model_specific_features(train_df)

            # 4. Split features and labels
            feature_cols = [
                col
                for col in train_df.columns
                if col not in ["label", "timestamp", "barrier_return", "volatility_at_entry"]
            ]
            X = train_df[feature_cols]
            y = train_df["label"]

            # 5. Handle class imbalance
            class_weights = self._calculate_class_weights(y)

            # 6. Get optimal parameters
            # Note: A proper hyperparameter optimization implementation (like with Optuna)
            # would be more complex and is simplified here.
            best_params = self._get_default_params()

            # 7. Perform walk-forward validation
            model, metrics, feature_importance = await self._walk_forward_validation(X, y, best_params, class_weights)

            if model is None:
                logger.error(f"Model training failed for {instrument_id} ({timeframe})")
                return None

            # 8. Train final model on all data
            final_model = await self._train_final_model(X, y, best_params, class_weights)

            # 9. Calculate and log metrics
            self._log_training_metrics(metrics, feature_importance, instrument_id, timeframe)

            # 10. Save model and artifacts
            model_path = await self._save_model_artifacts(
                final_model, feature_cols, feature_importance, metrics, best_params, instrument_id, timeframe
            )

            self.performance_metrics.stop_timer("model_training", timer_id, True)
            await self.health_monitor.record_successful_operation("model_training")

            return model_path

        except Exception as e:
            await self.error_handler.handle_error(
                "lgbm_trainer",
                f"Training failed for {instrument_id} ({timeframe}): {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe, "error": str(e)},
            )
            self.performance_metrics.stop_timer("model_training", timer_id, False)
            return None

    async def _fetch_training_data(self, instrument_id: int, timeframe: str) -> pd.DataFrame:
        """
        Fetches features and labels with proper time alignment.
        """
        # A more realistic lookback for training
        end_time = datetime.now()
        start_time = end_time - pd.DateOffset(years=2)

        features_records = await self.feature_repo.get_features_by_instrument(
            instrument_id=instrument_id, start_date=start_time, end_date=end_time
        )
        labels_records = await self.label_repo.get_labels_by_instrument(
            instrument_id=instrument_id, start_date=start_time, end_date=end_time
        )

        if not features_records or not labels_records:
            return pd.DataFrame()

        features_df = pd.DataFrame(features_records)
        labels_df = pd.DataFrame(labels_records)

        # Pivot features to wide format
        features_pivot = features_df.pivot_table(
            index="ts", columns="feature_name", values="feature_value"
        ).rename_axis(None, axis=1)  # remove axis name

        # Merge with labels
        merged_df = pd.merge(features_pivot, labels_df[["ts", "label"]], on="ts", how="inner")

        merged_df = merged_df.set_index("ts").sort_index()
        logger.info(
            f"Fetched {len(merged_df)} samples for {instrument_id} ({timeframe}) "
            f"with {len(features_pivot.columns)} features"
        )
        return merged_df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data by handling missing values.
        """
        df = df.dropna()
        df["label"] = df["label"].astype(int)
        logger.info(f"Data cleaned: {len(df)} samples remaining")
        return df

    def _calculate_class_weights(self, y: pd.Series) -> dict[int, float]:
        """
        Calculates class weights to handle imbalanced data.
        """
        class_counts = y.value_counts()
        total_samples = len(y)
        weights = {
            class_label: total_samples / (len(class_counts) * count) for class_label, count in class_counts.items()
        }
        logger.info(f"Class weights calculated: {weights}")
        return weights

    def _get_default_params(self) -> dict[str, Any]:
        """
        Returns default LightGBM parameters from config.
        """
        params = self.model_config.params.copy()
        params.update(
            {
                "objective": "multiclass",
                "num_class": 3,
                "metric": ["multi_logloss", "multi_error"],
                "boosting_type": "gbdt",
                "verbose": -1,
                "seed": 42,
                "deterministic": True,
            }
        )
        return params

    async def _walk_forward_validation(
        self, X: pd.DataFrame, y: pd.Series, params: dict[str, Any], class_weights: dict[int, float]
    ) -> tuple[Optional[lgb.Booster], dict[str, Any], dict[str, float]]:
        """
        Performs walk-forward validation.
        """
        tscv = TimeSeriesSplit(n_splits=self.model_config.walk_forward_splits)
        all_predictions = []
        all_actuals = []
        all_probabilities = []
        feature_importance_sum = np.zeros(len(X.columns))
        n_windows = 0
        model = None
        loop = asyncio.get_running_loop()

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            logger.info(
                f"Walk-forward window {n_windows + 1}: Train ({len(X_train)} samples), Val ({len(X_val)} samples)"
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            sample_weights = y_train.map(class_weights).values
            train_data = lgb.Dataset(X_train_scaled, label=y_train + 1, weight=sample_weights)
            val_data = lgb.Dataset(X_val_scaled, label=y_val + 1, reference=train_data)

            model = await loop.run_in_executor(
                None,
                lgb.train,
                params,
                train_data,
                1000,
                [val_data],
                [lgb.early_stopping(50, verbose=False)],
            )

            y_pred_proba = model.predict(X_val_scaled, num_iteration=model.best_iteration)
            y_pred = np.argmax(y_pred_proba, axis=1) - 1

            all_predictions.extend(y_pred)
            all_actuals.extend(y_val.values)
            all_probabilities.extend(y_pred_proba)

            feature_importance_sum += model.feature_importance(importance_type="gain")
            n_windows += 1

        if n_windows == 0:
            return None, {}, {}

        metrics = self._calculate_metrics(all_actuals, all_predictions, all_probabilities)
        feature_importance = dict(
            sorted(zip(X.columns, feature_importance_sum / n_windows), key=lambda x: x[1], reverse=True)
        )

        logger.info(f"Walk-forward validation completed with {n_windows} windows")
        return model, metrics, feature_importance

    def _calculate_metrics(self, y_true: list[int], y_pred: list[int], y_proba: list[np.ndarray]) -> dict[str, Any]:
        """
        Calculates comprehensive evaluation metrics.
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc_ovr_weighted": roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    async def _train_final_model(
        self, X: pd.DataFrame, y: pd.Series, params: dict[str, Any], class_weights: dict[int, float]
    ) -> lgb.Booster:
        """
        Trains the final model on all available data.
        """
        logger.info("Training final model on all data...")
        X_scaled = self.scaler.fit_transform(X)
        sample_weights = y.map(class_weights).values
        train_data = lgb.Dataset(X_scaled, label=y + 1, weight=sample_weights)

        final_params = params.copy()
        num_boost_round = final_params.pop("num_boost_round", 1000)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lgb.train, final_params, train_data, num_boost_round)

    def _log_training_metrics(
        self, metrics: dict[str, Any], feature_importance: dict[str, float], instrument_id: int, timeframe: str
    ):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training Results for {instrument_id} ({timeframe})")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  F1 Weighted: {metrics.get('f1_weighted', 0):.4f}")
        logger.info(f"  ROC AUC: {metrics.get('roc_auc_ovr_weighted', 0):.4f}")
        logger.info("\nTop 10 Features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
            logger.info(f"  {i + 1}. {feature}: {importance:.2f}")
        logger.info(f"{'=' * 60}\n")

    async def _save_model_artifacts(
        self,
        model: lgb.Booster,
        feature_names: list[str],
        feature_importance: dict[str, float],
        metrics: dict[str, Any],
        params: dict[str, Any],
        instrument_id: int,
        timeframe: str,
    ) -> Path:
        """
        Saves model and all associated artifacts with versioning.
        """
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"lgbm_{instrument_id}_{timeframe}_{version_id}"
        version_dir = self.artifacts_path / model_name
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "model.txt"
        model.save_model(str(model_path))

        scaler_path = version_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)

        metadata = {
            "version_id": version_id,
            "instrument_id": instrument_id,
            "timeframe": timeframe,
            "training_date": datetime.now().isoformat(),
            "model_type": "LightGBM",
            "features": {"names": feature_names, "count": len(feature_names), "importance": feature_importance},
            "metrics": metrics,
            "parameters": params,
        }
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4, default=str)

        latest_link = self.artifacts_path / f"latest_{instrument_id}_{timeframe}"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(version_dir.name)

        await self._cleanup_old_models(instrument_id, timeframe)
        logger.info(f"Model artifacts saved to {version_dir}")
        return model_path

    async def _cleanup_old_models(self, instrument_id: int, timeframe: str):
        """
        Removes old model versions keeping only the most recent N.
        """
        max_versions = 5  # Keep last 5 versions
        pattern = f"lgbm_{instrument_id}_{timeframe}_*"
        model_dirs = sorted(
            [d for d in self.artifacts_path.glob(pattern) if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True
        )

        for old_dir in model_dirs[max_versions:]:
            try:
                shutil.rmtree(old_dir)
                logger.info(f"Removed old model version: {old_dir.name}")
            except Exception as e:
                logger.error(f"Failed to remove old model {old_dir}: {e}")
