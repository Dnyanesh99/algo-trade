import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import pytz
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
from src.metrics import metrics_registry
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import ConfigLoader, ModelTrainingConfig
from src.utils.logger import LOGGER as logger

config = ConfigLoader().get_config()
app_timezone = pytz.timezone(config.system.timezone) if config.system else pytz.UTC


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
        feature_calculator: FeatureCalculator,
    ):
        self.feature_repo = feature_repo
        self.label_repo = label_repo
        self.error_handler = error_handler
        self.health_monitor = health_monitor
        self.feature_calculator = feature_calculator

        # NEW: Feature Engineering Pipeline Integration
        self.feature_engineering = self.feature_calculator.feature_engineering

        # Load configuration
        assert config.model_training is not None
        self.model_config: ModelTrainingConfig = config.model_training
        self.artifacts_path = Path(self.model_config.artifacts_path)
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
        training_start_time = datetime.now()

        try:
            logger.info(
                f"Starting LightGBM training for {instrument_id} ({timeframe}) "
                f"with hyperparameter optimization: {optimize_hyperparams}"
            )

            # 1. Fetch and prepare data
            train_df = await self._fetch_training_data(instrument_id, timeframe)
            min_data_for_training = self.model_config.min_data_for_training
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
            if optimize_hyperparams:
                best_params = await self._optimize_hyperparameters(X, y, class_weights)
            else:
                best_params = self._get_default_params()

            # 7. Perform walk-forward validation
            (
                model,
                metrics,
                feature_importance,
                last_fold_scaler,
                last_fold_X_train,
            ) = await self._walk_forward_validation(X, y, best_params, class_weights)

            if model is None or last_fold_scaler is None or last_fold_X_train is None:
                logger.error(f"Model training failed for {instrument_id} ({timeframe}) during walk-forward validation.")
                return None

            self.scaler = last_fold_scaler

            # 8. Train final model on all data
            final_model = await self._train_final_model(X, y, best_params, class_weights)

            # 9. Calculate and log metrics
            self._log_training_metrics(metrics, feature_importance, instrument_id, timeframe)

            # 10. Save model and artifacts
            feature_stats = self._get_feature_stats(last_fold_X_train)
            model_path = await self._save_model_artifacts(
                final_model,
                feature_cols,
                feature_importance,
                metrics,
                best_params,
                instrument_id,
                timeframe,
                feature_stats,
            )

            # NEW: Update feature selection based on training results
            await self._update_feature_selection(instrument_id, timeframe, feature_importance, metrics, model_path)

            # Record successful training metrics
            training_duration = (datetime.now() - training_start_time).total_seconds()
            metrics_registry.record_model_training(
                instrument=str(instrument_id),
                timeframe=timeframe,
                model_type="lightgbm",
                success=True,
                duration=training_duration,
            )
            self.health_monitor.record_successful_operation("model_training")

            return model_path

        except Exception as e:
            await self.error_handler.handle_error(
                "lgbm_trainer",
                f"Training failed for {instrument_id} ({timeframe}): {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe, "error": str(e)},
            )

            # Record failed training metrics
            training_duration = (datetime.now() - training_start_time).total_seconds()
            metrics_registry.record_model_training(
                instrument=str(instrument_id),
                timeframe=timeframe,
                model_type="lightgbm",
                success=False,
                duration=training_duration,
            )
            return None

    async def _fetch_training_data(self, instrument_id: int, timeframe: str) -> pd.DataFrame:
        """
        Fetches features and labels with proper time alignment.
        """
        # Use configured lookback for training
        end_time = app_timezone.localize(datetime.now()) if datetime.now().tzinfo is None else datetime.now()
        start_time = end_time - pd.DateOffset(days=self.model_config.historical_data_lookback_days)

        train_df = await self.feature_repo.get_training_data(
            instrument_id=instrument_id, timeframe=timeframe, start_time=start_time, end_time=end_time
        )

        if train_df.empty:
            return pd.DataFrame()

        return train_df

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
        assert self.model_config is not None
        params = self.model_config.lgbm_params.copy()
        params.update(
            {
                "objective": self.model_config.lgbm_params["objective"],
                "num_class": self.model_config.lgbm_params["num_class"],
                "metric": self.model_config.lgbm_params["metric"],
                "boosting_type": self.model_config.lgbm_params["boosting_type"],
                "verbose": self.model_config.lgbm_params["verbose"],
                "seed": self.model_config.lgbm_params["seed"],
                "deterministic": self.model_config.lgbm_params["deterministic"],
            }
        )
        return params

    def _get_feature_stats(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Calculates descriptive statistics for features."""
        stats = {}
        desc = df.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        for col in df.columns:
            stats[col] = {
                "mean": desc[col]["mean"],
                "std": desc[col]["std"],
                "min": desc[col]["min"],
                "max": desc[col]["max"],
                "p25": desc[col]["25%"],
                "p50": desc[col]["50%"],
                "p75": desc[col]["75%"],
            }
        return stats

    async def _walk_forward_validation(
        self, X: pd.DataFrame, y: pd.Series, params: dict[str, Any], class_weights: dict[int, float]
    ) -> tuple[
        Optional[lgb.Booster], dict[str, Any], dict[str, float], Optional[StandardScaler], Optional[pd.DataFrame]
    ]:
        """
        Performs walk-forward validation and returns model, metrics, feature importance,
        the scaler from the last fold, and the training features from the last fold.
        """
        assert self.model_config is not None
        tscv = TimeSeriesSplit(n_splits=self.model_config.walk_forward_validation.n_splits)
        all_predictions = []
        all_actuals = []
        all_probabilities: list[np.ndarray] = []
        feature_importance_sum = np.zeros(len(X.columns))
        n_windows = 0
        model = None
        loop = asyncio.get_running_loop()
        last_fold_scaler: Optional[StandardScaler] = None
        last_fold_X_train: Optional[pd.DataFrame] = None

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            logger.info(
                f"Walk-forward window {n_windows + 1}: Train ({len(X_train)} samples), Val ({len(X_val)} samples)"
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            last_fold_scaler = scaler
            last_fold_X_train = X_train

            sample_weights = y_train.map(class_weights).values
            train_data = lgb.Dataset(X_train_scaled, label=y_train + 1, weight=sample_weights)
            val_data = lgb.Dataset(X_val_scaled, label=y_val + 1, reference=train_data)

            def _train_lgb_model(train_data_arg: lgb.Dataset, val_data_arg: lgb.Dataset) -> lgb.Booster:
                return lgb.train(
                    params,
                    train_data_arg,
                    self.model_config.final_model.num_boost_round,
                    valid_sets=[val_data_arg],
                    callbacks=[lgb.early_stopping(self.model_config.optimization.early_stopping_rounds, verbose=False)],
                )

            model = await loop.run_in_executor(None, _train_lgb_model, train_data, val_data)

            assert model is not None
            y_pred_proba = model.predict(X_val_scaled, num_iteration=model.best_iteration)
            y_pred = np.argmax(y_pred_proba, axis=1) - 1

            all_predictions.extend(y_pred)
            all_actuals.extend(y_val.values)
            all_probabilities.extend(y_pred_proba)

            feature_importance_sum += model.feature_importance(importance_type="gain")
            n_windows += 1

        if n_windows == 0:
            return None, {}, {}, None, None

        metrics = self._calculate_metrics(all_actuals, all_predictions, all_probabilities)
        feature_importance = dict(
            sorted(zip(X.columns, feature_importance_sum / n_windows), key=lambda x: x[1], reverse=True)
        )

        logger.info(f"Walk-forward validation completed with {n_windows} windows")
        return model, metrics, feature_importance, last_fold_scaler, last_fold_X_train

    async def _optimize_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series, class_weights: dict[int, float]
    ) -> dict[str, Any]:
        """Performs hyperparameter optimization using Optuna."""

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", *self.model_config.optimization.n_estimators_range),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *self.model_config.optimization.learning_rate_range, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", *self.model_config.optimization.num_leaves_range),
                "max_depth": trial.suggest_int("max_depth", *self.model_config.optimization.max_depth_range),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", *self.model_config.optimization.min_child_samples_range
                ),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction", *self.model_config.optimization.feature_fraction_range
                ),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction", *self.model_config.optimization.bagging_fraction_range
                ),
                "bagging_freq": trial.suggest_int("bagging_freq", *self.model_config.optimization.bagging_freq_range),
                "lambda_l1": trial.suggest_float(
                    "lambda_l1", *self.model_config.optimization.lambda_l1_range, log=True
                ),
                "lambda_l2": trial.suggest_float(
                    "lambda_l2", *self.model_config.optimization.lambda_l2_range, log=True
                ),
                "verbose": -1,
                "seed": 42,
                "deterministic": True,
            }

            tscv = TimeSeriesSplit(n_splits=self.model_config.optimization.n_splits)
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                sample_weights = y_train.map(class_weights).values
                model = lgb.LGBMClassifier(**params)  # type: ignore[arg-type]
                model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                y_pred_proba = model.predict_proba(X_val_scaled)
                scores.append(
                    roc_auc_score(
                        y_val,
                        y_pred_proba,
                        multi_class=self.model_config.optimization.roc_auc_multi_class,
                        average=self.model_config.optimization.roc_auc_average,
                    )
                )

            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.model_config.optimization.n_trials)

        logger.info(f"Best trial for hyperparameter optimization: {study.best_trial.value}")
        best_params = self._get_default_params()
        best_params.update(study.best_params)
        return best_params

    def _calculate_metrics(self, y_true: list[int], y_pred: list[int], y_proba: list[np.ndarray]) -> dict[str, Any]:
        """
        Calculates comprehensive evaluation metrics and records them to monitoring system.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc_ovr_weighted": roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        # Calculate per-class metrics for detailed monitoring
        class_report = metrics["classification_report"]
        metrics["per_class_metrics"] = {}

        for class_label in ["-1", "0", "1"]:  # sell, neutral, buy
            if class_label in class_report:
                class_metrics = class_report[class_label]
                metrics["per_class_metrics"][class_label] = {
                    "precision": class_metrics.get("precision", 0.0),
                    "recall": class_metrics.get("recall", 0.0),
                    "f1_score": class_metrics.get("f1-score", 0.0),
                    "support": class_metrics.get("support", 0),
                }

        # Calculate prediction bias
        pred_counts = np.bincount(np.array(y_pred) + 1, minlength=3)  # Adjust for -1, 0, 1
        total_predictions = len(y_pred)
        if total_predictions > 0:
            metrics["prediction_bias"] = {
                "sell_ratio": pred_counts[0] / total_predictions,
                "neutral_ratio": pred_counts[1] / total_predictions,
                "buy_ratio": pred_counts[2] / total_predictions,
            }

        return metrics

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
        num_boost_round = self.model_config.final_model.num_boost_round

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lgb.train, final_params, train_data, num_boost_round)

    def _log_training_metrics(
        self, metrics: dict[str, Any], feature_importance: dict[str, float], instrument_id: int, timeframe: str
    ) -> None:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training Results for {instrument_id} ({timeframe})")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  F1 Weighted: {metrics.get('f1_weighted', 0):.4f}")
        logger.info(f"  ROC AUC: {metrics.get('roc_auc_ovr_weighted', 0):.4f}")
        logger.info("\nTop 10 Features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[: self.model_config.top_n_features]):
            logger.info(f"  {i + 1}. {feature}: {importance:.2f}")
        logger.info(f"{'=' * 60}\n")

        # Record metrics to monitoring system
        self._record_training_metrics_to_monitoring(metrics, feature_importance, instrument_id, timeframe)

    def _record_training_metrics_to_monitoring(
        self, metrics: dict[str, Any], feature_importance: dict[str, float], instrument_id: int, timeframe: str
    ) -> None:
        """Record training metrics to the monitoring system."""
        # Record overall performance metrics
        metrics_registry.record_model_performance(
            str(instrument_id), timeframe, "accuracy", "overall", metrics.get("accuracy", 0.0)
        )
        metrics_registry.record_model_performance(
            str(instrument_id), timeframe, "f1_weighted", "overall", metrics.get("f1_weighted", 0.0)
        )
        metrics_registry.record_model_performance(
            str(instrument_id), timeframe, "roc_auc", "overall", metrics.get("roc_auc_ovr_weighted", 0.0)
        )

        # Record per-class metrics
        if "per_class_metrics" in metrics:
            for class_label, class_metrics in metrics["per_class_metrics"].items():
                class_name = self.model_config.class_label_mapping.get(int(class_label), class_label)

                metrics_registry.record_model_performance(
                    str(instrument_id), timeframe, "precision", class_name, class_metrics.get("precision", 0.0)
                )
                metrics_registry.record_model_performance(
                    str(instrument_id), timeframe, "recall", class_name, class_metrics.get("recall", 0.0)
                )
                metrics_registry.record_model_performance(
                    str(instrument_id), timeframe, "f1_score", class_name, class_metrics.get("f1_score", 0.0)
                )

        # Record prediction bias
        if "prediction_bias" in metrics:
            bias_metrics = metrics["prediction_bias"]
            metrics_registry.record_prediction_bias(
                str(instrument_id), timeframe, "sell_bias", bias_metrics.get("sell_ratio", 0.0)
            )
            metrics_registry.record_prediction_bias(
                str(instrument_id), timeframe, "neutral_bias", bias_metrics.get("neutral_ratio", 0.0)
            )
            metrics_registry.record_prediction_bias(
                str(instrument_id), timeframe, "buy_bias", bias_metrics.get("buy_ratio", 0.0)
            )

        # Record feature importance (top N features)
        for feature_name, importance in list(feature_importance.items())[: self.model_config.top_n_features]:
            metrics_registry.record_feature_importance_shift(str(instrument_id), timeframe, feature_name, importance)

    async def _save_model_artifacts(
        self,
        model: lgb.Booster,
        feature_names: list[str],
        feature_importance: dict[str, float],
        metrics: dict[str, Any],
        params: dict[str, Any],
        instrument_id: int,
        timeframe: str,
        feature_stats: dict[str, dict[str, float]],
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

        stats_path = version_dir / "feature_stats.json"
        with open(stats_path, "w") as f:
            json.dump(feature_stats, f, indent=4)

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

    async def _cleanup_old_models(self, instrument_id: int, timeframe: str) -> None:
        """
        Removes old model versions keeping only the most recent N.
        """
        assert self.model_config is not None
        max_versions = self.model_config.retention.max_model_versions
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

    async def _update_feature_selection(
        self,
        instrument_id: int,
        timeframe: str,
        feature_importance: dict[str, float],
        metrics: dict[str, Any],
        model_path: Path,
    ) -> None:
        """
        Update feature selection based on training results.
        Called after successful model training to trigger automated feature selection.
        """
        try:
            # Prepare model metadata for feature selection
            model_metadata = {
                "features": {"importance": feature_importance, "count": len(feature_importance)},
                "metrics": {
                    "accuracy": metrics.get("accuracy", 0.0),
                    "f1_score": metrics.get("f1_score", 0.0),
                    "roc_auc": metrics.get("roc_auc_ovr_weighted", 0.0),
                },
                "version_id": model_path.name,
                "timestamp": datetime.now().isoformat(),
                "model_path": str(model_path),
            }

            # Call feature engineering pipeline to update selection
            selected_features = await self.feature_engineering.update_feature_selection(
                instrument_id, timeframe, model_metadata
            )

            if selected_features:
                logger.info(
                    f"Feature selection updated for {instrument_id}_{timeframe}: "
                    f"{len(selected_features)} features selected from {len(feature_importance)} total"
                )
            else:
                logger.warning(f"No feature selection update for {instrument_id}_{timeframe}")

        except Exception as e:
            logger.error(f"Failed to update feature selection for {instrument_id}_{timeframe}: {e}")
            await self.error_handler.handle_error(
                "feature_selection_update",
                f"Feature selection update failed: {e}",
                {"instrument_id": instrument_id, "timeframe": timeframe},
            )
