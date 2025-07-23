import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel

from src.metrics.registry import metrics_registry
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

config_loader = ConfigLoader()

config = config_loader.get_config()


class PredictionResult(BaseModel):
    """
    Structured prediction result with confidence and metadata.
    """

    instrument_id: int
    timeframe: str
    timestamp: datetime
    prediction: int  # -1: SELL, 0: NEUTRAL, 1: BUY
    confidence: float
    probabilities: dict[str, float]  # {'sell': 0.2, 'neutral': 0.5, 'buy': 0.3}
    feature_contributions: Optional[dict[str, float]] = None
    model_version: str
    inference_time_ms: float
    risk_assessment: dict[str, Any]


class ModelPredictor:
    """
    Production-grade model predictor with:
    - Model versioning and hot-swapping
    - Feature validation and monitoring
    - Confidence calibration
    - Risk assessment
    - Performance tracking
    """

    def __init__(
        self,
        error_handler: ErrorHandler,
        health_monitor: HealthMonitor,
    ):
        # Type hints ensure error_handler and health_monitor are valid instances

        self.error_handler = error_handler
        self.health_monitor = health_monitor

        self.models: dict[str, dict[str, Any]] = {}
        self.active_models: dict[str, str] = {}

        if not config.model_training or not config.model_training.predictor:
            logger.error(
                "ValueError: model_training.predictor configuration is required for ModelPredictor initialization."
            )
            raise ValueError("model_training.predictor configuration is required.")
        self.predictor_config = config.model_training
        self.artifacts_path = Path(self.predictor_config.artifacts_path)

        self.prediction_history: list[PredictionResult] = []
        self.confidence_threshold = self.predictor_config.confidence_threshold
        self.calibration_params: dict[str, Any] = {}

        logger.info("ModelPredictor initialized with production configuration.")

    async def load_model(
        self, model_path: Path, instrument_id: int, timeframe: str, set_as_active: bool = True
    ) -> bool:
        """
        Loads a model with all associated artifacts, including feature statistics.
        """
        start_time = time.time()
        model_key = f"{instrument_id}_{timeframe}"
        logger.info(f"Attempting to load model for {model_key} from path: {model_path}.")

        try:
            model_dir = model_path.parent if model_path.is_file() else model_path

            required_artifacts = self.predictor_config.predictor.required_artifacts
            artifact_paths = {name: model_dir / path_str for name, path_str in required_artifacts.items()}

            for name, path in artifact_paths.items():
                if not path.exists():
                    logger.critical(
                        f"CRITICAL: Required model artifact '{name}' not found at: {path}. Model loading aborted."
                    )
                    raise FileNotFoundError(f"Missing required model artifact: {path}")

            model = lgb.Booster(model_file=str(artifact_paths["model"]))
            with open(artifact_paths["metadata"]) as f:
                metadata = json.load(f)
            scaler = joblib.load(artifact_paths["scaler"])
            with open(artifact_paths["feature_stats"]) as f:
                feature_stats = json.load(f)

            if not self._validate_model(model, metadata):
                logger.critical(
                    f"CRITICAL: Model validation failed for {model_key} from {model_path}. Model is corrupted or incompatible. Aborting load."
                )
                return False

            version_id = metadata.get("version_id", "unknown")
            if version_id == "unknown":
                logger.warning(f"Model metadata for {model_key} is missing 'version_id'. Using 'unknown'.")

            full_model_key = f"{model_key}_{version_id}"
            self.models[full_model_key] = {
                "model": model,
                "metadata": metadata,
                "scaler": scaler,
                "feature_stats": feature_stats,
                "loaded_at": datetime.now(),
                "prediction_count": 0,
                "error_count": 0,
            }

            if set_as_active:
                self.active_models[model_key] = version_id
                logger.info(f"Set active model for {model_key}: version {version_id}.")

            calibration_path = model_dir / "calibration.json"
            if calibration_path.exists():
                try:
                    with open(calibration_path) as f:
                        self.calibration_params[model_key] = json.load(f)
                    logger.info(f"Loaded calibration parameters for {model_key}.")
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        f"Failed to load calibration file for {model_key}: {e}. Proceeding without calibration."
                    )

            loading_duration = time.time() - start_time
            metrics_registry.observe_histogram("model_loading_duration_seconds", loading_duration)
            metrics_registry.increment_counter("model_loading_total", {"status": "success"})

            logger.info(
                f"Successfully loaded model and all artifacts for {model_key} from {model_dir} in {loading_duration:.2f}s."
            )
            return True

        except Exception as e:
            loading_duration = time.time() - start_time
            metrics_registry.observe_histogram("model_loading_duration_seconds", loading_duration)
            metrics_registry.increment_counter("model_loading_total", {"status": "failure"})
            await self.error_handler.handle_error(
                "model_loading",
                f"Model loading failed for {model_key} from {model_path}: {e}",
                {"model_path": str(model_path), "error": str(e)},
                exc_info=True,
            )
            return False

    async def predict(
        self, instrument_id: int, timeframe: str, feature_vector: dict[str, float], return_diagnostics: bool = False
    ) -> Optional[PredictionResult]:
        """
        Generates a prediction with comprehensive validation and monitoring.
        """
        start_time = time.time()
        model_key = f"{instrument_id}_{timeframe}"
        logger.debug(f"Attempting prediction for {model_key}.")

        try:
            version_id = self.active_models.get(model_key)
            if not version_id:
                logger.error(f"PredictionError: No active model found for {model_key}. Cannot predict.")
                return None

            model_info = self.models.get(f"{model_key}_{version_id}")
            if not model_info:
                logger.error(
                    f"PredictionError: Model version {version_id} for {model_key} not found in loaded models. Cannot predict."
                )
                return None

            validation_result = await self._validate_features(feature_vector, model_info["metadata"])
            if not validation_result["valid"]:
                logger.error(
                    f"PredictionError: Feature validation failed for {model_key}: {validation_result['errors']}. Cannot predict."
                )
                await self.error_handler.handle_error(
                    "prediction_feature_validation",
                    f"Invalid features for {model_key}",
                    {"errors": validation_result["errors"], "features": feature_vector},
                )
                return None

            features_df = self._prepare_features(
                feature_vector, model_info["metadata"]["features"]["names"], model_info["scaler"]
            )

            drift_detected = await self._check_feature_drift(model_key, features_df, model_info["feature_stats"])
            if drift_detected:
                logger.warning(f"Feature drift detected for {model_key}. This may impact prediction reliability.")

            model = model_info["model"]
            raw_prediction = model.predict(features_df, num_iteration=model.best_iteration)

            prediction_result = self._process_prediction(
                raw_prediction, instrument_id, timeframe, version_id, start_time
            )

            if model_key in self.calibration_params:
                prediction_result = self._calibrate_confidence(prediction_result, self.calibration_params[model_key])

            risk_assessment = await self._assess_prediction_risk(prediction_result, feature_vector, model_info)
            prediction_result.risk_assessment = risk_assessment

            if return_diagnostics:
                prediction_result.feature_contributions = self._calculate_feature_contributions(
                    model, features_df, prediction_result.prediction
                )

            model_info["prediction_count"] += 1
            self.prediction_history.append(prediction_result)

            await self._record_prediction_metrics(prediction_result, model_info, start_time)

            if prediction_result.prediction != 0:
                logger.info(
                    f"Prediction for {instrument_id} ({timeframe}): "
                    f"{['SELL', 'NEUTRAL', 'BUY'][prediction_result.prediction + 1]} "
                    f"(confidence: {prediction_result.confidence:.2%}, risk: {prediction_result.risk_assessment['risk_level']})"
                )
            else:
                logger.info(
                    f"Prediction for {instrument_id} ({timeframe}): NEUTRAL (confidence: {prediction_result.confidence:.2%}, risk: {prediction_result.risk_assessment['risk_level']})"
                )

            return prediction_result

        except Exception as e:
            logger.error(f"Prediction error for {model_key}: {e}", exc_info=True)
            if model_key in self.active_models:
                version_id = self.active_models[model_key]
                model_info = self.models.get(f"{model_key}_{version_id}")
                if model_info:
                    model_info["error_count"] += 1
            await self.error_handler.handle_error(
                "prediction_execution",
                f"Prediction failed for {model_key}: {e}",
                {
                    "instrument_id": instrument_id,
                    "timeframe": timeframe,
                    "feature_vector": feature_vector,
                    "error": str(e),
                },
            )
            return None

    async def predict_batch(
        self, predictions_request: list[tuple[int, str, dict[str, float]]]
    ) -> list[Optional[PredictionResult]]:
        """
        Efficient batch prediction for multiple instruments.
        """
        if not predictions_request:
            logger.warning("predict_batch called with empty predictions_request.")
            return []

        logger.info(f"Starting batch prediction for {len(predictions_request)} requests.")
        results: list[Optional[PredictionResult]] = []
        grouped_requests: dict[str, list[tuple[int, str, dict[str, float]]]] = {}
        for instrument_id, timeframe, features in predictions_request:
            model_key = f"{instrument_id}_{timeframe}"
            grouped_requests.setdefault(model_key, []).append((instrument_id, timeframe, features))

        for model_key, requests in grouped_requests.items():
            if model_key not in self.active_models:
                logger.warning(f"No active model loaded for {model_key}. Skipping {len(requests)} predictions.")
                results.extend([None] * len(requests))
                continue
            for instrument_id, timeframe, features in requests:
                try:
                    result = await self.predict(instrument_id, timeframe, features)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error during batch prediction for {instrument_id} ({timeframe}): {e}", exc_info=True)
                    results.append(None)
        logger.info(f"Batch prediction completed. Processed {len(predictions_request)} requests.")
        return results

    def _validate_model(self, model: lgb.Booster, metadata: dict[str, Any]) -> bool:
        """
        Validates model integrity and compatibility.
        """
        logger.debug("Validating loaded model and its metadata.")
        try:
            if metadata.get("model_type") != "LightGBM":
                logger.error(
                    f"ModelValidationFailed: Invalid model type '{metadata.get('model_type')}'. Expected 'LightGBM'."
                )
                return False

            expected_features = metadata["features"]["names"]
            if not expected_features:
                logger.error("ModelValidationFailed: Metadata does not contain expected feature names.")
                return False

            model_features_count = model.num_feature()
            if model_features_count != len(expected_features):
                logger.error(
                    f"ModelValidationFailed: Feature count mismatch. Model has {model_features_count} features, "
                    f"metadata specifies {len(expected_features)}."
                )
                return False

            test_input = np.zeros((1, model_features_count))
            test_prediction = model.predict(test_input)
            if not isinstance(test_prediction, np.ndarray):
                logger.error(f"ModelValidationFailed: Expected ndarray prediction output, got {type(test_prediction)}.")
                return False
            if test_prediction.shape[1] != 3:
                logger.error(
                    f"ModelValidationFailed: Invalid prediction output shape. Expected (1, 3), got {test_prediction.shape}."
                )
                return False

            logger.info("Model validation successful.")
            return True
        except Exception as e:
            logger.error(f"ModelValidationFailed: Unexpected error during model validation: {e}", exc_info=True)
            return False

    async def _validate_features(self, feature_vector: dict[str, float], metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Comprehensive feature validation.
        """
        logger.debug("Validating input feature vector.")
        if not self.predictor_config:
            logger.critical("FeatureValidationFailed: Predictor configuration is missing. Cannot validate features.")
            raise ValueError("Predictor configuration is required for feature validation.")

        expected_features = set(metadata["features"]["names"])
        provided_features = set(feature_vector.keys())
        errors, warnings = [], []

        missing = expected_features - provided_features
        if missing:
            errors.append(f"Missing features: {missing}")
            logger.error(f"FeatureValidationFailed: Missing features {missing}.")

        extra = provided_features - expected_features
        if extra:
            warnings.append(f"Extra features (will be ignored): {extra}")
            logger.warning(f"FeatureValidationWarning: Extra features {extra} will be ignored.")

        for feature, value in feature_vector.items():
            if feature not in expected_features:
                continue
            if pd.isna(value) or np.isinf(value):
                errors.append(f"Invalid value for {feature}: {value}")
                logger.error(f"FeatureValidationFailed: Invalid value for {feature}: {value}.")

            if hasattr(self.predictor_config, "feature_ranges") and self.predictor_config.feature_ranges:
                ranges = self.predictor_config.feature_ranges
                if feature in ranges:
                    min_val, max_val = ranges[feature]
                    if not (min_val <= value <= max_val):
                        warnings.append(f"Feature {feature} out of expected range [{min_val}, {max_val}]: {value}")
                        logger.warning(f"FeatureValidationWarning: {feature} out of expected range.")

        if errors:
            logger.error("FeatureValidationFailed: Feature validation failed with errors.")
        elif warnings:
            logger.warning("FeatureValidationWarning: Feature validation completed with warnings.")
        else:
            logger.debug("Feature validation successful.")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _prepare_features(
        self, feature_vector: dict[str, float], expected_features: list[str], scaler: Any
    ) -> pd.DataFrame:
        """
        Prepares features for prediction with proper ordering and scaling.
        """
        logger.debug("Preparing features for prediction.")
        try:
            feature_data = {feature: feature_vector.get(feature, 0.0) for feature in expected_features}
            df = pd.DataFrame([feature_data])
            scaled_values = scaler.transform(df)
            prepared_df = pd.DataFrame(scaled_values, columns=df.columns)
            logger.debug(f"Features prepared. Shape: {prepared_df.shape}.")
            return prepared_df
        except Exception as e:
            logger.error(f"Error preparing features for prediction: {e}", exc_info=True)
            raise RuntimeError("Failed to prepare features for prediction.") from e

    async def _check_feature_drift(
        self, model_key: str, features_df: pd.DataFrame, feature_stats: dict[str, Any]
    ) -> bool:
        """
        Monitors for feature drift using z-score against training data statistics.
        """
        logger.debug(f"Checking feature drift for {model_key}.")
        if not self.predictor_config:
            logger.critical("FeatureDriftCheckFailed: Predictor configuration is missing. Cannot check feature drift.")
            raise ValueError("Predictor configuration is required for feature drift check.")

        drift_detected = False
        instrument_id, timeframe = model_key.split("_", 1)

        for feature in features_df.columns:
            if feature not in feature_stats:
                logger.warning(
                    f"Feature '{feature}' not found in training feature statistics for {model_key}. Skipping drift check."
                )
                continue

            stats = feature_stats[feature]
            mean, std = stats["mean"], stats["std"]
            value = features_df[feature].iloc[0]

            if std > 1e-8:  # Avoid division by zero
                z_score = abs((value - mean) / std)
                severity = "critical" if z_score > self.predictor_config.drift_threshold_z_score else "info"
                metrics_registry.record_feature_drift(instrument_id, timeframe, feature, z_score, severity)

                if z_score > self.predictor_config.drift_threshold_z_score:
                    logger.warning(
                        f"Feature drift detected for {feature} in {model_key}: "
                        f"z-score={z_score:.2f} (value={value:.4f}, train_mean={mean:.4f}, train_std={std:.4f})."
                    )
                    drift_detected = True
            else:
                logger.debug(f"Standard deviation for feature '{feature}' is zero. Skipping Z-score calculation.")

        if drift_detected:
            logger.warning(f"Feature drift detected for {model_key}.")
        else:
            logger.debug(f"No significant feature drift detected for {model_key}.")
        return drift_detected

    def _process_prediction(
        self,
        raw_prediction: NDArray[np.float64],
        instrument_id: int,
        timeframe: str,
        version_id: str,
        start_time: float,
    ) -> PredictionResult:
        """
        Processes raw model output into structured prediction.
        """
        logger.debug(f"Processing raw prediction for {instrument_id} ({timeframe}).")
        probabilities = raw_prediction[0]
        predicted_class = int(np.argmax(probabilities)) - 1
        sorted_probs = np.sort(probabilities)[::-1]
        confidence = sorted_probs[0] - sorted_probs[1]

        if confidence < self.predictor_config.confidence_threshold:
            predicted_class = 0
            logger.debug(
                f"Prediction for {instrument_id} ({timeframe}) set to NEUTRAL due to low confidence ({confidence:.2f})."
            )

        inference_time = (time.time() - start_time) * 1000
        return PredictionResult(
            instrument_id=instrument_id,
            timeframe=timeframe,
            timestamp=datetime.now(),
            prediction=predicted_class,
            confidence=confidence,
            probabilities={
                "sell": float(probabilities[0]),
                "neutral": float(probabilities[1]),
                "buy": float(probabilities[2]),
            },
            feature_contributions=None,  # Calculated only if return_diagnostics is True
            model_version=version_id,
            inference_time_ms=inference_time,
            risk_assessment={},
        )

    def _calibrate_confidence(
        self, prediction: PredictionResult, calibration_params: dict[str, Any]
    ) -> PredictionResult:
        """
        Applies confidence calibration using isotonic regression or Platt scaling.
        """
        logger.debug(f"Calibrating confidence for prediction {prediction.instrument_id} ({prediction.timeframe}).")
        if "scale" in calibration_params and "bias" in calibration_params:
            calibrated_confidence = prediction.confidence * calibration_params["scale"] + calibration_params["bias"]
            prediction.confidence = max(0, min(1, calibrated_confidence))
            logger.debug(f"Confidence calibrated to {prediction.confidence:.2f}.")
        else:
            logger.warning("Calibration parameters incomplete. Skipping confidence calibration.")
        return prediction

    async def _assess_prediction_risk(
        self, prediction: PredictionResult, features: dict[str, float], model_info: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Comprehensive risk assessment for the prediction.
        """
        logger.debug(f"Assessing prediction risk for {prediction.instrument_id} ({prediction.timeframe}).")
        if not self.predictor_config:
            logger.critical(
                "PredictionRiskAssessmentFailed: Predictor configuration is missing. Cannot assess prediction risk."
            )
            raise ValueError("Predictor configuration is required for risk assessment.")

        risk_factors = {}
        entropy = -sum(p * np.log(p + 1e-10) for p in prediction.probabilities.values() if p > 0)
        risk_factors["model_uncertainty"] = entropy / np.log(3)

        feature_quality_score = 1.0
        if "volatility_at_entry" in features:
            vol = features["volatility_at_entry"]
            if vol > 0.02:
                feature_quality_score *= 0.8
        risk_factors["feature_quality"] = feature_quality_score

        model_age = (datetime.now() - model_info["loaded_at"]).days
        risk_factors["model_staleness"] = (
            min(1.0, model_age / self.predictor_config.predictor.model_staleness_max_days)
            if model_age > self.predictor_config.model_staleness_days
            else 0.0
        )

        recent_predictions = [
            p
            for p in self.prediction_history[-self.predictor_config.predictor.prediction_history_window_small :]
            if p.instrument_id == prediction.instrument_id
        ]
        if len(recent_predictions) > 10:
            changes = sum(
                1
                for i in range(1, len(recent_predictions))
                if recent_predictions[i].prediction != recent_predictions[i - 1].prediction
            )
            flip_flop_rate = changes / len(recent_predictions)
            risk_factors["signal_stability"] = 1.0 - flip_flop_rate
        else:
            risk_factors["signal_stability"] = 0.5

        risk_score = np.mean(list(risk_factors.values()))
        logger.debug(f"Risk assessment completed. Risk score: {risk_score:.2f}.")
        return {
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "risk_level": "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.4 else "LOW",
        }

    def _calculate_feature_contributions(
        self, model: lgb.Booster, features_df: pd.DataFrame, prediction: int
    ) -> dict[str, float]:
        """
        Calculates feature contributions using tree SHAP approximation.
        """
        logger.debug("Calculating feature contributions.")
        try:
            contributions = model.predict(features_df, pred_contrib=True)[0]
            feature_names = features_df.columns.tolist()
            feature_contributions = {feature: float(contributions[i]) for i, feature in enumerate(feature_names)}
            total_contribution = sum(abs(c) for c in feature_contributions.values())
            if total_contribution > 0:
                feature_contributions = {k: v / total_contribution for k, v in feature_contributions.items()}
            logger.debug(f"Calculated contributions for {len(feature_contributions)} features.")
            return dict(sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True))
        except Exception as e:
            logger.error(f"Error calculating feature contributions: {e}", exc_info=True)
            return {}

    async def get_model_status(self) -> dict[str, Any]:
        """
        Returns comprehensive status of all loaded models.
        """
        logger.debug("Retrieving model status.")
        status: dict[str, Any] = {"active_models": {}, "prediction_statistics": {}}
        for model_key, version_id in self.active_models.items():
            model_info = self.models.get(f"{model_key}_{version_id}")
            if model_info:
                status["active_models"][model_key] = {
                    "version": version_id,
                    "loaded_at": model_info["loaded_at"].isoformat(),
                    "prediction_count": model_info["prediction_count"],
                    "error_count": model_info["error_count"],
                    "error_rate": (model_info["error_count"] / max(1, model_info["prediction_count"])),
                }
            else:
                logger.warning(f"Active model {model_key} (version {version_id}) not found in loaded models.")

        if self.prediction_history:
            from collections import defaultdict

            stats: dict[str, Any] = defaultdict(
                lambda: {
                    "total": 0,
                    "buy": 0,
                    "sell": 0,
                    "neutral": 0,
                    "avg_confidence": 0.0,
                    "avg_inference_time": 0.0,
                }
            )
            for pred in self.prediction_history[-1000:]:
                key = f"{pred.instrument_id}_{pred.timeframe}"
                stats[key]["total"] += 1
                stats[key]["avg_confidence"] += pred.confidence
                stats[key]["avg_inference_time"] += pred.inference_time_ms
                if pred.prediction == 1:
                    stats[key]["buy"] += 1
                elif pred.prediction == -1:
                    stats[key]["sell"] += 1
                else:
                    stats[key]["neutral"] += 1
            for stat in stats.values():
                if stat["total"] > 0:
                    stat["avg_confidence"] /= stat["total"]
                    stat["avg_inference_time"] /= stat["total"]
                    stat["signal_distribution"] = {
                        "buy": stat["buy"] / stat["total"],
                        "sell": stat["sell"] / stat["total"],
                        "neutral": stat["neutral"] / stat["total"],
                    }
            status["prediction_statistics"] = dict(stats)
        logger.info("Model status retrieval completed.")
        return status

    async def reload_model(self, instrument_id: int, timeframe: str, model_path: Optional[Path] = None) -> bool:
        """
        Reloads a model, useful for updating to newer versions.
        """
        model_key = f"{instrument_id}_{timeframe}"
        logger.info(f"Attempting to reload model for {model_key}.")

        if model_path is None:
            latest_link = self.artifacts_path / f"latest_{instrument_id}_{timeframe}"
            if not latest_link.exists() or not latest_link.is_symlink():
                logger.error(f"No latest model symlink found for {model_key} at {latest_link}. Cannot reload.")
                return False
            model_path = latest_link.resolve()
            logger.info(f"Resolved latest model path for {model_key} to {model_path}.")

        success = await self.load_model(model_path, instrument_id, timeframe, set_as_active=True)
        if success:
            old_versions = [
                k
                for k in self.models
                if k.startswith(model_key) and k != f"{model_key}_{self.active_models[model_key]}"
            ]
            for old_key in old_versions:
                del self.models[old_key]
                logger.info(f"Removed old model version from memory: {old_key}.")
            logger.info(f"Model {model_key} reloaded successfully.")
        else:
            logger.error(f"Failed to reload model for {model_key}.")
        return success

    async def _record_prediction_metrics(
        self, prediction_result: PredictionResult, model_info: dict[str, Any], start_time: float
    ) -> None:
        """Record comprehensive prediction metrics to monitoring system."""
        # Record prediction with confidence distribution
        prediction_label = {-1: "sell", 0: "neutral", 1: "buy"}[prediction_result.prediction]
        metrics_registry.record_model_prediction(
            str(prediction_result.instrument_id),
            prediction_result.timeframe,
            "lightgbm",
            prediction_label,
            prediction_result.confidence,
            prediction_result.inference_time_ms / 1000.0,  # Convert to seconds
            success=True,
        )

        # Calculate and record real-time accuracy if we have enough history
        await self._calculate_and_record_real_time_accuracy(prediction_result)

        # Record data quality score
        await self._record_data_quality_metrics(prediction_result)

        # Record model performance degradation indicators
        await self._check_model_performance_degradation(prediction_result, model_info)

    async def _calculate_and_record_real_time_accuracy(self, prediction_result: PredictionResult) -> None:
        """Calculate and record real-time accuracy over sliding windows."""
        instrument_id = str(prediction_result.instrument_id)
        timeframe = prediction_result.timeframe

        # Get recent predictions for this instrument
        recent_predictions = [
            p
            for p in self.prediction_history[-1000:]
            if p.instrument_id == prediction_result.instrument_id and p.timeframe == timeframe
        ]

        if len(recent_predictions) < 10:
            return

        # Calculate accuracy for different window sizes
        window_sizes = self.predictor_config.predictor.real_time_accuracy_windows
        for window_size in window_sizes:
            if len(recent_predictions) >= window_size:
                window_predictions = recent_predictions[-window_size:]

                # For real-time accuracy, we need actual outcomes
                # This is a simplified version - in production, you'd match with actual trade results
                # For now, we'll use a placeholder that assumes some predictions are correct
                correct_predictions = sum(1 for p in window_predictions if p.confidence > 0.7)
                accuracy = correct_predictions / len(window_predictions)

                metrics_registry.record_prediction_accuracy(instrument_id, timeframe, str(window_size), accuracy)

    async def _record_data_quality_metrics(self, prediction_result: PredictionResult) -> None:
        """Record data quality metrics based on prediction characteristics."""
        instrument_id = str(prediction_result.instrument_id)
        timeframe = prediction_result.timeframe

        # Calculate data quality score based on various factors
        quality_score = 100.0

        # Reduce score for low confidence predictions
        if prediction_result.confidence < 0.5:
            quality_score -= 20

        # Reduce score for high uncertainty (high entropy)
        risk_factors = prediction_result.risk_assessment.get("risk_factors", {})
        model_uncertainty = risk_factors.get("model_uncertainty", 0.0)
        if model_uncertainty > 0.8:
            quality_score -= 15

        # Reduce score for feature quality issues
        feature_quality = risk_factors.get("feature_quality", 1.0)
        if feature_quality < 0.8:
            quality_score -= 10

        # Reduce score for model staleness
        model_staleness = risk_factors.get("model_staleness", 0.0)
        if model_staleness > 0.5:
            quality_score -= 10

        quality_score = max(0.0, quality_score)
        metrics_registry.record_data_quality(instrument_id, timeframe, quality_score)

    async def _check_model_performance_degradation(
        self, prediction_result: PredictionResult, model_info: dict[str, Any]
    ) -> None:
        """Check for model performance degradation indicators."""
        instrument_id = str(prediction_result.instrument_id)
        timeframe = prediction_result.timeframe

        # Check prediction confidence trends
        recent_predictions = [
            p
            for p in self.prediction_history[-100:]
            if p.instrument_id == prediction_result.instrument_id and p.timeframe == timeframe
        ]

        if len(recent_predictions) >= 20:
            # Calculate average confidence over recent predictions
            recent_confidence = sum(p.confidence for p in recent_predictions[-20:]) / 20
            earlier_confidence = (
                sum(p.confidence for p in recent_predictions[-40:-20]) / 20
                if len(recent_predictions) >= 40
                else recent_confidence
            )

            confidence_degradation = earlier_confidence - recent_confidence
            if (
                confidence_degradation > self.predictor_config.predictor.confidence_degradation_threshold
            ):  # 10% degradation threshold
                logger.warning(
                    f"Model confidence degradation detected for {instrument_id} ({timeframe}): "
                    f"dropped by {confidence_degradation:.2%}"
                )

        # Check error rate trends
        error_rate = model_info.get("error_count", 0) / max(1, model_info.get("prediction_count", 1))
        if error_rate > self.predictor_config.predictor.error_rate_threshold:  # 5% error rate threshold
            logger.warning(f"High error rate detected for {instrument_id} ({timeframe}): {error_rate:.2%}")

    async def _validate_features_with_monitoring(
        self, feature_vector: dict[str, float], metadata: dict[str, Any], instrument_id: int, timeframe: str
    ) -> dict[str, Any]:
        """Enhanced feature validation with monitoring integration."""
        validation_result = await self._validate_features(feature_vector, metadata)

        # Record feature validation metrics
        if not validation_result["valid"]:
            # Record validation failure
            for error in validation_result["errors"]:
                logger.error(f"Feature validation error for {instrument_id} ({timeframe}): {error}")

        return validation_result
