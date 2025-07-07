import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import ModelTrainingConfig, config_loader
from src.utils.logger import LOGGER as logger
from src.utils.performance_metrics import PerformanceMetrics

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
        performance_metrics: PerformanceMetrics,
    ):
        self.error_handler = error_handler
        self.health_monitor = health_monitor
        self.performance_metrics = performance_metrics

        # Model storage
        self.models: dict[str, dict[str, Any]] = {}  # {model_key: {model, metadata, scaler}}
        self.active_models: dict[str, str] = {}  # {model_key: version_id}

        # Configuration
        self.predictor_config: ModelTrainingConfig
        assert config.model_training is not None
        self.predictor_config = config.model_training
        self.artifacts_path = Path(self.predictor_config.artifacts_path)

        # Feature monitoring
        self.feature_stats: dict[str, dict[str, Any]] = {}
        self.prediction_history: list[PredictionResult] = []

        # Confidence calibration
        assert self.predictor_config is not None
        self.confidence_threshold = self.predictor_config.confidence_threshold
        self.calibration_params: dict[str, Any] = {}

        logger.info("ModelPredictor initialized with production configuration")

    async def load_model(
        self, model_path: Path, instrument_id: int, timeframe: str, set_as_active: bool = True
    ) -> bool:
        """
        Loads a model with all associated artifacts.
        """
        try:
            timer_id = self.performance_metrics.start_timer("model_loading")

            # Determine model directory
            if model_path.is_file():
                model_dir = model_path.parent
            else:
                model_dir = model_path
                model_path = model_dir / "model.txt"

            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            # Load model
            model = lgb.Booster(model_file=str(model_path))

            # Load metadata
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                logger.error(f"Model metadata not found: {metadata_path}")
                return False

            with open(metadata_path) as f:
                metadata = json.load(f)

            # Load scaler
            scaler_path = model_dir / "scaler.pkl"
            scaler = None
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
            else:
                logger.warning(f"Scaler not found, predictions may be less accurate: {scaler_path}")

            # Validate model
            if not self._validate_model(model, metadata):
                logger.error("Model validation failed")
                return False

            # Store model
            model_key = f"{instrument_id}_{timeframe}"
            version_id = metadata.get("version_id", "unknown")

            self.models[f"{model_key}_{version_id}"] = {
                "model": model,
                "metadata": metadata,
                "scaler": scaler,
                "loaded_at": datetime.now(),
                "prediction_count": 0,
                "error_count": 0,
            }

            if set_as_active:
                self.active_models[model_key] = version_id
                logger.info(f"Set active model for {model_key}: version {version_id}")

            # Initialize feature monitoring
            self._initialize_feature_monitoring(model_key, metadata)

            # Load calibration parameters if available
            calibration_path = model_dir / "calibration.json"
            if calibration_path.exists():
                with open(calibration_path) as f:
                    self.calibration_params[model_key] = json.load(f)

            self.performance_metrics.stop_timer("model_loading", timer_id, True)
            logger.info(f"Successfully loaded model from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
            await self.error_handler.handle_error(
                "model_predictor", f"Model loading failed: {e}", {"model_path": str(model_path), "error": str(e)}
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

        try:
            # Check if model is loaded
            if model_key not in self.active_models:
                logger.error(f"No active model for {model_key}")
                return None

            version_id = self.active_models[model_key]
            model_info = self.models.get(f"{model_key}_{version_id}")

            if not model_info:
                logger.error(f"Model not found: {model_key}_{version_id}")
                return None

            # Validate features
            validation_result = await self._validate_features(feature_vector, model_info["metadata"])

            if not validation_result["valid"]:
                logger.error(f"Feature validation failed: {validation_result['errors']}")
                await self.error_handler.handle_error(
                    "model_predictor",
                    "Invalid features",
                    {"errors": validation_result["errors"], "features": feature_vector},
                )
                return None

            # Prepare features
            features_df = self._prepare_features(
                feature_vector, model_info["metadata"]["features"]["names"], model_info["scaler"]
            )

            # Monitor feature drift
            drift_detected = await self._check_feature_drift(model_key, feature_vector)

            if drift_detected:
                logger.warning(f"Feature drift detected for {model_key}")
                self.health_monitor.record_warning("feature_drift", {"model_key": model_key, "features": feature_vector}) # type: ignore

            # Make prediction
            model = model_info["model"]
            raw_prediction = model.predict(features_df, num_iteration=model.best_iteration)

            # Process prediction
            prediction_result = self._process_prediction(
                raw_prediction, instrument_id, timeframe, version_id, start_time
            )

            # Apply confidence calibration
            if model_key in self.calibration_params:
                prediction_result = self._calibrate_confidence(prediction_result, self.calibration_params[model_key])

            # Risk assessment
            risk_assessment = await self._assess_prediction_risk(prediction_result, feature_vector, model_info)
            prediction_result.risk_assessment = risk_assessment

            # Feature attribution (SHAP-like)
            if return_diagnostics:
                feature_contributions = self._calculate_feature_contributions(
                    model, features_df, prediction_result.prediction
                )
                prediction_result.feature_contributions = feature_contributions

            # Update statistics
            model_info["prediction_count"] += 1
            self.prediction_history.append(prediction_result)

            # Log prediction
            if prediction_result.prediction != 0:  # Log non-neutral predictions
                logger.info(
                    f"Prediction for {instrument_id} ({timeframe}): "
                    f"{['SELL', 'NEUTRAL', 'BUY'][prediction_result.prediction + 1]} "
                    f"(confidence: {prediction_result.confidence:.2%})"
                )

            return prediction_result

        except Exception as e:
            logger.error(f"Prediction error for {model_key}: {e}", exc_info=True)

            # Update error statistics
            if model_key in self.active_models:
                version_id = self.active_models[model_key]
                model_info = self.models.get(f"{model_key}_{version_id}")
                if model_info:
                    model_info["error_count"] += 1

            await self.error_handler.handle_error(
                "model_predictor",
                f"Prediction failed: {e}",
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
        results: list[Optional[PredictionResult]] = []

        # Group by model key for efficiency
        grouped_requests: dict[str, list[tuple[int, str, dict[str, float]]]] = {}
        for instrument_id, timeframe, features in predictions_request:
            model_key = f"{instrument_id}_{timeframe}"
            if model_key not in grouped_requests:
                grouped_requests[model_key] = []
            grouped_requests[model_key].append((instrument_id, timeframe, features))

        # Process each group
        for model_key, requests in grouped_requests.items():
            if model_key not in self.active_models:
                logger.warning(f"No model loaded for {model_key}")
                results.extend([None] * len(requests))
                continue

            # Batch process
            for instrument_id, timeframe, features in requests:
                result = await self.predict(instrument_id, timeframe, features)
                results.append(result)

        return results

    def _validate_model(self, model: lgb.Booster, metadata: dict[str, Any]) -> bool:
        """
        Validates model integrity and compatibility.
        """
        try:
            # Check model type
            if metadata.get("model_type") != "LightGBM":
                logger.error(f"Invalid model type: {metadata.get('model_type')}")
                return False

            # Check feature count
            expected_features = len(metadata["features"]["names"])
            model_features = model.num_feature()

            if model_features != expected_features:
                logger.error(
                    f"Feature count mismatch: model has {model_features}, metadata specifies {expected_features}"
                )
                return False

            # Check model can predict
            test_input = np.zeros((1, model_features))
            test_prediction = model.predict(test_input)
            assert isinstance(test_prediction, np.ndarray)

            if test_prediction.shape[1] != 3:  # 3 classes
                logger.error(f"Invalid prediction shape: {test_prediction.shape}")
                return False

            return True

        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return False

    async def _validate_features(self, feature_vector: dict[str, float], metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Comprehensive feature validation.
        """
        assert self.predictor_config is not None
        expected_features = set(metadata["features"]["names"])
        provided_features = set(feature_vector.keys())

        errors = []
        warnings = []

        # Check missing features
        missing = expected_features - provided_features
        if missing:
            errors.append(f"Missing features: {missing}")

        # Check extra features
        extra = provided_features - expected_features
        if extra:
            warnings.append(f"Extra features (will be ignored): {extra}")

        # Validate feature values
        for feature, value in feature_vector.items():
            if feature not in expected_features:
                continue

            # Check for invalid values
            if pd.isna(value) or np.isinf(value):
                errors.append(f"Invalid value for {feature}: {value}")

            # Check range (if defined in config)
            if hasattr(self.predictor_config, "feature_ranges"):
                ranges = self.predictor_config.feature_ranges
                if feature in ranges:
                    min_val, max_val = ranges[feature]
                    if not (min_val <= value <= max_val):
                        warnings.append(f"Feature {feature} out of expected range [{min_val}, {max_val}]: {value}")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _prepare_features(
        self, feature_vector: dict[str, float], expected_features: list[str], scaler: Optional[Any]
    ) -> pd.DataFrame:
        """
        Prepares features for prediction with proper ordering and scaling.
        """
        # Create DataFrame with expected feature order
        feature_data = {}
        for feature in expected_features:
            feature_data[feature] = feature_vector.get(feature, 0.0)

        df = pd.DataFrame([feature_data])

        # Apply scaling if available
        if scaler is not None:
            scaled_values = scaler.transform(df)
            df = pd.DataFrame(scaled_values, columns=df.columns)

        return df

    async def _check_feature_drift(self, model_key: str, feature_vector: dict[str, float]) -> bool:
        """
        Monitors for feature drift using statistical tests.
        """
        assert self.predictor_config is not None
        if model_key not in self.feature_stats:
            return False

        stats = self.feature_stats[model_key]
        drift_detected = False

        for feature, value in feature_vector.items():
            if feature not in stats:
                continue

            # Simple z-score based drift detection
            mean = stats[feature]["mean"]
            std = stats[feature]["std"]

            if std > 0:
                z_score = abs((value - mean) / std)
                if z_score > self.predictor_config.drift_threshold_z_score:
                    logger.warning(
                        f"Feature drift detected for {feature}: "
                        f"z-score={z_score:.2f}, value={value:.4f}, "
                        f"expected_mean={mean:.4f}"
                    )
                    drift_detected = True

        return drift_detected

    def _process_prediction(
        self, raw_prediction: np.ndarray, instrument_id: int, timeframe: str, version_id: str, start_time: float
    ) -> PredictionResult:
        """
        Processes raw model output into structured prediction.
        """
        # Convert probabilities (model outputs 0,1,2 for classes)
        probabilities = raw_prediction[0]

        # Map to -1, 0, 1
        predicted_class = int(np.argmax(probabilities)) - 1

        # Calculate confidence (difference between top 2 probabilities)
        sorted_probs = np.sort(probabilities)[::-1]
        confidence = sorted_probs[0] - sorted_probs[1]

        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            predicted_class = 0  # Default to NEUTRAL if not confident

        # Create result
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
        # Simple linear calibration for now
        if "scale" in calibration_params and "bias" in calibration_params:
            calibrated_confidence = prediction.confidence * calibration_params["scale"] + calibration_params["bias"]
            prediction.confidence = max(0, min(1, calibrated_confidence))

        return prediction

    async def _assess_prediction_risk(
        self, prediction: PredictionResult, features: dict[str, float], model_info: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Comprehensive risk assessment for the prediction.
        """
        assert self.predictor_config is not None
        risk_factors = {}

        # 1. Model uncertainty
        entropy = -sum(p * np.log(p + 1e-10) for p in prediction.probabilities.values() if p > 0)
        risk_factors["model_uncertainty"] = entropy / np.log(3)  # Normalize

        # 2. Feature reliability
        feature_quality_score = 1.0
        if "volatility_at_entry" in features:
            # High volatility increases risk
            vol = features["volatility_at_entry"]
            if vol > 0.02:  # 2% volatility threshold
                feature_quality_score *= 0.8

        risk_factors["feature_quality"] = feature_quality_score

        # 3. Model staleness
        model_age = (datetime.now() - model_info["loaded_at"]).days
        if model_age > self.predictor_config.model_staleness_days:
            risk_factors["model_staleness"] = min(1.0, model_age / 30)
        else:
            risk_factors["model_staleness"] = 0.0

        # 4. Prediction frequency
        recent_predictions = [p for p in self.prediction_history[-100:] if p.instrument_id == prediction.instrument_id]

        if len(recent_predictions) > 10:
            # Check for flip-flopping
            changes = sum(
                1
                for i in range(1, len(recent_predictions))
                if recent_predictions[i].prediction != recent_predictions[i - 1].prediction
            )
            flip_flop_rate = changes / len(recent_predictions)
            risk_factors["signal_stability"] = 1.0 - flip_flop_rate
        else:
            risk_factors["signal_stability"] = 0.5  # Unknown

        # 5. Overall risk score
        risk_score = np.mean(list(risk_factors.values()))

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
        # Get prediction contributions
        contributions = model.predict(features_df, pred_contrib=True)[0]

        # Map to feature names
        feature_names = features_df.columns.tolist()
        feature_contributions = {}

        # Contributions include bias term at the end
        for i, feature in enumerate(feature_names):
            feature_contributions[feature] = float(contributions[i])

        # Normalize by sum of absolute contributions
        total_contribution = sum(abs(c) for c in feature_contributions.values())
        if total_contribution > 0:
            feature_contributions = {k: v / total_contribution for k, v in feature_contributions.items()}

        # Sort by absolute contribution
        return dict(sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True))

    def _initialize_feature_monitoring(self, model_key: str, metadata: dict[str, Any]) -> None:
        """
        Initializes feature statistics for drift monitoring.
        """
        # In production, these stats would come from training data
        # For now, initialize with reasonable defaults
        self.feature_stats[model_key] = {}

        for feature in metadata["features"]["names"]:
            self.feature_stats[model_key][feature] = {"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0}

    async def get_model_status(self) -> dict[str, Any]:
        """
        Returns comprehensive status of all loaded models.
        """
        status: dict[str, Any] = {"active_models": {}, "model_performance": {}, "feature_drift_status": {}, "prediction_statistics": {}}

        # Active models
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

        # Recent prediction statistics
        if self.prediction_history:
            recent = self.prediction_history[-1000:]  # Last 1000 predictions

            # Group by instrument and timeframe
            from collections import defaultdict

            stats: dict[str, Any] = defaultdict(
                lambda: {"total": 0, "buy": 0, "sell": 0, "neutral": 0, "avg_confidence": 0.0, "avg_inference_time": 0.0}
            )

            for pred in recent:
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

            # Calculate averages
            for _key, stat in stats.items():
                if stat["total"] > 0:
                    stat["avg_confidence"] /= stat["total"]
                    stat["avg_inference_time"] /= stat["total"]
                    stat["signal_distribution"] = {
                        "buy": stat["buy"] / stat["total"],
                        "sell": stat["sell"] / stat["total"],
                        "neutral": stat["neutral"] / stat["total"],
                    }

            status["prediction_statistics"] = dict(stats)

        return status

    async def reload_model(self, instrument_id: int, timeframe: str, model_path: Optional[Path] = None) -> bool:
        """
        Reloads a model, useful for updating to newer versions.
        """
        model_key = f"{instrument_id}_{timeframe}"

        # If no path specified, look for latest
        if model_path is None:
            latest_link = self.artifacts_path / f"latest_{instrument_id}_{timeframe}"
            if latest_link.exists():
                model_path = latest_link.resolve()
            else:
                logger.error(f"No latest model found for {model_key}")
                return False

        # Load new model
        success = await self.load_model(model_path, instrument_id, timeframe, set_as_active=True)

        if success:
            # Clean up old version
            old_versions = [
                k
                for k in self.models
                if k.startswith(model_key) and k != f"{model_key}_{self.active_models[model_key]}"
            ]

            for old_key in old_versions:
                del self.models[old_key]
                logger.info(f"Removed old model version: {old_key}")

        return success
