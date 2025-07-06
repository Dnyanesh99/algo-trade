import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import pandas as pd

from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger
from src.utils.performance_metrics import PerformanceMetrics

config = config_loader.get_config()


class ModelPredictor:
    """
    Loads a trained LightGBM model and generates predictions.
    """

    def __init__(
        self,
        error_handler: ErrorHandler,
        health_monitor: HealthMonitor,
        performance_metrics: PerformanceMetrics,
    ):
        self.model: Optional[lgb.Booster] = None
        self.model_metadata: dict[str, Any] = {}
        self.error_handler = error_handler
        self.health_monitor = health_monitor
        self.performance_metrics = performance_metrics

        # TODO: Add model_artifacts_path to StorageConfig
        self.artifacts_path = Path(config.model_training.artifacts_path)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)

        logger.info("ModelPredictor initialized.")

    def load_model(self, model_path: Path) -> bool:
        """
        Loads a LightGBM model from the specified path.
        """
        try:
            self.model = lgb.Booster(model_file=str(model_path))
            metadata_path = model_path.with_suffix(".json")
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Model and metadata loaded from {model_path}")
                return True
            logger.warning(f"No metadata found for model at {model_path}")
            self.model_metadata = {}
            return True
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            self.model = None
            self.model_metadata = {}
            return False

    async def predict(self, feature_vector: dict[str, float]) -> int:
        """
        Generates a prediction (BUY/SELL/HOLD) from a feature vector.

        Args:
            feature_vector: A dictionary of feature names and their values.

        Returns:
            int: The predicted label (-1 for SELL, 0 for NEUTRAL, 1 for BUY).
        """
        if self.model is None:
            logger.error("Model not loaded. Cannot make prediction.")
            # TODO: Raise a specific exception or return a default/error value
            return 0  # Default to NEUTRAL if model not loaded

        # Ensure feature vector matches model's expected features
        expected_features = self.model_metadata.get("features", [])
        if not expected_features:
            logger.warning("Model metadata does not contain expected features. Prediction might be unreliable.")
            # Attempt to use features provided, but warn
            input_df = pd.DataFrame([feature_vector])
        else:
            # Create a DataFrame with all expected features, filling missing with 0 or NaN
            input_data = {f: feature_vector.get(f, 0.0) for f in expected_features}
            input_df = pd.DataFrame([input_data])

        try:
            # LightGBM expects a 2D array-like input
            prediction_proba = self.model.predict(input_df)

            # Assuming a classification model with 3 classes (-1, 0, 1)
            # The output of predict_proba depends on how the model was trained.
            # For simplicity, let's assume it returns probabilities for classes [0, 1, 2] mapping to [-1, 0, 1]
            # Or, if it's a binary classifier, it might return probabilities for 0 and 1.
            # For now, let's assume it returns the class directly if it's a classifier.
            # If it's predict_proba, we need to convert probabilities to labels.

            # For a multi-class classifier, predict() usually returns the class with highest probability
            predicted_label = int(prediction_proba[0])

            # Map internal model labels to external system labels if necessary
            # E.g., if model predicts 0, 1, 2 for NEUTRAL, BUY, SELL
            # This mapping should be part of model_metadata or config
            # For now, assume model directly predicts -1, 0, 1

            logger.debug(f"Prediction: {predicted_label} for features: {feature_vector}")
            return predicted_label
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            await self.error_handler.handle_error(
                "model_predictor", f"Prediction failed: {e}", {"feature_vector": feature_vector, "error": str(e)}
            )
            return 0  # Default to NEUTRAL on error



