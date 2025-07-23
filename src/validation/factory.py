# src/validation/factory.py
from src.utils.config_loader import AppConfig
from src.utils.logger import LOGGER as logger
from src.validation.candle_validator import CandleValidator
from src.validation.feature_validator import FeatureValidator
from src.validation.interfaces import IValidator


class ValidationFactory:
    def __init__(self, config: AppConfig):
        self.config = config

    def get_validator(self, validator_type: str) -> IValidator:
        if validator_type == "candle":
            return CandleValidator(self.config.data_quality)
        if validator_type == "feature":
            return FeatureValidator(self.config)
        logger.error(f"ValueError: Unknown validator type requested: {validator_type}")
        raise ValueError(f"Unknown validator type: {validator_type}")
