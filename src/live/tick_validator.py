from datetime import datetime
from typing import Any

from src.metrics import metrics_registry
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

config = ConfigLoader().get_config()


class TickValidator:
    """
    Validates incoming tick data for integrity and correctness.
    """

    def __init__(self) -> None:
        if not config.data_quality:
            raise ValueError("data_quality configuration is required.")
        self.lpt_threshold = config.data_quality.live_data_lpt_threshold
        self._last_sequence_numbers: dict[int, int] = {}
        logger.info(f"TickValidator initialized with LPT threshold: {self.lpt_threshold}")

    def validate_tick(self, tick: dict[str, Any]) -> bool:
        """
        Performs comprehensive validation on a single tick.
        """
        instrument_token = tick.get("instrument_token")
        logger.debug(f"Validating tick for instrument {instrument_token}.")

        try:
            if not config.live_aggregator or not config.live_aggregator.required_tick_fields:
                raise ValueError("live_aggregator.required_tick_fields configuration is required.")

            for field in config.live_aggregator.required_tick_fields:
                if field not in tick:
                    logger.warning(f"Missing required field '{field}' in tick: {tick}")
                    return False

            if not isinstance(instrument_token, int):
                logger.warning(f"Invalid instrument_token type: {type(instrument_token)}.")
                return False

            timestamp_raw = tick["timestamp"]
            if isinstance(timestamp_raw, (int, float)):
                tick["timestamp"] = datetime.fromtimestamp(timestamp_raw)
            elif isinstance(timestamp_raw, str):
                tick["timestamp"] = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
            else:
                logger.warning(f"Invalid timestamp type: {type(timestamp_raw)}.")
                return False

            for field in ["last_traded_price", "volume", "buy_quantity", "sell_quantity"]:
                if field in tick and not isinstance(tick[field], (int, float)):
                    logger.warning(f"Invalid type for numeric field '{field}': {type(tick[field])}.")
                    return False
                if field in tick and tick[field] < 0:
                    logger.warning(f"Negative value for numeric field '{field}': {tick[field]}.")
                    return False

            if not (0 < tick["last_traded_price"] < self.lpt_threshold):
                logger.warning(f"LTP {tick['last_traded_price']} is out of realistic range (0, {self.lpt_threshold}).")
                return False

            if "sequence_number" in tick and not self._validate_sequence(tick, instrument_token):
                return False

            logger.debug(f"Tick for {instrument_token} passed validation.")
            return True

        except (ValueError, TypeError) as e:
            logger.error(f"Validation failed for tick {tick} due to: {e}", exc_info=True)
            return False

    def _validate_sequence(self, tick: dict[str, Any], instrument_token: int) -> bool:
        """Validates the sequence number of a tick."""
        current_sequence = tick["sequence_number"]
        if not isinstance(current_sequence, int):
            logger.warning(f"Invalid sequence_number type: {type(current_sequence)}.")
            return False

        last_sequence = self._last_sequence_numbers.get(instrument_token)
        if last_sequence is not None:
            if current_sequence <= last_sequence:
                logger.warning(
                    f"Out-of-sequence tick for {instrument_token}: current {current_sequence} <= last {last_sequence}."
                )
                metrics_registry.increment_counter("tick_validation_errors_total", {"error_type": "out_of_sequence"})
                return False
            if current_sequence > last_sequence + 1:
                logger.warning(
                    f"Sequence gap detected for {instrument_token}: current {current_sequence} > last {last_sequence} + 1."
                )
                # This is a warning, not an error, as we can continue processing.

        self._last_sequence_numbers[instrument_token] = current_sequence
        return True
