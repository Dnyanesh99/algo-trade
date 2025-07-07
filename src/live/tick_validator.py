from datetime import datetime
from typing import Any

from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

config = config_loader.get_config()


class TickValidator:
    """
    Validates incoming tick data for integrity and correctness.
    """

    def __init__(self) -> None:
        self.lpt_threshold = config.get_data_quality_config().live_data_lpt_threshold
        self._last_sequence_numbers: dict[int, int] = {}
        logger.info(f"TickValidator initialized with LPT threshold: {self.lpt_threshold}")

    def validate_tick(self, tick: dict[str, Any]) -> bool:
        """
        Performs comprehensive validation on a single tick.

        Args:
            tick (Dict[str, Any]): The tick data dictionary.

        Returns:
            bool: True if the tick is valid, False otherwise.
        """
        # The tick is expected to be a dict[str, Any] from the TickQueue.
        # If it's not, it indicates a fundamental issue upstream, and mypy should catch it.
        # No need for isinstance(tick, dict) check here.

        # 1. Check for essential fields
        required_fields = ["instrument_token", "timestamp", "last_traded_price"]
        for field in required_fields:
            if field not in tick:
                logger.warning(f"Missing required field '{field}' in tick: {tick}")
                return False

        # 2. Validate instrument_token type
        instrument_token = tick["instrument_token"]
        if not isinstance(instrument_token, int):
            logger.warning(
                f"Invalid instrument_token type: Expected int, got {type(instrument_token)} for {instrument_token}."
            )
            return False

        # 3. Validate timestamp type and convert to datetime
        timestamp_raw = tick["timestamp"]
        try:
            if isinstance(timestamp_raw, (int, float)):
                # Assume Unix timestamp (seconds)
                timestamp = datetime.fromtimestamp(timestamp_raw)
            elif isinstance(timestamp_raw, str):
                # Attempt to parse ISO format string
                timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
            else:
                logger.warning(
                    f"Invalid timestamp type: Expected int, float or str, got {type(timestamp_raw)}. Tick: {tick}"
                )
                return False
            tick["timestamp"] = timestamp  # Update tick with datetime object
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse timestamp '{timestamp_raw}': {e}. Tick: {tick}")
            return False

        # 4. Validate numeric values (LTP, volume, etc.)
        numeric_fields = ["last_traded_price", "volume", "buy_quantity", "sell_quantity"]
        for field in numeric_fields:
            if field in tick and not isinstance(tick[field], (int, float)):
                logger.warning(
                    f"Invalid numeric type for field '{field}': {tick[field]} ({type(tick[field])}). Tick: {tick}"
                )
                return False
            if field in tick and tick[field] < 0:
                logger.warning(f"Negative value for field '{field}': {tick[field]}. Tick: {tick}")
                return False

        # 5. Filter junk values (e.g., LTP = 0 or extremely large)
        if tick["last_traded_price"] <= 0 or tick["last_traded_price"] > self.lpt_threshold:
            logger.warning(f"LTP out of realistic range: {tick['last_traded_price']}. Tick: {tick}")
            return False

        # 6. Sequence validation
        if "sequence_number" in tick:
            current_sequence = tick["sequence_number"]
            if not isinstance(current_sequence, int):
                logger.warning(
                    f"Invalid sequence_number type: Expected int, got {type(current_sequence)}. Tick: {tick}"
                )
                return False

            last_sequence = self._last_sequence_numbers.get(instrument_token)

            if last_sequence is not None:
                if current_sequence <= last_sequence:
                    logger.warning(
                        f"Out-of-sequence tick for {instrument_token}: current {current_sequence} <= last {last_sequence}. Tick: {tick}"
                    )
                    # Decide whether to drop or process out-of-sequence ticks
                    # For now, we'll drop it as it indicates a data integrity issue
                    return False
                if current_sequence > last_sequence + 1:
                    logger.warning(
                        f"Sequence gap detected for {instrument_token}: current {current_sequence} > last {last_sequence} + 1. Tick: {tick}"
                    )
                    # This indicates dropped ticks. Depending on policy, might trigger backfill or just log.
                    # For now, we'll allow it but log a warning.

            self._last_sequence_numbers[instrument_token] = current_sequence
        else:
            logger.debug(f"No sequence_number in tick for {instrument_token}. Skipping sequence validation.")

        logger.debug(f"Tick for {instrument_token} passed validation.")
        return True
