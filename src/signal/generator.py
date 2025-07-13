from datetime import datetime
from typing import Any, Callable, Optional

from src.database.models import SignalData
from src.database.signal_repo import SignalRepository
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

config_loader = ConfigLoader()

# PerformanceMetrics module was removed - using metrics registry instead

config = config_loader.get_config()


class SignalGenerator:
    """
    Translates raw model predictions into definitive trading signals (BUY/SELL/HOLD).
    Applies confidence thresholds and logs signals to the database and alert system.
    """

    def __init__(
        self,
        signal_repo: SignalRepository,
        error_handler: ErrorHandler,
        health_monitor: HealthMonitor,
        on_signal_generated: Callable[[dict[str, Any]], Any],
    ):
        if config.signal_generation is None:
            raise ValueError("Signal generation configuration is required")
        self.signal_repo = signal_repo
        self.error_handler = error_handler
        self.health_monitor = health_monitor
        self.on_signal_generated = on_signal_generated

        self.signal_config = config.signal_generation

        logger.info("SignalGenerator initialized.")

    async def generate_signal(
        self,
        instrument_id: int,
        timeframe: str,
        timestamp: datetime,
        prediction: int,
        confidence_score: float,
        current_price: float,
        source_feature_name: Optional[str] = None,
        source_feature_value: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Generates a trading signal based on model prediction and confidence.

        Args:
            instrument_id: The ID of the instrument.
            timeframe: The timeframe of the signal (e.g., "15min").
            timestamp: The timestamp of the signal.
            prediction: The raw model prediction (-1, 0, 1).
            confidence_score: The confidence score of the prediction (0.0 to 1.0).
            current_price: The current market price at the time of signal generation.
            source_feature_name: Optional name of the feature that triggered the signal.
            source_feature_value: Optional value of the feature that triggered the signal.

        Returns:
            Dict[str, Any]: The generated signal dictionary, or None if no signal is generated.
        """

        signal_type: str = "HOLD"
        direction: str = "Neutral"

        try:
            # Performance metrics moved to metrics registry

            if prediction == 1 and confidence_score >= self.signal_config.buy_threshold:
                signal_type = "Entry"
                direction = "Long"
            elif prediction == -1 and confidence_score >= self.signal_config.sell_threshold:
                signal_type = "Entry"
                direction = "Short"
            else:
                signal_type = "Hold"
                direction = "Neutral"

            if signal_type != "Hold":
                signal_data_model = SignalData(
                    ts=timestamp,
                    signal_type=signal_type,
                    direction=direction,
                    confidence_score=confidence_score,
                    source_feature_name=source_feature_name,
                    price_at_signal=current_price,
                    source_feature_value=source_feature_value,
                    details={"timeframe": timeframe, "prediction": prediction},
                )
                await self.signal_repo.insert_signal(instrument_id, signal_data_model)
                logger.info(
                    f"Generated signal: {direction} for {instrument_id} at {current_price} (Confidence: {confidence_score:.2f})"
                )

                # Dispatch signal to alert system
                signal_data_dict = signal_data_model.model_dump()
                await self.on_signal_generated(signal_data_dict)

                # Performance metrics moved to metrics registry
                return signal_data_dict
            logger.info(
                f"No actionable signal generated for {instrument_id} (Hold). Confidence: {confidence_score:.2f}"
            )
            # Performance metrics moved to metrics registry
            return None

        except Exception as e:
            # Performance metrics moved to metrics registry
            await self.error_handler.handle_error(
                "signal_generator",
                f"Error generating signal for {instrument_id}: {e}",
                {
                    "instrument_id": instrument_id,
                    "prediction": prediction,
                    "confidence": confidence_score,
                    "error": str(e),
                },
            )
            return None
