from typing import Any

from src.state.error_handler import ErrorHandler
from src.utils.logger import LOGGER as logger


class AlertSystem:
    """
    Handles logging and dispatching trading signals to various output channels.
    """

    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        logger.info("AlertSystem initialized.")

    async def dispatch_signal(self, signal: dict[str, Any]) -> None:
        """
        Dispatches a trading signal to configured alert channels.
        For now, logs to console and file.
        """
        try:
            signal_message = (
                f"SIGNAL: {signal.get('direction')} {signal.get('instrument_id')} "
                f"@ {signal.get('price_at_signal'):.2f} "
                f"(Confidence: {signal.get('confidence_score'):.2f}, Type: {signal.get('signal_type')}) "
                f"at {signal.get('timestamp')}"
            )
            logger.info(signal_message)

            # TODO: Implement other alert channels (e.g., email, Telegram, PagerDuty)

        except Exception as e:
            await self.error_handler.handle_error(
                "alert_system",
                f"Error dispatching signal: {e}",
                {"signal": signal, "error": str(e)},
            )
