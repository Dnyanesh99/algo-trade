from typing import Any

from src.metrics import metrics_registry
from src.utils.logger import LOGGER as logger


class AlertSystem:
    """
    Handles logging and dispatching trading signals and system alerts.
    """

    def __init__(self) -> None:
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
            metrics_registry.increment_counter(
                "alerts_sent_total", {"channel": "log", "signal_type": str(signal.get("signal_type"))}
            )

            # TODO: Implement other alert channels (e.g., email, Telegram, PagerDuty)

        except Exception as e:
            logger.error(f"Error dispatching signal in AlertSystem: {e}", exc_info=True)

    async def dispatch_system_alert(self, level: str, component: str, message: str, details: dict[str, Any]) -> None:
        """
        Dispatches a system alert.
        """
        try:
            alert_message = f"SYSTEM ALERT ({level.upper()}): [{component}] {message} | Details: {details}"

            if level.upper() == "CRITICAL":
                logger.critical(alert_message)
            elif level.upper() == "ERROR":
                logger.error(alert_message)
            elif level.upper() == "WARNING":
                logger.warning(alert_message)
            else:
                logger.info(alert_message)

            metrics_registry.increment_counter(
                "system_alerts_sent_total", {"channel": "log", "level": level.lower(), "component": component}
            )

            # TODO: Implement other alert channels (e.g., email, Telegram, PagerDuty) for critical/error alerts

        except Exception as e:
            logger.error(f"Critical error within AlertSystem itself while dispatching system alert: {e}", exc_info=True)
