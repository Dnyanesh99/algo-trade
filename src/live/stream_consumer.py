import asyncio
import time
from typing import Optional

from src.live.candle_buffer import CandleBuffer
from src.live.tick_queue import TickQueue
from src.live.tick_validator import TickValidator
from src.metrics import metrics_registry
from src.utils.logger import LOGGER as logger


class StreamConsumer:
    """
    Consumes ticks from the TickQueue, validates them, and passes them to the CandleBuffer.
    """

    def __init__(
        self,
        tick_queue: TickQueue,
        tick_validator: TickValidator,
        candle_buffer: CandleBuffer,
    ):
        if not all(
            isinstance(obj, (TickQueue, TickValidator, CandleBuffer))
            for obj in [tick_queue, tick_validator, candle_buffer]
        ):
            raise TypeError("All arguments must be valid instances of their respective classes.")

        self.tick_queue = tick_queue
        self.tick_validator = tick_validator
        self.candle_buffer = candle_buffer
        self._consumer_task: Optional[asyncio.Task[None]] = None
        logger.info("StreamConsumer initialized.")

    async def _consume_ticks(self) -> None:
        """
        Continuously consumes ticks from the queue, validates, and processes them.
        """
        logger.info("StreamConsumer starting tick consumption loop.")
        while True:
            try:
                start_time = time.monotonic()
                tick = await self.tick_queue.get()
                if not tick:
                    logger.debug("Received None tick, continuing...")
                    continue

                if self.tick_validator.validate_tick(tick):
                    await self.candle_buffer.process_tick(tick)
                else:
                    logger.warning(f"Invalid tick received, skipping: {tick}")
                    metrics_registry.increment_counter(
                        "invalid_ticks_total", {"instrument_token": tick.get("instrument_token", "unknown")}
                    )

                duration = time.monotonic() - start_time
                metrics_registry.observe_histogram("live_tick_processing_duration_seconds", duration)

            except asyncio.CancelledError:
                logger.info("StreamConsumer tick consumption loop cancelled.")
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred in the tick consumption loop: {e}", exc_info=True)
                # Optional: Add a small delay to prevent rapid-fire error loops
                await asyncio.sleep(1)

    def start_consuming(self) -> None:
        """
        Starts the background task for tick consumption.
        """
        if self._consumer_task and not self._consumer_task.done():
            logger.warning("Attempted to start StreamConsumer, but it is already running.")
            return
        self._consumer_task = asyncio.create_task(self._consume_ticks())
        logger.info("StreamConsumer consumption task has been started.")

    async def stop_consuming(self) -> None:
        """
        Stops the background task for tick consumption gracefully.
        """
        if not self._consumer_task or self._consumer_task.done():
            logger.info("StreamConsumer is not running.")
            return

        logger.info("Stopping StreamConsumer consumption task...")
        self._consumer_task.cancel()
        try:
            await self._consumer_task
        except asyncio.CancelledError:
            logger.info("StreamConsumer consumption task stopped successfully.")
        except Exception as e:
            logger.error(f"An error occurred while stopping the consumption task: {e}", exc_info=True)
        finally:
            self._consumer_task = None
            logger.info("StreamConsumer consumption task cleaned up.")
