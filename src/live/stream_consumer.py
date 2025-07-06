import asyncio
from datetime import datetime
from typing import Any, Optional

import pytz

from src.live.candle_buffer import CandleBuffer
from src.live.tick_queue import TickQueue
from src.live.tick_validator import TickValidator
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

config = config_loader.get_config()


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
        self.tick_queue = tick_queue
        self.tick_validator = tick_validator
        self.candle_buffer = candle_buffer
        self._consumer_task: Optional[asyncio.Task] = None
        logger.info("StreamConsumer initialized.")

    async def _consume_ticks(self) -> None:
        """
        Continuously consumes ticks from the queue, validates, and processes them.
        """
        logger.info("StreamConsumer: Starting tick consumption.")
        while True:
            try:
                tick = await self.tick_queue.get()
                if tick:
                    # KiteTicker's on_ticks callback already provides parsed ticks.
                    # So, we just need to validate and pass to candle_buffer.
                    # If raw binary parsing was needed, it would happen here.
                    if self.tick_validator.validate_tick(tick):
                        # Ensure timestamp is a datetime object and UTC localized
                        timestamp = tick["timestamp"]
                        if isinstance(timestamp, (int, float)):
                            # Assuming Unix timestamp in seconds
                            timestamp = datetime.fromtimestamp(timestamp, tz=datetime.now().astimezone().tzinfo) # Localize to system timezone first
                        elif isinstance(timestamp, str):
                            try:
                                timestamp = datetime.fromisoformat(timestamp)
                            except ValueError:
                                logger.error(
                                    f"StreamConsumer: Invalid timestamp string format: {timestamp}. Skipping tick."
                                )
                                continue

                        if not isinstance(timestamp, datetime):
                            logger.error(f"StreamConsumer: Unexpected timestamp type after conversion: {type(timestamp)}. Skipping tick.")
                            continue

                        # Ensure timestamp is UTC for consistency in the pipeline
                        if timestamp.tzinfo is None:
                            # Assume it's IST if no timezone info, then convert to UTC
                            ist_tz = config_loader.get_config().system.timezone
                            timestamp = pytz.timezone(ist_tz).localize(timestamp).astimezone(pytz.utc)
                        else:
                            timestamp = timestamp.astimezone(pytz.utc)

                        tick["timestamp"] = timestamp
                        await self.candle_buffer.process_tick(tick)
                    else:
                        logger.warning(f"StreamConsumer: Invalid tick received, skipping: {tick}")
                else:
                    logger.debug("StreamConsumer: Received None tick, continuing.")
            except asyncio.CancelledError:
                logger.info("StreamConsumer: Tick consumption cancelled.")
                break
            except Exception as e:
                logger.error(f"StreamConsumer: Error processing tick: {e}", exc_info=True)
                # Continue processing other ticks despite error

    def start_consuming(self) -> None:
        """
        Starts the background task for tick consumption.
        """
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._consume_ticks())
            logger.info("StreamConsumer: Consumption started.")

    async def stop_consuming(self) -> None:
        """
        Stops the background task for tick consumption gracefully.
        """
        if self._consumer_task:
            logger.info("StreamConsumer: Stopping consumption...")
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                logger.info("StreamConsumer: Consumption stopped successfully.")
            except Exception as e:
                logger.error(f"StreamConsumer: Error stopping consumption task: {e}")
            finally:
                self._consumer_task = None



