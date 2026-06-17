import asyncio
from typing import Any, ClassVar, Optional

from src.metrics import metrics_registry
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

config = ConfigLoader().get_config()


class TickQueue:
    """
    An asynchronous queue that immediately stores incoming ticks from the WebSocket
    on_ticks callback. This prevents the callback from blocking and ensures no ticks
    are lost during processing spikes.
    """

    _instance: ClassVar[Optional["TickQueue"]] = None
    _is_initialized: bool = False
    _queue: asyncio.Queue[dict[str, Any]]

    def __new__(cls) -> "TickQueue":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._is_initialized:
            return
        if (
            not config.performance
            or not config.performance.processing
            or not config.performance.processing.tick_queue_max_size
        ):
            raise ValueError("performance.processing.tick_queue_max_size configuration is required.")

        max_size = config.performance.processing.tick_queue_max_size
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("tick_queue_max_size must be a positive integer.")

        self._queue = asyncio.Queue(maxsize=max_size)
        TickQueue._is_initialized = True
        logger.info(f"TickQueue initialized with a max size of {max_size}.")

    async def put(self, tick: dict[str, Any]) -> None:
        """
        Puts a single tick into the queue. If the queue is full, it will log a warning
        and drop the tick to prevent blocking the producer (WebSocket callback).
        """
        try:
            self._queue.put_nowait(tick)
            logger.debug(f"Tick for {tick.get('instrument_token')} added to queue. Current size: {self._queue.qsize()}")
            metrics_registry.increment_counter("ticks_added_to_queue_total")
        except asyncio.QueueFull:
            logger.error(
                "CRITICAL: Tick queue is full. Dropping tick to prevent WebSocket callback blockage. This indicates a performance bottleneck."
            )
            metrics_registry.increment_counter("tick_queue_overflow_total")

    async def put_many(self, ticks: list[dict[str, Any]]) -> None:
        """
        Puts multiple ticks into the queue. Tries to put all, but may drop some if queue is full.
        """
        for tick in ticks:
            await self.put(tick)

    async def get(self) -> dict[str, Any]:
        """
        Retrieves a single tick from the queue. Awaits if the queue is empty.
        """
        tick = await self._queue.get()
        metrics_registry.set_gauge("tick_queue_size", self._queue.qsize())
        logger.debug(f"Tick retrieved from queue. Current size: {self._queue.qsize()}")
        return tick

    async def get_nowait(self) -> Optional[dict[str, Any]]:
        """
        Retrieves a single tick from the queue without waiting. Returns None if empty.
        """
        try:
            tick = self._queue.get_nowait()
            logger.debug(f"Tick retrieved from queue (non-blocking). Current size: {self._queue.qsize()}")
            return tick
        except asyncio.QueueEmpty:
            logger.debug("Tick queue is empty (non-blocking call).")
            return None

    def qsize(self) -> int:
        """
        Returns the current size of the queue.
        """
        return self._queue.qsize()

    def empty(self) -> bool:
        """
        Returns True if the queue is empty, False otherwise.
        """
        return self._queue.empty()

    def full(self) -> bool:
        """
        Returns True if the queue is full, False otherwise.
        """
        return self._queue.full()


# Singleton instance
tick_queue = TickQueue()
