import asyncio
from typing import Any, ClassVar, Optional

from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

config = config_loader.get_config()


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
        self._queue = asyncio.Queue(maxsize=config.performance.processing.tick_queue_max_size)
        TickQueue._is_initialized = True
        logger.info("TickQueue initialized.")

    async def put(self, tick: dict[str, Any]) -> None:
        """
        Puts a single tick into the queue. If the queue is full, it will log a warning
        and potentially drop the tick to prevent blocking the producer (WebSocket callback).
        """
        try:
            self._queue.put_nowait(tick)
            logger.debug(f"Tick added to queue. Current size: {self._queue.qsize()}")
        except asyncio.QueueFull:
            logger.warning("Tick queue is full. Dropping tick to prevent blocking WebSocket callback.")

    async def put_many(self, ticks: list[dict[str, Any]]) -> None:
        """
        Puts multiple ticks into the queue. Tries to put all, but may drop some if queue is full.
        """
        successfully_put = 0
        for tick in ticks:
            try:
                self._queue.put_nowait(tick)
                successfully_put += 1
            except asyncio.QueueFull:
                logger.warning(
                    "Tick queue is full while putting multiple ticks. Dropping remaining ticks in this batch."
                )
                break  # Stop putting more ticks from this batch if queue is full
        logger.debug(
            f"Attempted to put {len(ticks)} ticks, successfully put {successfully_put}. Current queue size: {self._queue.qsize()}"
        )

    async def get(self) -> dict[str, Any]:
        """
        Retrieves a single tick from the queue. Awaits if the queue is empty.
        """
        tick = await self._queue.get()
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
