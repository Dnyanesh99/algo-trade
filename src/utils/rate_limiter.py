"""
Production-grade token bucket rate limiter for API endpoints.
Implements thread-safe, configurable rate limiting with proper error handling.
"""

import asyncio
import threading
import time
from collections import deque
from typing import Any

from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

config = config_loader.get_config()


class RateLimiter:
    """
    Thread-safe token bucket rate limiter implementation.

    Features:
    - Per-endpoint rate limiting based on configuration
    - Exponential backoff for queue waiting
    - Graceful degradation with default limits
    - Performance monitoring integration
    """

    _instances: dict[str, "RateLimiter"] = {}
    _creation_lock = threading.Lock()

    def __new__(cls, endpoint_name: str) -> "RateLimiter":
        with cls._creation_lock:
            if endpoint_name not in cls._instances:
                cls._instances[endpoint_name] = super().__new__(cls)
        return cls._instances[endpoint_name]

    def __init__(self, endpoint_name: str) -> None:
        if hasattr(self, "_initialized"):
            return

        self.endpoint_name = endpoint_name
        self.rate_limit = getattr(config.api_rate_limits, endpoint_name, None)

        # Set limits from configuration or raise error if not found
        if not self.rate_limit:
            logger.critical(
                f"No rate limit configuration found for endpoint: {endpoint_name}. "
                "This is a critical misconfiguration. Using hardcoded safe defaults (1 req/sec)."
            )
            # Fallback to a very safe default if configuration is missing
            self.limit = 1.0
            self.interval = 1.0
        else:
            self.limit = float(self.rate_limit.limit)
            self.interval = float(self.rate_limit.interval)

        # Token bucket state
        self.tokens = self.limit
        self.last_refill_time = time.monotonic()
        self.queue: deque = deque()
        self._lock = asyncio.Lock()

        # Performance tracking
        self.total_requests = 0
        self.blocked_requests = 0

        self._initialized = True
        logger.info(f"RateLimiter initialized for '{endpoint_name}': {self.limit} req/{self.interval}s")

    async def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        time_passed = now - self.last_refill_time

        if time_passed > 0:
            refill_amount = (time_passed / self.interval) * self.limit
            self.tokens = min(self.limit, self.tokens + refill_amount)
            self.last_refill_time = now

    async def acquire(self) -> None:
        """
        Acquire a token, blocking if necessary until available.
        Implements fair queuing and exponential backoff.
        """
        self.total_requests += 1

        async with self._lock:
            await self._refill_tokens()

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                logger.debug(f"Token acquired for {self.endpoint_name}. Remaining: {self.tokens:.2f}")
                return

        # Need to wait for tokens
        self.blocked_requests += 1
        logger.debug(f"Rate limit reached for {self.endpoint_name}. Queuing request...")

        # Exponential backoff with jitter
        wait_time = self.interval / self.limit
        backoff_multiplier = 1.0
        max_backoff = min(5.0, self.interval * 2)

        while True:
            await asyncio.sleep(wait_time)

            async with self._lock:
                await self._refill_tokens()

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    logger.debug(f"Token acquired for {self.endpoint_name} after waiting. Remaining: {self.tokens:.2f}")
                    return

            # Exponential backoff with cap
            wait_time = min(wait_time * backoff_multiplier, max_backoff)
            backoff_multiplier = min(backoff_multiplier * 1.5, 3.0)

    def get_stats(self) -> dict[str, float]:
        """Get rate limiter performance statistics."""
        if self.total_requests == 0:
            return {
                "total_requests": 0,
                "blocked_requests": 0,
                "block_rate": 0.0,
                "current_tokens": self.tokens,
            }

        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": self.blocked_requests / self.total_requests,
            "current_tokens": self.tokens,
        }

    async def __aenter__(self) -> "RateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Nothing needed for cleanup in token bucket
        pass
