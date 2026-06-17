import asyncio
import time
from types import TracebackType
from typing import Optional

from src.utils.config_loader import ApiRateLimit, AppConfig
from src.utils.logger import LOGGER as logger


class RateLimiter:
    """
    A token bucket rate limiter for controlling access to API endpoints.
    Ensures that requests do not exceed the defined rate limits.
    """

    def __init__(self, endpoint_name: str, rate_limit_config: ApiRateLimit) -> None:
        self.endpoint_name = endpoint_name
        self.limit = float(rate_limit_config.limit)
        self.interval = rate_limit_config.interval

        # Token bucket state
        self.tokens: float = self.limit
        self.last_refill_time = time.monotonic()
        self._lock = asyncio.Lock()

        logger.info(
            f"RateLimiter for '{endpoint_name}' initialized: "
            f"Limit={self.limit} requests, Interval={self.interval} seconds"
        )

    async def acquire(self) -> None:
        """
        Acquires a token from the bucket, waiting if necessary.
        This method is a coroutine and should be awaited.
        """
        async with self._lock:
            self._refill()

            while self.tokens < 1:
                # Calculate wait time based on when the next token will be available
                time_to_refill = self.interval / self.limit
                await asyncio.sleep(time_to_refill)
                self._refill()

            self.tokens -= 1

    async def __aenter__(self) -> "RateLimiter":
        await self.acquire()
        return self

    async def __aexit__(
        self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        pass

    def _refill(self) -> None:
        """
        Refills the token bucket based on the elapsed time.
        """
        now = time.monotonic()
        elapsed = now - self.last_refill_time
        tokens_to_add = elapsed * (self.limit / self.interval)

        if tokens_to_add > 0:
            self.tokens = min(self.limit, self.tokens + tokens_to_add)
            self.last_refill_time = now


_rate_limiters: dict[str, RateLimiter] = {}
_rate_limiter_lock = asyncio.Lock()


async def get_rate_limiter(endpoint_name: str, config: Optional[AppConfig] = None) -> RateLimiter:
    """
    Factory function to get a RateLimiter instance for a given endpoint.
    This ensures that there is only one RateLimiter per endpoint.
    """
    async with _rate_limiter_lock:
        if endpoint_name not in _rate_limiters:
            if config is None:
                from src.utils.config_loader import ConfigLoader

                config_loader = ConfigLoader()
                config = config_loader.get_config()

            if not config.api_rate_limits:
                logger.error(
                    "ConfigurationError: api_rate_limits configuration is missing. Cannot initialize rate limiter."
                )
                raise ValueError("api_rate_limits configuration is missing.")

            rate_limit_config = getattr(config.api_rate_limits, endpoint_name, None)
            if not rate_limit_config:
                logger.error(
                    f"ConfigurationError: No rate limit configuration found for endpoint: {endpoint_name}. Cannot initialize rate limiter."
                )
                raise ValueError(f"No rate limit configuration found for endpoint: {endpoint_name}")

            _rate_limiters[endpoint_name] = RateLimiter(endpoint_name, rate_limit_config)

        return _rate_limiters[endpoint_name]
