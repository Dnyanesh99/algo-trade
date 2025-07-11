import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable
from typing import Any, Callable, Optional

from src.metrics import metrics_registry
from src.signal.alert_system import AlertSystem
from src.state.system_state import SystemState
from src.utils.config_loader import CircuitBreakerConfig
from src.utils.logger import LOGGER as logger


class CircuitBreaker:
    """
    Implements a basic circuit breaker pattern to prevent repeated failures
    from a component from cascading throughout the system.
    States: CLOSED, OPEN, HALF_OPEN.
    """

    def __init__(
        self,
        name: str,
        breaker_config: CircuitBreakerConfig,
        on_trip: Optional[Callable[[str], Awaitable[Any]]] = None,
        on_reset: Optional[Callable[[str], Awaitable[Any]]] = None,
    ):
        self.name = name
        self.failure_threshold = breaker_config.failure_threshold
        self.recovery_timeout = breaker_config.recovery_timeout
        self.half_open_attempts = breaker_config.half_open_attempts
        self.on_trip = on_trip
        self.on_reset = on_reset

        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_test_count = 0
        self._lock = asyncio.Lock()

        logger.info(f"Circuit Breaker '{self.name}' initialized in CLOSED state.")

    @property
    def state(self) -> str:
        return self._state

    async def _check_and_transition(self) -> None:
        """
        Internal method to check state transitions based on failures and time.
        """
        if self._state == "CLOSED":
            if self._failure_count >= self.failure_threshold:
                self._state = "OPEN"
                self._last_failure_time = time.monotonic()
                logger.warning(f"Circuit Breaker '{self.name}' tripped to OPEN state.")
                if self.on_trip:
                    await self.on_trip(self.name)
        elif self._state == "OPEN":
            if (time.monotonic() - self._last_failure_time) > self.recovery_timeout:
                self._state = "HALF_OPEN"
                self._half_open_test_count = 0
                logger.info(f"Circuit Breaker '{self.name}' transitioned to HALF_OPEN state.")
        elif self._state == "HALF_OPEN":
            # Logic for HALF_OPEN is handled by `attempt_operation`
            pass

    async def record_failure(self) -> None:
        """
        Records a failure and updates the circuit breaker state.
        """
        async with self._lock:
            if self._state == "CLOSED":
                self._failure_count += 1
                logger.debug(f"Circuit Breaker '{self.name}' recorded failure. Count: {self._failure_count}")
            elif self._state == "HALF_OPEN":
                # If a failure occurs in HALF_OPEN, immediately go back to OPEN
                self._state = "OPEN"
                self._last_failure_time = time.monotonic()
                self._failure_count = 1  # Reset failure count for next OPEN state
                logger.warning(f"Circuit Breaker '{self.name}' failed in HALF_OPEN, returning to OPEN state.")
                if self.on_trip:
                    await self.on_trip(self.name)
            await self._check_and_transition()

    async def record_success(self) -> None:
        """
        Records a success and updates the circuit breaker state.
        """
        async with self._lock:
            if self._state == "HALF_OPEN":
                self._half_open_test_count += 1
                if self._half_open_test_count >= self.half_open_attempts:
                    self._state = "CLOSED"
                    self._failure_count = 0
                    self._half_open_test_count = 0
                    logger.info(f"Circuit Breaker '{self.name}' reset to CLOSED state.")
                    if self.on_reset:
                        await self.on_reset(self.name)
            elif self._state == "CLOSED":
                self._failure_count = 0  # Reset failure count on success in CLOSED state
                logger.debug(f"Circuit Breaker '{self.name}' recorded success. Failure count reset.")

    async def allow_request(self) -> bool:
        """
        Checks if a request is allowed based on the current circuit breaker state.
        """
        async with self._lock:
            await self._check_and_transition()
            if self._state == "OPEN":
                logger.warning(f"Circuit Breaker '{self.name}' is OPEN. Request denied.")
                return False
            return True


class CircuitBreakerOpenError(Exception):
    """
    Custom exception raised when a circuit breaker is open and blocks a request.
    """


class ErrorHandler:
    """
    Implements a centralized error handling mechanism with circuit breakers
    and alert escalation.
    """

    def __init__(
        self, system_state: SystemState, alert_system: AlertSystem, error_handler_config: CircuitBreakerConfig
    ) -> None:
        self.system_state = system_state
        self.alert_system = alert_system
        self.breaker_config = error_handler_config
        self.circuit_breakers: dict[str, CircuitBreaker] = defaultdict(self._create_default_breaker)

    def _create_default_breaker(self) -> CircuitBreaker:
        """
        Creates a default circuit breaker instance using configuration values.
        """
        return CircuitBreaker(
            name="default",  # Name will be overridden when added to dict
            breaker_config=self.breaker_config,
            on_trip=self._on_breaker_trip,
            on_reset=self._on_breaker_reset,
        )

    async def _on_breaker_trip(self, breaker_name: str) -> None:
        """
        Callback when a circuit breaker trips to OPEN state.
        Escalates alert and updates system state.
        """
        logger.critical(f"Circuit Breaker for '{breaker_name}' has TRIPPED! Escalating alert.")
        self.system_state.set_system_status(f"ERROR: {breaker_name}_BREAKER_TRIPPED")
        await self.alert_system.dispatch_system_alert(
            level="CRITICAL",
            component=breaker_name,
            message=f"Circuit Breaker for '{breaker_name}' has TRIPPED!",
            details={},
        )

    async def _on_breaker_reset(self, breaker_name: str) -> None:
        """
        Callback when a circuit breaker resets to CLOSED state.
        """
        logger.info(f"Circuit Breaker for '{breaker_name}' has RESET. System recovering.")
        # Potentially revert system status if no other errors
        if self.system_state.get_system_status() == f"ERROR: {breaker_name}_BREAKER_TRIPPED":
            self.system_state.set_system_status("LIVE")  # Or previous state
        await self.alert_system.dispatch_system_alert(
            level="INFO", component=breaker_name, message=f"Circuit Breaker for '{breaker_name}' has RESET.", details={}
        )

    async def handle_error(self, component_name: str, message: str, details: dict[str, Any]) -> None:
        """
        Centralized method to handle errors, log them, and update system state.
        """
        logger.error(f"Error in {component_name}: {message}. Details: {details}")
        metrics_registry.increment_counter(
            "error_recovery_attempts_total", {"component": component_name, "error_type": "handled_error"}
        )
        self.system_state.set_system_status(f"ERROR: {component_name}_FAILURE")
        await self.alert_system.dispatch_system_alert(
            level="ERROR", component=component_name, message=message, details=details
        )

    def get_breaker(self, component_name: str) -> CircuitBreaker:
        """
        Retrieves or creates a circuit breaker for a given component.
        """
        if component_name not in self.circuit_breakers:
            breaker = self._create_default_breaker()
            breaker.name = component_name
            self.circuit_breakers[component_name] = breaker
        return self.circuit_breakers[component_name]

    async def execute_safely(
        self, component_name: str, operation: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Executes an operation safely using the circuit breaker pattern.
        """
        breaker = self.get_breaker(component_name)
        if not await breaker.allow_request():
            raise CircuitBreakerOpenError(f"Operation for {component_name} blocked by circuit breaker.")

        try:
            result = await operation(*args, **kwargs)
            await breaker.record_success()
            return result
        except Exception as e:
            await breaker.record_failure()
            logger.error(f"Operation for {component_name} failed: {e}")
            raise
