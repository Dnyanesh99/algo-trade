"""
Production-grade performance metrics tracking for latency, throughput, and success rates.
Thread-safe singleton implementation with configurable monitoring.
"""

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

# Load configuration
config = config_loader.get_config()


@dataclass
class MetricData:
    total_duration: float = 0.0
    call_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    last_call_time: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = float("-inf")


class PerformanceMetrics:
    """
    A utility class for tracking and reporting performance metrics such as latency,
    throughput, and success rates across different operations.
    """

    _instance: Optional["PerformanceMetrics"] = None
    _lock: threading.Lock = threading.Lock()
    _metrics: dict[str, MetricData]  # Declare _metrics here

    def __new__(cls) -> "PerformanceMetrics":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._metrics = defaultdict(MetricData)
                logger.info("PerformanceMetrics initialized.")
            return cls._instance

    def start_timer(self, operation_name: str) -> float:
        """
        Starts a timer for a given operation.
        Returns the current time (monotonic) to be used as a start marker.
        """
        if not config.performance.metrics.enabled:
            return 0.0
        start_time = time.monotonic()
        logger.debug(f"Timer started for '{operation_name}'.")
        return start_time

    def stop_timer(self, operation_name: str, start_time: float, success: bool = True) -> None:
        """
        Stops the timer for a given operation and updates metrics.

        Args:
            operation_name (str): The name of the operation.
            start_time (float): The start time returned by start_timer.
            success (bool): True if the operation was successful, False otherwise.
        """
        if not config.performance.metrics.enabled:
            return
        end_time = time.monotonic()
        duration = end_time - start_time

        with self._lock:
            metrics = self._metrics[operation_name]
            metrics.total_duration += duration
            metrics.call_count += 1
            metrics.last_call_time = end_time
            metrics.min_duration = min(metrics.min_duration, duration)
            metrics.max_duration = max(metrics.max_duration, duration)

            if success:
                metrics.success_count += 1
            else:
                metrics.fail_count += 1

        logger.debug(f"Timer stopped for '{operation_name}'. Duration: {duration:.4f}s, Success: {success}.")

    def get_metrics(self, operation_name: Optional[str] = None) -> dict[str, Any]:
        """
        Retrieves calculated metrics for a specific operation or all operations.

        Args:
            operation_name (str, optional): The name of the operation. If None, returns all metrics.

        Returns:
            Dict[str, Any]: A dictionary containing the metrics.
        """
        if not config.performance.metrics.enabled:
            return {}

        with self._lock:
            if operation_name:
                metrics = self._metrics.get(operation_name)
                if metrics:
                    return self._calculate_derived_metrics(metrics)
                return {}
            all_metrics = {}
            for op, metrics in self._metrics.items():
                all_metrics[op] = self._calculate_derived_metrics(metrics)
            return all_metrics

    def _calculate_derived_metrics(self, metrics: MetricData) -> dict[str, Any]:
        """
        Calculates derived metrics like average duration, success rate, etc.
        """
        derived = {
            "total_duration": metrics.total_duration,
            "call_count": metrics.call_count,
            "success_count": metrics.success_count,
            "fail_count": metrics.fail_count,
            "last_call_time": metrics.last_call_time,
            "min_duration": metrics.min_duration,
            "max_duration": metrics.max_duration,
        }
        call_count = derived["call_count"]

        if call_count > 0:
            derived["average_duration"] = derived["total_duration"] / call_count
            derived["success_rate"] = derived["success_count"] / call_count
            derived["fail_rate"] = derived["fail_count"] / call_count
        else:
            derived["average_duration"] = 0.0
            derived["success_rate"] = 0.0
            derived["fail_rate"] = 0.0

        # Clean up min/max if no calls were made
        if derived["min_duration"] == float("inf"):
            derived["min_duration"] = 0.0
        if derived["max_duration"] == float("-inf"):
            derived["max_duration"] = 0.0

        return derived

    def reset_metrics(self, operation_name: Optional[str] = None) -> None:
        """
        Resets metrics for a specific operation or all operations.
        """
        if not config.performance.metrics.enabled:
            return

        with self._lock:
            if operation_name:
                if operation_name in self._metrics:
                    self._metrics[operation_name] = MetricData()
                    logger.info(f"Metrics for '{operation_name}' reset.")
            else:
                self._metrics.clear()
                logger.info("All performance metrics reset.")

    @contextmanager
    def measure_execution(self, operation_name: str, success_on_exit: bool = True) -> Any:
        """
        Context manager to measure the execution time of a block of code.
        Usage:
            with performance_metrics.measure_execution("my_operation"):
                # code to measure
        """
        start_time = self.start_timer(operation_name)
        try:
            yield
        except Exception as e:
            self.stop_timer(operation_name, start_time, success=False)
            raise e
        else:
            self.stop_timer(operation_name, start_time, success=success_on_exit)


# Singleton instance
performance_metrics = PerformanceMetrics()



