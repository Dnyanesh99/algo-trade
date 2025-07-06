import threading
from typing import Any, Optional

from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger  # Centralized logger

# Load configuration
config = config_loader.get_config()

# No need for direct logging configuration here, it's handled by src.utils.logger


class SystemState:
    """
    A central state machine that tracks the overall system status.
    It holds flags like `60_min_data_available`.
    """

    _instance: Optional["SystemState"] = None
    _lock: threading.Lock = threading.Lock()
    _60_min_data_available: bool
    _system_status: str
    _component_health: dict[str, Any]

    def __new__(cls) -> "SystemState":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._60_min_data_available = False
                cls._instance._system_status = "INITIALIZING"
                cls._instance._component_health = {}
                logger.info("SystemState initialized.")
        return cls._instance

    def is_60_min_data_available(self) -> bool:
        """
        Checks if 60-minute data is available for prediction.
        """
        status = self._60_min_data_available
        logger.debug(f"60-min data availability check: {status}")
        return status

    def set_60_min_data_available(self, status: bool) -> None:
        """
        Sets the status of 60-minute data availability.
        """
        self._60_min_data_available = status
        logger.info(f"60-min data availability set to: {status}")

    def get_system_status(self) -> str:
        """
        Returns the overall system status.
        """
        status = self._system_status
        logger.debug(f"Current system status: {status}")
        return status

    def set_system_status(self, status: str) -> None:
        """
        Sets the overall system status.
        """
        self._system_status = status
        logger.info(f"System status updated to: {status}")

    def update_component_health(self, component_name: str, health_info: Any) -> None:
        """
        Updates the health information for a specific component.
        """
        self._component_health[component_name] = health_info
        logger.debug(f"Component {component_name} health updated: {health_info}")

    def get_component_health(self, component_name: Optional[str] = None) -> Any:
        """
        Returns health information for a specific component or all components.
        """
        if component_name:
            health = self._component_health.get(component_name)
            logger.debug(f"Retrieved health for {component_name}: {health}")
            return health
        logger.debug(f"Retrieved all component health: {self._component_health}")
        return self._component_health



