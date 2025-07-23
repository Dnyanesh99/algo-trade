"""
FastAPI dependencies for the trading system API
"""

from typing import Any, Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.utils.logger import logger

# Global application state - will be populated by main.py
_app_state: dict[str, Any] = {}
_websocket_manager = None


def set_app_state(state: dict[str, Any]) -> None:
    """Set the global application state"""
    global _app_state
    _app_state = state


def set_websocket_manager(manager: Any) -> None:
    """Set the global WebSocket manager"""
    global _websocket_manager
    _websocket_manager = manager


def get_app_state() -> dict[str, Any]:
    """Get the global application state"""
    if not _app_state:
        raise HTTPException(status_code=500, detail="Application state not initialized")
    return _app_state


def get_websocket_manager() -> Any:
    """Get the global WebSocket manager"""
    if not _websocket_manager:
        raise HTTPException(status_code=500, detail="WebSocket manager not initialized")
    return _websocket_manager


def get_config() -> Any:
    """Get the application configuration"""
    state = get_app_state()
    return state.get("config")


def get_db_manager() -> Any:
    """Get the database manager"""
    state = get_app_state()
    return state.get("db_manager")


def get_system_state() -> Any:
    """Get the system state"""
    state = get_app_state()
    return state.get("system_state")


def get_health_monitor() -> Any:
    """Get the health monitor"""
    state = get_app_state()
    return state.get("health_monitor")


def get_repositories() -> dict[str, Any]:
    """Get all repositories"""
    state = get_app_state()
    return state.get("repositories", {})  # type: ignore[no-any-return]


def get_instrument_repository() -> Any:
    """Get the instrument repository"""
    repositories = get_repositories()
    return repositories.get("instrument")


def get_ohlcv_repository() -> Any:
    """Get the OHLCV repository"""
    repositories = get_repositories()
    return repositories.get("ohlcv")


def get_feature_repository() -> Any:
    """Get the feature repository"""
    repositories = get_repositories()
    return repositories.get("feature")


def get_signal_repository() -> Any:
    """Get the signal repository"""
    repositories = get_repositories()
    return repositories.get("signal")


def get_label_repository() -> Any:
    """Get the label repository"""
    repositories = get_repositories()
    return repositories.get("label")


# Authentication dependencies
security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security), config: Any = Depends(get_config)
) -> dict[str, str]:
    """Get the current authenticated user"""
    # For now, we'll implement a simple token-based authentication
    # In production, you'd want to integrate with your authentication system

    if not credentials:
        # For development, allow unauthenticated access
        return {"user_id": "admin", "role": "admin"}

    token = credentials.credentials
    admin_token = config.api.admin_token

    # Validate token (implement your token validation logic here)
    if token == admin_token:
        return {"user_id": "admin", "role": "admin"}
    raise HTTPException(status_code=401, detail="Invalid authentication token")


def require_admin_role(current_user: dict[str, str] = Depends(get_current_user)) -> dict[str, str]:
    """Require admin role for the current user"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return current_user


def check_system_running() -> bool:
    """Check if the system is running"""
    system_state = get_system_state()
    if system_state.status != "running":
        raise HTTPException(status_code=400, detail=f"System is not running. Current status: {system_state.status}")
    return True


def check_system_mode(required_mode: str) -> bool:
    """Check if the system is in the required mode"""
    config = get_config()
    if config.system.mode != required_mode:
        raise HTTPException(
            status_code=400, detail=f"System must be in {required_mode} mode. Current mode: {config.system.mode}"
        )
    return True


def validate_instrument_id(instrument_id: int) -> Any:
    """Validate that an instrument ID exists"""
    try:
        instrument_repo = get_instrument_repository()
        instrument = instrument_repo.get_instrument_by_id(instrument_id)
        if not instrument:
            raise HTTPException(status_code=404, detail=f"Instrument {instrument_id} not found")
        return instrument
    except Exception as e:
        logger.error(f"Failed to validate instrument ID {instrument_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate instrument") from e


def validate_timeframe(timeframe: str) -> int:
    """Validate that a timeframe is supported"""
    config = get_config()
    supported_timeframes = config.trading.aggregation_timeframes

    # Convert timeframe string to minutes for validation
    timeframe_minutes = _timeframe_to_minutes(timeframe)

    if timeframe_minutes not in supported_timeframes:
        raise HTTPException(
            status_code=400, detail=f"Timeframe {timeframe} not supported. Supported: {supported_timeframes}"
        )
    return timeframe_minutes


def _timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes"""
    if timeframe.endswith("m"):
        return int(timeframe[:-1])
    if timeframe.endswith("h"):
        return int(timeframe[:-1]) * 60
    if timeframe.endswith("d"):
        return int(timeframe[:-1]) * 1440
    # Assume it's already in minutes
    return int(timeframe)


def get_rate_limiter() -> None:
    """Get rate limiter for API endpoints"""
    # Implement rate limiting logic here
    # For now, return a placeholder
    return


def log_api_call(endpoint: str, user_id: Optional[str] = None) -> None:
    """Log API call for auditing"""
    logger.info(f"API call: {endpoint}, User: {user_id}")


# Pagination dependencies
def get_pagination_params(page: int = 1, size: int = 100) -> dict[str, int]:
    """Get pagination parameters"""
    if page < 1:
        raise HTTPException(status_code=400, detail="Page must be >= 1")
    if size < 1 or size > 1000:
        raise HTTPException(status_code=400, detail="Size must be between 1 and 1000")

    offset = (page - 1) * size
    return {"offset": offset, "limit": size, "page": page, "size": size}


# Validation dependencies
def validate_date_range(start_date: str, end_date: str) -> dict[str, Any]:
    """Validate date range parameters"""
    from datetime import datetime

    try:
        start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format.") from e

    if start >= end:
        raise HTTPException(status_code=400, detail="Start date must be before end date")

    # Check if date range is not too large (e.g., max 1 year)
    if (end - start).days > 365:
        raise HTTPException(status_code=400, detail="Date range cannot exceed 1 year")

    return {"start_date": start, "end_date": end}


# Health check dependencies
def check_database_health() -> bool:
    """Check database health"""
    try:
        get_db_manager()
        # Perform a simple health check query
        # This would depend on your database implementation
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(status_code=503, detail="Database unhealthy") from e


def check_broker_connection() -> bool:
    """Check broker connection health"""
    try:
        # Check if broker connection is healthy
        # This would depend on your broker implementation
        return True
    except Exception as e:
        logger.error(f"Broker health check failed: {e}")
        raise HTTPException(status_code=503, detail="Broker connection unhealthy") from e


# Configuration validation dependencies
def validate_config_section(section: str) -> str:
    """Validate that a configuration section exists"""
    config = get_config()
    if not hasattr(config, section):
        raise HTTPException(status_code=404, detail=f"Configuration section '{section}' not found")
    return section


def validate_config_update(section: str, data: dict[str, Any]) -> dict[str, Any]:
    """Validate configuration update data"""
    config = get_config()
    section_obj = getattr(config, section)

    # Validate that all keys exist in the section
    for key in data:
        if not hasattr(section_obj, key):
            raise HTTPException(status_code=400, detail=f"Unknown configuration key: {section}.{key}")

    return data


# Model dependencies
def validate_model_id(model_id: str) -> str:
    """Validate that a model ID exists"""
    # This would check your model registry
    # For now, return a placeholder validation
    if not model_id:
        raise HTTPException(status_code=400, detail="Model ID is required")
    return model_id


# Signal filtering dependencies
def validate_signal_filters(
    signal_type: Optional[str] = None, direction: Optional[str] = None, min_confidence: Optional[float] = None
) -> dict[str, Any]:
    """Validate signal filtering parameters"""
    filters = {}

    if signal_type:
        filters["signal_type"] = signal_type

    if direction:
        valid_directions = ["BUY", "SELL", "HOLD", "NEUTRAL"]
        if direction not in valid_directions:
            raise HTTPException(status_code=400, detail=f"Invalid direction. Must be one of: {valid_directions}")
        filters["direction"] = direction

    if min_confidence is not None:
        if min_confidence < 0 or min_confidence > 1:
            raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1")
        filters["min_confidence"] = str(min_confidence)

    return filters


# File upload dependencies
def validate_file_upload(file_type: str, max_size: int = 10 * 1024 * 1024) -> Any:  # 10MB default
    """Validate file upload parameters"""

    def validator(file: Any) -> Any:
        if file.size > max_size:
            raise HTTPException(status_code=400, detail=f"File size exceeds maximum limit of {max_size} bytes")

        if file_type == "csv" and not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be a CSV file")

        return file

    return validator
