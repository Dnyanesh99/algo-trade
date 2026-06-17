"""
Environment variable utilities for consistent parsing across the application.
"""

import os
from typing import Union


def get_bool_env(env_var_name: str, default: bool = False) -> bool:
    """
    Parse boolean environment variables with consistent logic across the application.

    This utility eliminates redundant _get_bool_env methods scattered throughout the codebase.

    Args:
        env_var_name: Name of the environment variable
        default: Default value if env var is not set

    Returns:
        Boolean value parsed from environment variable

    Examples:
        >>> os.environ['DEBUG'] = 'true'
        >>> get_bool_env('DEBUG')
        True
        >>> get_bool_env('NONEXISTENT', False)
        False
    """
    env_value = os.getenv(env_var_name)
    if env_value is None:
        return default
    return env_value.lower() in ("true", "1", "yes", "on")


def get_int_env(env_var_name: str, default: int = 0) -> int:
    """
    Parse integer environment variables with error handling.

    Args:
        env_var_name: Name of the environment variable
        default: Default value if env var is not set or invalid

    Returns:
        Integer value parsed from environment variable
    """
    env_value = os.getenv(env_var_name)
    if env_value is None:
        return default

    try:
        return int(env_value)
    except (ValueError, TypeError):
        return default


def get_float_env(env_var_name: str, default: float = 0.0) -> float:
    """
    Parse float environment variables with error handling.

    Args:
        env_var_name: Name of the environment variable
        default: Default value if env var is not set or invalid

    Returns:
        Float value parsed from environment variable
    """
    env_value = os.getenv(env_var_name)
    if env_value is None:
        return default

    try:
        return float(env_value)
    except (ValueError, TypeError):
        return default


def get_str_env(env_var_name: str, default: str = "") -> str:
    """
    Get string environment variable with default.

    Args:
        env_var_name: Name of the environment variable
        default: Default value if env var is not set

    Returns:
        String value from environment variable
    """
    return os.getenv(env_var_name, default)


def get_list_env(env_var_name: str, separator: str = ",", default: Union[list[str], None] = None) -> list[str]:
    """
    Parse list environment variables (comma-separated by default).

    Args:
        env_var_name: Name of the environment variable
        separator: Separator character for splitting
        default: Default list if env var is not set

    Returns:
        List of strings parsed from environment variable
    """
    if default is None:
        default = []

    env_value = os.getenv(env_var_name)
    if env_value is None:
        return default

    return [item.strip() for item in env_value.split(separator) if item.strip()]
