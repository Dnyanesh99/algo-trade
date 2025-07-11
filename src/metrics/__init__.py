"""
Centralized metrics system for the algo trading application.
Provides Prometheus metrics with config-driven definitions.
"""

from .registry import metrics_registry

__all__ = ["metrics_registry"]
