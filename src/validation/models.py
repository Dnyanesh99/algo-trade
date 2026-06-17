from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DataQualityReport:
    """Data quality assessment results."""

    total_rows: int
    valid_rows: int
    nan_counts: dict[str, int]
    ohlc_violations: int
    duplicate_timestamps: int
    time_gaps: int
    outliers: dict[str, int]
    quality_score: float
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class ValidationRule:
    """Feature validation rule definition."""

    feature_pattern: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    validator_type: str = "range"
    custom_logic: Optional[str] = None
    priority: int = 1
    description: str = ""


@dataclass
class FeatureValidationResult:
    """Result of feature validation."""

    feature_name: str
    is_valid: bool
    violations_count: int
    applied_strategy: str
    validation_rule: Optional[ValidationRule] = None
    error_message: Optional[str] = None
