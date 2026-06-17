"""
Simple test configuration to avoid complex Pydantic validation errors.
"""
from dataclasses import dataclass
from typing import Optional
from src.utils.config_loader import DataQualityConfig


def create_test_config(required_columns: Optional[list[str]] = None) -> DataQualityConfig:
    """Create a minimal test configuration for validation tests."""
    
    @dataclass 
    class MockValidationConfig:
        required_columns: list[str] = None
        min_valid_rows: int = 1
        quality_score_threshold: float = 70.0
        
        def __post_init__(self):
            if self.required_columns is None:
                self.required_columns = ["ts", "open", "high", "low", "close", "volume"]
    
    @dataclass
    class MockTimeSeriesConfig:
        gap_multiplier: float = 1.5
    
    @dataclass 
    class MockOutlierConfig:
        handling_strategy: str = "clip"
        iqr_multiplier: float = 1.5
    
    @dataclass
    class MockPenaltiesConfig:
        outlier_penalty: float = 0.1
        gap_penalty: float = 0.2  
        ohlc_violation_penalty: float = 0.15
        duplicate_penalty: float = 0.1
    
    @dataclass
    class MockDataQualityConfig:
        validation: MockValidationConfig
        time_series: MockTimeSeriesConfig
        outlier_detection: MockOutlierConfig
        penalties: MockPenaltiesConfig
    
    # Set required columns if provided
    if required_columns is not None:
        validation_config = MockValidationConfig(required_columns=required_columns)
    else:
        validation_config = MockValidationConfig()
    
    return MockDataQualityConfig(
        validation=validation_config,
        time_series=MockTimeSeriesConfig(),
        outlier_detection=MockOutlierConfig(),
        penalties=MockPenaltiesConfig()
    )