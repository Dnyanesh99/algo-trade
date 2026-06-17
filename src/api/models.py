"""
Pydantic models for API requests and responses
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, validator


# System Models
class SystemStatus(BaseModel):
    mode: str = Field(..., description="System operation mode")
    version: str = Field(..., description="System version")
    timezone: str = Field(..., description="System timezone")
    uptime: str = Field(..., description="System uptime")
    status: str = Field(..., description="System status")
    health_score: float = Field(..., ge=0.0, le=1.0, description="Overall health score")
    components: dict[str, Any] = Field(..., description="Component status")
    memory_usage: dict[str, float] = Field(..., description="Memory usage statistics")
    last_updated: datetime = Field(..., description="Last update timestamp")


class SystemModeRequest(BaseModel):
    mode: str = Field(..., description="Target system mode")

    @validator("mode")
    def validate_mode(cls, v: str) -> str:
        if v not in ["HISTORICAL_MODE", "LIVE_MODE"]:
            raise ValueError("Mode must be either HISTORICAL_MODE or LIVE_MODE")
        return v


class SystemControlRequest(BaseModel):
    action: str = Field(..., description="Control action")
    parameters: Optional[dict[str, Any]] = Field(None, description="Additional parameters")


# Configuration Models
class ConfigSection(BaseModel):
    section: str = Field(..., description="Configuration section name")
    data: dict[str, Any] = Field(..., description="Configuration data")
    last_modified: datetime = Field(..., description="Last modification timestamp")


class ConfigUpdateRequest(BaseModel):
    data: dict[str, Any] = Field(..., description="Configuration data to update")
    validate_only: bool = Field(False, description="Only validate, don't save")


# Instrument Models
class InstrumentData(BaseModel):
    id: Optional[int] = Field(None, description="Instrument database ID")
    label: str = Field(..., description="Instrument display label")
    tradingsymbol: str = Field(..., description="Trading symbol")
    exchange: str = Field(..., description="Exchange (NSE/BSE)")
    instrument_type: str = Field(..., description="Instrument type")
    segment: str = Field(..., description="Market segment")
    enabled: bool = Field(False, description="Whether instrument is enabled for trading")
    last_price: Optional[float] = Field(None, description="Last traded price")
    volume: Optional[int] = Field(None, description="Trading volume")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")

    @validator("exchange")
    def validate_exchange(cls, v: str) -> str:
        if v not in ["NSE", "BSE"]:
            raise ValueError("Exchange must be either NSE or BSE")
        return v


class InstrumentCreateRequest(BaseModel):
    instruments: list[InstrumentData] = Field(..., description="List of instruments to create")


class InstrumentUpdateRequest(BaseModel):
    label: Optional[str] = Field(None, description="Instrument display label")
    enabled: Optional[bool] = Field(None, description="Whether instrument is enabled")


class InstrumentFilterRequest(BaseModel):
    segment: Optional[str] = Field(None, description="Filter by segment")
    exchange: Optional[str] = Field(None, description="Filter by exchange")
    enabled_only: bool = Field(False, description="Show only enabled instruments")
    search: Optional[str] = Field(None, description="Search term")


# Signal Models
class SignalData(BaseModel):
    id: Optional[int] = Field(None, description="Signal database ID")
    timestamp: datetime = Field(..., description="Signal timestamp")
    instrument_id: int = Field(..., description="Instrument ID")
    instrument_symbol: Optional[str] = Field(None, description="Instrument symbol")
    signal_type: str = Field(..., description="Signal type")
    direction: str = Field(..., description="Signal direction (BUY/SELL/HOLD)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Signal confidence")
    price_at_signal: Optional[float] = Field(None, description="Price when signal generated")
    source_feature_name: Optional[str] = Field(None, description="Source feature name")
    source_feature_value: Optional[float] = Field(None, description="Source feature value")
    details: Optional[dict[str, Any]] = Field(None, description="Additional signal details")

    @validator("direction")
    def validate_direction(cls, v: str) -> str:
        if v not in ["BUY", "SELL", "HOLD", "NEUTRAL"]:
            raise ValueError("Direction must be one of: BUY, SELL, HOLD, NEUTRAL")
        return v


class SignalFilterRequest(BaseModel):
    instrument_id: Optional[int] = Field(None, description="Filter by instrument ID")
    signal_type: Optional[str] = Field(None, description="Filter by signal type")
    direction: Optional[str] = Field(None, description="Filter by direction")
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence score")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")


# Model Management Models
class ModelMetrics(BaseModel):
    model_id: str = Field(..., description="Model identifier")
    version: str = Field(..., description="Model version")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Model accuracy")
    precision: float = Field(..., ge=0.0, le=1.0, description="Model precision")
    recall: float = Field(..., ge=0.0, le=1.0, description="Model recall")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    training_date: datetime = Field(..., description="Training completion date")
    is_active: bool = Field(..., description="Whether model is currently active")
    training_duration: Optional[int] = Field(None, description="Training duration in seconds")
    feature_count: Optional[int] = Field(None, description="Number of features used")
    training_samples: Optional[int] = Field(None, description="Number of training samples")


class ModelTrainingRequest(BaseModel):
    instrument_id: int = Field(..., description="Instrument ID for training")
    timeframe: str = Field(..., description="Timeframe for training")
    retrain: bool = Field(False, description="Whether to retrain existing model")
    hyperparameter_optimization: bool = Field(False, description="Enable hyperparameter optimization")

    @validator("timeframe")
    def validate_timeframe(cls, v: str) -> str:
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "1d"]
        if v not in valid_timeframes:
            raise ValueError(f"Timeframe must be one of: {valid_timeframes}")
        return v


class ModelDeploymentRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier to deploy")
    version: str = Field(..., description="Model version to deploy")


# Authentication Models
class AuthStatus(BaseModel):
    authenticated: bool = Field(..., description="Whether user is authenticated")
    token_expires: Optional[datetime] = Field(None, description="Token expiration time")
    last_login: Optional[datetime] = Field(None, description="Last login time")
    broker_status: Optional[str] = Field(None, description="Broker connection status")


class AuthLoginRequest(BaseModel):
    force_reauth: bool = Field(False, description="Force re-authentication")


# Health Monitoring Models
class HealthStatus(BaseModel):
    component: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status")
    health_score: float = Field(..., ge=0.0, le=1.0, description="Health score")
    last_check: datetime = Field(..., description="Last health check time")
    error_message: Optional[str] = Field(None, description="Error message if any")
    metrics: Optional[dict[str, Any]] = Field(None, description="Component-specific metrics")


class AlertData(BaseModel):
    id: Optional[int] = Field(None, description="Alert ID")
    level: str = Field(..., description="Alert level (INFO/WARNING/ERROR/CRITICAL)")
    component: str = Field(..., description="Component that generated the alert")
    message: str = Field(..., description="Alert message")
    timestamp: datetime = Field(..., description="Alert timestamp")
    acknowledged: bool = Field(False, description="Whether alert is acknowledged")
    details: Optional[dict[str, Any]] = Field(None, description="Additional alert details")

    @validator("level")
    def validate_level(cls, v: str) -> str:
        if v not in ["INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Level must be one of: INFO, WARNING, ERROR, CRITICAL")
        return v


# Data Models
class OHLCVData(BaseModel):
    timestamp: datetime = Field(..., description="Candle timestamp")
    instrument_id: int = Field(..., description="Instrument ID")
    timeframe: str = Field(..., description="Timeframe (1m, 5m, 15m, etc.)")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: int = Field(..., description="Volume")
    oi: Optional[int] = Field(None, description="Open interest")


class FeatureData(BaseModel):
    timestamp: datetime = Field(..., description="Feature timestamp")
    instrument_id: int = Field(..., description="Instrument ID")
    timeframe: str = Field(..., description="Timeframe")
    feature_name: str = Field(..., description="Feature name")
    feature_value: float = Field(..., description="Feature value")


class DataQueryRequest(BaseModel):
    instrument_id: int = Field(..., description="Instrument ID")
    timeframe: str = Field(..., description="Timeframe")
    start_time: datetime = Field(..., description="Start time")
    end_time: datetime = Field(..., description="End time")
    limit: Optional[int] = Field(None, ge=1, le=10000, description="Maximum number of records")


# WebSocket Models
class WebSocketMessage(BaseModel):
    type: str = Field(..., description="Message type")
    channel: Optional[str] = Field(None, description="Channel name")
    data: Optional[dict[str, Any]] = Field(None, description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class WebSocketSubscription(BaseModel):
    channel: str = Field(..., description="Channel to subscribe to")
    filters: Optional[dict[str, Any]] = Field(None, description="Subscription filters")


# File Upload Models
class FileUploadResponse(BaseModel):
    filename: str = Field(..., description="Uploaded filename")
    records_processed: int = Field(..., description="Number of records processed")
    records_created: int = Field(..., description="Number of records created")
    records_updated: int = Field(..., description="Number of records updated")
    errors: list[str] = Field(default_factory=list, description="Processing errors")
    warnings: list[str] = Field(default_factory=list, description="Processing warnings")


# Metrics Models
class MetricData(BaseModel):
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    timestamp: datetime = Field(..., description="Metric timestamp")
    labels: Optional[dict[str, str]] = Field(None, description="Metric labels")


class MetricsResponse(BaseModel):
    metrics: list[MetricData] = Field(..., description="List of metrics")
    last_updated: datetime = Field(..., description="Last update timestamp")


# Holiday Calendar Models
class HolidayData(BaseModel):
    date: str = Field(..., description="Holiday date (YYYY-MM-DD)")
    name: str = Field(..., description="Holiday name")
    type: str = Field(..., description="Holiday type (national/religious/market)")

    @validator("type")
    def validate_type(cls, v: str) -> str:
        if v not in ["national", "religious", "market"]:
            raise ValueError("Type must be one of: national, religious, market")
        return v


class TradingSession(BaseModel):
    date: str = Field(..., description="Session date (YYYY-MM-DD)")
    name: str = Field(..., description="Session name")
    open_time: str = Field(..., description="Opening time (HH:MM)")
    close_time: str = Field(..., description="Closing time (HH:MM)")


class CalendarUpdateRequest(BaseModel):
    holidays: Optional[list[HolidayData]] = Field(None, description="Holidays to add/update")
    special_sessions: Optional[list[TradingSession]] = Field(None, description="Special sessions to add/update")
    market_hours: Optional[dict[str, str]] = Field(None, description="Regular market hours")


# Error Models
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


# Success Models
class SuccessResponse(BaseModel):
    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Success message")
    data: Optional[dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# Batch Operation Models
class BatchOperation(BaseModel):
    operation: str = Field(..., description="Batch operation type")
    items: list[dict[str, Any]] = Field(..., description="Items to process")
    options: Optional[dict[str, Any]] = Field(None, description="Operation options")


class BatchResult(BaseModel):
    total_items: int = Field(..., description="Total items processed")
    successful: int = Field(..., description="Successfully processed items")
    failed: int = Field(..., description="Failed items")
    errors: list[str] = Field(default_factory=list, description="Processing errors")
    results: list[dict[str, Any]] = Field(default_factory=list, description="Individual results")


# Indicator Configuration Models
class IndicatorConfig(BaseModel):
    name: str = Field(..., description="Indicator name")
    function: str = Field(..., description="TA-Lib function name")
    enabled: bool = Field(True, description="Whether indicator is enabled")
    params: dict[str, dict[str, Any]] = Field(..., description="Parameters by timeframe")
    category: Optional[str] = Field(None, description="Indicator category")


class IndicatorUpdateRequest(BaseModel):
    indicators: list[IndicatorConfig] = Field(..., description="Indicators to update")


class IndicatorTestRequest(BaseModel):
    indicator_name: str = Field(..., description="Indicator to test")
    instrument_id: int = Field(..., description="Instrument ID for testing")
    timeframe: str = Field(..., description="Timeframe for testing")
    start_time: datetime = Field(..., description="Start time for test")
    end_time: datetime = Field(..., description="End time for test")
