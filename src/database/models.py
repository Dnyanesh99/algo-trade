from datetime import datetime
from typing import Any, Optional, cast

from pydantic import BaseModel, field_validator


class InstrumentData(BaseModel):
    instrument_token: int
    exchange_token: Optional[str] = None
    tradingsymbol: str
    name: Optional[str] = None
    last_price: Optional[float] = None
    expiry: Optional[datetime] = None
    strike: Optional[float] = None
    tick_size: Optional[float] = None
    lot_size: Optional[int] = None
    instrument_type: str
    segment: Optional[str] = None
    exchange: str

    @field_validator("expiry", mode="before")
    @classmethod
    def validate_expiry(cls, v: Any) -> Optional[datetime]:
        if v is None or v == "" or v == "None":
            return None
        return cast("Optional[datetime]", v)


class InstrumentRecord(InstrumentData):
    instrument_id: int
    last_updated: datetime


class OHLCVData(BaseModel):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: Optional[int] = None


class FeatureData(BaseModel):
    ts: datetime
    feature_name: str
    feature_value: float


class LabelData(BaseModel):
    ts: datetime
    timeframe: str  # e.g., '15min'
    label: int  # -1, 0, 1 for SELL, NEUTRAL, BUY
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_bar_offset: Optional[int] = None
    barrier_return: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    volatility_at_entry: Optional[float] = None


class SignalData(BaseModel):
    ts: datetime
    signal_type: str  # e.g., 'Entry', 'Exit', 'Trend_Change'
    direction: str  # e.g., 'Long', 'Short', 'Neutral'
    confidence_score: Optional[float] = None
    source_feature_name: Optional[str] = None
    price_at_signal: float
    source_feature_value: Optional[float] = None
    details: Optional[dict[str, Any]] = None


class LabelingStatsData(BaseModel):
    symbol: str
    timeframe: str
    total_bars: int
    labeled_bars: int
    label_distribution: dict[str, int]
    avg_return_by_label: dict[str, float]
    exit_reasons: dict[str, int]
    avg_holding_period: float
    processing_time_ms: float
    data_quality_score: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    run_timestamp: datetime = datetime.now()


class ProcessingStateData(BaseModel):
    """Model for tracking processing completion state."""

    instrument_id: int
    process_type: str  # historical_fetch, aggregation, features, labeling
    completed_at: datetime
    metadata: dict[str, Any] = {}
    status: str = "completed"  # completed, failed, in_progress


class DataRangeData(BaseModel):
    """Model for tracking data ranges per instrument and timeframe."""

    instrument_id: int
    timeframe: str  # 1min, 5min, 15min, 60min
    earliest_ts: datetime
    latest_ts: datetime
    last_updated: datetime = datetime.now()
    record_count: Optional[int] = None
