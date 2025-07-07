from datetime import datetime
from typing import Any, Optional

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
    def validate_expiry(cls, v):
        if v is None or v == "" or v == "None":
            return None
        return v


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
