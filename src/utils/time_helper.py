from datetime import datetime, timedelta
from typing import Optional

import pytz

# Default timezone for the application
APP_TIMEZONE = pytz.timezone("Asia/Kolkata")


def get_candle_start_time(
    timestamp: datetime, timeframe_minutes: int, timezone: Optional[pytz.BaseTzInfo] = None
) -> datetime:
    """
    Calculates the start time of the candle for a given timestamp and timeframe.

    Args:
        timestamp: The timestamp to calculate candle start for
        timeframe_minutes: The timeframe in minutes (e.g., 15 for 15-minute candles)
        timezone: Optional timezone to use for localization
    """
    if timezone and timestamp.tzinfo is None:
        timestamp = timezone.localize(timestamp)
    elif timezone and timestamp.tzinfo != timezone:
        timestamp = timestamp.astimezone(timezone)

    timestamp_in_minutes = timestamp.hour * 60 + timestamp.minute
    floored_minute = (timestamp_in_minutes // timeframe_minutes) * timeframe_minutes
    return timestamp.replace(hour=floored_minute // 60, minute=floored_minute % 60, second=0, microsecond=0)


def get_next_candle_boundary(
    current_time: datetime, candle_interval_minutes: int, timezone: Optional[pytz.BaseTzInfo] = None
) -> datetime:
    """
    Calculate the next candle boundary timestamp for a given time and interval.

    Args:
        current_time: Current time (timezone-aware or naive)
        candle_interval_minutes: Candle interval in minutes (e.g., 15 for 15-minute candles)
        timezone: Optional timezone to use for localization

    Returns:
        Next candle boundary datetime
    """
    # Ensure we're working with timezone-aware datetime if timezone is provided
    if timezone:
        if current_time.tzinfo is None:
            current_time = timezone.localize(current_time)
        elif current_time.tzinfo != timezone:
            current_time = current_time.astimezone(timezone)

    # Calculate next boundary
    current_minute = current_time.minute
    minutes_to_add = candle_interval_minutes - (current_minute % candle_interval_minutes)

    # If current_minute is already a multiple of candle_interval_minutes
    # and we are exactly at 0 seconds, the next boundary is current time + interval
    if minutes_to_add == candle_interval_minutes:
        minutes_to_add = 0

    next_boundary = current_time.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)

    # If the calculated next_boundary is in the past or exactly now (and we want the *next* one)
    # and we are not exactly at the start of the current boundary
    if next_boundary <= current_time and (current_time.second > 0 or current_time.microsecond > 0):
        next_boundary += timedelta(minutes=candle_interval_minutes)

    return next_boundary
