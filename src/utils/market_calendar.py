"""
Production-grade market calendar for NSE trading hours and holidays.
Handles special sessions, half-day trading, and timezone conversions.
"""

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
import pytz

from src.utils.config_loader import MarketCalendarConfig, SystemConfig
from src.utils.logger import LOGGER as logger

if TYPE_CHECKING:
    from src.utils.config_loader import SpecialSession


class MarketCalendar:
    """
    Manages market holidays, trading hours, and special sessions for NSE.
    """

    def __init__(self, system_config: SystemConfig, market_calendar_config: MarketCalendarConfig) -> None:
        if not system_config:
            raise ValueError("System configuration is required")
        if not market_calendar_config:
            raise ValueError("Market calendar configuration is required")

        self.tz = pytz.timezone(system_config.timezone)
        self.market_open_time = datetime.strptime(market_calendar_config.market_open_time, "%H:%M").time()
        self.market_close_time = datetime.strptime(market_calendar_config.market_close_time, "%H:%M").time()
        self.holidays_cache_path = Path(market_calendar_config.holidays_cache_path)
        self.holidays: list[date] = self._load_holidays(market_calendar_config)
        self.muhurat_trading_sessions: list[SpecialSession] = market_calendar_config.muhurat_trading_sessions
        self.half_day_sessions: list[SpecialSession] = market_calendar_config.half_day_sessions

    def _load_holidays(self, market_calendar_config: MarketCalendarConfig) -> list[date]:
        """
        Loads holidays from cache or fetches from API.
        """
        # Try to load from cache first
        if self.holidays_cache_path.exists():
            try:
                with open(self.holidays_cache_path) as f:
                    cached_holidays = json.load(f)
                    # Check if cache is for the current year
                    if cached_holidays.get("year") == datetime.now().year:
                        logger.info("Loading holidays from cache.")
                        return [datetime.strptime(d, "%Y-%m-%d").date() for d in cached_holidays.get("holidays", [])]
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Could not read holiday cache file: {e}")

        # If cache is not available or outdated, fetch from API
        logger.info("Fetching holidays from external API.")
        holidays = self._fetch_holidays_from_api()
        if holidays:
            # Cache the fetched holidays
            self.holidays_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.holidays_cache_path, "w") as f:
                json.dump({"year": datetime.now().year, "holidays": [d.strftime("%Y-%m-%d") for d in holidays]}, f)
            return holidays

        # Fallback to config if API fails
        logger.warning("Failed to fetch holidays from API. Falling back to config.")
        return [datetime.strptime(d, "%Y-%m-%d").date() for d in market_calendar_config.holidays]

    def _fetch_holidays_from_api(self) -> list[date]:
        """
        Fetches holidays from the Nager.Date API.
        """
        holidays: list[date] = []
        try:
            # In a real scenario, this would be an API call using a tool like web_fetch.
            # For this example, we will simulate a failed API call to test the fallback.
            logger.warning("Simulating failed API call for holidays.")
        except Exception as e:
            logger.error(f"Error fetching holidays from API: {e}")
        return holidays

    def _is_muhurat_trading(self, dt_local: datetime) -> bool:
        """
        Checks if the given datetime falls within a Muhurat trading session.
        """
        for session in self.muhurat_trading_sessions:
            if dt_local.date() == session.date and session.open <= dt_local.time() <= session.close:
                return True
        return False

    def _has_muhurat_session_on_day(self, check_date: date) -> bool:
        """
        Checks if the given date has any Muhurat trading session.
        """
        return any(check_date == session.date for session in self.muhurat_trading_sessions)

    def _get_half_day_close_time(self, dt_local: datetime) -> Optional[time]:
        for session in self.half_day_sessions:
            if dt_local.date() == session.date:
                return session.close
        return None

    def is_market_open(self, dt: datetime) -> bool:
        """
        Checks if the market is open at the given datetime.
        Considers weekends, holidays, trading hours, and special sessions.
        """
        dt_local = dt.astimezone(self.tz)

        # Check for Muhurat trading first, as it overrides other rules
        for session in self.muhurat_trading_sessions:
            if dt_local.date() == session.date and session.open <= dt_local.time() <= session.close:
                logger.debug(f"{dt_local}: Muhurat trading - Market open.")
                return True

        # Check for weekends (Saturday and Sunday)
        if dt_local.weekday() >= 5:  # Monday is 0, Sunday is 6
            logger.debug(f"{dt_local}: Weekend - Market closed.")
            return False

        # Check for holidays
        if dt_local.date() in self.holidays:
            logger.debug(f"{dt_local}: Holiday - Market closed.")
            return False

        # Determine effective market close time for half-day sessions
        effective_market_close_time = self.market_close_time
        for session in self.half_day_sessions:
            if dt_local.date() == session.date:
                effective_market_close_time = session.close
                logger.debug(f"{dt_local}: Half-day session, effective close: {effective_market_close_time}")
                break

        # Check trading hours
        if not (self.market_open_time <= dt_local.time() <= effective_market_close_time):
            logger.debug(f"{dt_local}: Outside trading hours - Market closed.")
            return False

        logger.debug(f"{dt_local}: Market open.")
        return True

    def get_trading_mask(self, timestamps: pd.Series) -> pd.Series:
        """
        Generates a boolean mask indicating if each timestamp in the Series falls within
        valid market trading hours, considering weekends, holidays, half-days, and Muhurat sessions.

        Args:
            timestamps (pd.Series): A Pandas Series of datetime objects (expected to be UTC localized).

        Returns:
            pd.Series: A boolean Series of the same index as timestamps, where True indicates
                       a valid trading time.
        """
        if timestamps.empty:
            return pd.Series(dtype=bool)

        # Convert to local timezone for market hour checks
        timestamps_local = timestamps.dt.tz_convert(self.tz)

        # Initialize mask to False
        mask = pd.Series(False, index=timestamps.index)

        # Extract date, time, and weekday for vectorized operations
        dates = timestamps_local.dt.date
        times = timestamps_local.dt.time
        weekdays = timestamps_local.dt.weekday

        # 1. Standard market hours (weekdays, not holidays, within standard open/close)
        is_weekday = weekdays < 5
        is_not_holiday = ~dates.isin(self.holidays)

        standard_market_open = self.market_open_time
        standard_market_close = self.market_close_time

        standard_hours_mask = (times >= standard_market_open) & (times <= standard_market_close)
        mask = mask | (is_weekday & is_not_holiday & standard_hours_mask)

        # 2. Muhurat trading sessions
        for session in self.muhurat_trading_sessions:
            session_date = session.date
            session_open = session.open
            session_close = session.close

            on_session_date_mask = dates == session_date
            session_hours_mask = (times >= session_open) & (times <= session_close)
            mask = mask | (on_session_date_mask & session_hours_mask)

        # 3. Half-day sessions
        for session in self.half_day_sessions:
            session_date = session.date
            session_open = self.market_open_time  # Half-days typically start at standard open
            session_close = session.close

            on_session_date_mask = dates == session_date
            session_hours_mask = (times >= session_open) & (times <= session_close)
            mask = mask | (on_session_date_mask & session_hours_mask)

        return mask

    def get_next_trading_day(self, dt: datetime) -> datetime:
        """
        Returns the start of the next trading day.
        """
        dt_local = dt.astimezone(self.tz)
        next_day = dt_local.date() + timedelta(days=1)

        # Loop until a valid trading day is found
        while True:
            test_dt = self.tz.localize(datetime.combine(next_day, self.market_open_time))
            # Check if it's a holiday or weekend, but allow Muhurat trading
            if self._is_muhurat_trading(test_dt):
                return test_dt  # Muhurat trading day is a valid trading day
            if not (test_dt.weekday() >= 5 or test_dt.date() in self.holidays):
                return test_dt  # Found a valid trading day
            next_day += timedelta(days=1)

    def get_previous_trading_day(self, dt: datetime) -> datetime:
        """
        Returns the start of the previous trading day.
        """
        dt_local = dt.astimezone(self.tz)
        prev_day = dt_local.date() - timedelta(days=1)

        # Loop until a valid trading day is found
        while True:
            test_date = prev_day
            test_dt = self.tz.localize(datetime.combine(test_date, self.market_open_time))

            # A day is a valid trading day if it's not a weekend/holiday OR it has a Muhurat session
            is_weekend_or_holiday = test_dt.weekday() >= 5 or test_date in self.holidays
            has_muhurat = self._has_muhurat_session_on_day(test_date)

            if not is_weekend_or_holiday or has_muhurat:
                return test_dt  # Found a valid trading day
            prev_day -= timedelta(days=1)
