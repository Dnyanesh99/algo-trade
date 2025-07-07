"""
Production-grade market calendar for NSE trading hours and holidays.
Handles special sessions, half-day trading, and timezone conversions.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

import pandas as pd
import pytz

from src.utils.config_loader import SpecialSession, config_loader
from src.utils.logger import LOGGER as logger

# Load configuration
config = config_loader.get_config()


class MarketCalendar:
    """
    Manages market holidays, trading hours, and special sessions for NSE.
    """

    def __init__(self) -> None:
        if config.system:
            self.tz = pytz.timezone(config.system.timezone)
        else:
            logger.warning("System configuration not found. Using default timezone 'Asia/Kolkata'.")
            self.tz = pytz.timezone("Asia/Kolkata")

        if config.market_calendar:
            self.market_open_time = datetime.strptime(config.market_calendar.market_open_time, "%H:%M").time()
            self.market_close_time = datetime.strptime(config.market_calendar.market_close_time, "%H:%M").time()
            self.holidays: list[date] = self._load_holidays()
            self.muhurat_trading_sessions: list[SpecialSession] = config.market_calendar.muhurat_trading_sessions
            self.half_day_sessions: list[SpecialSession] = config.market_calendar.half_day_sessions
        else:
            logger.warning("Market calendar configuration not found. Using default market hours and no holidays.")
            self.market_open_time = time(9, 15)
            self.market_close_time = time(15, 30)
            self.holidays = []
            self.muhurat_trading_sessions = []
            self.half_day_sessions = []

    def _load_holidays(self) -> list[date]:
        if config.market_calendar:
            return [datetime.strptime(d, "%Y-%m-%d").date() for d in config.market_calendar.holidays]
        return []

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
