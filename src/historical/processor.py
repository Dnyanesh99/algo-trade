from typing import Any

import pandas as pd

from src.utils.data_quality import DataQualityReport, DataValidator
from src.utils.logger import LOGGER as logger
from src.utils.market_calendar import MarketCalendar


class HistoricalProcessor:
    """
    Processes raw historical data, performing validation, cleaning, and filtering.
    """

    def __init__(self) -> None:
        self.market_calendar = MarketCalendar()
        self.data_validator = DataValidator()

    def filter_market_hours_and_holidays(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out data points that fall outside market hours or on holidays/weekends using a vectorized approach.
        Assumes 'timestamp' column is already in datetime objects.
        """
        initial_rows = len(df)
        if initial_rows == 0:
            return df

        # Ensure timestamp is in the correct timezone (UTC for consistency with MarketCalendar.get_trading_mask)
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        # Get the trading mask from MarketCalendar
        trading_mask = self.market_calendar.get_trading_mask(df["timestamp"])

        filtered_df = df[trading_mask].copy()

        rows_removed = initial_rows - len(filtered_df)
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} rows outside market hours or on holidays/weekends.")
        return filtered_df

    def process(self, raw_data: list[dict[str, Any]], symbol: str, instrument_type: str = "UNKNOWN") -> tuple[pd.DataFrame, DataQualityReport]:
        """
        Main processing method for historical data.
        Converts raw data to DataFrame, validates, cleans, and filters.

        Args:
            raw_data (List[Dict[str, Any]]): List of raw historical data dictionaries.
            symbol (str): The instrument symbol for logging and reporting.
            instrument_type (str): The instrument type/segment (e.g., 'INDICES', 'EQ').

        Returns:
            Tuple[pd.DataFrame, DataQualityReport]: A tuple containing the processed DataFrame and a data quality report.
        """
        if not raw_data:
            logger.warning(f"No raw data provided for instrument {symbol}. Returning empty DataFrame.")
            return pd.DataFrame(), DataQualityReport(0, 0, {}, 0, 0, 0, {}, 0.0, ["No raw data"])

        df = pd.DataFrame(raw_data)

        # Rename 'date' to 'timestamp' for consistency
        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        is_valid, df_clean, quality_report = self.data_validator.validate_ohlcv(df, symbol, instrument_type)

        if not is_valid:
            logger.error(f"Initial data validation failed for {symbol}. Skipping further processing.")
            return pd.DataFrame(), quality_report

        df_filtered = self.filter_market_hours_and_holidays(df_clean)

        logger.info(f"Successfully processed historical data for {symbol}. Valid rows: {len(df_filtered)}")
        return df_filtered, quality_report
