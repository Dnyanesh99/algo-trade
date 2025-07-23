from typing import Any

import pandas as pd

from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger
from src.utils.market_calendar import MarketCalendar
from src.validation.factory import ValidationFactory
from src.validation.models import DataQualityReport

config_loader = ConfigLoader()


class HistoricalProcessor:
    """
    Processes raw historical data, performing validation, cleaning, and filtering.
    Can also run as an async service for continuous processing.
    """

    def __init__(self) -> None:
        config = config_loader.get_config()
        if not all([config.system, config.market_calendar, config.data_quality, config.data_quality.validation]):
            raise ValueError("System, market_calendar, and data_quality configurations are required.")

        if config.market_calendar is None:
            raise ValueError("Market calendar configuration is missing")
        self.market_calendar = MarketCalendar(config.system, config.market_calendar)
        validation_factory = ValidationFactory(config)
        self.candle_validator = validation_factory.get_validator("candle")
        self.expected_columns = config.data_quality.validation.expected_columns
        logger.info("HistoricalProcessor initialized with production configuration.")

    def filter_market_hours_and_holidays(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out data points that fall outside market hours or on holidays/weekends using a vectorized approach.
        Assumes 'ts' column is already in datetime objects.
        """
        if df.empty:
            logger.debug("Input DataFrame is empty, no filtering needed.")
            return df

        initial_rows = len(df)
        logger.debug(f"Filtering market hours and holidays for {initial_rows} rows.")

        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize("UTC")
        else:
            df["ts"] = df["ts"].dt.tz_convert("UTC")

        trading_mask = self.market_calendar.get_trading_mask(df["ts"])
        filtered_df = df[trading_mask].copy()

        rows_removed = initial_rows - len(filtered_df)
        if rows_removed > 0:
            logger.info(
                f"Removed {rows_removed} rows ({rows_removed / initial_rows:.2%}) outside of market hours or on holidays."
            )
        else:
            logger.debug("No rows were removed for market hours or holidays.")

        return filtered_df

    def process(
        self, raw_data: list[dict[str, Any]], symbol: str, instrument_type: str
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        """
        Main processing method for historical data.
        Converts raw data to DataFrame, validates, cleans, and filters.
        """
        logger.info(f"Processing {len(raw_data)} raw data records for instrument {symbol}.")
        if not raw_data:
            logger.warning(f"No raw data provided for instrument {symbol}. Returning empty DataFrame.")
            return pd.DataFrame(), DataQualityReport(0, 0, {}, 0, 0, 0, {}, 0.0, ["No raw data"])

        try:
            df = pd.DataFrame(raw_data)

            for col, dtype in self.expected_columns.items():
                if col not in df.columns:
                    logger.error(f"Missing expected column '{col}' in historical data for {symbol}.")
                    raise KeyError(f"Missing required column: {col}")
                try:
                    if col == "date" and dtype == "datetime64[ns]":
                        df[col] = pd.to_datetime(df[col], utc=True)
                    else:
                        df[col] = df[col].astype(dtype)
                except (TypeError, ValueError) as e:
                    logger.error(f"Error casting column '{col}' to {dtype} for {symbol}: {e}", exc_info=True)
                    raise TypeError(f"Invalid data type for column: {col}") from e

            df = df.rename(columns={"date": "ts"})

            is_valid, df_clean, quality_report = self.candle_validator.validate(
                df, symbol=symbol, instrument_type=instrument_type
            )

            if not is_valid:
                logger.error(f"Initial data validation failed for {symbol}. Issues: {quality_report.issues}")
                # Depending on the severity, we might still proceed with cleaned data if available.
                if df_clean.empty:
                    logger.critical(f"Data for {symbol} is unusable after validation. Aborting processing.")
                    return pd.DataFrame(), quality_report
                logger.warning(f"Proceeding with {len(df_clean)} cleaned rows for {symbol}.")

            df_filtered = self.filter_market_hours_and_holidays(df_clean)

            logger.info(f"Successfully processed historical data for {symbol}. Final valid rows: {len(df_filtered)}.")
            return df_filtered, quality_report

        except (KeyError, TypeError) as e:
            # These are configuration/data integrity errors, so we return an empty dataframe and a report.
            return pd.DataFrame(), DataQualityReport(0, 0, {}, 0, 0, 0, {}, 0.0, [str(e)])
        except Exception as e:
            logger.critical(
                f"An unexpected error occurred during historical processing for {symbol}: {e}", exc_info=True
            )
            # For unexpected errors, we should not hide the failure.
            raise
