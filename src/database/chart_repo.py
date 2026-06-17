from datetime import datetime
from typing import Any, Optional

from psycopg2.extensions import AsIs

from src.database.db_utils import db_manager
from src.database.models import InstrumentRecord, OHLCVData
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

queries = ConfigLoader().get_queries()


class ChartRepository:
    """
    Repository for managing chart data operations.
    Handles OHLCV data retrieval for different timeframes.
    """

    def __init__(self) -> None:
        self.db_manager = db_manager
        self.valid_timeframes = ["1min", "5min", "15min", "60min"]
        logger.info(f"ChartRepository initialized with valid timeframes: {self.valid_timeframes}")

    def _get_table_name(self, timeframe: str) -> str:
        """Get the appropriate table name for the given timeframe."""
        if timeframe not in self.valid_timeframes:
            logger.error(f"Invalid timeframe '{timeframe}' requested. Must be one of {self.valid_timeframes}")
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {self.valid_timeframes}")
        table_name = f"ohlcv_{timeframe}"
        logger.debug(f"Resolved timeframe '{timeframe}' to table name '{table_name}'.")
        return table_name

    async def get_chart_data(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> list[OHLCVData]:
        """
        Retrieve OHLCV chart data for a specific instrument and timeframe.

        Args:
            instrument_id: The instrument ID
            timeframe: The timeframe (1min, 5min, 15min, 60min)
            start_time: Start timestamp for data retrieval
            end_time: End timestamp for data retrieval

        Returns:
            A list of OHLCVData objects, or an empty list if an error occurs.
        """
        try:
            table_name = self._get_table_name(timeframe)
            query = queries.chart_repo["get_chart_data"].format(table_name=table_name)
            logger.debug(f"Executing get_chart_data for instrument {instrument_id} on table '{table_name}'.")

            records = await self.db_manager.fetch_rows(query, instrument_id, start_time, end_time)
            logger.info(f"Fetched {len(records)} chart data records for instrument {instrument_id} ({timeframe}).")
            return [OHLCVData.model_validate(dict(record)) for record in records]
        except ValueError as e:
            # This catches invalid timeframe errors from _get_table_name
            logger.error(f"Cannot get chart data: {e}")
            return []
        except Exception as e:
            logger.error(
                f"Database error fetching chart data for instrument {instrument_id} ({timeframe}): {e}", exc_info=True
            )
            # In a production system, you might want to distinguish between recoverable errors
            # and critical failures. For now, we return an empty list to prevent crashes.
            return []

    async def get_chart_data_latest(self, instrument_id: int, timeframe: str, limit: int = 1000) -> list[OHLCVData]:
        """
        Retrieve the latest OHLCV chart data for a specific instrument and timeframe.

        Args:
            instrument_id: The instrument ID
            timeframe: The timeframe (1min, 5min, 15min, 60min)
            limit: Number of latest records to retrieve

        Returns:
            A list of OHLCVData objects, or an empty list on error.
        """
        if limit <= 0:
            logger.warning(f"Requested latest chart data with non-positive limit: {limit}. Returning empty list.")
            return []
        try:
            table_name = self._get_table_name(timeframe)
            query = queries.chart_repo["get_chart_data_latest"].format(table_name=table_name)
            logger.debug(
                f"Executing get_chart_data_latest for instrument {instrument_id} on table '{table_name}' with limit {limit}."
            )

            records = await self.db_manager.fetch_rows(query, instrument_id, limit)
            logger.info(
                f"Fetched {len(records)} latest chart data records for instrument {instrument_id} ({timeframe})."
            )
            return [OHLCVData.model_validate(dict(record)) for record in records]
        except ValueError as e:
            logger.error(f"Cannot get latest chart data: {e}")
            return []
        except Exception as e:
            logger.error(
                f"Database error fetching latest chart data for instrument {instrument_id} ({timeframe}): {e}",
                exc_info=True,
            )
            return []

    async def get_chart_data_with_indicators(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """
        Retrieve OHLCV chart data joined with corresponding technical indicators.

        Args:
            instrument_id: The instrument ID
            timeframe: The timeframe (1min, 5min, 15min, 60min)
            start_time: Start timestamp for data retrieval
            end_time: End timestamp for data retrieval

        Returns:
            A list of dictionaries containing OHLCV data and a nested 'indicators' dictionary.
        """
        try:
            table_name = self._get_table_name(timeframe)
            query = queries.chart_repo["get_chart_data_with_indicators"].format(table_name=table_name)
            logger.debug(
                f"Executing get_chart_data_with_indicators for instrument {instrument_id} on table '{table_name}'."
            )

            records = await self.db_manager.fetch_rows(query, instrument_id, start_time, end_time, timeframe)
            logger.info(
                f"Fetched {len(records)} raw records for chart data with indicators for instrument {instrument_id}."
            )

            chart_data: dict[datetime, dict[str, Any]] = {}
            for record in records:
                ts = record["ts"]
                if ts not in chart_data:
                    chart_data[ts] = {
                        "ts": ts,
                        "open": float(record["open"]),
                        "high": float(record["high"]),
                        "low": float(record["low"]),
                        "close": float(record["close"]),
                        "volume": int(record["volume"]),
                        "oi": int(record["oi"]) if record["oi"] is not None else None,
                        "indicators": {},
                    }

                if record.get("feature_name") and record.get("feature_value") is not None:
                    chart_data[ts]["indicators"][record["feature_name"]] = float(record["feature_value"])

            final_data = sorted(chart_data.values(), key=lambda x: x["ts"])
            logger.info(f"Processed into {len(final_data)} unique timestamp data points with indicators.")
            return final_data
        except ValueError as e:
            logger.error(f"Cannot get chart data with indicators: {e}")
            return []
        except Exception as e:
            logger.error(
                f"Database error fetching chart data with indicators for instrument {instrument_id} ({timeframe}): {e}",
                exc_info=True,
            )
            return []

    async def get_realtime_chart_data(
        self, instrument_id: int, timeframe: str, since_time: datetime, limit: int = 100
    ) -> list[OHLCVData]:
        """
        Retrieve real-time chart data since a specific timestamp.

        Args:
            instrument_id: The instrument ID
            timeframe: The timeframe (1min, 5min, 15min, 60min)
            since_time: Timestamp to get data since
            limit: Maximum number of records to retrieve

        Returns:
            A list of OHLCVData objects, or an empty list on error.
        """
        try:
            table_name = self._get_table_name(timeframe)
            query = queries.chart_repo["get_realtime_chart_data"].format(table_name=table_name)
            logger.debug(
                f"Executing get_realtime_chart_data for instrument {instrument_id} on table '{table_name}' since {since_time}."
            )

            records = await self.db_manager.fetch_rows(query, instrument_id, since_time, limit)
            logger.info(
                f"Fetched {len(records)} real-time chart data records for instrument {instrument_id} ({timeframe})."
            )
            return [OHLCVData.model_validate(dict(record)) for record in records]
        except ValueError as e:
            logger.error(f"Cannot get real-time chart data: {e}")
            return []
        except Exception as e:
            logger.error(
                f"Database error fetching real-time chart data for instrument {instrument_id} ({timeframe}): {e}",
                exc_info=True,
            )
            return []

    async def get_instruments_for_chart(self, search_term: str, limit: int = 50) -> list[InstrumentRecord]:
        """
        Search for instruments that can be used for charting.

        Args:
            search_term: Search term for instrument symbol or name.
            limit: Maximum number of results to return.

        Returns:
            A list of InstrumentRecord objects, or an empty list on error.
        """
        if not search_term or not isinstance(search_term, str):
            logger.warning("Instrument search called with an empty or invalid search term.")
            return []
        if limit <= 0:
            logger.warning(f"Instrument search called with non-positive limit: {limit}. Returning empty list.")
            return []

        query = queries.chart_repo["get_instruments_for_chart"]
        search_pattern = f"%{search_term.upper()}%"  # Use uppercase for case-insensitive matching
        logger.debug(f"Executing instrument search with pattern: '{search_pattern}' and limit: {limit}.")

        try:
            records = await self.db_manager.fetch_rows(query, search_pattern, limit)
            logger.info(f"Found {len(records)} instruments matching search term '{search_term}'.")
            # Note: This returns a simplified InstrumentRecord for performance.
            return [InstrumentRecord.model_validate(dict(record)) for record in records]
        except Exception as e:
            logger.error(f"Database error searching for instruments with term '{search_term}': {e}", exc_info=True)
            return []

    async def get_chart_data_summary(
        self, instrument_id: int, timeframe: str, start_time: datetime, end_time: datetime
    ) -> Optional[dict[str, Any]]:
        """
        Get summary statistics for chart data.

        Args:
            instrument_id: The instrument ID
            timeframe: The timeframe (1min, 5min, 15min, 60min)
            start_time: Start timestamp for data retrieval
            end_time: End timestamp for data retrieval

        Returns:
            A dictionary containing summary statistics, or None on error.
        """
        try:
            table_name = self._get_table_name(timeframe)
            query = queries.chart_repo["get_chart_data_summary"].format(table_name=table_name)
            logger.debug(f"Executing get_chart_data_summary for instrument {instrument_id} on table '{table_name}'.")

            record = await self.db_manager.fetch_row(query, instrument_id, start_time, end_time)
            if record and record["total_candles"] > 0:
                summary = {
                    "total_candles": int(record["total_candles"]),
                    "start_time": record["start_time"],
                    "end_time": record["end_time"],
                    "min_price": float(record["min_price"]) if record["min_price"] is not None else None,
                    "max_price": float(record["max_price"]) if record["max_price"] is not None else None,
                    "total_volume": int(record["total_volume"]) if record["total_volume"] is not None else None,
                }
                logger.info(f"Successfully generated chart data summary for instrument {instrument_id} ({timeframe}).")
                return summary
            logger.warning(
                f"No chart data found to summarize for instrument {instrument_id} ({timeframe}) in the given range."
            )
            return None
        except ValueError as e:
            logger.error(f"Cannot get chart data summary: {e}")
            return None
        except Exception as e:
            logger.error(
                f"Database error fetching chart data summary for instrument {instrument_id} ({timeframe}): {e}",
                exc_info=True,
            )
            return None

    async def get_available_timeframes(self, instrument_id: int) -> list[str]:
        """
        Get available timeframes for a specific instrument by checking for data presence.

        Args:
            instrument_id: The instrument ID

        Returns:
            A list of available timeframes (e.g., ['1min', '5min']).
        """
        logger.debug(f"Checking available timeframes for instrument {instrument_id}.")
        available_timeframes = []

        for timeframe in self.valid_timeframes:
            try:
                table_name = self._get_table_name(timeframe)
                # Use a more efficient query to check for existence.
                query = queries.chart_repo["check_table_exists"]
                exists = await self.db_manager.fetchval(query, table_name)
                if exists:
                    # Now that we know the table exists, we can safely query it.
                    data_exists_query = "SELECT 1 FROM %s WHERE instrument_id = %s LIMIT 1"
                    if await self.db_manager.fetchval(data_exists_query, (AsIs(table_name), instrument_id)):
                        available_timeframes.append(timeframe)
            except Exception as e:
                logger.error(
                    f"Error checking timeframe '{timeframe}' for instrument {instrument_id}: {e}", exc_info=True
                )
                # Do not add the timeframe if there was an error checking it.

        logger.info(f"Found available timeframes for instrument {instrument_id}: {available_timeframes}")
        return available_timeframes
