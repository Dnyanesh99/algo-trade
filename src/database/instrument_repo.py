from typing import Optional

from src.database.db_utils import db_manager
from src.database.models import InstrumentData, InstrumentRecord
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

queries = ConfigLoader().get_queries()


class InstrumentRepository:
    """
    Repository for managing instrument data in the TimescaleDB `instruments` table.
    """

    def __init__(self) -> None:
        self.db_manager = db_manager
        logger.info("InstrumentRepository initialized.")

    async def insert_instrument(self, instrument_data: InstrumentData) -> int:
        """
        Inserts a new instrument record into the database.

        Returns:
            The instrument_id of the newly inserted record.

        Raises:
            ValueError: If the instrument_data is invalid.
            RuntimeError: If the database insertion fails.
        """
        if not isinstance(instrument_data, InstrumentData):
            raise ValueError("Invalid InstrumentData object provided.")

        query = queries.instrument_repo["insert_instrument"]
        logger.debug(f"Inserting new instrument: {instrument_data.tradingsymbol}")
        try:
            instrument_id = await self.db_manager.fetchval(
                query,
                instrument_data.instrument_token,
                instrument_data.exchange_token,
                instrument_data.tradingsymbol,
                instrument_data.name,
                instrument_data.last_price,
                instrument_data.expiry,
                instrument_data.strike,
                instrument_data.tick_size,
                instrument_data.lot_size,
                instrument_data.instrument_type,
                instrument_data.segment,
                instrument_data.exchange,
            )
            if instrument_id is None:
                raise RuntimeError("Database did not return an instrument_id after insertion.")

            logger.info(f"Successfully inserted instrument {instrument_data.tradingsymbol} with ID: {instrument_id}")
            return int(instrument_id)
        except Exception as e:
            logger.error(f"Database error inserting instrument {instrument_data.tradingsymbol}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to insert instrument {instrument_data.tradingsymbol}") from e

    async def update_instrument(self, instrument_id: int, instrument_data: InstrumentData) -> None:
        """
        Updates an existing instrument record in the database.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If the database update fails.
        """
        if not isinstance(instrument_id, int) or not isinstance(instrument_data, InstrumentData):
            raise ValueError("Invalid instrument_id or InstrumentData object provided.")

        query = queries.instrument_repo["update_instrument"]
        logger.debug(f"Updating instrument ID: {instrument_id} with data for {instrument_data.tradingsymbol}")
        try:
            status = await self.db_manager.execute(
                query,
                instrument_data.instrument_token,
                instrument_data.exchange_token,
                instrument_data.tradingsymbol,
                instrument_data.name,
                instrument_data.last_price,
                instrument_data.expiry,
                instrument_data.strike,
                instrument_data.tick_size,
                instrument_data.lot_size,
                instrument_data.instrument_type,
                instrument_data.segment,
                instrument_data.exchange,
                instrument_id,
            )
            # The status from execute is a string like 'UPDATE 1'. We check if a row was affected.
            if int(status.split()[-1]) == 0:
                logger.warning(f"Update command for instrument ID {instrument_id} did not affect any rows.")
            else:
                logger.info(f"Successfully updated instrument ID: {instrument_id}")
        except Exception as e:
            logger.error(f"Database error updating instrument ID {instrument_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to update instrument ID {instrument_id}") from e

    async def get_instrument_by_tradingsymbol(self, tradingsymbol: str, exchange: str) -> Optional[InstrumentRecord]:
        """
        Retrieves an instrument by its trading symbol and exchange.
        """
        query = queries.instrument_repo["get_instrument_by_tradingsymbol"]
        logger.debug(f"Fetching instrument by tradingsymbol: {tradingsymbol} on exchange: {exchange}")
        try:
            record = await self.db_manager.fetch_row(query, tradingsymbol, exchange)
            if record:
                logger.debug(f"Found instrument {tradingsymbol} with ID: {record['instrument_id']}")
                return InstrumentRecord.model_validate(dict(record))
            logger.debug(f"Instrument {tradingsymbol} not found on exchange {exchange}.")
            return None
        except Exception as e:
            logger.error(f"Database error retrieving instrument by tradingsymbol {tradingsymbol}: {e}", exc_info=True)
            raise RuntimeError(f"Could not retrieve instrument {tradingsymbol}") from e

    async def get_instrument_by_token(self, instrument_token: int) -> Optional[InstrumentRecord]:
        """
        Retrieves an instrument by its instrument token.
        """
        query = queries.instrument_repo["get_instrument_by_token"]
        logger.debug(f"Fetching instrument by token: {instrument_token}")
        try:
            record = await self.db_manager.fetch_row(query, instrument_token)
            if record:
                logger.debug(f"Found instrument with token {instrument_token}: {record['tradingsymbol']}")
                return InstrumentRecord.model_validate(dict(record))
            logger.debug(f"Instrument with token {instrument_token} not found.")
            return None
        except Exception as e:
            logger.error(f"Database error retrieving instrument by token {instrument_token}: {e}", exc_info=True)
            raise RuntimeError(f"Could not retrieve instrument with token {instrument_token}") from e

    async def get_instrument_by_id(self, instrument_id: int) -> Optional[InstrumentRecord]:
        """
        Retrieves an instrument by its primary key instrument_id.
        """
        query = queries.instrument_repo["get_instrument_by_id"]
        logger.debug(f"Fetching instrument by ID: {instrument_id}")
        try:
            record = await self.db_manager.fetch_row(query, instrument_id)
            if record:
                logger.debug(f"Found instrument with ID {instrument_id}: {record['tradingsymbol']}")
                return InstrumentRecord.model_validate(dict(record))
            logger.debug(f"Instrument with ID {instrument_id} not found.")
            return None
        except Exception as e:
            logger.error(f"Database error retrieving instrument by ID {instrument_id}: {e}", exc_info=True)
            raise RuntimeError(f"Could not retrieve instrument with ID {instrument_id}") from e

    async def get_all_instruments(self) -> list[InstrumentRecord]:
        """
        Retrieves all instruments stored in the database.
        """
        query = queries.instrument_repo["get_all_instruments"]
        logger.debug("Fetching all instruments from the database.")
        try:
            records = await self.db_manager.fetch_rows(query)
            logger.info(f"Fetched {len(records)} instruments from the database.")
            return [InstrumentRecord.model_validate(dict(record)) for record in records]
        except Exception as e:
            logger.error(f"Database error fetching all instruments: {e}", exc_info=True)
            raise RuntimeError("Could not retrieve all instruments.") from e

    async def get_instruments_by_type(self, instrument_type: str, exchange: str) -> list[InstrumentRecord]:
        """
        Retrieves instruments by type and exchange.
        """
        query = queries.instrument_repo["get_instruments_by_type"]
        logger.debug(f"Fetching instruments of type '{instrument_type}' on exchange '{exchange}'.")
        try:
            records = await self.db_manager.fetch_rows(query, instrument_type, exchange)
            logger.info(f"Fetched {len(records)} instruments of type '{instrument_type}' on exchange '{exchange}'.")
            return [InstrumentRecord.model_validate(dict(record)) for record in records]
        except Exception as e:
            logger.error(
                f"Database error fetching instruments by type {instrument_type} on {exchange}: {e}", exc_info=True
            )
            raise RuntimeError(f"Could not retrieve instruments by type {instrument_type} on {exchange}") from e
