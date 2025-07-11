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

    async def insert_instrument(self, instrument_data: InstrumentData) -> Optional[int]:
        """
        Inserts a new instrument record into the database.
        Returns the instrument_id if successful, None otherwise.
        """
        query = queries.instrument_repo["insert_instrument"]
        try:
            async with self.db_manager.get_connection() as conn:
                instrument_id = await conn.fetchval(
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
                logger.info(f"Inserted instrument {instrument_data.tradingsymbol} with ID: {instrument_id}")
                return int(instrument_id) if instrument_id is not None else None
        except Exception as e:
            logger.error(f"Error inserting instrument {instrument_data.tradingsymbol}: {e}")
            return None

    async def update_instrument(self, instrument_id: int, instrument_data: InstrumentData) -> bool:
        """
        Updates an existing instrument record in the database.
        Returns True if successful, False otherwise.
        """
        query = queries.instrument_repo["update_instrument"]
        try:
            async with self.db_manager.get_connection() as conn:
                await conn.execute(
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
                logger.info(f"Updated instrument ID: {instrument_id}")
                return True
        except Exception as e:
            logger.error(f"Error updating instrument ID {instrument_id}: {e}")
            return False

    async def get_instrument_by_tradingsymbol(self, tradingsymbol: str, exchange: str) -> Optional[InstrumentRecord]:
        """
        Retrieves an instrument by its trading symbol and exchange.
        """
        query = queries.instrument_repo["get_instrument_by_tradingsymbol"]
        try:
            async with self.db_manager.get_connection() as conn:
                record = await conn.fetchrow(query, tradingsymbol, exchange)
                return InstrumentRecord.model_validate(dict(record)) if record else None
        except Exception as e:
            logger.error(f"Error retrieving instrument by tradingsymbol {tradingsymbol}: {e}")
            return None

    async def get_instrument_by_token(self, instrument_token: int) -> Optional[InstrumentRecord]:
        """
        Retrieves an instrument by its instrument token.
        """
        query = queries.instrument_repo["get_instrument_by_token"]
        try:
            async with self.db_manager.get_connection() as conn:
                record = await conn.fetchrow(query, instrument_token)
                return InstrumentRecord.model_validate(dict(record)) if record else None
        except Exception as e:
            logger.error(f"Error retrieving instrument by token {instrument_token}: {e}")
            return None

    async def get_all_instruments(self) -> list[InstrumentRecord]:
        """
        Retrieves all instruments stored in the database.
        """
        query = queries.instrument_repo["get_all_instruments"]
        try:
            async with self.db_manager.get_connection() as conn:
                records = await conn.fetch(query)
                return [InstrumentRecord.model_validate(dict(record)) for record in records]
        except Exception as e:
            logger.error(f"Error fetching all instruments: {e}")
            return []

    async def get_instruments_by_type(self, instrument_type: str, exchange: str) -> list[InstrumentData]:
        """
        Retrieves instruments by type and exchange.
        """
        query = queries.instrument_repo["get_instruments_by_type"]
        try:
            async with self.db_manager.get_connection() as conn:
                records = await conn.fetch(query, instrument_type, exchange)
                return [InstrumentRecord.model_validate(dict(record)) for record in records]
        except Exception as e:
            logger.error(f"Error fetching instruments by type {instrument_type} on {exchange}: {e}")
            return []
