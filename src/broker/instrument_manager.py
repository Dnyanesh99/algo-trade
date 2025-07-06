from typing import Optional

from src.broker.rest_client import KiteRESTClient
from src.database.instrument_repo import InstrumentRepository
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

config = config_loader.get_config()


class InstrumentManager:
    """
    Manages the synchronization of instrument data between the broker and the local database.
    """

    def __init__(self, rest_client: KiteRESTClient, instrument_repo: InstrumentRepository):
        self.rest_client = rest_client
        self.instrument_repo = instrument_repo
        self.configured_instruments = config.trading.instruments

    async def sync_instruments(self) -> None:
        """
        Fetches all instruments from the broker, and updates/inserts them into the database.
        It also ensures that the configured instruments have their tokens updated.
        """
        logger.info("Starting instrument synchronization...")
        broker_instruments = await self.rest_client.get_instruments()

        if not broker_instruments:
            logger.error("Failed to fetch instruments from broker. Synchronization aborted.")
            return

        logger.info(f"Fetched {len(broker_instruments)} instruments from broker.")

        for configured_inst in self.configured_instruments:
            label = configured_inst.label
            exchange = configured_inst.exchange
            instrument_type = configured_inst.instrument_type
            tradingsymbol = configured_inst.tradingsymbol

            found_match = False
            for broker_inst in broker_instruments:
                # Match based on tradingsymbol, exchange, and instrument_type
                # For INDEX, tradingsymbol is usually the name itself
                if (
                    broker_inst.get("tradingsymbol") == tradingsymbol
                    and broker_inst.get("exchange") == exchange
                    and broker_inst.get("instrument_type") == instrument_type
                ):
                    # Prepare data for DB insertion/update
                    db_data = {
                        "instrument_token": broker_inst["instrument_token"],
                        "exchange_token": broker_inst.get("exchange_token"),
                        "tradingsymbol": broker_inst["tradingsymbol"],
                        "name": broker_inst.get("name"),
                        "last_price": broker_inst.get("last_price"),
                        "expiry": broker_inst.get("expiry"),
                        "strike": broker_inst.get("strike"),
                        "tick_size": broker_inst.get("tick_size"),
                        "lot_size": broker_inst.get("lot_size"),
                        "instrument_type": broker_inst["instrument_type"],
                        "segment": broker_inst.get("segment"),
                        "exchange": broker_inst["exchange"],
                    }

                    # Check if instrument already exists in DB
                    existing_inst = await self.instrument_repo.get_instrument_by_tradingsymbol(tradingsymbol, exchange)

                    if existing_inst:
                        # Update existing instrument
                        await self.instrument_repo.update_instrument(existing_inst["instrument_id"], db_data)
                        logger.info(f"Updated instrument: {label} ({tradingsymbol})")
                    else:
                        # Insert new instrument
                        await self.instrument_repo.insert_instrument(db_data)
                        logger.info(f"Inserted new instrument: {label} ({tradingsymbol})")
                    found_match = True
                    break
            if not found_match:
                logger.warning(f"Configured instrument {label} ({tradingsymbol}) not found on broker.")
        logger.info("Instrument synchronization completed.")

    async def get_instrument_token_by_label(self, label: str) -> Optional[int]:
        """
        Retrieves the instrument token for a given configured instrument label from the database.
        """
        for configured_inst in self.configured_instruments:
            if configured_inst.label == label:
                tradingsymbol = configured_inst.tradingsymbol
                exchange = configured_inst.exchange
                instrument = await self.instrument_repo.get_instrument_by_tradingsymbol(tradingsymbol, exchange)
                if instrument:
                    return int(instrument["instrument_token"])
                logger.warning(f"Instrument token for label '{label}' ({tradingsymbol}) not found in DB.")
                return None
        logger.warning(f"Label '{label}' not found in configured instruments.")
        return None

    async def get_configured_instrument_tokens(self) -> list[int]:
        """
        Retrieves instrument tokens for all configured instruments.
        """
        tokens = []
        for configured_inst in self.configured_instruments:
            tradingsymbol = configured_inst.tradingsymbol
            exchange = configured_inst.exchange
            instrument = await self.instrument_repo.get_instrument_by_tradingsymbol(tradingsymbol, exchange)
            if instrument:
                tokens.append(instrument["instrument_token"])
            else:
                logger.warning(f"Instrument token for {tradingsymbol} not found in DB. Skipping.")
        return tokens
