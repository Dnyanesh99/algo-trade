from typing import Any, Optional

from src.broker.rest_client import KiteRESTClient
from src.database.instrument_repo import InstrumentRepository
from src.database.models import InstrumentData, InstrumentRecord
from src.utils.config_loader import TradingConfig
from src.utils.logger import LOGGER as logger


class InstrumentManager:
    """
    Manages the synchronization of instrument data between the broker and the local database.
    """

    def __init__(
        self, rest_client: KiteRESTClient, instrument_repo: InstrumentRepository, trading_config: TradingConfig
    ):
        self.rest_client = rest_client
        self.instrument_repo = instrument_repo
        self.trading_config = trading_config
        if trading_config:
            self.configured_instruments = trading_config.instruments
        else:
            logger.warning("Trading configuration not found. No instruments configured.")
            self.configured_instruments = []

    async def sync_instruments(self) -> None:
        """
        Fetches all instruments from the broker using optimized CSV approach,
        and updates/inserts configured instruments into the database.
        Uses lookup optimization for handling large instrument datasets efficiently.
        """
        logger.info("Starting optimized instrument synchronization...")
        broker_instruments = await self.rest_client.get_instruments()

        if not broker_instruments:
            logger.error("Failed to fetch instruments from broker. Synchronization aborted.")
            return

        logger.info(f"Fetched {len(broker_instruments)} instruments from broker using optimized CSV approach.")

        # Create optimized lookup index for fast instrument matching
        # Key format: "tradingsymbol|exchange|instrument_type"
        logger.info("Building instrument lookup index for fast matching...")
        instrument_lookup = {}
        for instrument_data in broker_instruments:
            tradingsymbol = instrument_data.get("tradingsymbol")
            exchange = instrument_data.get("exchange")
            instrument_type = instrument_data.get("instrument_type")

            if tradingsymbol and exchange and instrument_type:
                lookup_key = f"{tradingsymbol}|{exchange}|{instrument_type}"
                instrument_lookup[lookup_key] = instrument_data

        logger.info(f"Built lookup index with {len(instrument_lookup)} unique instrument keys.")

        # Process configured instruments using lookup index
        processed_count = 0
        matched_count = 0

        for configured_inst in self.configured_instruments:
            processed_count += 1
            label = configured_inst.label
            exchange = configured_inst.exchange
            instrument_type = configured_inst.instrument_type
            tradingsymbol = configured_inst.tradingsymbol

            # Use lookup index for O(1) instrument matching
            lookup_key = f"{tradingsymbol}|{exchange}|{instrument_type}"
            broker_inst: Optional[dict[str, Any]] = instrument_lookup.get(lookup_key)

            if broker_inst is not None:
                matched_count += 1

                # Extract and validate required fields
                instrument_token_val = broker_inst.get("instrument_token")
                tradingsymbol_val = broker_inst.get("tradingsymbol")
                instrument_type_val = broker_inst.get("instrument_type")
                exchange_val = broker_inst.get("exchange")

                if not all([instrument_token_val, tradingsymbol_val, instrument_type_val, exchange_val]):
                    logger.warning(f"Skipping instrument due to missing critical data: {broker_inst}")
                    continue

                # Prepare data for DB insertion/update
                instrument_data_obj = InstrumentData(
                    instrument_token=int(instrument_token_val) if instrument_token_val is not None else 0,
                    exchange_token=broker_inst.get("exchange_token"),
                    tradingsymbol=str(tradingsymbol_val),
                    name=broker_inst.get("name"),
                    last_price=broker_inst.get("last_price"),
                    expiry=broker_inst.get("expiry"),
                    strike=broker_inst.get("strike"),
                    tick_size=broker_inst.get("tick_size"),
                    lot_size=broker_inst.get("lot_size"),
                    instrument_type=str(instrument_type_val),
                    segment=broker_inst.get("segment"),
                    exchange=str(exchange_val),
                )

                # Check if instrument already exists in DB
                existing_inst: Optional[InstrumentRecord] = await self.instrument_repo.get_instrument_by_tradingsymbol(
                    tradingsymbol, exchange
                )

                if existing_inst:
                    # Update existing instrument
                    await self.instrument_repo.update_instrument(existing_inst.instrument_id, instrument_data_obj)
                    logger.info(f"Updated instrument: {label} ({tradingsymbol}) - Token: {instrument_token_val}")
                else:
                    # Insert new instrument
                    await self.instrument_repo.insert_instrument(instrument_data_obj)
                    logger.info(f"Inserted new instrument: {label} ({tradingsymbol}) - Token: {instrument_token_val}")
            else:
                logger.warning(
                    f"Configured instrument {label} ({tradingsymbol}|{exchange}|{instrument_type}) not found in broker data."
                )

        logger.info(
            f"Instrument synchronization completed. "
            f"Processed: {processed_count}, Matched: {matched_count}, Success Rate: {matched_count / processed_count * 100:.1f}%"
        )

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
                    return int(instrument.instrument_token)
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
                tokens.append(instrument.instrument_token)
            else:
                logger.warning(f"Instrument token for {tradingsymbol} not found in DB. Skipping.")
        return tokens
