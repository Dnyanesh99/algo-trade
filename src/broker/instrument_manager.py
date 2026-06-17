from typing import Any

from src.broker.rest_client import KiteRESTClient
from src.database.instrument_repo import InstrumentRepository
from src.database.models import InstrumentData
from src.utils.config_loader import TradingConfig
from src.utils.logger import LOGGER as logger


class InstrumentManager:
    """
    Manages the synchronization of instrument data between the broker and the local database.
    """

    def __init__(
        self, rest_client: KiteRESTClient, instrument_repo: InstrumentRepository, trading_config: TradingConfig
    ):
        # Type hints ensure rest_client and instrument_repo are valid instances

        self.rest_client = rest_client
        self.instrument_repo = instrument_repo
        self.trading_config = trading_config

        if not trading_config or not trading_config.instruments:
            logger.error("CRITICAL: Trading configuration ('trading.instruments') is missing or empty in config.yaml.")
            raise ValueError("InstrumentManager cannot operate without configured instruments.")

        self.configured_instruments = trading_config.instruments
        logger.info(f"InstrumentManager initialized with {len(self.configured_instruments)} configured instruments.")

    async def sync_instruments(self) -> None:
        """
        Fetches all instruments from the broker, validates them against the configuration,
        and updates the local database. This is a critical startup and daily operation.

        Raises:
            RuntimeError: If fetching instruments from the broker fails critically.
        """
        logger.info("Starting instrument synchronization process...")
        try:
            broker_instruments = await self.rest_client.get_instruments()
            if not broker_instruments or not isinstance(broker_instruments, list):
                logger.error("Failed to fetch instruments from broker or invalid format. Synchronization aborted.")
                raise RuntimeError("Could not retrieve any instruments from the broker.")
        except Exception as e:
            logger.critical(f"A critical error occurred while fetching instruments from the broker: {e}", exc_info=True)
            if "429" in str(e):
                logger.error("Rate limiting error detected. Please wait and try again later.")
            raise RuntimeError("Synchronization failed due to an error in fetching broker instruments.") from e

        logger.info(f"Successfully fetched {len(broker_instruments)} total instruments from the broker.")

        logger.info("Building a fast lookup index for all broker instruments...")
        instrument_lookup: dict[str, dict[str, Any]] = {
            f"{inst.get('tradingsymbol')}|{inst.get('exchange')}|{inst.get('instrument_type')}": inst
            for inst in broker_instruments
            if inst.get("tradingsymbol") and inst.get("exchange") and inst.get("instrument_type")
        }
        logger.info(f"Built lookup index with {len(instrument_lookup)} unique instrument keys.")

        processed_count = 0
        matched_count = 0
        updated_count = 0
        inserted_count = 0
        failed_count = 0

        for conf_inst in self.configured_instruments:
            processed_count += 1
            lookup_key = f"{conf_inst.tradingsymbol}|{conf_inst.exchange}|{conf_inst.instrument_type}"
            broker_inst = instrument_lookup.get(lookup_key)

            if not broker_inst:
                logger.error(
                    f"CRITICAL: Configured instrument '{conf_inst.label}' ({lookup_key}) not found in broker's master list. "
                    f"Please check configuration."
                )
                failed_count += 1
                continue

            try:
                matched_count += 1
                instrument_data = self._validate_and_build_instrument_data(broker_inst)

                existing_inst = await self.instrument_repo.get_instrument_by_tradingsymbol(
                    instrument_data.tradingsymbol, instrument_data.exchange
                )

                if existing_inst:
                    await self.instrument_repo.update_instrument(existing_inst.instrument_id, instrument_data)
                    logger.info(
                        f"Updated instrument: {conf_inst.label} ({instrument_data.tradingsymbol}) - Token: {instrument_data.instrument_token}"
                    )
                    updated_count += 1
                else:
                    await self.instrument_repo.insert_instrument(instrument_data)
                    logger.info(
                        f"Inserted new instrument: {conf_inst.label} ({instrument_data.tradingsymbol}) - Token: {instrument_data.instrument_token}"
                    )
                    inserted_count += 1
            except (ValueError, TypeError) as e:
                logger.error(
                    f"Failed to process instrument '{conf_inst.label}' due to validation error: {e}", exc_info=True
                )
                failed_count += 1
            except Exception as e:
                logger.error(
                    f"An unexpected database error occurred for instrument '{conf_inst.label}': {e}", exc_info=True
                )
                failed_count += 1

        success_rate = (matched_count - failed_count) / processed_count * 100 if processed_count > 0 else 0
        logger.info(
            f"Instrument synchronization completed. "
            f"Processed: {processed_count}, Matched: {matched_count}, Inserted: {inserted_count}, Updated: {updated_count}, Failed: {failed_count}. "
            f"Success Rate: {success_rate:.2f}%"
        )
        if failed_count > 0:
            logger.error(f"{failed_count} instruments failed to sync. Please review logs for critical errors.")

    def _validate_and_build_instrument_data(self, broker_inst: dict[str, Any]) -> InstrumentData:
        """Validates raw broker data and constructs a typed InstrumentData object."""
        required_fields = ["instrument_token", "tradingsymbol", "instrument_type", "exchange"]
        for field in required_fields:
            if not broker_inst.get(field):
                raise ValueError(f"Missing critical field '{field}' in broker data: {broker_inst}")

        return InstrumentData(
            instrument_token=int(broker_inst["instrument_token"]),
            exchange_token=broker_inst.get("exchange_token"),
            tradingsymbol=str(broker_inst["tradingsymbol"]),
            name=broker_inst.get("name"),
            last_price=float(broker_inst["last_price"]) if broker_inst.get("last_price") is not None else None,
            expiry=broker_inst.get("expiry"),
            strike=float(broker_inst["strike"]) if broker_inst.get("strike") is not None else None,
            tick_size=float(broker_inst["tick_size"]) if broker_inst.get("tick_size") is not None else None,
            lot_size=int(broker_inst["lot_size"]) if broker_inst.get("lot_size") is not None else None,
            instrument_type=str(broker_inst["instrument_type"]),
            segment=broker_inst.get("segment"),
            exchange=str(broker_inst["exchange"]),
        )

    async def get_instrument_token_by_label(self, label: str) -> int:
        """
        Retrieves the instrument token for a given configured instrument label from the database.

        Raises:
            ValueError: If the label is not found in the configuration or the instrument is not in the database.
        """
        logger.debug(f"Attempting to retrieve instrument token for label: '{label}'")
        if not label:
            raise ValueError("Instrument label cannot be empty.")

        for conf_inst in self.configured_instruments:
            if conf_inst.label == label:
                try:
                    instrument = await self.instrument_repo.get_instrument_by_tradingsymbol(
                        conf_inst.tradingsymbol, conf_inst.exchange
                    )
                    if instrument and instrument.instrument_token:
                        logger.debug(f"Found token {instrument.instrument_token} for label '{label}'.")
                        return instrument.instrument_token
                    logger.error(
                        f"Instrument for label '{label}' ({conf_inst.tradingsymbol}) not found in the database."
                    )
                    raise ValueError(f"Instrument for label '{label}' not found in DB.")
                except Exception as e:
                    logger.error(f"Database error while fetching token for label '{label}': {e}", exc_info=True)
                    raise ValueError(f"Could not retrieve token for label '{label}' due to a database error.") from e

        logger.error(f"Label '{label}' not found in any configured instruments.")
        raise ValueError(f"Label '{label}' is not a configured instrument.")

    async def get_configured_instrument_tokens(self) -> list[int]:
        """
        Retrieves instrument tokens for all instruments listed in the configuration.
        This method is critical for subscribing to the correct WebSocket feeds.

        Returns:
            A list of all instrument tokens that were successfully found in the database.
            If an instrument is configured but not found, it will be logged and skipped.
        """
        logger.info("Retrieving instrument tokens for all configured instruments.")
        tokens: list[int] = []
        missing_instruments: list[str] = []

        for conf_inst in self.configured_instruments:
            try:
                instrument = await self.instrument_repo.get_instrument_by_tradingsymbol(
                    conf_inst.tradingsymbol, conf_inst.exchange
                )
                if instrument and instrument.instrument_token:
                    tokens.append(instrument.instrument_token)
                else:
                    logger.error(
                        f"Configured instrument '{conf_inst.label}' ({conf_inst.tradingsymbol}) not found in database."
                    )
                    missing_instruments.append(conf_inst.label)
            except Exception as e:
                logger.error(f"Database error while fetching token for '{conf_inst.label}': {e}", exc_info=True)
                missing_instruments.append(conf_inst.label)

        if missing_instruments:
            logger.warning(
                f"Could not retrieve tokens for {len(missing_instruments)} configured instruments: {missing_instruments}. "
                f"These will be excluded from operations."
            )

        logger.info(f"Successfully retrieved {len(tokens)} instrument tokens for subscription.")
        return tokens
