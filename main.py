import asyncio

from src.auth.authenticator import KiteAuthenticator
from src.auth.token_manager import TokenManager
from src.broker.rest_client import KiteRESTClient
from src.broker.websocket_client import KiteWebSocketClient
from src.core.instrument_manager import InstrumentManager
from src.database.instrument_repo import InstrumentRepo
from src.state.system_state import SystemState
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger  # Centralized logger
from src.utils.market_calendar import MarketCalendar
from src.utils.scheduler import Scheduler
from src.utils.time_synchronizer import TimeSynchronizer

# Load configuration
config = config_loader.get_config()


async def prediction_pipeline_callback():
    """
    Placeholder for the 15-minute prediction pipeline.
    This function will be called by the scheduler.
    """
    logger.info("Executing 15-minute prediction pipeline...")
    # TODO: Implement feature calculation, model prediction, and signal generation
    await asyncio.sleep(0.1)  # Simulate some async work


async def main_async():
    logger.info("Starting Automated Intraday Trading Signal Generation System...")
    logger.info(f"System Version: {config.system.version}")
    logger.info(f"System Mode: {config.system.mode}")
    logger.info(f"Timezone: {config.system.timezone}")

    authenticator = KiteAuthenticator()
    token_manager = TokenManager()
    rest_client = KiteRESTClient()
    instrument_repo = InstrumentRepo()
    instrument_manager = InstrumentManager(rest_client, instrument_repo)

    # Authentication Flow
    if not token_manager.is_token_available():
        logger.info("Access token not found. Initiating automated authentication flow.")
        from src.auth.auth_server import start_auth_server_and_wait
        auth_success = start_auth_server_and_wait()
        if not auth_success:
            logger.critical("Automated authentication failed. Exiting system.")
            return
        logger.info("Authentication successful and access token stored.")
    else:
        logger.info("Access token already available. Skipping authentication flow.")

    # Synchronize instruments before proceeding
    await instrument_manager.sync_instruments()

    if config.system.mode == "HISTORICAL_MODE":
        logger.info("Running in HISTORICAL_MODE. Orchestrating historical data processing and training.")
        # Placeholder for historical pipeline orchestration
        # historical_fetcher.fetch_data()
        # historical_aggregator.aggregate_data()
        # feature_calculator.calculate_features()
        # labeler.label_data()
        # lgbm_trainer.train_model()
        logger.info("Historical data processing and training complete.")
    elif config.system.mode == "LIVE_MODE":
        logger.info("Running in LIVE_MODE. Orchestrating live trading and signal generation.")

        from src.live.tick_queue import TickQueue
        from src.live.tick_validator import TickValidator
        from src.live.candle_buffer import CandleBuffer
        from src.live.stream_consumer import StreamConsumer

        system_state = SystemState()
        market_calendar = MarketCalendar()
        time_synchronizer = TimeSynchronizer()

        tick_queue = TickQueue()
        tick_validator = TickValidator()
        candle_buffer = CandleBuffer(on_candle_complete_callback=lambda candle: logger.info(f"Completed candle: {candle}")) # TODO: Replace with actual LiveAggregator callback

        stream_consumer = StreamConsumer(tick_queue, tick_validator, candle_buffer)

        ws_client = KiteWebSocketClient(
            on_ticks_callback=lambda ticks: asyncio.create_task(tick_queue.put_many(ticks)), # Use put_many if implemented, else loop put
            on_reconnect_callback=lambda attempt: logger.info(f"KWS reconnected: {attempt}"),
            on_noreconnect_callback=lambda: logger.critical("KWS failed to reconnect!"),
        )
        scheduler = Scheduler(market_calendar, system_state, prediction_pipeline_callback)

        try:
            logger.info("Connecting to WebSocket...")
            await ws_client.connect()
            if ws_client.is_connected():
                logger.info("WebSocket connected. Starting scheduler.")
                # Subscribe to configured instruments
                configured_tokens = await instrument_manager.get_configured_instrument_tokens()
                if configured_tokens:
                    ws_client.subscribe(configured_tokens)
                    logger.info(f"Subscribed to {len(configured_tokens)} instruments.")
                else:
                    logger.warning("No instruments configured or found in DB for subscription.")

                scheduler.start()
                # Keep the main loop running to allow scheduler and WebSocket to operate
                while True:
                    await asyncio.sleep(3600)  # Sleep for a long duration, or until interrupted
            else:
                logger.error("WebSocket connection failed. Cannot start live mode.")
        except Exception as e:
            logger.critical(f"Error in live mode orchestration: {e}. Exiting system.")
        finally:
            await scheduler.stop()
            await ws_client.disconnect()

    else:
        logger.error(f"Invalid system mode specified in config: {config.system.mode}")

    logger.info("System shutdown.")

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()