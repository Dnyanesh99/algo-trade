import asyncio
from typing import Any

from src.auth.authenticator import KiteAuthenticator
from src.auth.token_manager import TokenManager
from src.broker.instrument_manager import InstrumentManager
from src.broker.rest_client import KiteRESTClient
from src.broker.websocket_client import KiteWebSocketClient
from src.database.instrument_repo import InstrumentRepository
from src.state.system_state import SystemState
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger  # Centralized logger
from src.utils.market_calendar import MarketCalendar
from src.utils.scheduler import Scheduler
from src.utils.time_synchronizer import TimeSynchronizer

# Load configuration
config = config_loader.get_config()


async def prediction_pipeline_callback() -> None:
    """
    Placeholder for the 15-minute prediction pipeline.
    This function will be called by the scheduler.
    """
    logger.info("Executing 15-minute prediction pipeline...")
    # TODO: Implement feature calculation, model prediction, and signal generation
    await asyncio.sleep(0.1)  # Simulate some async work


async def main_async() -> None:
    logger.info("Starting Automated Intraday Trading Signal Generation System...")
    system_config = config.get_system_config()
    logger.info(f"System Version: {system_config.version}")
    logger.info(f"System Mode: {system_config.mode}")
    logger.info(f"Timezone: {system_config.timezone}")

    # Initialize database manager early for all repository operations
    from src.database.db_utils import db_manager
    await db_manager.initialize()

    authenticator = KiteAuthenticator()
    token_manager = TokenManager()
    rest_client = KiteRESTClient()
    instrument_repo = InstrumentRepository()
    instrument_manager = InstrumentManager(rest_client, instrument_repo)

    # Authentication Flow
    if not token_manager.is_token_available():
        logger.info("Access token not found. Initiating automated authentication flow.")
        from src.auth.auth_server import start_auth_server_and_wait

        auth_success = await asyncio.to_thread(start_auth_server_and_wait)
        if not auth_success:
            logger.critical("Automated authentication failed. Exiting system.")
            return
        logger.info("Authentication successful and access token stored.")
    else:
        logger.info("Access token already available. Skipping authentication flow.")

    # Synchronize instruments before proceeding (only if flag is set)
    broker_config = config.get_broker_config()
    if broker_config.should_fetch_instruments:
        logger.info("should_fetch_instruments=True. Starting instrument synchronization...")
        await instrument_manager.sync_instruments()
        logger.info("Instrument synchronization completed.")
    else:
        logger.info("should_fetch_instruments=False. Skipping instrument synchronization.")

    if system_config.mode == "HISTORICAL_MODE":
        logger.info("Running in HISTORICAL_MODE. Orchestrating historical data processing and training.")

        # Import historical pipeline components
        from src.core.feature_calculator import FeatureCalculator
        from src.core.historical_aggregator import HistoricalAggregator
        from src.core.labeler import BarrierConfig, OptimizedTripleBarrierLabeler
        from src.database.db_utils import DatabaseManager
        from src.database.feature_repo import FeatureRepository
        from src.database.label_repo import LabelRepository
        from src.database.ohlcv_repo import OHLCVRepository
        from src.historical.fetcher import HistoricalFetcher
        from src.historical.processor import HistoricalProcessor
        from src.model.lgbm_trainer import LGBMTrainer
        from src.state.error_handler import ErrorHandler
        from src.state.health_monitor import HealthMonitor
        from src.utils.performance_metrics import PerformanceMetrics

        # Initialize components
        system_state = SystemState()
        error_handler = ErrorHandler(system_state)
        # No WebSocket client needed for historical mode
        health_monitor = HealthMonitor(
            system_state, None, db_manager
        )  # WebSocket client not needed for historical mode
        performance_metrics = PerformanceMetrics()

        # Initialize repositories
        ohlcv_repo = OHLCVRepository()
        feature_repo = FeatureRepository()
        label_repo = LabelRepository()

        # Initialize pipeline components
        historical_fetcher = HistoricalFetcher(rest_client)
        historical_processor = HistoricalProcessor()
        historical_aggregator = HistoricalAggregator(ohlcv_repo, error_handler, health_monitor, performance_metrics)
        feature_calculator = FeatureCalculator(
            ohlcv_repo, feature_repo, error_handler, health_monitor, performance_metrics
        )
        # Get labeling configuration
        trading_config = config.get_trading_config()
        labeling_config = trading_config.labeling
        barrier_config = BarrierConfig(
            atr_period=labeling_config.atr_period,
            tp_atr_multiplier=labeling_config.tp_atr_multiplier,
            sl_atr_multiplier=labeling_config.sl_atr_multiplier,
            max_holding_periods=labeling_config.max_holding_periods,
            min_bars_required=labeling_config.min_bars_required,
            atr_smoothing=labeling_config.atr_smoothing,
            epsilon=labeling_config.epsilon,
            use_dynamic_barriers=labeling_config.use_dynamic_barriers,
            volatility_lookback=labeling_config.volatility_lookback,
        )
        labeler = OptimizedTripleBarrierLabeler(barrier_config, label_repo)
        lgbm_trainer = LGBMTrainer(
            feature_repo, label_repo, error_handler, health_monitor, performance_metrics, feature_calculator
        )

        try:
            # Get configured instruments from config
            if not trading_config.instruments:
                logger.error("No instruments configured for historical processing.")
                return

            logger.info(f"Processing {len(trading_config.instruments)} instruments for historical data pipeline.")

            # Process each instrument
            for configured_inst in trading_config.instruments:
                # Get instrument details from database
                instrument = await instrument_repo.get_instrument_by_tradingsymbol(
                    configured_inst.tradingsymbol, configured_inst.exchange
                )
                if not instrument:
                    logger.warning(f"Instrument {configured_inst.tradingsymbol} not found in database. Skipping.")
                    continue

                instrument_id = instrument.instrument_id
                instrument_token = instrument.instrument_token
                symbol = instrument.tradingsymbol

                logger.info(f"Processing instrument: {symbol} (ID: {instrument_id}, Token: {instrument_token})")

                # 1. Fetch historical data (configured period of 1-minute data)
                from datetime import datetime, timedelta

                model_training_config = config.get_model_training_config()
                lookback_days = model_training_config.historical_data_lookback_days
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)

                historical_data = await historical_fetcher.fetch_historical_data(instrument_token, start_date, end_date)

                if not historical_data:
                    logger.warning(f"No historical data fetched for {symbol}. Skipping.")
                    continue

                logger.info(f"Fetched {len(historical_data)} 1-minute candles for {symbol}")

                # 2. Process and validate historical data
                # Get instrument type from configured instrument
                instrument_segment = getattr(configured_inst, 'segment', "UNKNOWN")
                logger.info(f"Processing {symbol} with segment: {instrument_segment}")
                df_processed, quality_report = historical_processor.process(historical_data, symbol, instrument_segment)
                
                if df_processed.empty:
                    logger.warning(f"No valid data after processing for {symbol}. Skipping.")
                    continue

                logger.info(f"Processed {len(df_processed)} valid 1-minute candles for {symbol}")

                # Store 1-minute data first
                from src.database.models import OHLCVData

                # Transform the processed data to match OHLCVData model format
                ohlcv_records = []
                for _, row in df_processed.iterrows():
                    candle_data = {
                        'ts': row['timestamp'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume'],
                        'oi': row.get('oi', None)
                    }
                    ohlcv_records.append(OHLCVData(**candle_data))
                
                await ohlcv_repo.insert_ohlcv_data(instrument_id, "1min", ohlcv_records)

                # 3. Prepare DataFrame for aggregation (rename timestamp back to date)
                df_for_aggregation = df_processed.rename(columns={'timestamp': 'date'})

                # Aggregate to higher timeframes
                aggregation_results = await historical_aggregator.aggregate_and_store(instrument_id, df_for_aggregation)
                successful_aggregations = [r for r in aggregation_results if r.success]

                logger.info(
                    f"Successfully aggregated {len(successful_aggregations)}/{len(aggregation_results)} timeframes for {symbol}"
                )

                # 3. Calculate features for each timeframe
                for timeframe in trading_config.feature_timeframes:
                    timeframe_str = f"{timeframe}min"

                    # Get latest candle timestamp for this timeframe
                    latest_candle = await ohlcv_repo.get_latest_candle(instrument_id, timeframe_str)
                    if not latest_candle:
                        logger.warning(f"No {timeframe_str} data found for {symbol}. Skipping feature calculation.")
                        continue

                    latest_timestamp = latest_candle.ts

                    # Calculate features
                    feature_results = await feature_calculator.calculate_and_store_features(
                        instrument_id, latest_timestamp
                    )

                    successful_features = [r for r in feature_results if r.success]
                    logger.info(
                        f"Successfully calculated features for {len(successful_features)} timeframes for {symbol}"
                    )

                # 4. Label data using triple barrier method
                for timeframe in trading_config.feature_timeframes:
                    timeframe_str = f"{timeframe}min"

                    # Get OHLCV data for labeling
                    ohlcv_data = await ohlcv_repo.get_ohlcv_data_for_features(instrument_id, timeframe, 1000)
                    if len(ohlcv_data) < 100:  # Need sufficient data for labeling
                        logger.warning(f"Insufficient {timeframe_str} data for labeling {symbol}. Skipping.")
                        continue

                    # Convert to DataFrame for labeling
                    import pandas as pd
                    labeling_df = pd.DataFrame([o.model_dump() for o in ohlcv_data])
                    labeling_df["timestamp"] = labeling_df["ts"]

                    # Process labeling
                    labeling_stats = await labeler.process_symbol(instrument_id, symbol, labeling_df, timeframe_str)

                    if labeling_stats:
                        logger.info(
                            f"Successfully labeled {labeling_stats.labeled_bars} bars for {symbol} ({timeframe_str})"
                        )
                    else:
                        logger.warning(f"Failed to label data for {symbol} ({timeframe_str})")

                # 5. Train model for each timeframe
                for timeframe in trading_config.feature_timeframes:
                    timeframe_str = f"{timeframe}min"

                    # Train model
                    model_path = await lgbm_trainer.train_model(
                        instrument_id, timeframe_str, optimize_hyperparams=False
                    )

                    if model_path:
                        logger.info(f"Successfully trained model for {symbol} ({timeframe_str}): {model_path}")
                    else:
                        logger.warning(f"Failed to train model for {symbol} ({timeframe_str})")

                logger.info(f"Completed historical processing for {symbol}")

            logger.info("Historical data processing and training complete.")

        except Exception as e:
            logger.critical(f"Error in historical mode orchestration: {e}", exc_info=True)
            await error_handler.handle_error(
                "historical_pipeline", f"Historical processing failed: {e}", {"error": str(e)}
            )

        finally:
            # Cleanup
            await labeler.shutdown()
            await db_manager.close()
    elif system_config.mode == "LIVE_MODE":
        logger.info("Running in LIVE_MODE. Orchestrating live trading and signal generation.")

        from src.live.candle_buffer import CandleBuffer
        from src.live.stream_consumer import StreamConsumer
        from src.live.tick_queue import TickQueue
        from src.live.tick_validator import TickValidator

        system_state = SystemState()
        market_calendar = MarketCalendar()
        time_synchronizer = TimeSynchronizer()

        tick_queue = TickQueue()
        tick_validator = TickValidator()
        async def candle_complete_callback(candle: dict[str, Any]) -> None:
            logger.info(f"Completed candle: {candle}")

        candle_buffer = CandleBuffer(
            on_candle_complete_callback=candle_complete_callback
        )  # TODO: Replace with actual LiveAggregator callback

        stream_consumer = StreamConsumer(tick_queue, tick_validator, candle_buffer)

        def ticks_callback(ticks: list[dict[str, Any]]) -> None:
            asyncio.create_task(tick_queue.put_many(ticks))

        ws_client = KiteWebSocketClient(
            on_ticks_callback=ticks_callback,  # Use put_many if implemented, else loop put
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
        logger.error(f"Invalid system mode specified in config: {system_config.mode}")

    logger.info("System shutdown.")
    
    # Cleanup database connection
    await db_manager.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
