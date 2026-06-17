# OpenMP library configuration will be set after loading config
import asyncio
import importlib
import importlib.util
import os
from datetime import datetime
from typing import Any, Optional

from src.auth.auth_server import start_auth_server_and_wait
from src.auth.token_manager import TokenManager
from src.broker.backfill_manager import BackfillManager
from src.broker.connection_manager import ConnectionManager
from src.broker.instrument_manager import InstrumentManager
from src.broker.rest_client import KiteRESTClient
from src.broker.websocket_client import KiteWebSocketClient
from src.core.feature_calculator import FeatureCalculator
from src.core.historical_aggregator import HistoricalAggregator
from src.core.labeler import BarrierConfig, OptimizedTripleBarrierLabeler
from src.database.db_utils import db_manager
from src.database.feature_repo import FeatureRepository
from src.database.instrument_repo import InstrumentRepository
from src.database.label_repo import LabelRepository
from src.database.label_stats_repo import LabelingStatsRepository
from src.database.ohlcv_repo import OHLCVRepository
from src.database.processing_state_repo import ProcessingStateRepository
from src.database.signal_repo import SignalRepository
from src.historical.fetcher import HistoricalFetcher
from src.historical.processor import HistoricalProcessor
from src.live.candle_buffer import CandleBuffer
from src.live.stream_consumer import StreamConsumer
from src.live.tick_queue import TickQueue
from src.live.tick_validator import TickValidator
from src.metrics import metrics_registry
from src.model.lgbm_trainer import LGBMTrainer
from src.model.predictor import ModelPredictor
from src.signal.alert_system import AlertSystem
from src.signal.generator import SignalGenerator
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.state.system_state import SystemState
from src.utils.config_loader import ConfigLoader
from src.utils.dependency_container import HistoricalPipelineDependencies
from src.utils.env_utils import get_bool_env
from src.utils.historical_pipeline_manager import HistoricalPipelineManager
from src.utils.live_pipeline_manager import LivePipelineManager
from src.utils.logger import LOGGER as logger
from src.utils.logger import LoggerSetup
from src.utils.market_calendar import MarketCalendar
from src.utils.scheduler import Scheduler


async def ensure_valid_authentication(token_manager: TokenManager, config: Any) -> bool:
    """
    Ensures valid authentication is available. Handles token validation,
    failure detection, and automatic re-authentication.

    Returns:
        True if authentication is valid, False if failed
    """
    # Step 1: Check if token exists
    if not token_manager.is_token_available():
        logger.info("🔐 No access token found. Initiating authentication flow.")
        return await perform_authentication(token_manager, config)

    # Step 2: Validate existing token (with caching to avoid excessive API calls)
    logger.info("🔍 Access token found. Validating with Kite API...")
    token_valid = await token_manager.validate_token_with_api(config.broker.api_key)

    if token_valid:
        logger.info("✅ Access token is valid. Proceeding with application startup.")
        return True

    # Step 3: Token is invalid/expired, clear it and get fresh token
    logger.warning(
        "❌ Access token is invalid or expired. Clearing old token and initiating fresh authentication flow."
    )

    # Clear the invalid token so we don't keep trying to use it
    try:
        token_manager.clear_tokens()
        logger.info("🗑️ Cleared invalid token from storage")
    except Exception as e:
        logger.warning(f"Failed to clear invalid token: {e}")

    return await perform_authentication(token_manager, config)


async def perform_authentication(token_manager: TokenManager, config: Any) -> bool:
    """
    Perform the authentication flow using the auth server.

    Returns:
        True if authentication successful, False otherwise
    """
    if config.auth_server is None:
        logger.critical("Auth server configuration is missing. Cannot proceed with authentication.")
        return False

    logger.info("🚀 Starting authentication flow - browser will open automatically...")

    try:
        auth_success = await asyncio.to_thread(start_auth_server_and_wait, config.broker, config.auth_server)

        if auth_success:
            logger.info("✅ Authentication successful and access token stored.")
            token_manager.reset_auth_failures()
            return True
        logger.error("❌ Authentication failed.")
        return False

    except Exception as e:
        logger.error(f"❌ Authentication flow failed with error: {e}", exc_info=True)
        return False


async def handle_auth_failure(token_manager: TokenManager, config: Any, error_context: str = "") -> bool:
    """
    Handle authentication failures from WebSocket or API calls.

    Args:
        token_manager: TokenManager instance
        config: Application config
        error_context: Context about where the failure occurred

    Returns:
        True if re-authentication was successful, False otherwise
    """
    logger.warning(f"🔥 Authentication failure detected: {error_context}")

    # Record the failure
    should_reauth = token_manager.record_auth_failure()

    if should_reauth:
        logger.error("🔄 Maximum authentication failures reached. Triggering re-authentication.")
        # We don't clear the token file to preserve it across restarts
        success = await perform_authentication(token_manager, config)
        if success:
            token_manager.reset_auth_failures()
        else:
            logger.critical("CRITICAL: Re-authentication failed after maximum attempts. System cannot proceed.")
        return success
    logger.info("⚠️ Authentication failure recorded, but not triggering re-auth yet.")
    return False


# async def process_historical_instrument(
#     configured_inst: Any,
#     instrument_repo: InstrumentRepository,
#     historical_fetcher: HistoricalFetcher,
#     historical_processor: HistoricalProcessor,
#     ohlcv_repo: OHLCVRepository,
#     historical_aggregator: HistoricalAggregator,
#     feature_calculator: FeatureCalculator,
#     labeler: OptimizedTripleBarrierLabeler,
#     lgbm_trainer: LGBMTrainer,
#     config: Any,
# ) -> None:
#     """Process historical data for a single instrument."""
#     try:
#         # Get instrument details from database
#         instrument = await instrument_repo.get_instrument_by_tradingsymbol(
#             configured_inst.tradingsymbol, configured_inst.exchange
#         )
#         if not instrument:
#             logger.warning(f"Instrument {configured_inst.tradingsymbol} not found in database. Skipping.")
#             return

#         instrument_id = instrument.instrument_id
#         instrument_token = instrument.instrument_token
#         symbol = instrument.tradingsymbol

#         logger.info(f"Processing instrument: {symbol} (ID: {instrument_id}, Token: {instrument_token})")

#         # 1. Fetch historical data (configured period of 1-minute data)
#         try:
#             model_training_config = config.model_training
#             lookback_days = model_training_config.historical_data_lookback_days

#             end_date = datetime.now()
#             start_date = end_date - timedelta(days=lookback_days)

#             historical_data = await historical_fetcher.fetch_historical_data(instrument_token, start_date, end_date)

#             if not historical_data:
#                 logger.warning(f"No historical data fetched for {symbol}. Skipping.")
#                 return

#             logger.info(f"Fetched {len(historical_data)} 1-minute candles for {symbol}")
#         except Exception as e:
#             logger.error(f"Error fetching historical data for {symbol}: {e}", exc_info=True)
#             return

#         # 2. Process and validate historical data
#         try:
#             instrument_segment = configured_inst.segment
#             logger.info(f"Processing {symbol} with segment: {instrument_segment}")
#             df_processed, quality_report = historical_processor.process(historical_data, symbol, instrument_segment)

#             if df_processed.empty:
#                 logger.warning(f"No valid data after processing for {symbol}. Skipping.")
#                 return

#             logger.info(f"Processed {len(df_processed)} valid 1-minute candles for {symbol}")
#         except Exception as e:
#             logger.error(f"Error processing historical data for {symbol}: {e}", exc_info=True)
#             return

#         # Store 1-minute data first
#         try:
#             ohlcv_records = []
#             for _, row in df_processed.iterrows():
#                 candle_data = {
#                     "ts": row["ts"],
#                     "open": row["open"],
#                     "high": row["high"],
#                     "low": row["low"],
#                     "close": row["close"],
#                     "volume": row["volume"],
#                     "oi": row.get("oi", None),
#                 }
#                 ohlcv_records.append(OHLCVData(**candle_data))

#             await ohlcv_repo.insert_ohlcv_data(instrument_id, "1min", ohlcv_records)
#         except Exception as e:
#             logger.error(f"Error storing 1-minute OHLCV data for {symbol}: {e}", exc_info=True)
#             return

#         # 3. Prepare DataFrame for aggregation (already has ts column from processor)
#         df_for_aggregation = df_processed

#         # Aggregate to higher timeframes
#         try:
#             aggregation_results = await historical_aggregator.aggregate_and_store(instrument_id, df_for_aggregation)
#             successful_aggregations = [r for r in aggregation_results if r.success]

#             logger.info(
#                 f"Successfully aggregated {len(successful_aggregations)}/{len(aggregation_results)} timeframes for {symbol}"
#             )
#         except Exception as e:
#             logger.error(f"Error aggregating historical data for {symbol}: {e}", exc_info=True)
#             return

#         # 4. Calculate features for each timeframe
#         for timeframe in config.trading.feature_timeframes:
#             timeframe_str = f"{timeframe}min"
#             try:
#                 latest_candle = await ohlcv_repo.get_latest_candle(instrument_id, timeframe_str)
#                 if not latest_candle:
#                     logger.warning(f"No {timeframe_str} data found for {symbol}. Skipping feature calculation.")
#                     continue

#                 latest_timestamp = latest_candle.ts

#                 feature_results = await feature_calculator.calculate_and_store_features(instrument_id, latest_timestamp)

#                 successful_features = [r for r in feature_results if r.success]
#                 logger.info(f"Successfully calculated features for {len(successful_features)} timeframes for {symbol}")
#             except Exception as e:
#                 logger.error(f"Error calculating features for {symbol} ({timeframe_str}): {e}", exc_info=True)
#                 continue

#         # 5. Label data using triple barrier method (only for target signal timeframes)
#         for timeframe in config.trading.labeling_timeframes:
#             timeframe_str = f"{timeframe}min"
#             try:
#                 ohlcv_data = await ohlcv_repo.get_ohlcv_data_for_features(
#                     instrument_id, timeframe, config.trading.labeling.ohlcv_data_limit_for_labeling
#                 )
#                 if len(ohlcv_data) < config.trading.labeling.minimum_ohlcv_data_for_labeling:
#                     logger.warning(f"Insufficient {timeframe_str} data for labeling {symbol}. Skipping.")
#                     continue

#                 labeling_df = pd.DataFrame([o.model_dump() for o in ohlcv_data])
#                 labeling_df["timestamp"] = labeling_df["ts"]

#                 labeling_stats = await labeler.process_symbol(instrument_id, symbol, labeling_df, timeframe_str)

#                 if labeling_stats:
#                     logger.info(
#                         f"Successfully labeled {labeling_stats.labeled_bars} bars for {symbol} ({timeframe_str})"
#                     )
#                 else:
#                     logger.warning(f"Failed to label data for {symbol} ({timeframe_str})")
#             except Exception as e:
#                 logger.error(f"Error labeling data for {symbol} ({timeframe_str}): {e}", exc_info=True)
#                 continue

#         # 6. Train model for each labeling timeframe (need both features and labels)
#         for timeframe in config.trading.labeling_timeframes:
#             timeframe_str = f"{timeframe}min"
#             try:
#                 model_path = await lgbm_trainer.train_model(
#                     instrument_id, timeframe_str, optimize_hyperparams=config.model_training.optimize_hyperparams
#                 )

#                 if model_path:
#                     logger.info(f"Successfully trained model for {symbol} ({timeframe_str}): {model_path}")
#                 else:
#                     logger.warning(f"Failed to train model for {symbol} ({timeframe_str})")
#             except Exception as e:
#                 logger.error(f"Error training model for {symbol} ({timeframe_str}): {e}", exc_info=True)
#                 continue

#         logger.info(f"Completed historical processing for {symbol}")

#     except Exception as e:
#         symbol = getattr(configured_inst, "tradingsymbol", "Unknown")
#         logger.error(f"Error processing instrument {symbol}: {e}", exc_info=True)
#         logger.info("Continuing with next instrument...")


async def initialize_historical_mode_enhanced(
    config: Any,
    system_state: SystemState,
    alert_system: AlertSystem,
    error_handler: ErrorHandler,
    instrument_repo: InstrumentRepository,
) -> None:
    """Enhanced historical mode with state-aware processing to prevent data duplication."""
    logger.info("Running ENHANCED HISTORICAL_MODE with intelligent state management.")

    # Initialize components (same as original)
    health_monitor = HealthMonitor(system_state, None, db_manager, config.health_monitor, config.system)
    metrics_registry.start_metrics_server()

    # Initialize repositories
    ohlcv_repo = OHLCVRepository()
    feature_repo = FeatureRepository()
    label_repo = LabelRepository()
    stats_repo = LabelingStatsRepository()
    processing_state_repo = ProcessingStateRepository()

    # Initialize pipeline components (same as original)
    token_manager = TokenManager()
    rest_client = KiteRESTClient(token_manager, config.broker.api_key)
    await rest_client.initialize(config)

    # Initialize instrument manager for smart synchronization
    instrument_manager = InstrumentManager(rest_client, instrument_repo, config.trading)

    historical_fetcher = HistoricalFetcher(rest_client)
    historical_processor = HistoricalProcessor()
    historical_aggregator = HistoricalAggregator(ohlcv_repo, error_handler, health_monitor)
    feature_calculator = FeatureCalculator(ohlcv_repo, feature_repo, instrument_repo, error_handler, health_monitor)

    labeler = OptimizedTripleBarrierLabeler(
        BarrierConfig.from_config(config.trading.labeling),
        label_repo,
        instrument_repo,
        stats_repo,
        config.trading.labeling_timeframes,
    )
    lgbm_trainer = LGBMTrainer(feature_repo, label_repo, error_handler, health_monitor, feature_calculator)

    # Initialize enhanced pipeline manager
    pipeline_manager = HistoricalPipelineManager(
        system_state=system_state,
        health_monitor=health_monitor,
        error_handler=error_handler,
        processing_state_repo=processing_state_repo,
        config=config,
    )

    try:
        if not config.trading.instruments:
            logger.error("No instruments configured for historical processing.")
            return

        logger.info(f"Processing {len(config.trading.instruments)} instruments with intelligent state management.")

        # Smart instrument synchronization (only sync if needed)
        sync_success = await pipeline_manager.ensure_instruments_synchronized(instrument_manager)
        if not sync_success:
            logger.error("Failed to ensure instruments are synchronized. Cannot proceed.")
            return

        # Create dependency container
        dependencies = HistoricalPipelineDependencies(
            instrument_repo=instrument_repo,
            historical_fetcher=historical_fetcher,
            historical_processor=historical_processor,
            ohlcv_repo=ohlcv_repo,
            historical_aggregator=historical_aggregator,
            feature_calculator=feature_calculator,
            labeler=labeler,
            lgbm_trainer=lgbm_trainer,
        )

        successful_instruments = 0
        # Process each instrument with state awareness
        for configured_inst in config.trading.instruments:
            success = await pipeline_manager.process_instrument_smart(configured_inst, dependencies)

            if success:
                successful_instruments += 1
                logger.info(f"✅ Successfully processed {configured_inst.tradingsymbol} (Smart Mode)")
            else:
                logger.warning(f"⚠️ Failed to process {configured_inst.tradingsymbol} (Smart Mode)")

        logger.info(
            f"Enhanced historical processing complete: {successful_instruments}/{len(config.trading.instruments)} successful"
        )

    except Exception as e:
        safe_error = str(e).replace("{", "{{").replace("}", "}}")
        logger.critical(f"Error in enhanced historical mode: {safe_error}", exc_info=True)
        await error_handler.handle_error(
            "enhanced_historical_pipeline", f"Enhanced processing failed: {e}", {"error": str(e)}
        )

    finally:
        await labeler.shutdown()


# async def initialize_historical_mode(
#     config: Any,
#     system_state: SystemState,
#     alert_system: AlertSystem,
#     error_handler: ErrorHandler,
#     instrument_repo: InstrumentRepository,
# ) -> None:
#     """Initialize and run historical mode processing (Original - Fully Functional)."""
#     logger.info("Running in HISTORICAL_MODE. Orchestrating historical data processing and training.")

#     # Initialize components
#     health_monitor = HealthMonitor(system_state, None, db_manager, config.health_monitor, config.system)
#     metrics_registry.start_metrics_server()

#     # Initialize repositories
#     ohlcv_repo = OHLCVRepository()
#     feature_repo = FeatureRepository()
#     label_repo = LabelRepository()
#     stats_repo = LabelingStatsRepository()

#     # Initialize pipeline components
#     token_manager = TokenManager()
#     rest_client = KiteRESTClient(token_manager, config.broker.api_key)
#     await rest_client.initialize(config)

#     historical_fetcher = HistoricalFetcher(rest_client)
#     historical_processor = HistoricalProcessor()
#     historical_aggregator = HistoricalAggregator(ohlcv_repo, error_handler, health_monitor)
#     feature_calculator = FeatureCalculator(ohlcv_repo, feature_repo, instrument_repo, error_handler, health_monitor)

#     labeler = OptimizedTripleBarrierLabeler(
#         config.trading.labeling, label_repo, instrument_repo, stats_repo, config.trading.labeling_timeframes
#     )
#     lgbm_trainer = LGBMTrainer(feature_repo, label_repo, error_handler, health_monitor, feature_calculator)

#     try:
#         # Get configured instruments from config
#         if not config.trading.instruments:
#             logger.error("No instruments configured for historical processing.")
#             return

#         logger.info(f"Processing {len(config.trading.instruments)} instruments for historical data pipeline.")

#         # Process each instrument
#         for configured_inst in config.trading.instruments:
#             await process_historical_instrument(
#                 configured_inst,
#                 instrument_repo,
#                 historical_fetcher,
#                 historical_processor,
#                 ohlcv_repo,
#                 historical_aggregator,
#                 feature_calculator,
#                 labeler,
#                 lgbm_trainer,
#                 config,
#             )

#         logger.info("Historical data processing and training complete.")

#     except Exception as e:
#         logger.critical(f"Error in historical mode orchestration: {e}", exc_info=True)
#         await error_handler.handle_error("historical_pipeline", f"Historical processing failed: {e}", {"error": str(e)})

#     finally:
#         await labeler.shutdown()


async def initialize_live_mode(
    config: Any,
    system_state: SystemState,
    alert_system: AlertSystem,
    error_handler: ErrorHandler,
    instrument_manager: InstrumentManager,
    processing_state_repo: ProcessingStateRepository,
    market_calendar: MarketCalendar,
) -> None:
    """Initialize and run live mode processing."""
    logger.info("Running in LIVE_MODE. Orchestrating live trading and signal generation.")

    ohlcv_repo = OHLCVRepository()
    tick_queue = TickQueue()
    token_manager = TokenManager()

    # Track connection failures for auth retry logic
    connection_failure_count = 0
    max_connection_failures = 3

    live_pipeline_manager = LivePipelineManager(system_state, processing_state_repo)

    def on_ticks(ticks: list[dict[str, Any]]) -> None:
        now = datetime.now()
        asyncio.create_task(live_pipeline_manager.update_tick_stream_status("streaming", now))
        asyncio.create_task(tick_queue.put_many(ticks))

    def on_ws_error(error_code: int, error_message: str) -> None:
        """Handle WebSocket errors and trigger auth retry if needed."""
        nonlocal connection_failure_count
        asyncio.create_task(
            live_pipeline_manager.update_websocket_status("error", {"code": error_code, "message": error_message})
        )

        # Check for authentication-related errors
        if error_code == 1006 and "403" in error_message:
            logger.warning(f"WebSocket authentication error detected: {error_code} - {error_message}")
            connection_failure_count += 1

            if connection_failure_count >= max_connection_failures:
                logger.error("Multiple WebSocket auth failures detected. Will trigger re-authentication.")
                asyncio.create_task(
                    handle_auth_failure(
                        token_manager, config, f"WebSocket connection failures: {connection_failure_count}"
                    )
                )

    def on_ws_reconnect(attempt: int, last_connect_time: Optional[float] = None) -> None:
        logger.info(f"KWS reconnected: {attempt}")
        asyncio.create_task(live_pipeline_manager.update_websocket_status("reconnected", {"attempt": attempt}))
        # Reset failure count on successful reconnection
        nonlocal connection_failure_count
        connection_failure_count = 0

    def on_ws_connect() -> None:
        logger.info("WebSocket connected successfully.")
        asyncio.create_task(live_pipeline_manager.update_websocket_status("connected"))

    ws_client = KiteWebSocketClient(
        broker_config=config.broker,
        on_ticks_callback=on_ticks,
        on_reconnect_callback=on_ws_reconnect,
        on_noreconnect_callback=lambda: logger.critical("KWS failed to reconnect!"),
        on_connect_callback=on_ws_connect,
    )

    # Add error callback if WebSocket client supports it
    if hasattr(ws_client, "on_error_callback"):
        ws_client.on_error_callback = on_ws_error

    health_monitor = HealthMonitor(system_state, ws_client, db_manager, config.health_monitor, config.system)
    feature_repo = FeatureRepository()
    instrument_repo = InstrumentRepository()
    feature_calculator = FeatureCalculator(ohlcv_repo, feature_repo, instrument_repo, error_handler, health_monitor)
    model_predictor = ModelPredictor(error_handler, health_monitor)
    signal_repo = SignalRepository()
    signal_generator = SignalGenerator(
        signal_repo,
        error_handler,
        health_monitor,
        on_signal_generated=lambda signal_data: logger.info(f"Signal generated: {signal_data}"),
    )

    async def candle_complete_callback(candle: dict[str, Any]) -> None:
        logger.info(f"Completed candle: {candle}")
        await prediction_pipeline_callback(
            instrument_manager=instrument_manager,
            feature_calculator=feature_calculator,
            model_predictor=model_predictor,
            signal_generator=signal_generator,
            feature_repo=feature_repo,
            error_handler=error_handler,
            config=config,
        )

    tick_validator = TickValidator()
    candle_buffer = CandleBuffer(
        on_candle_complete_callback=candle_complete_callback, live_pipeline_manager=live_pipeline_manager
    )
    stream_consumer = StreamConsumer(tick_queue, tick_validator, candle_buffer)

    # Create rest_client for this function (token_manager should be passed from main)
    rest_client_local = KiteRESTClient(token_manager, config.broker.api_key)
    await rest_client_local.initialize(config)

    # Instantiate and connect the resilience components
    backfill_manager = BackfillManager(rest_client_local)

    async def on_data_gap_detected() -> None:
        logger.warning("Data gap detected due to reconnection. Initiating backfill.")
        try:
            last_known_timestamp = await processing_state_repo.get_last_processing_timestamp(
                0, "live_candle_aggregation_1min"
            )
            if not last_known_timestamp:
                logger.error("Cannot determine last known timestamp for backfill. Aborting.")
                return

            configured_tokens = await instrument_manager.get_configured_instrument_tokens()
            for token in configured_tokens:
                try:
                    missing_data = await backfill_manager.fetch_missing_data(token, last_known_timestamp)
                    if missing_data:
                        # This would feed into the StreamConsumer or a dedicated backfill processor
                        logger.info(f"Would merge {len(missing_data)} backfilled candles for {token}")
                except Exception as e:
                    logger.error(f"Error fetching missing data for token {token}: {e}", exc_info=True)
                    continue
        except Exception as e:
            logger.error(f"Critical error during data gap detection and backfill initiation: {e}", exc_info=True)
            await error_handler.handle_error(
                "data_gap_backfill", f"Critical error in backfill process: {e}", {"error": str(e)}
            )

    def on_critical_disconnection() -> None:
        logger.critical("CRITICAL: WebSocket disconnected and could not reconnect. Shutting down live mode.")
        system_state.set_system_status("ERROR_CRITICAL_DISCONNECT")

    connection_manager = ConnectionManager(
        websocket_client=ws_client,
        on_data_gap_detected=on_data_gap_detected,
        on_noreconnect_critical=on_critical_disconnection,
    )

    # Hand off control to the scheduler
    scheduler = Scheduler(
        market_calendar=market_calendar,
        system_state=system_state,
        prediction_pipeline_callback=lambda: prediction_pipeline_callback(
            instrument_manager=instrument_manager,
            feature_calculator=feature_calculator,
            model_predictor=model_predictor,
            signal_generator=signal_generator,
            feature_repo=feature_repo,
            error_handler=error_handler,
            config=config,
        ),
        scheduler_config=config.scheduler,
    )

    # Start all background tasks
    stream_consumer.start_consuming()
    connection_manager.start_monitoring()
    metrics_registry.start_metrics_server()
    await scheduler.start()  # This will now run the main live loop

    # Keep the main thread alive to wait for shutdown
    try:
        while scheduler.is_running():
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    finally:
        await scheduler.stop()
        await connection_manager.stop_monitoring()
        await stream_consumer.stop_consuming()
        await ws_client.disconnect()


async def prediction_pipeline_callback(
    instrument_manager: InstrumentManager,
    feature_calculator: FeatureCalculator,
    model_predictor: ModelPredictor,
    signal_generator: SignalGenerator,
    feature_repo: FeatureRepository,
    error_handler: ErrorHandler,
    config: Any,
) -> None:
    """
    15-minute prediction pipeline that calculates features, runs model prediction, and generates signals.
    This function will be called by the scheduler.
    """
    logger.info("Executing 15-minute prediction pipeline...")

    try:
        active_instruments = await instrument_manager.get_configured_instrument_tokens()

        if not active_instruments:
            logger.warning("No active instruments found for prediction")
            return

        logger.info(f"Running prediction pipeline for {len(active_instruments)} instruments")

        successful_predictions = 0

        for instrument_token in active_instruments:
            try:
                prediction_timeframe = config.scheduler.prediction_timeframe
                latest_candle = await feature_calculator.ohlcv_repo.get_latest_candle(
                    instrument_token, prediction_timeframe
                )
                if not latest_candle:
                    logger.warning(f"No latest candle found for instrument {instrument_token}")
                    continue

                feature_results = await feature_calculator.calculate_and_store_features(
                    instrument_token, latest_candle.ts
                )

                if not any(fr.success for fr in feature_results):
                    logger.warning(f"Feature calculation failed for instrument {instrument_token}")
                    continue

                latest_features = await feature_repo.get_latest_features(instrument_token, prediction_timeframe)
                if not latest_features:
                    logger.warning(f"No latest features found for instrument {instrument_token}")
                    continue

                feature_vector = {f.feature_name: f.feature_value for f in latest_features}

                prediction_result = await model_predictor.predict(
                    instrument_token, prediction_timeframe, feature_vector
                )

                if not prediction_result:
                    logger.warning(f"Prediction failed for instrument {instrument_token}")
                    continue

                signal_result = await signal_generator.generate_signal(
                    instrument_id=prediction_result.instrument_id,
                    timeframe=prediction_result.timeframe,
                    timestamp=prediction_result.timestamp,
                    prediction=prediction_result.prediction,
                    confidence_score=prediction_result.confidence,
                    current_price=latest_candle.close,
                )

                if signal_result:
                    successful_predictions += 1

            except Exception as e:
                logger.error(f"Error processing instrument {instrument_token}: {e}")
                await error_handler.handle_error(
                    "prediction_pipeline_instrument",
                    f"Error processing instrument {instrument_token}: {e}",
                    {"instrument_token": instrument_token},
                )
                continue

        logger.info(
            f"Prediction pipeline completed. {successful_predictions}/{len(active_instruments)} successful predictions"
        )

    except Exception as e:
        logger.error(f"Critical error in prediction pipeline: {e}")
        await error_handler.handle_error(
            "prediction_pipeline_critical", f"Critical error in prediction pipeline: {e}", {}
        )
        raise


async def hydrate_state(system_state: SystemState, processing_state_repo: ProcessingStateRepository) -> None:
    """Load persistent processing state from DB into in-memory SystemState."""
    logger.info("Hydrating system state from database...")
    all_states = await processing_state_repo.get_all_processing_states()
    for state in all_states:
        system_state.mark_processing_complete(
            instrument_id=state.instrument_id, process_type=state.processing_type, metadata=state.metadata
        )

    all_ranges = await processing_state_repo.get_all_data_ranges()
    for dr in all_ranges:
        system_state.set_data_range(
            instrument_id=dr.instrument_id, timeframe=dr.timeframe, start_ts=dr.earliest_ts, end_ts=dr.latest_ts
        )
    logger.info(f"Hydrated {len(all_states)} processing states and {len(all_ranges)} data ranges.")


async def main_async() -> None:
    """
    Main asynchronous function to run the trading system.
    """
    # Load configuration
    config_loader = ConfigLoader()
    try:
        config = config_loader.get_config()
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}", exc_info=True)
        raise RuntimeError("System cannot start without valid configuration.") from e

    # Fix OpenMP library conflict between LightGBM and scikit-learn
    os.environ["OMP_NUM_THREADS"] = str(config.system.openmp_threads)
    os.environ["MKL_NUM_THREADS"] = str(config.system.mkl_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(config.system.numexpr_threads)

    # Initialize logger with configuration
    LoggerSetup.setup_logger(config.logging)

    # Check for TA-Lib
    if importlib.util.find_spec("talib") is None:
        logger.warning(
            "TA-Lib not found. Feature calculation will use pandas-based implementations, which may be slower."
        )

    # Initialize Metrics Registry
    from src.metrics.registry import metrics_registry

    metrics_registry.initialize(config.metrics)

    logger.info("Starting Automated Intraday Trading Signal Generation System...")
    logger.info(f"System Version: {config.system.version}")
    logger.info(f"System Mode: {config.system.mode}")
    logger.info(f"Timezone: {config.system.timezone}")

    # Initialize database manager early for all repository operations
    try:
        await db_manager.initialize()
    except Exception as e:
        logger.critical(f"Failed to initialize database manager: {e}", exc_info=True)
        raise RuntimeError("System cannot proceed without database connection.") from e

    # Initialize authentication components
    token_manager = TokenManager()
    rest_client = KiteRESTClient(token_manager, config.broker.api_key)
    try:
        await rest_client.initialize(config)
    except Exception as e:
        logger.critical(f"Failed to initialize REST client: {e}", exc_info=True)
        raise RuntimeError("System cannot proceed without REST client initialization.") from e

    instrument_repo = InstrumentRepository()
    instrument_manager = InstrumentManager(rest_client, instrument_repo, config.trading)

    # Ensure market_calendar config is present before instantiation
    if not config.market_calendar:
        logger.critical("Market calendar configuration is missing. Exiting system.")
        raise RuntimeError("Market calendar configuration is missing.")
    market_calendar = MarketCalendar(config.system, config.market_calendar)

    # Authentication Flow - Check both availability and validity
    auth_success = await ensure_valid_authentication(token_manager, config)
    if not auth_success:
        logger.critical("Authentication failed. Exiting system.")
        raise RuntimeError("Authentication failed. Cannot proceed.")

    # Log successful authentication and proceed with system initialization
    logger.info("✅ Authentication successful. Proceeding with system initialization.")

    # Skip old instrument synchronization - handled by smart processing pipeline
    logger.info("🧠 Instrument synchronization will be handled by smart processing pipeline")

    # Initialize core components
    system_state = SystemState()
    alert_system = AlertSystem()
    error_handler = ErrorHandler(system_state, alert_system, config.error_handler)

    # Hydrate state from database
    try:
        processing_state_repo = ProcessingStateRepository()
        await hydrate_state(system_state, processing_state_repo)
    except Exception as e:
        logger.error(f"Failed to hydrate system state from database: {e}", exc_info=True)
        # This might not be critical enough to stop the system, but worth logging.

    if config.system.mode == "HISTORICAL_MODE":
        # Check for smart processing mode via environment variable
        use_smart_processing = get_bool_env("USE_SMART_PROCESSING", True)

        if use_smart_processing:
            logger.info("🧠 USE_SMART_PROCESSING=true - Using enhanced state-aware processing")
            await initialize_historical_mode_enhanced(
                config, system_state, alert_system, error_handler, instrument_repo
            )
        else:
            logger.info("📊 USE_SMART_PROCESSING=false - Using original historical processing (default)")
            # await initialize_historical_mode(config, system_state, alert_system, error_handler, instrument_repo)
    else:  # config.system.mode == "LIVE_MODE"
        await initialize_live_mode(
            config,
            system_state,
            alert_system,
            error_handler,
            instrument_manager,
            processing_state_repo,
            market_calendar,
        )

    logger.info("System shutdown.")
    try:
        await db_manager.close()
    except Exception as e:
        logger.error(f"Error closing database connection: {e}", exc_info=True)
        # Log, but don't prevent shutdown


def main() -> None:
    """
    Main entry point for the application.
    """
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("System interrupted by user. Shutting down.")
    except Exception as e:
        safe_error = str(e).replace("{", "{{").replace("}", "}}")
        logger.critical(f"Unhandled exception at top level: {safe_error}", exc_info=True)
    finally:
        logger.complete()


if __name__ == "__main__":
    # Test comment to verify Docker build optimization
    main()
