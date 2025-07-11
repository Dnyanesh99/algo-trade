import os

# Fix OpenMP library conflict between LightGBM and scikit-learn
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import asyncio
import importlib
from typing import TYPE_CHECKING, Any

from src.auth.token_manager import TokenManager
from src.broker.instrument_manager import InstrumentManager
from src.broker.rest_client import KiteRESTClient
from src.database.db_utils import db_manager
from src.database.instrument_repo import InstrumentRepository
from src.signal.alert_system import AlertSystem
from src.state.error_handler import ErrorHandler
from src.state.system_state import SystemState
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger
from src.utils.logger import LoggerSetup

if TYPE_CHECKING:
    from src.core.feature_calculator import FeatureCalculator
    from src.database.feature_repo import FeatureRepository
    from src.model.predictor import ModelPredictor
    from src.signal.generator import SignalGenerator


async def prediction_pipeline_callback(
    instrument_manager: InstrumentManager,
    feature_calculator: "FeatureCalculator",
    model_predictor: "ModelPredictor",
    signal_generator: "SignalGenerator",
    feature_repo: "FeatureRepository",
    error_handler: "ErrorHandler",
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
                latest_candle = await feature_calculator.ohlcv_repo.get_latest_candle(instrument_token, "15min")
                if not latest_candle:
                    logger.warning(f"No latest candle found for instrument {instrument_token}")
                    continue

                feature_results = await feature_calculator.calculate_and_store_features(
                    instrument_token, latest_candle.ts
                )

                if not any(fr.success for fr in feature_results):
                    logger.warning(f"Feature calculation failed for instrument {instrument_token}")
                    continue

                latest_features = await feature_repo.get_latest_features(instrument_token, "15min")
                if not latest_features:
                    logger.warning(f"No latest features found for instrument {instrument_token}")
                    continue

                feature_vector = {f.feature_name: f.feature_value for f in latest_features}

                prediction_result = await model_predictor.predict(instrument_token, "15min", feature_vector)

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
        await error_handler.handle_error("prediction_pipeline_critical", f"Critical error in prediction pipeline: {e}")
        raise


async def main_async() -> None:
    """
    Main asynchronous function to run the trading system.
    """
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.get_config()

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
    await db_manager.initialize(config.database)

    # Initialize authentication components
    token_manager = TokenManager()
    rest_client = KiteRESTClient(token_manager, config.broker.api_key)
    await rest_client.initialize(config)

    instrument_repo = InstrumentRepository()
    instrument_manager = InstrumentManager(rest_client, instrument_repo, config.trading)

    # Authentication Flow
    if not token_manager.is_token_available():
        logger.info("Access token not found. Initiating automated authentication flow.")
        from src.auth.auth_server import start_auth_server_and_wait

        auth_success = await asyncio.to_thread(start_auth_server_and_wait, config.broker, config.auth_server)
        if not auth_success:
            logger.critical("Automated authentication failed. Exiting system.")
            return
        logger.info("Authentication successful and access token stored.")
    else:
        logger.info("Access token already available. Skipping authentication flow.")

    # Synchronize instruments before proceeding (only if flag is set)
    if config.broker.should_fetch_instruments:
        logger.info("should_fetch_instruments=True. Starting instrument synchronization...")
        await instrument_manager.sync_instruments()
        logger.info("Instrument synchronization completed.")
    else:
        logger.info("should_fetch_instruments=False. Skipping instrument synchronization.")

    # Initialize core components
    system_state = SystemState()
    alert_system = AlertSystem()
    error_handler = ErrorHandler(system_state, alert_system, config.error_handler)

    if config.system.mode == "HISTORICAL_MODE":
        logger.info("Running in HISTORICAL_MODE. Orchestrating historical data processing and training.")
        from src.core.feature_calculator import FeatureCalculator
        from src.core.historical_aggregator import HistoricalAggregator
        from src.core.labeler import BarrierConfig, OptimizedTripleBarrierLabeler
        from src.database.feature_repo import FeatureRepository
        from src.database.label_repo import LabelRepository
        from src.database.ohlcv_repo import OHLCVRepository
        from src.historical.fetcher import HistoricalFetcher
        from src.historical.processor import HistoricalProcessor
        from src.metrics import metrics_registry
        from src.model.lgbm_trainer import LGBMTrainer
        from src.state.health_monitor import HealthMonitor

        # Initialize components
        health_monitor = HealthMonitor(system_state, None, db_manager, config.health_monitor, config.system)
        metrics_registry.start_metrics_server()

        # Initialize repositories
        ohlcv_repo = OHLCVRepository()
        feature_repo = FeatureRepository()
        label_repo = LabelRepository()

        # Initialize pipeline components
        historical_fetcher = HistoricalFetcher(rest_client)
        historical_processor = HistoricalProcessor()
        historical_aggregator = HistoricalAggregator(ohlcv_repo, error_handler, health_monitor)
        feature_calculator = FeatureCalculator(ohlcv_repo, feature_repo, error_handler, health_monitor)
        # Get labeling configuration
        labeling_config = config.trading.labeling
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
        lgbm_trainer = LGBMTrainer(feature_repo, label_repo, error_handler, health_monitor, feature_calculator)

        try:
            # Get configured instruments from config
            if not config.trading.instruments:
                logger.error("No instruments configured for historical processing.")
                return

            logger.info(f"Processing {len(config.trading.instruments)} instruments for historical data pipeline.")

            # Process each instrument
            for configured_inst in config.trading.instruments:
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

                model_training_config = config.model_training
                lookback_days = model_training_config.historical_data_lookback_days

                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)

                historical_data = await historical_fetcher.fetch_historical_data(instrument_token, start_date, end_date)

                if not historical_data:
                    logger.warning(f"No historical data fetched for {symbol}. Skipping.")
                    continue

                logger.info(f"Fetched {len(historical_data)} 1-minute candles for {symbol}")

                # 2. Process and validate historical data
                instrument_segment = configured_inst.segment
                logger.info(f"Processing {symbol} with segment: {instrument_segment}")
                df_processed, quality_report = historical_processor.process(historical_data, symbol, instrument_segment)

                if df_processed.empty:
                    logger.warning(f"No valid data after processing for {symbol}. Skipping.")
                    continue

                logger.info(f"Processed {len(df_processed)} valid 1-minute candles for {symbol}")

                # Store 1-minute data first
                from src.database.models import OHLCVData

                ohlcv_records = []
                for _, row in df_processed.iterrows():
                    candle_data = {
                        "ts": row["timestamp"],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                        "oi": row.get("oi", None),
                    }
                    ohlcv_records.append(OHLCVData(**candle_data))

                await ohlcv_repo.insert_ohlcv_data(instrument_id, "1min", ohlcv_records)

                # 3. Prepare DataFrame for aggregation (rename timestamp back to date)
                df_for_aggregation = df_processed.rename(columns={"timestamp": "date"})

                # Aggregate to higher timeframes
                aggregation_results = await historical_aggregator.aggregate_and_store(instrument_id, df_for_aggregation)
                successful_aggregations = [r for r in aggregation_results if r.success]

                logger.info(
                    f"Successfully aggregated {len(successful_aggregations)}/{len(aggregation_results)} timeframes for {symbol}"
                )

                # 3. Calculate features for each timeframe
                for timeframe in config.trading.feature_timeframes:
                    timeframe_str = f"{timeframe}min"

                    latest_candle = await ohlcv_repo.get_latest_candle(instrument_id, timeframe_str)
                    if not latest_candle:
                        logger.warning(f"No {timeframe_str} data found for {symbol}. Skipping feature calculation.")
                        continue

                    latest_timestamp = latest_candle.ts

                    feature_results = await feature_calculator.calculate_and_store_features(
                        instrument_id, latest_timestamp
                    )

                    successful_features = [r for r in feature_results if r.success]
                    logger.info(
                        f"Successfully calculated features for {len(successful_features)} timeframes for {symbol}"
                    )

                # 4. Label data using triple barrier method
                for timeframe in config.trading.feature_timeframes:
                    timeframe_str = f"{timeframe}min"

                    ohlcv_data = await ohlcv_repo.get_ohlcv_data_for_features(instrument_id, timeframe, 1000)
                    if len(ohlcv_data) < 100:
                        logger.warning(f"Insufficient {timeframe_str} data for labeling {symbol}. Skipping.")
                        continue

                    import pandas as pd

                    labeling_df = pd.DataFrame([o.model_dump() for o in ohlcv_data])
                    labeling_df["timestamp"] = labeling_df["ts"]

                    labeling_stats = await labeler.process_symbol(instrument_id, symbol, labeling_df, timeframe_str)

                    if labeling_stats:
                        logger.info(
                            f"Successfully labeled {labeling_stats.labeled_bars} bars for {symbol} ({timeframe_str})"
                        )
                    else:
                        logger.warning(f"Failed to label data for {symbol} ({timeframe_str})")

                # 5. Train model for each timeframe
                for timeframe in config.trading.feature_timeframes:
                    timeframe_str = f"{timeframe}min"

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
            await labeler.shutdown()
            await db_manager.close()

    elif config.system.mode == "LIVE_MODE":
        logger.info("Running in LIVE_MODE. Orchestrating live trading and signal generation.")

        from src.broker.websocket_client import KiteWebSocketClient
        from src.core.feature_calculator import FeatureCalculator
        from src.database.ohlcv_repo import OHLCVRepository
        from src.database.signal_repo import SignalRepository
        from src.live.candle_buffer import CandleBuffer
        from src.live.stream_consumer import StreamConsumer
        from src.live.tick_queue import TickQueue
        from src.live.tick_validator import TickValidator
        from src.metrics import metrics_registry
        from src.model.predictor import ModelPredictor
        from src.signal.generator import SignalGenerator

        ohlcv_repo = OHLCVRepository()
        tick_queue = TickQueue()

        def on_ticks(ticks: list[dict[str, Any]]) -> None:
            asyncio.create_task(tick_queue.put_many(ticks))

        ws_client = KiteWebSocketClient(
            broker_config=config.broker,
            on_ticks_callback=on_ticks,
            on_reconnect_callback=lambda attempt: logger.info(f"KWS reconnected: {attempt}"),
            on_noreconnect_callback=lambda: logger.critical("KWS failed to reconnect!"),
        )
        health_monitor = HealthMonitor(system_state, ws_client, db_manager, config.health_monitor, config.system)
        feature_repo = FeatureRepository()
        feature_calculator = FeatureCalculator(ohlcv_repo, feature_repo, error_handler, health_monitor)
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
            )

        tick_validator = TickValidator()
        candle_buffer = CandleBuffer(on_candle_complete_callback=candle_complete_callback)
        stream_consumer = StreamConsumer(tick_queue, tick_validator, candle_buffer)
        stream_consumer.start_consuming()

        metrics_registry.start_metrics_server()

        try:
            logger.info("Connecting to WebSocket...")
            await ws_client.connect()
            if ws_client.is_connected():
                logger.info("WebSocket connected. Subscribing to instruments.")
                configured_tokens = await instrument_manager.get_configured_instrument_tokens()
                if configured_tokens:
                    ws_client.subscribe(configured_tokens)
                    logger.info(f"Subscribed to {len(configured_tokens)} instruments.")
                else:
                    logger.warning("No instruments configured or found in DB for subscription.")

                while True:
                    await asyncio.sleep(3600)
            else:
                logger.error("WebSocket connection failed. Cannot start live mode.")
        except Exception as e:
            logger.critical(f"Error in live mode orchestration: {e}. Exiting system.")
        finally:
            await stream_consumer.stop_consuming()
            await ws_client.disconnect()

    else:
        logger.error(f"Invalid system mode specified in config: {config.system.mode}")

    logger.info("System shutdown.")
    await db_manager.close()


def main() -> None:
    """
    Main entry point for the application.
    """
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("System interrupted by user. Shutting down.")
    except Exception as e:
        logger.critical(f"Unhandled exception at top level: {e}", exc_info=True)


if __name__ == "__main__":
    main()
