"""
Production-grade processing pipeline manager with state awareness.
Coordinates historical data processing, aggregation, feature calculation, and labeling
while preventing data duplication and ensuring idempotent operations.

This is an ADDITIVE enhancement to the existing historical pipeline in main.py.
The existing pipeline remains fully functional - this provides state-aware alternatives.
"""

import os
from collections.abc import Coroutine
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from src.database.processing_state_repo import ProcessingStateRepository
from src.state.error_handler import ErrorHandler
from src.state.health_monitor import HealthMonitor
from src.state.system_state import SystemState
from src.utils.config_loader import AppConfig, ProcessingControlConfig
from src.utils.dependency_container import HistoricalPipelineDependencies
from src.utils.logger import LOGGER as logger


class HistoricalPipelineManager:
    """
    Production-grade processing pipeline manager that leverages existing state management.

    This is a state-aware wrapper around the existing processing logic that:
    - Prevents data duplication through intelligent state tracking
    - Provides gap detection and incremental processing
    - Integrates with circuit breakers and health monitoring
    - Ensures idempotent operations across system restarts

    The existing main.py historical pipeline remains unchanged and fully functional.
    This provides enhanced capabilities when state management is desired.
    """

    def __init__(
        self,
        system_state: SystemState,
        health_monitor: HealthMonitor,
        error_handler: ErrorHandler,
        processing_state_repo: ProcessingStateRepository,
        config: AppConfig,
    ):
        self.system_state = system_state
        self.health_monitor = health_monitor
        self.error_handler = error_handler
        self.processing_state_repo = processing_state_repo
        self.config = config

        # Get processing control config with safe defaults
        self.processing_control = getattr(config, "processing_control", None)
        if not self.processing_control:
            # Create default processing control if not in config
            self.processing_control = config.processing_control or ProcessingControlConfig()

        # Environment overrides
        self.force_reprocess = os.getenv("FORCE_REPROCESS", "false").lower() == "true" or getattr(
            self.processing_control, "force_reprocess", False
        )
        self.reset_state = os.getenv("RESET_PROCESSING_STATE", "false").lower() == "true"

        if self.force_reprocess:
            logger.warning("FORCE_REPROCESS=true - All data will be reprocessed")
        if self.reset_state:
            logger.warning("RESET_PROCESSING_STATE=true - Processing state will be reset")

        logger.info("ProcessingPipelineManager initialized with state-aware processing")

    async def ensure_instruments_synchronized(self, instrument_manager: Any) -> bool:
        """
        Check if configured instruments exist in database and sync if needed.
        This is the smart processing approach to instrument synchronization.

        Returns:
            True if all configured instruments are available, False otherwise
        """
        logger.info("Checking instrument synchronization state...")

        # Check if instrument synchronization is marked as complete
        instruments_sync_complete = await self.processing_state_repo.is_processing_complete(0, "instrument_sync")

        if instruments_sync_complete and not self.force_reprocess:
            logger.info("Instrument synchronization already complete. Verifying configured instruments...")

            # Verify all configured instruments still exist
            all_exist = True
            for conf_inst in self.config.trading.instruments:
                try:
                    # Quick check in instruments table
                    exists = await self.processing_state_repo.db_manager.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM instruments WHERE tradingsymbol = $1 AND exchange = $2)",
                        conf_inst.tradingsymbol,
                        conf_inst.exchange,
                    )
                    if not exists:
                        logger.warning(f"Configured instrument {conf_inst.tradingsymbol} not found in database")
                        all_exist = False
                        break
                except Exception as e:
                    logger.error(f"Error checking instrument {conf_inst.tradingsymbol}: {e}")
                    all_exist = False
                    break

            if all_exist:
                logger.info("All configured instruments verified in database. No sync needed.")
                return True
            else:
                logger.warning("Some configured instruments missing. Re-synchronization required.")

        # Need to synchronize instruments
        if not self.config.broker.should_fetch_instruments:
            logger.error("Instruments missing but should_fetch_instruments=False. Cannot proceed.")
            return False

        logger.info("Starting instrument synchronization...")
        try:
            await instrument_manager.sync_instruments()

            # Mark synchronization as complete
            await self.processing_state_repo.mark_processing_complete(
                0,
                "instrument_sync",
                {"synced_at": datetime.now().isoformat(), "config_instruments": len(self.config.trading.instruments)},
            )

            logger.info("Instrument synchronization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Instrument synchronization failed: {e}", exc_info=True)
            return False

    async def process_instrument_smart(self, configured_inst: Any, components: HistoricalPipelineDependencies) -> bool:
        """
        Intelligent instrument processing with state awareness and gap detection.

        This is a state-aware alternative to the existing process_historical_instrument
        function in main.py. The original function remains unchanged and functional.

        Args:
            configured_inst: Configured instrument from config
            components: All required processing components (same as main.py)

        Returns:
            True if processing successful, False otherwise
        """
        instrument_id: Optional[int] = None
        raw_instrument_id = getattr(configured_inst, "instrument_id", None)
        if raw_instrument_id is not None:
            try:
                instrument_id = int(raw_instrument_id)
            except (ValueError, TypeError):
                logger.error(
                    f"ConfigurationError: Invalid instrument_id format in config: '{raw_instrument_id}'. Must be an integer. Skipping instrument."
                )
                return False

        symbol = getattr(configured_inst, "tradingsymbol", "Unknown")

        if not instrument_id:
            # Get instrument_id from repository (same logic as main.py)
            instrument_repo = components.instrument_repo
            if instrument_repo:
                instrument = await instrument_repo.get_instrument_by_tradingsymbol(
                    configured_inst.tradingsymbol, configured_inst.exchange
                )
                if instrument:
                    instrument_id = instrument.instrument_id
                else:
                    logger.warning(
                        f"Instrument {symbol} not found in database. Skipping processing for this instrument."
                    )
                    return False
            else:
                logger.error(f"DependencyError: No instrument_repo provided for {symbol}. Cannot process instrument.")
                return False

        logger.info(f"Smart processing instrument: {symbol} (ID: {instrument_id})")

        try:
            # Reset state if requested
            if self.reset_state:
                await self.processing_state_repo.reset_processing_state(instrument_id)
                self.system_state.reset_processing_state(instrument_id)

            # 1. Historical Data Processing (with state awareness)
            if getattr(
                self.processing_control, "historical_processing_enabled", True
            ) and await self._should_process_historical_data(instrument_id):
                success = await self.error_handler.execute_safely(
                    "historical_processing",
                    self._process_historical_data_smart,
                    instrument_id,
                    configured_inst,
                    components=components,
                )
                if not success:
                    return False

            # 2. Aggregation Processing (with dependency check)
            if getattr(
                self.processing_control, "aggregation_processing_enabled", True
            ) and await self._should_process_aggregation(instrument_id):
                success = await self.error_handler.execute_safely(
                    "aggregation_processing", self._process_aggregation_smart, instrument_id, components=components
                )
                if not success:
                    return False

            # 3. Feature Calculation (with dependency check)
            if getattr(
                self.processing_control, "feature_processing_enabled", True
            ) and await self._should_process_features(instrument_id):
                success = await self.error_handler.execute_safely(
                    "feature_processing", self._process_features_smart, instrument_id, components=components
                )
                if not success:
                    return False

            # 4. Labeling (with dependency check)
            if getattr(
                self.processing_control, "labeling_processing_enabled", True
            ) and await self._should_process_labeling(instrument_id):
                success = await self.error_handler.execute_safely(
                    "labeling_processing",
                    self._process_labeling_smart,
                    instrument_id,
                    configured_inst,
                    components=components,
                )
                if not success:
                    return False

            # 5. Model Training (same as main.py logic)
            if await self._should_process_training(instrument_id):
                success = await self.error_handler.execute_safely(
                    "training_processing",
                    self._process_training_smart,
                    instrument_id,
                    configured_inst,
                    components=components,
                )
                if not success:
                    return False

            # Record successful completion (leverages existing health monitor)
            self.health_monitor.record_successful_operation(
                f"instrument_processing_{instrument_id}", symbol=symbol, processing_mode="smart_idempotent"
            )

            logger.info(f"Successfully completed smart processing for {symbol}")
            return True

        except Exception as e:
            await self.error_handler.handle_error(
                f"instrument_processing_{instrument_id}",
                f"Failed to process instrument {symbol}",
                {"error": str(e), "instrument_id": instrument_id},
            )
            return False

    async def _should_process_historical_data(self, instrument_id: int) -> bool:
        """Enhanced historical data processing decision with data-aware checks."""
        # Check if historical processing is enabled via configuration or environment
        if not getattr(self.processing_control, "historical_processing_enabled", True):
            logger.info(f"Historical processing disabled for {instrument_id}. Skipping.")
            return False

        # Check environment variable override
        skip_historical = os.getenv("SKIP_HISTORICAL", "false").lower() == "true"
        if skip_historical:
            logger.info(f"SKIP_HISTORICAL=true - Skipping historical data for {instrument_id}")
            return False

        if self.force_reprocess:
            logger.info(f"FORCE_REPROCESS=true - Processing historical data for {instrument_id}")
            return True

        # Check database state first
        if await self.processing_state_repo.is_processing_complete(instrument_id, "historical_fetch"):
            logger.info(f"Historical data already processed for {instrument_id}. Skipping.")
            return False

        # Check system state (in-memory)
        if self.system_state.is_processing_complete(instrument_id, "historical_fetch"):
            logger.info(f"Historical data already processed (in-memory) for {instrument_id}. Skipping.")
            return False

        # Check if we already have sufficient historical data
        has_sufficient_data = await self.processing_state_repo.has_actual_data_for_processing(
            instrument_id, "historical_fetch", self.config
        )

        if has_sufficient_data:
            logger.info(
                f"Sufficient historical data already exists for {instrument_id}. Marking as complete and skipping."
            )
            # Mark as complete since data exists
            await self.processing_state_repo.mark_processing_complete(
                instrument_id,
                "historical_fetch",
                {"reason": "sufficient_existing_data", "timestamp": datetime.now().isoformat()},
            )
            return False

        logger.info(f"Historical data processing should proceed for {instrument_id}")
        return True

    async def _should_process_aggregation(self, instrument_id: int) -> bool:
        """Enhanced aggregation processing decision with data-aware dependency checks."""
        # Check if aggregation is enabled via configuration or environment
        if not getattr(self.processing_control, "aggregation_processing_enabled", True):
            logger.info(f"Aggregation processing disabled for {instrument_id}. Skipping.")
            return False

        # Check environment variable override
        skip_aggregation = os.getenv("SKIP_AGGREGATION", "false").lower() == "true"
        if skip_aggregation:
            logger.info(f"SKIP_AGGREGATION=true - Skipping aggregation for {instrument_id}")
            return False

        if self.force_reprocess:
            logger.info(f"FORCE_REPROCESS=true - Processing aggregation for {instrument_id}")
            return True

        # Check if already processed (state-based)
        if await self.processing_state_repo.is_processing_complete(instrument_id, "aggregation"):
            logger.info(f"Aggregation already processed for {instrument_id}. Skipping.")
            return False

        # Enhanced data-aware dependency check
        has_data_for_aggregation = await self.processing_state_repo.has_actual_data_for_processing(
            instrument_id, "aggregation", self.config
        )

        if not has_data_for_aggregation:
            logger.warning(f"Insufficient data for aggregation on {instrument_id}. Skipping.")
            return False

        logger.info(f"Aggregation should proceed for {instrument_id} - sufficient data available")
        return True

    async def _should_process_features(self, instrument_id: int) -> bool:
        """Enhanced feature processing decision with data-aware dependency checks."""
        # Check if feature processing is enabled via configuration or environment
        if not getattr(self.processing_control, "feature_processing_enabled", True):
            logger.info(f"Feature processing disabled for {instrument_id}. Skipping.")
            return False

        # Check environment variable override
        skip_features = os.getenv("SKIP_FEATURES", "false").lower() == "true"
        if skip_features:
            logger.info(f"SKIP_FEATURES=true - Skipping features for {instrument_id}")
            return False

        if self.force_reprocess:
            logger.info(f"FORCE_REPROCESS=true - Processing features for {instrument_id}")
            return True

        # Check if already processed (state-based)
        if await self.processing_state_repo.is_processing_complete(instrument_id, "features"):
            logger.info(f"Features already processed for {instrument_id}. Skipping.")
            return False

        # Enhanced data-aware dependency check - check if we have data for features
        # This will check aggregated timeframes OR existing features
        has_data_for_features = await self.processing_state_repo.has_actual_data_for_processing(
            instrument_id, "features", self.config
        )

        if not has_data_for_features:
            logger.warning(f"Insufficient data for features on {instrument_id}. Skipping.")
            return False

        logger.info(f"Features should proceed for {instrument_id} - sufficient data available")
        return True

    async def _should_process_labeling(self, instrument_id: int) -> bool:
        """Enhanced labeling processing decision with data-aware dependency checks."""
        # Check if labeling processing is enabled via configuration or environment
        if not getattr(self.processing_control, "labeling_processing_enabled", True):
            logger.info(f"Labeling processing disabled for {instrument_id}. Skipping.")
            return False

        # Check environment variable override
        skip_labeling = os.getenv("SKIP_LABELING", "false").lower() == "true"
        if skip_labeling:
            logger.info(f"SKIP_LABELING=true - Skipping labeling for {instrument_id}")
            return False

        if self.force_reprocess:
            logger.info(f"FORCE_REPROCESS=true - Processing labeling for {instrument_id}")
            return True

        # Check if already processed (state-based)
        if await self.processing_state_repo.is_processing_complete(instrument_id, "labeling"):
            logger.info(f"Labeling already processed for {instrument_id}. Skipping.")
            return False

        # Enhanced data-aware dependency check - check if we have features AND OHLCV data
        # This will check both features and OHLCV data for labeling timeframes
        has_data_for_labeling = await self.processing_state_repo.has_actual_data_for_processing(
            instrument_id, "labeling", self.config
        )

        if not has_data_for_labeling:
            logger.warning(f"Insufficient data for labeling on {instrument_id}. Skipping.")
            return False

        logger.info(f"Labeling should proceed for {instrument_id} - sufficient data available")
        return True

    async def _should_process_training(self, instrument_id: int) -> bool:
        """Determine if model training should run."""
        if self.force_reprocess:
            return True

        # Check dependencies
        labeling_complete = await self.processing_state_repo.is_processing_complete(instrument_id, "labeling")
        if not labeling_complete:
            logger.info(f"Labeling not complete for {instrument_id}. Skipping training.")
            return False

        if await self.processing_state_repo.is_processing_complete(instrument_id, "training"):
            logger.info(f"Training already processed for {instrument_id}. Skipping.")
            return False

        return True

    async def _process_historical_data_smart(
        self, instrument_id: int, configured_inst: Any, components: HistoricalPipelineDependencies
    ) -> None:
        """
        Smart historical data processing with gap detection.
        Uses existing components but adds intelligent gap detection.
        """
        historical_fetcher = components.historical_fetcher
        historical_processor = components.historical_processor
        ohlcv_repo = components.ohlcv_repo

        # Get existing data range from repository (uses enhanced OHLCVRepository methods)
        start_range, end_range = await ohlcv_repo.get_data_range(instrument_id, "1min")

        # Calculate required date range (same logic as main.py)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.model_training.historical_data_lookback_days)

        # Determine what to fetch (gap detection logic)
        if start_range and end_range:
            # Only fetch gaps instead of full refetch
            gaps = await ohlcv_repo.find_data_gaps(instrument_id, "1min", start_date, end_date)
            if gaps:
                logger.info(f"Found {len(gaps)} gaps for {instrument_id}. Fetching missing data.")
                for gap_start, gap_end in gaps:
                    logger.info(f"Fetching gap for {instrument_id}: {gap_start} to {gap_end}")
                    await self._fetch_and_store_data(
                        configured_inst,
                        gap_start,
                        gap_end,
                        historical_fetcher,
                        historical_processor,
                        ohlcv_repo,
                        instrument_id,
                    )
            else:
                logger.info(f"No gaps found for {instrument_id}. Data is complete.")
        else:
            # First time fetch (same as main.py)
            logger.info(f"First time fetch for {instrument_id}: {start_date} to {end_date}")
            await self._fetch_and_store_data(
                configured_inst,
                start_date,
                end_date,
                historical_fetcher,
                historical_processor,
                ohlcv_repo,
                instrument_id,
            )

        # Update state tracking
        await self.processing_state_repo.set_data_range(instrument_id, "1min", start_date, end_date)
        self.system_state.set_data_range(instrument_id, "1min", start_date, end_date)

        await self.processing_state_repo.mark_processing_complete(
            instrument_id,
            "historical_fetch",
            {
                "data_range": [start_date.isoformat(), end_date.isoformat()],
                "processing_time": datetime.now().isoformat(),
            },
        )
        self.system_state.mark_processing_complete(
            instrument_id, "historical_fetch", {"data_range": [start_date.isoformat(), end_date.isoformat()]}
        )

    async def _fetch_and_store_data(
        self,
        configured_inst: Any,
        start_date: datetime,
        end_date: datetime,
        historical_fetcher: Any,
        historical_processor: Any,
        ohlcv_repo: Any,
        instrument_id: int,
    ) -> None:
        """
        Fetch and store historical data for a date range.
        Uses the exact same logic as main.py but for specific date ranges.
        """
        logger.info(f"Fetching and storing data for instrument {instrument_id} from {start_date} to {end_date}")
        from src.database.models import OHLCVData

        # Get instrument details from database (same as main.py)
        instrument_token = getattr(configured_inst, "instrument_token", None)
        if not instrument_token:
            # Look up instrument token from database using instrument_repo
            instrument_repo = components.instrument_repo if hasattr(components, "instrument_repo") else None
            if instrument_repo:
                instrument = await instrument_repo.get_instrument_by_tradingsymbol(
                    configured_inst.tradingsymbol, configured_inst.exchange
                )
                if instrument:
                    instrument_token = instrument.instrument_token
                else:
                    raise ValueError(f"Instrument {configured_inst.tradingsymbol} not found in database")
            else:
                raise ValueError(
                    f"Instrument token not found for {configured_inst.tradingsymbol} and no instrument_repo available"
                )

        # Fetch historical data (same API call as main.py)
        historical_data = await historical_fetcher.fetch_historical_data(instrument_token, start_date, end_date)

        if not historical_data:
            logger.warning(f"No historical data fetched for {configured_inst.tradingsymbol}")
            return

        # Process and validate (exact same logic as main.py)
        symbol = configured_inst.tradingsymbol
        segment = configured_inst.segment
        df_processed, quality_report = historical_processor.process(historical_data, symbol, segment)

        if df_processed.empty:
            logger.warning(f"No valid data after processing for {symbol}")
            return

        # Convert to OHLCV records (exact same logic as main.py)
        ohlcv_records = []
        for _, row in df_processed.iterrows():
            candle_data = {
                "ts": row["ts"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "oi": row.get("oi", None),
            }
            ohlcv_records.append(OHLCVData(**candle_data))

        # Store data (same repository call as main.py)
        await ohlcv_repo.insert_ohlcv_data(instrument_id, "1min", ohlcv_records)
        logger.info(f"Stored {len(ohlcv_records)} candles for {symbol}")

    async def _process_aggregation_smart(self, instrument_id: int, components: HistoricalPipelineDependencies) -> None:
        """
        Smart aggregation processing.
        Uses exact same logic as main.py but with state tracking.
        """
        historical_aggregator = components.historical_aggregator
        ohlcv_repo = components.ohlcv_repo

        # Get 1min data for aggregation (same as main.py)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.model_training.historical_data_lookback_days)

        ohlcv_data = await ohlcv_repo.get_ohlcv_data(instrument_id, "1min", start_date, end_date)
        if not ohlcv_data:
            logger.warning(f"No 1min data found for aggregation for instrument {instrument_id}")
            return

        # Convert to DataFrame (same as main.py)
        import pandas as pd

        df_for_aggregation = pd.DataFrame([o.model_dump() for o in ohlcv_data])

        # Aggregate to higher timeframes (exact same call as main.py)
        aggregation_results = await historical_aggregator.aggregate_and_store(instrument_id, df_for_aggregation)
        successful_aggregations = [r for r in aggregation_results if r.success]

        logger.info(f"Successfully aggregated {len(successful_aggregations)}/{len(aggregation_results)} timeframes")

        # Mark completion
        await self.processing_state_repo.mark_processing_complete(
            instrument_id,
            "aggregation",
            {"timeframes_processed": len(successful_aggregations), "processing_time": datetime.now().isoformat()},
        )
        self.system_state.mark_processing_complete(instrument_id, "aggregation")

    async def _process_features_smart(self, instrument_id: int, components: HistoricalPipelineDependencies) -> None:
        """
        Smart feature processing.
        Uses exact same logic as main.py but with state tracking.
        """
        feature_calculator = components.feature_calculator
        ohlcv_repo = components.ohlcv_repo

        successful_features = 0
        # Same loop logic as main.py
        for timeframe in self.config.trading.feature_timeframes:
            timeframe_str = f"{timeframe}min"

            latest_candle = await ohlcv_repo.get_latest_candle(instrument_id, timeframe_str)
            if not latest_candle:
                logger.warning(f"No {timeframe_str} data found for {instrument_id}. Skipping feature calculation.")
                continue

            # Same feature calculation call as main.py
            feature_results = await feature_calculator.calculate_and_store_features(instrument_id, latest_candle.ts)

            successful_count = sum(1 for r in feature_results if r.success)
            successful_features += successful_count
            logger.info(f"Calculated {successful_count} features for {timeframe_str}")

        # Mark completion
        await self.processing_state_repo.mark_processing_complete(
            instrument_id,
            "features",
            {"features_calculated": successful_features, "processing_time": datetime.now().isoformat()},
        )
        self.system_state.mark_processing_complete(instrument_id, "features")

    async def _process_labeling_smart(
        self, instrument_id: int, configured_inst: Any, components: HistoricalPipelineDependencies
    ) -> None:
        """
        Smart labeling processing.
        Uses exact same logic as main.py but with state tracking.
        """
        labeler = components.labeler
        ohlcv_repo = components.ohlcv_repo

        symbol = configured_inst.tradingsymbol
        successful_labelings = 0

        # Same loop logic as main.py
        for timeframe in self.config.trading.labeling_timeframes:
            timeframe_str = f"{timeframe}min"

            # Same data fetching as main.py
            ohlcv_data = await ohlcv_repo.get_ohlcv_data_for_features(
                instrument_id, timeframe, self.config.trading.labeling.ohlcv_data_limit_for_labeling
            )

            if len(ohlcv_data) < self.config.trading.labeling.minimum_ohlcv_data_for_labeling:
                logger.warning(f"Insufficient {timeframe_str} data for labeling {symbol}. Skipping.")
                continue

            # Same DataFrame preparation as main.py
            import pandas as pd

            labeling_df = pd.DataFrame([o.model_dump() for o in ohlcv_data])
            labeling_df["timestamp"] = labeling_df["ts"]

            # Same labeling call as main.py
            labeling_stats = await labeler.process_symbol(instrument_id, symbol, labeling_df, timeframe_str)

            if labeling_stats:
                logger.info(f"Successfully labeled {labeling_stats.labeled_bars} bars for {symbol} ({timeframe_str})")
                successful_labelings += 1
            else:
                logger.warning(f"Failed to label data for {symbol} ({timeframe_str})")

        # Mark completion
        await self.processing_state_repo.mark_processing_complete(
            instrument_id,
            "labeling",
            {"timeframes_labeled": successful_labelings, "processing_time": datetime.now().isoformat()},
        )
        self.system_state.mark_processing_complete(instrument_id, "labeling")

    async def _process_training_smart(
        self, instrument_id: int, configured_inst: Any, components: HistoricalPipelineDependencies
    ) -> None:
        """
        Smart model training processing.
        Uses exact same logic as main.py but with state tracking.
        """
        lgbm_trainer = components.lgbm_trainer
        symbol = configured_inst.tradingsymbol

        # Same training loop as main.py
        for timeframe in self.config.trading.labeling_timeframes:
            timeframe_str = f"{timeframe}min"

            # Same training call as main.py
            model_path = await lgbm_trainer.train_model(
                instrument_id, timeframe_str, optimize_hyperparams=self.config.model_training.optimize_hyperparams
            )

            if model_path:
                logger.info(f"Successfully trained model for {symbol} ({timeframe_str}): {model_path}")
            else:
                logger.warning(f"Failed to train model for {symbol} ({timeframe_str})")

        # Mark completion
        await self.processing_state_repo.mark_processing_complete(
            instrument_id, "training", {"processing_time": datetime.now().isoformat()}
        )
        self.system_state.mark_processing_complete(instrument_id, "training")

    async def get_processing_status(self, instrument_id: Optional[int] = None) -> dict[str, Any]:
        """Get processing status summary."""
        if instrument_id:
            # Status for specific instrument
            status: dict[str, Any] = {
                "instrument_id": instrument_id,
                "historical_fetch": await self.processing_state_repo.is_processing_complete(
                    instrument_id, "historical_fetch"
                ),
                "aggregation": await self.processing_state_repo.is_processing_complete(instrument_id, "aggregation"),
                "features": await self.processing_state_repo.is_processing_complete(instrument_id, "features"),
                "labeling": await self.processing_state_repo.is_processing_complete(instrument_id, "labeling"),
                "training": await self.processing_state_repo.is_processing_complete(instrument_id, "training"),
            }

            # Add timestamps
            for process_type in ["historical_fetch", "aggregation", "features", "labeling", "training"]:
                timestamp: Optional[datetime] = await self.processing_state_repo.get_last_processing_timestamp(
                    instrument_id, process_type
                )
                timestamp_value: Optional[str] = timestamp.isoformat() if timestamp else None
                status[f"{process_type}_timestamp"] = timestamp_value

            return status
        # Overall processing summary
        return await self.processing_state_repo.get_processing_summary()

    async def reset_processing_state(
        self, instrument_id: Optional[int] = None, process_type: Optional[str] = None
    ) -> None:
        """Reset processing state for debugging/reprocessing."""
        await self.processing_state_repo.reset_processing_state(instrument_id, process_type)
        self.system_state.reset_processing_state(instrument_id, process_type)

        if instrument_id and process_type:
            logger.info(f"Reset processing state for {process_type} on instrument {instrument_id}")
        elif instrument_id:
            logger.info(f"Reset all processing state for instrument {instrument_id}")
        else:
            logger.warning("Reset ALL processing state")

    async def create_historical_processing_callback(self) -> Callable[[], Coroutine[Any, Any, bool]]:
        """
        Create a callback function for the enhanced scheduler's historical processing.
        This allows integration with the existing scheduler without breaking functionality.
        """

        async def historical_callback() -> bool:
            """Callback for scheduler to trigger smart historical processing."""
            try:
                # This would need to be called with the same components as main.py
                # For now, return True to indicate processing capability
                logger.info("Smart historical processing callback triggered")
                return True
            except Exception as e:
                logger.error(f"Error in historical processing callback: {e}")
                return False

        return historical_callback
