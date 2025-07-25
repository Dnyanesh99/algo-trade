"""
Production-grade processing pipeline manager with state awareness.
Coordinates historical data processing, aggregation, feature calculation, and labeling
while preventing data duplication and ensuring idempotent operations.

This is an ADDITIVE enhancement to the existing historical pipeline in main.py.
The existing pipeline remains fully functional - this provides state-aware alternatives.
"""

import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import pytz

from src.database.models import OHLCVData
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
            self.processing_control = ProcessingControlConfig()

        # Environment overrides for frequently changing controls
        self.force_reprocess = self._get_bool_env("FORCE_REPROCESS")
        self.reset_state = self._get_bool_env("RESET_PROCESSING_STATE")
        self.truncate_all_data = self._get_bool_env("TRUNCATE_ALL_DATA")

        # Generate unique session ID for this pipeline run
        self._session_id = str(uuid.uuid4())

        if self.force_reprocess:
            logger.warning("FORCE_REPROCESS=true - All data will be reprocessed")
        if self.reset_state:
            logger.warning("RESET_PROCESSING_STATE=true - Processing state will be reset")
        if self.truncate_all_data:
            logger.critical("🚨 TRUNCATE_ALL_DATA=true - ALL CONFIGURED DATABASE TABLES WILL BE TRUNCATED ONCE! 🚨")

        # Initialize timezone for datetime normalization
        self.timezone = pytz.timezone(config.system.timezone)

        logger.info("ProcessingPipelineManager initialized with state-aware processing")

    def _normalize_datetime(self, dt: datetime) -> datetime:
        """
        Normalize datetime to be timezone-aware using system timezone.

        Args:
            dt: DateTime object (timezone-aware or naive)

        Returns:
            Timezone-aware datetime object
        """
        if dt.tzinfo is None:
            # If timezone-naive, assume it's in system timezone
            return self.timezone.localize(dt)
        # If already timezone-aware, convert to system timezone
        return dt.astimezone(self.timezone)

    @staticmethod
    def _get_bool_env(env_var_name: str, default: bool = False) -> bool:
        """
        Helper to parse boolean environment variables.

        Args:
            env_var_name: Name of the environment variable
            default: Default value if env var is not set

        Returns:
            Boolean value parsed from environment variable
        """
        env_value = os.getenv(env_var_name)
        if env_value is None:
            return default
        return env_value.lower() in ("true", "1", "yes", "on")

    @staticmethod
    def _validate_timeframe(timeframe: str) -> bool:
        """
        Validate timeframe parameter against a whitelist to prevent SQL injection.

        Args:
            timeframe: The timeframe string to validate

        Returns:
            True if timeframe is valid, False otherwise
        """
        # Whitelist of allowed timeframes (numeric values only)
        allowed_timeframes = {"1", "3", "5", "10", "15", "30", "60", "240", "1440"}
        return timeframe in allowed_timeframes

    async def ensure_instruments_synchronized(self, instrument_manager: Any) -> bool:
        """
        Check if configured instruments exist in database and sync if needed.
        This is the smart processing approach to instrument synchronization.

        Returns:
            True if all configured instruments are available, False otherwise
        """
        logger.info("Checking instrument synchronization state...")

        # Check if instrument synchronization is marked as complete
        # Use system operations table for global operations
        instruments_sync_complete = await self.processing_state_repo.is_system_operation_complete("instrument_sync")

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
            logger.warning("Some configured instruments missing. Re-synchronization required.")

        # Need to synchronize instruments
        if not self.config.broker.should_fetch_instruments:
            logger.error("Instruments missing but should_fetch_instruments=False. Cannot proceed.")
            return False

        logger.info("Starting instrument synchronization...")
        try:
            await instrument_manager.sync_instruments()

            # Mark synchronization as complete
            await self.processing_state_repo.mark_system_operation_complete(
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
                try:
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
                except Exception as e:
                    logger.error(f"Error retrieving instrument {symbol} from database: {e}")
                    return False
            else:
                logger.error(f"DependencyError: No instrument_repo provided for {symbol}. Cannot process instrument.")
                return False

        logger.info(f"Smart processing instrument: {symbol} (ID: {instrument_id})")

        try:
            # Truncate all data if requested (DANGEROUS - happens ONCE per system run)
            if self.truncate_all_data:
                (
                    truncation_requested,
                    truncation_performed,
                ) = await self.processing_state_repo.check_and_perform_truncation_atomic(self._session_id)
                if truncation_requested and truncation_performed:
                    # After truncation, reset in-memory state to be consistent
                    self.system_state.reset_processing_state()
                    logger.info(f"✅ Global truncation completed. Continuing with {symbol} using fresh state.")
                elif truncation_requested and not truncation_performed:
                    logger.error(f"💥 Global truncation failed. Aborting processing for {symbol}")
                    return False
                else:
                    logger.info(f"⏭️ Truncation not requested or failed to check. Continuing with {symbol}...")

            # Reset state if requested (only affects state tracking, not actual data)
            if self.reset_state:
                await self.processing_state_repo.reset_processing_state(instrument_id)
                self.system_state.reset_processing_state(instrument_id)

            # 1. Historical Data Processing (with state awareness)
            logger.info(f"🔄 Stage 1/5: Evaluating historical data processing for {symbol} (ID: {instrument_id})")
            if self._get_bool_env("HISTORICAL_PROCESSING_ENABLED", True) and await self._should_process_historical_data(
                instrument_id
            ):
                logger.info(f"📊 Processing historical data for {symbol} (ID: {instrument_id})")
                success = await self.error_handler.execute_safely(
                    "historical_processing",
                    self._process_historical_data_smart,
                    instrument_id,
                    configured_inst,
                    components=components,
                )
                if not success:
                    logger.error(f"Historical data processing failed for {symbol} (ID: {instrument_id})")
                    return False
                logger.info(f"✅ Stage 1/5: Historical data processing completed for {symbol}")
            else:
                logger.info(f"⏭️  Stage 1/5: Skipping historical data processing for {symbol}")

            # 2. Aggregation Processing (with dependency check)
            logger.info(f"🔄 Stage 2/5: Evaluating aggregation processing for {symbol} (ID: {instrument_id})")
            if self._get_bool_env("AGGREGATION_PROCESSING_ENABLED", True) and await self._should_process_aggregation(
                instrument_id
            ):
                logger.info(f"📈 Processing aggregation for {symbol} (ID: {instrument_id})")
                success = await self.error_handler.execute_safely(
                    "aggregation_processing", self._process_aggregation_smart, instrument_id, components=components
                )
                if not success:
                    logger.error(f"Aggregation processing failed for {symbol} (ID: {instrument_id})")
                    return False
                logger.info(f"✅ Stage 2/5: Aggregation processing completed for {symbol}")
            else:
                logger.info(f"⏭️  Stage 2/5: Skipping aggregation processing for {symbol}")

            # Continue with remaining stages only if aggregation was successful or skipped

            # 3. Feature Calculation (with dependency check)
            logger.info(f"🔄 Stage 3/5: Evaluating feature calculation for {symbol} (ID: {instrument_id})")
            if self._get_bool_env("FEATURE_PROCESSING_ENABLED", True) and await self._should_process_features(
                instrument_id
            ):
                logger.info(f"🧮 Processing features for {symbol} (ID: {instrument_id})")
                success = await self.error_handler.execute_safely(
                    "feature_processing", self._process_features_smart, instrument_id, components=components
                )
                if not success:
                    logger.error(f"Feature processing failed for {symbol} (ID: {instrument_id})")
                    return False
                logger.info(f"✅ Stage 3/5: Feature calculation completed for {symbol}")
            else:
                logger.info(f"⏭️  Stage 3/5: Skipping feature calculation for {symbol}")

            # 4. Labeling (with dependency check)
            logger.info(f"🔄 Stage 4/5: Evaluating labeling for {symbol} (ID: {instrument_id})")
            if self._get_bool_env("LABELING_PROCESSING_ENABLED", True) and await self._should_process_labeling(
                instrument_id
            ):
                logger.info(f"🏷️  Processing labeling for {symbol} (ID: {instrument_id})")
                success = await self.error_handler.execute_safely(
                    "labeling_processing",
                    self._process_labeling_smart,
                    instrument_id,
                    configured_inst,
                    components=components,
                )
                if not success:
                    logger.error(f"❌ Labeling processing failed for {symbol} (ID: {instrument_id})")
                    return False
                logger.info(f"✅ Stage 4/5: Labeling completed for {symbol}")
            else:
                # Check if labeling was skipped due to failed dependencies
                try:
                    dependencies_satisfied = await self.processing_state_repo.validate_processing_dependencies(
                        instrument_id, "labeling"
                    )
                    if not dependencies_satisfied:
                        # Check if features failed
                        features_failed = await self.processing_state_repo.is_processing_failed(
                            instrument_id, "features"
                        )
                        if features_failed:
                            logger.error(
                                f"🚫 Stopping pipeline for {symbol} (ID: {instrument_id}) - "
                                f"feature processing failed and labeling cannot proceed"
                            )
                            return False
                except Exception as e:
                    logger.error(f"Error checking labeling dependencies for {symbol}: {e}")

                logger.info(f"⏭️  Stage 4/5: Skipping labeling for {symbol}")

            # 5. Model Training (same as main.py logic)
            logger.info(f"🔄 Stage 5/5: Evaluating model training for {symbol} (ID: {instrument_id})")
            if await self._should_process_training(instrument_id):
                logger.info(f"🤖 Processing model training for {symbol} (ID: {instrument_id})")
                success = await self.error_handler.execute_safely(
                    "training_processing",
                    self._process_training_smart,
                    instrument_id,
                    configured_inst,
                    components=components,
                )
                if not success:
                    logger.error(f"❌ Model training failed for {symbol} (ID: {instrument_id})")
                    return False
                logger.info(f"✅ Stage 5/5: Model training completed for {symbol}")
            else:
                # Check if training was skipped due to failed dependencies
                try:
                    dependencies_satisfied = await self.processing_state_repo.validate_processing_dependencies(
                        instrument_id, "training"
                    )
                    if not dependencies_satisfied:
                        # Check if labeling failed
                        labeling_failed = await self.processing_state_repo.is_processing_failed(
                            instrument_id, "labeling"
                        )
                        if labeling_failed:
                            logger.error(
                                f"🚫 Stopping pipeline for {symbol} (ID: {instrument_id}) - "
                                f"labeling processing failed and training cannot proceed"
                            )
                            return False
                except Exception as e:
                    logger.error(f"Error checking training dependencies for {symbol}: {e}")

                logger.info(f"⏭️  Stage 5/5: Skipping model training for {symbol}")

            # Record successful completion (leverages existing health monitor)
            logger.info(f"🎉 All stages completed successfully for {symbol} (ID: {instrument_id})")
            self.health_monitor.record_successful_operation(
                f"instrument_processing_{instrument_id}", symbol=symbol, processing_mode="smart_idempotent"
            )

            # Update the system state to reflect successful completion
            self.system_state.mark_processing_complete(instrument_id, "smart_processing")
            logger.info(f"Successfully completed smart processing for {symbol}")
            return True

        except Exception as e:
            error_msg = f"Failed to process instrument {symbol}: {str(e)}"
            logger.error(error_msg)
            await self.error_handler.handle_error(
                f"instrument_processing_{instrument_id}",
                error_msg,
                {"error": str(e), "instrument_id": instrument_id},
            )
            # Mark the processing as failed in the system state
            await self.processing_state_repo.mark_processing_failed(
                instrument_id,
                "smart_processing",
                {
                    "error": str(e),
                    "failed_at": datetime.now().isoformat(),
                },
            )
            return False

    async def _should_process_historical_data(self, instrument_id: int) -> bool:
        """Enhanced historical data processing decision with data-aware checks."""
        # Check if historical processing is enabled via environment variable
        if not self._get_bool_env("HISTORICAL_PROCESSING_ENABLED", True):
            logger.info(f"Historical processing disabled for {instrument_id}. Skipping.")
            return False

        # Check database state first - but proceed if FORCE_REPROCESS=true
        try:
            is_complete = await self.processing_state_repo.is_processing_complete(instrument_id, "historical_fetch")
            if is_complete and not self.force_reprocess:
                logger.info(f"Historical data already processed for {instrument_id}. Skipping.")
                return False
            if is_complete and self.force_reprocess:
                logger.info(
                    f"FORCE_REPROCESS=true - Reprocessing historical data for {instrument_id} despite being complete"
                )
        except Exception as e:
            logger.error(f"Error checking processing state for {instrument_id}: {e}")
            # Continue processing if we can't check the state

        # Check if we already have sufficient historical data - but respect FORCE_REPROCESS
        try:
            has_sufficient_data = await self.processing_state_repo.has_actual_data_for_processing(
                instrument_id, "historical_fetch", self.config
            )

            if has_sufficient_data and not self.force_reprocess:
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
            if has_sufficient_data and self.force_reprocess:
                logger.info(
                    f"FORCE_REPROCESS=true - Reprocessing historical data for {instrument_id} despite sufficient existing data"
                )
        except Exception as e:
            logger.error(f"Error checking data sufficiency for {instrument_id}: {e}")
            # Continue processing if we can't check data sufficiency

        logger.info(f"Historical data processing should proceed for {instrument_id}")
        return True

    async def _should_process_aggregation(self, instrument_id: int) -> bool:
        """Enhanced aggregation processing decision with dependency validation."""
        # Check if aggregation is enabled via environment variable
        if not self._get_bool_env("AGGREGATION_PROCESSING_ENABLED", True):
            logger.info(f"Aggregation processing disabled for {instrument_id}. Skipping.")
            return False

        # Check if already processed (state-based) - but proceed if FORCE_REPROCESS=true
        try:
            is_complete = await self.processing_state_repo.is_processing_complete(instrument_id, "aggregation")
            if is_complete and not self.force_reprocess:
                logger.info(f"Aggregation already processed for {instrument_id}. Skipping.")
                return False
            if is_complete and self.force_reprocess:
                logger.info(
                    f"FORCE_REPROCESS=true - Reprocessing aggregation for {instrument_id} despite being complete"
                )
        except Exception as e:
            logger.error(f"Error checking aggregation processing state for {instrument_id}: {e}")
            # Continue processing if we can't check the state

        # Validate processing dependencies (handles environment variables and data checks)
        try:
            dependencies_satisfied = await self.processing_state_repo.validate_processing_dependencies(
                instrument_id, "aggregation"
            )
            if not dependencies_satisfied:
                logger.warning(f"Dependencies not satisfied for aggregation on {instrument_id}. Skipping.")
                return False
        except Exception as e:
            logger.error(f"Error validating dependencies for aggregation on {instrument_id}: {e}")
            logger.warning(f"Skipping aggregation for {instrument_id} due to dependency check failure - fail-safe mode")
            return False

        logger.info(f"Aggregation should proceed for {instrument_id} - dependencies satisfied")
        return True

    async def _should_process_features(self, instrument_id: int) -> bool:
        """Enhanced feature processing decision with dependency validation."""
        # Check if feature processing is enabled via environment variable
        if not self._get_bool_env("FEATURE_PROCESSING_ENABLED", True):
            logger.info(f"Feature processing disabled for {instrument_id}. Skipping.")
            return False

        # Check if already processed (state-based) - but proceed if FORCE_REPROCESS=true
        try:
            is_complete = await self.processing_state_repo.is_processing_complete(instrument_id, "features")
            if is_complete and not self.force_reprocess:
                logger.info(f"Features already processed for {instrument_id}. Skipping.")
                return False
            if is_complete and self.force_reprocess:
                logger.info(f"FORCE_REPROCESS=true - Reprocessing features for {instrument_id} despite being complete")
        except Exception as e:
            logger.error(f"Error checking features processing state for {instrument_id}: {e}")
            # Continue processing if we can't check the state

        # Validate processing dependencies (handles environment variables and data checks)
        try:
            dependencies_satisfied = await self.processing_state_repo.validate_processing_dependencies(
                instrument_id, "features"
            )
            if not dependencies_satisfied:
                logger.warning(f"Dependencies not satisfied for features on {instrument_id}. Skipping.")
                return False
        except Exception as e:
            logger.error(f"Error validating dependencies for features on {instrument_id}: {e}")
            logger.warning(f"Skipping features for {instrument_id} due to dependency check failure - fail-safe mode")
            return False

        logger.info(f"Features should proceed for {instrument_id} - dependencies satisfied")
        return True

    async def _should_process_labeling(self, instrument_id: int) -> bool:
        """Enhanced labeling processing decision with dependency validation."""
        # Check if labeling processing is enabled via environment variable
        if not self._get_bool_env("LABELING_PROCESSING_ENABLED", True):
            logger.info(f"Labeling processing disabled for {instrument_id}. Skipping.")
            return False

        # Check if already processed (state-based) - but proceed if FORCE_REPROCESS=true
        try:
            is_complete = await self.processing_state_repo.is_processing_complete(instrument_id, "labeling")
            if is_complete and not self.force_reprocess:
                logger.info(f"Labeling already processed for {instrument_id}. Skipping.")
                return False
            if is_complete and self.force_reprocess:
                logger.info(f"FORCE_REPROCESS=true - Reprocessing labeling for {instrument_id} despite being complete")
        except Exception as e:
            logger.error(f"Error checking labeling processing state for {instrument_id}: {e}")
            # Continue processing if we can't check the state

        # Validate processing dependencies (handles environment variables and data checks)
        try:
            dependencies_satisfied = await self.processing_state_repo.validate_processing_dependencies(
                instrument_id, "labeling"
            )
            if not dependencies_satisfied:
                logger.warning(f"Dependencies not satisfied for labeling on {instrument_id}. Skipping.")
                # Log detailed status for debugging
                await self.processing_state_repo.log_processing_summary(instrument_id)
                return False
        except Exception as e:
            logger.error(f"Error validating dependencies for labeling on {instrument_id}: {e}")
            logger.warning(f"Skipping labeling for {instrument_id} due to dependency check failure - fail-safe mode")
            return False

        logger.info(f"Labeling should proceed for {instrument_id} - dependencies satisfied")
        return True

    async def _should_process_training(self, instrument_id: int) -> bool:
        """Determine if model training should run."""
        # Check if training processing is enabled via environment variable
        if not self._get_bool_env("TRAINING_PROCESSING_ENABLED", True):
            logger.info(f"Training processing disabled for {instrument_id}. Skipping.")
            return False

        # Check if already processed (state-based) - but proceed if FORCE_REPROCESS=true
        try:
            training_complete = await self.processing_state_repo.is_processing_complete(instrument_id, "training")
            if training_complete and not self.force_reprocess:
                logger.info(f"Training already processed for {instrument_id}. Skipping.")
                return False
            if training_complete and self.force_reprocess:
                logger.info(f"FORCE_REPROCESS=true - Reprocessing training for {instrument_id} despite being complete")
        except Exception as e:
            logger.error(f"Error checking training completion for {instrument_id}: {e}")
            # Continue with training if we can't check if it's already done

        # Validate processing dependencies (handles environment variables and data checks)
        if not self.force_reprocess:
            try:
                dependencies_satisfied = await self.processing_state_repo.validate_processing_dependencies(
                    instrument_id, "training"
                )
                if not dependencies_satisfied:
                    logger.warning(f"Dependencies not satisfied for training on {instrument_id}. Skipping.")
                    # Log detailed status for debugging
                    await self.processing_state_repo.log_processing_summary(instrument_id)
                    return False
            except Exception as e:
                logger.error(f"Error validating dependencies for training on {instrument_id}: {e}")
                logger.warning(
                    f"Skipping training for {instrument_id} due to dependency check failure - fail-safe mode"
                )
                return False
        else:
            logger.warning(f"FORCE_REPROCESS=true - Bypassing dependency checks for training on {instrument_id}")
            # Still log the status for awareness
            await self.processing_state_repo.log_processing_summary(instrument_id)

        logger.info(f"Training should proceed for {instrument_id}")
        return True

    async def _process_historical_data_smart(
        self, instrument_id: int, configured_inst: Any, components: HistoricalPipelineDependencies
    ) -> bool:
        """
        Smart historical data processing with gap detection.
        Uses existing components but adds intelligent gap detection.
        """
        historical_fetcher = components.historical_fetcher
        historical_processor = components.historical_processor
        ohlcv_repo = components.ohlcv_repo

        # Mark processing as in progress
        try:
            await self.processing_state_repo.mark_processing_in_progress(
                instrument_id, "historical_fetch", {"started_at": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Failed to mark historical processing as in-progress: {e}")
            # Continue processing even if we can't update the state

        # Get existing data range from repository (uses enhanced OHLCVRepository methods)
        try:
            start_range, end_range = await ohlcv_repo.get_data_range(instrument_id, "1min")
        except Exception as e:
            logger.error(f"Error getting data range for {instrument_id}: {e}")
            start_range, end_range = None, None

        # Initialize date variables to prevent race condition
        end_date = self._normalize_datetime(datetime.now())
        start_date = self._normalize_datetime(
            end_date - timedelta(days=self.config.model_training.historical_data_lookback_days)
        )

        # Determine fetch strategy based on existing data and reset state
        fetch_performed = False

        if self.reset_state or not start_range or not end_range:
            # Fresh start or no existing data - fetch full historical range
            logger.info(f"Fetching full historical range for {instrument_id}: {start_date.date()} to {end_date.date()}")
            await self._fetch_and_store_data(
                configured_inst,
                start_date,
                end_date,
                historical_fetcher,
                historical_processor,
                ohlcv_repo,
                instrument_id,
                components,
            )
            fetch_performed = True
        else:
            # Incremental fetch - get data from last known point to current time
            latest_candle = await ohlcv_repo.get_latest_candle(instrument_id, "1min")
            if latest_candle:
                # Fetch from day after last candle to current time
                incremental_start = latest_candle.ts.date() + timedelta(days=1)
                incremental_start_date = self._normalize_datetime(
                    datetime.combine(incremental_start, datetime.min.time())
                )
                incremental_end_date = self._normalize_datetime(datetime.now())

                # Only fetch if there are potentially new trading days
                if incremental_start_date.date() < datetime.now().date():
                    logger.info(
                        f"Performing incremental fetch for {instrument_id}: {incremental_start_date.date()} to {incremental_end_date.date()}"
                    )
                    await self._fetch_and_store_data(
                        configured_inst,
                        incremental_start_date,
                        incremental_end_date,
                        historical_fetcher,
                        historical_processor,
                        ohlcv_repo,
                        instrument_id,
                        components,
                    )
                    # Update dates for state tracking
                    start_date = incremental_start_date
                    end_date = incremental_end_date
                    fetch_performed = True
                else:
                    logger.info(f"Data for {instrument_id} is up-to-date. No incremental fetch needed.")
            else:
                logger.warning(
                    f"Data range exists but no latest candle found for {instrument_id}. Performing full fetch."
                )
                # Fallback to full fetch if data range exists but no candles found
                await self._fetch_and_store_data(
                    configured_inst,
                    start_date,
                    end_date,
                    historical_fetcher,
                    historical_processor,
                    ohlcv_repo,
                    instrument_id,
                    components,
                )
                fetch_performed = True

        try:
            # Update state tracking only if data was actually fetched
            if fetch_performed:
                try:
                    await self.processing_state_repo.set_data_range(instrument_id, "1min", start_date, end_date)
                    self.system_state.set_data_range(instrument_id, "1min", start_date, end_date)
                except Exception as e:
                    logger.error(f"Failed to update data range for {instrument_id}: {e}")

            # Mark processing as complete
            try:
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
                logger.info(f"Successfully completed historical data processing for instrument {instrument_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to mark historical processing as complete: {e}")
                return False
        except asyncio.CancelledError:
            # Re-raise cancellation to allow graceful shutdown
            raise
        except Exception as e:
            # Mark processing as failed
            error_message = str(e)
            logger.error(f"Historical data processing failed for {instrument_id}: {error_message}", exc_info=True)
            try:
                await self.processing_state_repo.mark_processing_failed(
                    instrument_id,
                    "historical_fetch",
                    {
                        "error": error_message,
                        "failed_at": datetime.now().isoformat(),
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as inner_e:
                logger.error(f"Failed to mark historical processing as failed: {inner_e}")
            return False

    async def _fetch_and_store_data(
        self,
        configured_inst: Any,
        start_date: datetime,
        end_date: datetime,
        historical_fetcher: Any,
        historical_processor: Any,
        ohlcv_repo: Any,
        instrument_id: int,
        components: HistoricalPipelineDependencies,
    ) -> None:
        """
        Fetch and store historical data for a date range.
        Uses the exact same logic as main.py but for specific date ranges.
        """
        logger.info(f"Fetching and storing data for instrument {instrument_id} from {start_date} to {end_date}")

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

        # Log data quality issues if any
        if quality_report and hasattr(quality_report, "issues") and quality_report.issues:
            logger.warning(f"Data quality issues for {symbol}: {quality_report.issues}")

        if df_processed.empty:
            logger.warning(f"No valid data after processing for {symbol}")
            return

        # Convert to OHLCV records (exact same logic as main.py)
        ohlcv_records = []
        for _, row in df_processed.iterrows():
            # Normalize timestamp to ensure timezone awareness
            normalized_ts = self._normalize_datetime(row["ts"])
            candle_data = {
                "ts": normalized_ts,
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

    async def _process_aggregation_smart(self, instrument_id: int, components: HistoricalPipelineDependencies) -> bool:
        """
        Smart aggregation processing with proper state tracking.

        Args:
            instrument_id: The ID of the instrument to process
            components: Required processing components

        Returns:
            bool: True if aggregation was successful, False otherwise
        """
        historical_aggregator = components.historical_aggregator
        ohlcv_repo = components.ohlcv_repo

        # Mark processing as in progress
        try:
            await self.processing_state_repo.mark_processing_in_progress(
                instrument_id, "aggregation", {"started_at": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Failed to mark aggregation processing as in-progress: {e}")
            # Continue processing even if we can't update the state

        try:
            # Get 1min data for aggregation (same as main.py)
            # Ensure timezone-aware datetime objects to prevent comparison errors
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.model_training.historical_data_lookback_days)

            # Normalize datetimes to ensure timezone consistency
            end_date = self._normalize_datetime(end_date)
            start_date = self._normalize_datetime(start_date)

            ohlcv_data = await ohlcv_repo.get_ohlcv_data(instrument_id, "1min", start_date, end_date)
            if not ohlcv_data:
                logger.warning(f"No 1min data found for aggregation for instrument {instrument_id}")
                await self.processing_state_repo.mark_processing_failed(
                    instrument_id,
                    "aggregation",
                    {
                        "error": "No 1min data found for aggregation",
                        "failed_at": datetime.now().isoformat(),
                    },
                )
                return False

            # Convert to DataFrame (same as main.py)
            df_for_aggregation = pd.DataFrame([o.model_dump() for o in ohlcv_data])

            # Aggregate to higher timeframes (exact same call as main.py)
            aggregation_results = await historical_aggregator.aggregate_and_store(instrument_id, df_for_aggregation)
            successful_aggregations = [r for r in aggregation_results if r.success]
            expected_timeframes = len(self.config.trading.aggregation_timeframes)

            logger.info(f"Successfully aggregated {len(successful_aggregations)}/{len(aggregation_results)} timeframes")

            # Only mark as complete if ALL timeframes were successfully aggregated
            if len(successful_aggregations) == expected_timeframes:
                await self.processing_state_repo.mark_processing_complete(
                    instrument_id,
                    "aggregation",
                    {
                        "timeframes_processed": len(successful_aggregations),
                        "processing_time": datetime.now().isoformat(),
                    },
                )
                self.system_state.mark_processing_complete(instrument_id, "aggregation")
                logger.info(f"Successfully completed aggregation for instrument {instrument_id}")
                return True
            # Mark as failed if any timeframe failed
            error_message = (
                f"Only {len(successful_aggregations)}/{expected_timeframes} timeframes were successfully aggregated"
            )
            await self.processing_state_repo.mark_processing_failed(
                instrument_id,
                "aggregation",
                {
                    "error": error_message,
                    "timeframes_processed": len(successful_aggregations),
                    "expected_timeframes": expected_timeframes,
                    "failed_at": datetime.now().isoformat(),
                },
            )
            logger.error(f"Aggregation incomplete for instrument {instrument_id}: {error_message}")
            return False
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Mark processing as failed
            error_message = str(e)
            logger.error(f"Aggregation processing failed for {instrument_id}: {error_message}", exc_info=True)
            try:
                await self.processing_state_repo.mark_processing_failed(
                    instrument_id,
                    "aggregation",
                    {
                        "error": error_message,
                        "failed_at": datetime.now().isoformat(),
                    },
                )
            except Exception as inner_e:
                logger.error(f"Failed to mark aggregation processing as failed: {inner_e}")
            return False

    async def _process_features_smart(self, instrument_id: int, components: HistoricalPipelineDependencies) -> bool:
        """
        Smart feature processing.
        Uses exact same logic as main.py but with state tracking.

        Returns:
            bool: True if feature processing was successful, False otherwise
        """
        feature_calculator = components.feature_calculator
        ohlcv_repo = components.ohlcv_repo

        # Mark processing as in progress
        try:
            await self.processing_state_repo.mark_processing_in_progress(
                instrument_id, "features", {"started_at": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Failed to mark feature processing as in-progress: {e}")
            # Continue processing even if we can't update the state

        try:
            successful_features = 0
            expected_features = 0
            # Same loop logic as main.py
            for timeframe in self.config.trading.feature_timeframes:
                # Validate timeframe to prevent injection
                if not self._validate_timeframe(str(timeframe)):
                    logger.error(f"Invalid timeframe value: {timeframe}. Skipping.")
                    continue
                timeframe_str = f"{timeframe}min"
                expected_features += 1

                latest_candle = await ohlcv_repo.get_latest_candle(instrument_id, timeframe_str)
                if not latest_candle:
                    logger.warning(f"No {timeframe_str} data found for {instrument_id}. Skipping feature calculation.")
                    continue

                # Same feature calculation call as main.py
                try:
                    feature_results = await feature_calculator.calculate_and_store_features(
                        instrument_id, latest_candle.ts
                    )

                    # Safely check results - handle cases where results might be exceptions or malformed
                    successful_count = 0
                    for r in feature_results:
                        try:
                            if hasattr(r, "success") and r.success:
                                successful_count += 1
                        except Exception as result_error:
                            logger.warning(f"Error checking feature result for {timeframe_str}: {result_error}")
                            continue

                    successful_features += 1 if successful_count > 0 else 0
                    logger.info(f"Calculated {successful_count} features for {timeframe_str}")

                except Exception as calc_error:
                    logger.error(f"Feature calculation failed for {timeframe_str}: {calc_error}")
                    # Continue with next timeframe instead of failing completely
                    continue

            # Mark completion
            if successful_features > 0:
                await self.processing_state_repo.mark_processing_complete(
                    instrument_id,
                    "features",
                    {"features_calculated": successful_features, "processing_time": datetime.now().isoformat()},
                )
                self.system_state.mark_processing_complete(instrument_id, "features")
                logger.info(f"Successfully completed feature calculation for instrument {instrument_id}")
                return True
            error_message = f"No features were successfully calculated for instrument {instrument_id}"
            logger.error(error_message)
            await self.processing_state_repo.mark_processing_failed(
                instrument_id,
                "features",
                {
                    "error": error_message,
                    "failed_at": datetime.now().isoformat(),
                },
            )
            return False
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Mark processing as failed
            error_message = str(e)
            logger.error(f"Feature processing failed for {instrument_id}: {error_message}", exc_info=True)
            try:
                await self.processing_state_repo.mark_processing_failed(
                    instrument_id,
                    "features",
                    {
                        "error": error_message,
                        "failed_at": datetime.now().isoformat(),
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as inner_e:
                logger.error(f"Failed to mark feature processing as failed: {inner_e}")
            return False

    async def _process_labeling_smart(
        self, instrument_id: int, configured_inst: Any, components: HistoricalPipelineDependencies
    ) -> bool:
        """
        Smart labeling processing.
        Uses exact same logic as main.py but with state tracking.

        Returns:
            bool: True if labeling was successful, False otherwise
        """
        labeler = components.labeler
        ohlcv_repo = components.ohlcv_repo

        # Mark processing as in progress
        try:
            await self.processing_state_repo.mark_processing_in_progress(
                instrument_id, "labeling", {"started_at": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Failed to mark labeling processing as in-progress: {e}")
            # Continue processing even if we can't update the state

        try:
            symbol = configured_inst.tradingsymbol
            successful_labelings = 0
            expected_labelings = 0

            # Same loop logic as main.py
            for timeframe in self.config.trading.labeling_timeframes:
                # Validate timeframe to prevent injection
                if not self._validate_timeframe(str(timeframe)):
                    logger.error(f"Invalid timeframe value: {timeframe}. Skipping.")
                    continue
                timeframe_str = f"{timeframe}min"
                expected_labelings += 1

                # Same data fetching as main.py
                ohlcv_data = await ohlcv_repo.get_ohlcv_data_for_features(
                    instrument_id, timeframe, self.config.trading.labeling.ohlcv_data_limit_for_labeling
                )

                if len(ohlcv_data) < self.config.trading.labeling.minimum_ohlcv_data_for_labeling:
                    logger.warning(f"Insufficient {timeframe_str} data for labeling {symbol}. Skipping.")
                    continue

                # Same DataFrame preparation as main.py
                labeling_df = pd.DataFrame([o.model_dump() for o in ohlcv_data])
                labeling_df["timestamp"] = labeling_df["ts"]

                # Same labeling call as main.py
                labeling_stats = await labeler.process_symbol(instrument_id, symbol, labeling_df, timeframe_str)

                if labeling_stats:
                    logger.info(
                        f"Successfully labeled {labeling_stats.labeled_bars} bars for {symbol} ({timeframe_str})"
                    )
                    successful_labelings += 1
                else:
                    logger.warning(f"Failed to label data for {symbol} ({timeframe_str})")

            # Mark completion
            if successful_labelings > 0:
                await self.processing_state_repo.mark_processing_complete(
                    instrument_id,
                    "labeling",
                    {"timeframes_labeled": successful_labelings, "processing_time": datetime.now().isoformat()},
                )
                self.system_state.mark_processing_complete(instrument_id, "labeling")
                logger.info(f"Successfully completed labeling for instrument {instrument_id}")
                return True
            error_message = f"No timeframes were successfully labeled for instrument {instrument_id}"
            logger.error(error_message)
            await self.processing_state_repo.mark_processing_failed(
                instrument_id,
                "labeling",
                {
                    "error": error_message,
                    "failed_at": datetime.now().isoformat(),
                },
            )
            return False
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Mark processing as failed
            error_message = str(e)
            logger.error(f"Labeling processing failed for {instrument_id}: {error_message}", exc_info=True)
            try:
                await self.processing_state_repo.mark_processing_failed(
                    instrument_id,
                    "labeling",
                    {
                        "error": error_message,
                        "failed_at": datetime.now().isoformat(),
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as inner_e:
                logger.error(f"Failed to mark labeling processing as failed: {inner_e}")
            return False

    async def _process_training_smart(
        self, instrument_id: int, configured_inst: Any, components: HistoricalPipelineDependencies
    ) -> bool:
        """
        Smart model training processing.
        Uses exact same logic as main.py but with state tracking.

        Returns:
            bool: True if training was successful, False otherwise
        """
        lgbm_trainer = components.lgbm_trainer
        symbol = configured_inst.tradingsymbol

        # Mark processing as in progress
        try:
            await self.processing_state_repo.mark_processing_in_progress(
                instrument_id, "training", {"started_at": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Failed to mark training processing as in-progress: {e}")
            # Continue processing even if we can't update the state

        try:
            successful_models = 0
            expected_models = 0
            # Same training loop as main.py
            for timeframe in self.config.trading.labeling_timeframes:
                # Validate timeframe to prevent injection
                if not self._validate_timeframe(str(timeframe)):
                    logger.error(f"Invalid timeframe value: {timeframe}. Skipping.")
                    continue
                timeframe_str = f"{timeframe}min"
                expected_models += 1

                # Same training call as main.py
                model_path = await lgbm_trainer.train_model(
                    instrument_id, timeframe_str, optimize_hyperparams=self.config.model_training.optimize_hyperparams
                )

                if model_path:
                    logger.info(f"Successfully trained model for {symbol} ({timeframe_str}): {model_path}")
                    successful_models += 1
                else:
                    logger.warning(f"Failed to train model for {symbol} ({timeframe_str})")

            # Mark completion
            if successful_models > 0:
                await self.processing_state_repo.mark_processing_complete(
                    instrument_id,
                    "training",
                    {
                        "processing_time": datetime.now().isoformat(),
                        "successful_models": successful_models,
                        "timeframes": [f"{tf}min" for tf in self.config.trading.labeling_timeframes],
                    },
                )
                self.system_state.mark_processing_complete(instrument_id, "training")
                logger.info(f"Successfully completed model training for instrument {instrument_id}")
                return True
            error_message = f"No models were successfully trained for instrument {instrument_id}"
            logger.error(error_message)
            await self.processing_state_repo.mark_processing_failed(
                instrument_id,
                "training",
                {
                    "error": error_message,
                    "failed_at": datetime.now().isoformat(),
                },
            )
            return False
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Mark processing as failed
            error_message = str(e)
            logger.error(f"Training processing failed for {instrument_id}: {error_message}", exc_info=True)
            try:
                await self.processing_state_repo.mark_processing_failed(
                    instrument_id,
                    "training",
                    {
                        "error": error_message,
                        "failed_at": datetime.now().isoformat(),
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as inner_e:
                logger.error(f"Failed to mark training processing as failed: {inner_e}")
            return False

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
        # Overall processing summary - create from available data
        all_states = await self.processing_state_repo.get_all_processing_states()
        summary: dict[str, Any] = {
            "total_processing_records": len(all_states),
            "processing_by_type": dict[str, int](),
            "processing_by_status": dict[str, int](),
        }

        # Group by process type and status
        for state in all_states:
            process_type = state.processing_type
            state_status = state.status

            if process_type not in summary["processing_by_type"]:
                summary["processing_by_type"][process_type] = 0
            summary["processing_by_type"][process_type] += 1

            if state_status not in summary["processing_by_status"]:
                summary["processing_by_status"][state_status] = 0
            summary["processing_by_status"][state_status] += 1

        return summary

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
