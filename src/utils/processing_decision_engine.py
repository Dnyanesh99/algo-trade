"""
Production-grade unified processing decision engine.

Eliminates redundancy across _should_process_* methods with a single,
robust, configurable decision framework that handles all processing stages.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.database.processing_state_repo import ProcessingStateRepository
from src.utils.config_loader import AppConfig
from src.utils.env_utils import get_bool_env
from src.utils.logger import LOGGER as logger


class ProcessingStage(Enum):
    """Enumeration of all processing stages in dependency order."""

    HISTORICAL_FETCH = "historical_fetch"
    AGGREGATION = "aggregation"
    FEATURES = "features"
    LABELING = "labeling"
    TRAINING = "training"


@dataclass
class ProcessingDecision:
    """Result of processing decision with detailed reasoning."""

    should_process: bool
    reason: str
    stage: ProcessingStage
    instrument_id: int
    bypass_reason: Optional[str] = None
    data_sufficient: Optional[bool] = None


class ProcessingDecisionEngine:
    """
    Unified production-grade decision engine for all processing stages.

    Eliminates redundant _should_process_* methods by providing a single,
    configurable framework that handles environment variables, state checks,
    dependency validation, and data sufficiency analysis.

    Key Features:
    - Single point of truth for processing decisions
    - Configurable via environment variables
    - Robust error handling and logging
    - Force reprocess capability
    - Dependency chain validation
    - Data sufficiency analysis
    """

    # Environment variable mapping for each stage
    ENV_VARIABLE_MAP: dict[ProcessingStage, str] = {
        ProcessingStage.HISTORICAL_FETCH: "HISTORICAL_PROCESSING_ENABLED",
        ProcessingStage.AGGREGATION: "AGGREGATION_PROCESSING_ENABLED",
        ProcessingStage.FEATURES: "FEATURE_PROCESSING_ENABLED",
        ProcessingStage.LABELING: "LABELING_PROCESSING_ENABLED",
        ProcessingStage.TRAINING: "TRAINING_PROCESSING_ENABLED",
    }

    def __init__(
        self,
        processing_state_repo: ProcessingStateRepository,
        config: AppConfig,
        force_reprocess: bool = False,
    ):
        """
        Initialize the decision engine.

        Args:
            processing_state_repo: Repository for state management
            config: Application configuration
            force_reprocess: Whether to force reprocessing regardless of completion state
        """
        self.processing_state_repo = processing_state_repo
        self.config = config
        self.force_reprocess = force_reprocess

        logger.info("ProcessingDecisionEngine initialized with unified decision framework")

    async def should_process(
        self, stage: ProcessingStage, instrument_id: int, enable_data_checks: bool = True
    ) -> ProcessingDecision:
        """
        Unified processing decision for any stage.

        This single method replaces all _should_process_* methods with consistent logic:
        1. Check if stage is enabled via environment variable
        2. Check completion state (respect force_reprocess flag)
        3. Validate dependencies if applicable
        4. Check data sufficiency if enabled

        Args:
            stage: The processing stage to evaluate
            instrument_id: The instrument ID to check
            enable_data_checks: Whether to perform data sufficiency checks

        Returns:
            ProcessingDecision with detailed reasoning
        """
        try:
            # 1. Check if processing is enabled via environment variable
            env_var = self.ENV_VARIABLE_MAP[stage]
            if not get_bool_env(env_var, True):
                return ProcessingDecision(
                    should_process=False,
                    reason=f"{stage.value} processing disabled via {env_var}=false",
                    stage=stage,
                    instrument_id=instrument_id,
                )

            # 2. Check completion state (with force reprocess handling)
            completion_decision = await self._check_completion_state(stage, instrument_id)
            if not completion_decision.should_process:
                return completion_decision

            # 3. Validate dependencies for stages that have them
            if stage != ProcessingStage.HISTORICAL_FETCH:
                dependency_decision = await self._validate_dependencies(stage, instrument_id)
                if not dependency_decision.should_process:
                    return dependency_decision

            # 4. Check data sufficiency if enabled
            if enable_data_checks:
                data_decision = await self._check_data_sufficiency(stage, instrument_id)
                if not data_decision.should_process:
                    return data_decision

            # All checks passed
            return ProcessingDecision(
                should_process=True,
                reason=f"{stage.value} should proceed - all conditions satisfied",
                stage=stage,
                instrument_id=instrument_id,
                data_sufficient=True,
            )

        except Exception as e:
            logger.error(f"Error in processing decision for {stage.value} on {instrument_id}: {e}")
            # Fail-safe: block processing on decision errors
            return ProcessingDecision(
                should_process=False,
                reason=f"Decision error: {str(e)} - blocking for safety",
                stage=stage,
                instrument_id=instrument_id,
            )

    async def _check_completion_state(self, stage: ProcessingStage, instrument_id: int) -> ProcessingDecision:
        """Check if processing is already complete, respecting force_reprocess flag."""
        try:
            is_complete = await self.processing_state_repo.is_processing_complete(instrument_id, stage.value)

            if is_complete and not self.force_reprocess:
                return ProcessingDecision(
                    should_process=False,
                    reason=f"{stage.value} already completed for {instrument_id}",
                    stage=stage,
                    instrument_id=instrument_id,
                )

            if is_complete and self.force_reprocess:
                return ProcessingDecision(
                    should_process=True,
                    reason=f"FORCE_REPROCESS=true - reprocessing {stage.value} despite completion",
                    stage=stage,
                    instrument_id=instrument_id,
                    bypass_reason="force_reprocess",
                )

            # Not complete, should process
            return ProcessingDecision(
                should_process=True, reason=f"{stage.value} not yet completed", stage=stage, instrument_id=instrument_id
            )

        except Exception as e:
            logger.error(f"Error checking completion state for {stage.value}: {e}")
            # Continue processing if we can't check state
            return ProcessingDecision(
                should_process=True,
                reason="Could not verify completion state - proceeding with caution",
                stage=stage,
                instrument_id=instrument_id,
            )

    async def _validate_dependencies(self, stage: ProcessingStage, instrument_id: int) -> ProcessingDecision:
        """Validate processing dependencies using the repository's validation logic."""
        try:
            dependencies_satisfied = await self.processing_state_repo.validate_processing_dependencies(
                instrument_id, stage.value
            )

            if not dependencies_satisfied:
                return ProcessingDecision(
                    should_process=False,
                    reason=f"Dependencies not satisfied for {stage.value} on {instrument_id}",
                    stage=stage,
                    instrument_id=instrument_id,
                )

            return ProcessingDecision(
                should_process=True,
                reason=f"Dependencies satisfied for {stage.value}",
                stage=stage,
                instrument_id=instrument_id,
            )

        except Exception as e:
            logger.error(f"Error validating dependencies for {stage.value}: {e}")
            # Fail-safe: block processing on dependency check failure
            return ProcessingDecision(
                should_process=False,
                reason=f"Dependency validation failed - blocking for safety: {str(e)}",
                stage=stage,
                instrument_id=instrument_id,
            )

    async def _check_data_sufficiency(self, stage: ProcessingStage, instrument_id: int) -> ProcessingDecision:
        """Check if sufficient data exists for processing (used for smart skipping)."""
        try:
            # Special handling for historical_fetch - check actual data existence
            if stage == ProcessingStage.HISTORICAL_FETCH:
                has_sufficient_data = await self.processing_state_repo.has_actual_data_for_processing(
                    instrument_id, stage.value, self.config
                )

                if has_sufficient_data and not self.force_reprocess:
                    # Mark as complete since data exists
                    await self.processing_state_repo.mark_processing_complete(
                        instrument_id, stage.value, {"reason": "sufficient_existing_data", "auto_completed": True}
                    )

                    return ProcessingDecision(
                        should_process=False,
                        reason=f"Sufficient {stage.value} data already exists - marked complete",
                        stage=stage,
                        instrument_id=instrument_id,
                        data_sufficient=True,
                    )

            # For other stages, data sufficiency is checked as part of dependency validation
            return ProcessingDecision(
                should_process=True,
                reason=f"Data sufficiency check passed for {stage.value}",
                stage=stage,
                instrument_id=instrument_id,
                data_sufficient=True,
            )

        except Exception as e:
            logger.error(f"Error checking data sufficiency for {stage.value}: {e}")
            # Continue processing if we can't check data sufficiency
            return ProcessingDecision(
                should_process=True,
                reason="Could not verify data sufficiency - proceeding",
                stage=stage,
                instrument_id=instrument_id,
            )

    async def get_processing_plan(self, instrument_id: int) -> dict[ProcessingStage, ProcessingDecision]:
        """
        Get complete processing plan for an instrument across all stages.

        Args:
            instrument_id: The instrument to analyze

        Returns:
            dictionary mapping each stage to its processing decision
        """
        plan = {}

        for stage in ProcessingStage:
            decision = await self.should_process(stage, instrument_id)
            plan[stage] = decision

        logger.info(f"Generated processing plan for instrument {instrument_id}")
        return plan

    def log_decision(self, decision: ProcessingDecision, level: str = "info") -> None:
        """
        Log processing decision with appropriate emoji and formatting.

        Args:
            decision: The processing decision to log
            level: Log level (info, warning, error)
        """
        emoji = "✅" if decision.should_process else "⏭️"
        action = "Processing" if decision.should_process else "Skipping"

        log_message = (
            f"{emoji} {action} {decision.stage.value} for instrument {decision.instrument_id}: {decision.reason}"
        )

        if decision.bypass_reason:
            log_message += f" (bypass: {decision.bypass_reason})"

        if level == "warning":
            logger.warning(log_message)
        elif level == "error":
            logger.error(log_message)
        else:
            logger.info(log_message)
