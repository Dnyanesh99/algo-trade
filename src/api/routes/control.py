"""
Secure, production-grade API for runtime system control and observability.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.database.instrument_repo import InstrumentRepository
from src.state.system_state import SystemState
from src.utils.config_loader import ConfigLoader
from src.utils.historical_pipeline_manager import HistoricalPipelineManager

router = APIRouter(
    prefix="/api/v1/control",
    tags=["System Control"],
)


# --- Dependency Injection --- #
def get_config_loader() -> ConfigLoader:
    return ConfigLoader()


def get_system_state() -> SystemState:
    return SystemState()


def get_historical_pipeline_manager() -> HistoricalPipelineManager:
    # This is a placeholder. In a real app, this would be a singleton
    # managed by a proper dependency injection container.
    # For now, we re-initialize it, but it will share state via SystemState singleton.
    from src.database.db_utils import db_manager
    from src.database.processing_state_repo import ProcessingStateRepository
    from src.signal.alert_system import AlertSystem
    from src.state.error_handler import ErrorHandler
    from src.state.health_monitor import HealthMonitor

    config = get_config_loader().get_config()
    return HistoricalPipelineManager(
        system_state=get_system_state(),
        health_monitor=HealthMonitor(get_system_state(), None, db_manager, config.health_monitor, config.system),
        error_handler=ErrorHandler(get_system_state(), AlertSystem(), config.error_handler),
        processing_state_repo=ProcessingStateRepository(),
        config=config,
    )


# --- API Endpoints --- #


@router.get("/status")
async def get_full_system_status(state: SystemState = Depends(get_system_state)) -> dict[str, Any]:
    """Get a comprehensive, unified status of the entire system."""
    return state.get_processing_summary()


@router.post("/reprocess")
async def reprocess_instrument(
    instrument_symbol: str,
    step: str,
    manager: HistoricalPipelineManager = Depends(get_historical_pipeline_manager),
    instrument_repo: InstrumentRepository = Depends(InstrumentRepository),
) -> dict[str, str]:
    """Trigger a reprocessing job for a specific instrument and step."""
    instrument = await instrument_repo.get_instrument_by_tradingsymbol(instrument_symbol, "NSE")
    if not instrument:
        raise HTTPException(status_code=404, detail=f"Instrument '{instrument_symbol}' not found.")

    await manager.reset_processing_state(instrument.instrument_id, step)
    return {
        "message": f"Successfully scheduled '{step}' for reprocessing on instrument '{instrument_symbol}' on the next run."
    }


@router.get("/config")
async def get_current_config(config_loader: ConfigLoader = Depends(get_config_loader)) -> dict[str, Any]:
    """Retrieve the current, live application configuration."""
    return config_loader.get_config().model_dump()


@router.put("/config")
async def update_config(
    section: str, key: str, value: str, config_loader: ConfigLoader = Depends(get_config_loader)
) -> dict[str, str]:
    """Update a specific key in the application's configuration at runtime."""
    try:
        # A more robust implementation would have a detailed mapping of key to type.
        # For now, we attempt to cast to the correct type.
        current_config = config_loader.get_config_section(section)
        original_value = current_config.get(key)

        if original_value is None:
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found in section '{section}'.")

        typed_value = type(original_value)(value)

        config_loader.update_config_section(section, {key: typed_value})
        return {"message": f"Successfully updated {section}.{key} to {value}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {e}") from e
