"""API endpoint for system status and observability."""

from typing import Any

from fastapi import APIRouter, Depends

from src.state.system_state import SystemState
from src.utils.scheduler import Scheduler

router = APIRouter()


@router.get("/status")
def get_system_status(
    system_state: SystemState = Depends(SystemState), scheduler: Scheduler = Depends(Scheduler)
) -> dict[str, Any]:
    """Returns the current status of the system."""
    return {
        "system_status": system_state.get_system_status(),
        "scheduler_status": scheduler.get_status(),
        "component_health": system_state.get_component_health(),
        "processing_summary": system_state.get_processing_summary(),
    }
