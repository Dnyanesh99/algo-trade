"""
System control and monitoring routes
"""

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.utils.logger import logger

if TYPE_CHECKING:
    from src.utils.config_loader import AppConfig

from ..dependencies import get_app_state, get_websocket_manager
from ..models import (
    AlertData,
    HealthStatus,
    SuccessResponse,
    SystemControlRequest,
    SystemModeRequest,
    SystemStatus,
)

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/status", response_model=SystemStatus)
async def get_system_status(app_state: dict[str, Any] = Depends(get_app_state)) -> SystemStatus:
    """Get comprehensive system status"""
    try:
        system_state = app_state["system_state"]
        health_monitor = app_state["health_monitor"]
        config: AppConfig = app_state["config"]

        # Calculate uptime
        uptime = datetime.now() - system_state.start_time
        uptime_str = f"{uptime.days}d {uptime.seconds // 3600}h {(uptime.seconds % 3600) // 60}m"

        # Get component health
        components = health_monitor.get_component_status()
        health_score = health_monitor.get_overall_health_score()
        memory_usage = health_monitor.get_memory_usage()

        return SystemStatus(
            mode=config.system.mode,
            version=config.system.version,
            timezone=config.system.timezone,
            uptime=uptime_str,
            status=system_state.status,
            health_score=health_score,
            components=components,
            memory_usage=memory_usage,
            last_updated=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/mode", response_model=SuccessResponse)
async def set_system_mode(
    request: SystemModeRequest,
    app_state: dict[str, Any] = Depends(get_app_state),
    websocket_manager: Any = Depends(get_websocket_manager),
) -> SuccessResponse:
    """Set system operation mode"""
    try:
        config: AppConfig = app_state["config"]
        system_state = app_state["system_state"]

        # Validate mode transition
        current_mode = config.system.mode
        if current_mode == request.mode:
            return SuccessResponse(
                success=True, message=f"System is already in {request.mode}", data={"mode": request.mode}
            )

        # Check if system is running
        if system_state.status == "running":
            return SuccessResponse(success=False, message="Cannot change mode while system is running", data={})

        # Update configuration
        config.system.mode = cast("Literal['HISTORICAL_MODE', 'LIVE_MODE']", request.mode)

        # Save configuration
        from src.utils.config_loader import ConfigLoader

        config_loader = ConfigLoader()
        config_loader.save_config(config)

        # Update system state
        system_state.mode = request.mode
        system_state.last_mode_change = datetime.now()

        # Broadcast to WebSocket clients
        await websocket_manager.broadcast(
            json.dumps({"type": "system_mode_changed", "mode": request.mode, "timestamp": datetime.now().isoformat()})
        )

        logger.info(f"System mode changed from {current_mode} to {request.mode}")

        return SuccessResponse(
            success=True,
            message=f"System mode changed to {request.mode}",
            data={"mode": request.mode, "previous_mode": current_mode},
        )

    except Exception as e:
        logger.error(f"Failed to set system mode: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/start", response_model=SuccessResponse)
async def start_system(
    background_tasks: BackgroundTasks,
    app_state: dict[str, Any] = Depends(get_app_state),
    websocket_manager: Any = Depends(get_websocket_manager),
) -> SuccessResponse:
    """Start the trading system"""
    try:
        system_state = app_state["system_state"]
        config: AppConfig = app_state["config"]

        # Check if system is already running
        if system_state.status == "running":
            return SuccessResponse(success=True, message="System is already running", data={"status": "running"})

        # Validate authentication
        from src.auth.token_manager import TokenManager

        token_manager = TokenManager()
        if not token_manager.is_token_available():
            return SuccessResponse(success=False, message="Authentication required before starting system", data={})

        async def start_system_components() -> None:
            """Start system components in background"""
            try:
                system_state.status = "starting"
                system_state.start_time = datetime.now()

                # Broadcast status change
                await websocket_manager.broadcast(
                    json.dumps(
                        {"type": "system_status_changed", "status": "starting", "timestamp": datetime.now().isoformat()}
                    )
                )

                # Initialize components based on mode
                if config.system.mode == "HISTORICAL_MODE":
                    # Start historical data processing using main.py architecture
                    from main import initialize_historical_mode_enhanced

                    from src.database.instrument_repo import InstrumentRepository
                    from src.signal.alert_system import AlertSystem
                    from src.state.error_handler import ErrorHandler

                    # Start historical mode in background
                    task = asyncio.create_task(
                        initialize_historical_mode_enhanced(
                            config,
                            system_state,
                            AlertSystem(),
                            ErrorHandler(system_state, AlertSystem(), config.error_handler),
                            InstrumentRepository(),
                        )
                    )
                    app_state["background_tasks"].add(task)

                elif config.system.mode == "LIVE_MODE":
                    # Start live data processing using main.py architecture
                    from main import initialize_live_mode

                    from src.broker.instrument_manager import InstrumentManager
                    from src.broker.rest_client import KiteRESTClient
                    from src.database.processing_state_repo import ProcessingStateRepository
                    from src.utils.market_calendar import MarketCalendar

                    if config.market_calendar is None:
                        raise ValueError("Market calendar configuration is missing")

                    # Start live mode in background
                    task = asyncio.create_task(
                        initialize_live_mode(
                            config,
                            system_state,
                            AlertSystem(),
                            ErrorHandler(system_state, AlertSystem(), config.error_handler),
                            InstrumentManager(
                                KiteRESTClient(token_manager, config.broker.api_key),
                                InstrumentRepository(),
                                config.trading,
                            ),
                            ProcessingStateRepository(),
                            MarketCalendar(config.system, config.market_calendar),
                        )
                    )
                    app_state["background_tasks"].add(task)

                    # Live mode components are managed within initialize_live_mode

                # Update system state
                system_state.status = "running"

                # Broadcast status change
                await websocket_manager.broadcast(
                    json.dumps(
                        {"type": "system_status_changed", "status": "running", "timestamp": datetime.now().isoformat()}
                    )
                )

                logger.info(f"System started successfully in {config.system.mode}")

            except Exception as e:
                logger.error(f"Failed to start system: {e}")
                system_state.status = "error"
                system_state.error_message = str(e)

                # Broadcast error
                await websocket_manager.broadcast(
                    json.dumps({"type": "system_error", "error": str(e), "timestamp": datetime.now().isoformat()})
                )

        # Start system in background
        background_tasks.add_task(start_system_components)

        return SuccessResponse(
            success=True, message="System startup initiated", data={"status": "starting", "mode": config.system.mode}
        )

    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/stop", response_model=SuccessResponse)
async def stop_system(
    app_state: dict[str, Any] = Depends(get_app_state), websocket_manager: Any = Depends(get_websocket_manager)
) -> SuccessResponse:
    """Stop the trading system"""
    try:
        system_state = app_state["system_state"]

        # Check if system is running
        if system_state.status != "running":
            return SuccessResponse(success=True, message="System is not running", data={"status": system_state.status})

        # Stop all background tasks
        for task in app_state["background_tasks"]:
            if not task.done():
                task.cancel()

        # Stop connection manager if exists
        if "connection_manager" in app_state:
            await app_state["connection_manager"].disconnect()

        # Stop stream consumer if exists
        if "stream_consumer" in app_state:
            await app_state["stream_consumer"].stop()

        # Update system state
        system_state.status = "stopped"
        system_state.stop_time = datetime.now()

        # Broadcast status change
        await websocket_manager.broadcast(
            json.dumps({"type": "system_status_changed", "status": "stopped", "timestamp": datetime.now().isoformat()})
        )

        logger.info("System stopped successfully")

        return SuccessResponse(success=True, message="System stopped successfully", data={"status": "stopped"})

    except Exception as e:
        logger.error(f"Failed to stop system: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/restart", response_model=SuccessResponse)
async def restart_system(
    background_tasks: BackgroundTasks,
    app_state: dict[str, Any] = Depends(get_app_state),
    websocket_manager: Any = Depends(get_websocket_manager),
) -> SuccessResponse:
    """Restart the trading system"""
    try:
        # Stop system first
        await stop_system(app_state, websocket_manager)

        # Wait a bit before restarting
        await asyncio.sleep(2)

        # Start system again
        return await start_system(background_tasks, app_state, websocket_manager)

    except Exception as e:
        logger.error(f"Failed to restart system: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/health", response_model=list[HealthStatus])
async def get_system_health(app_state: dict[str, Any] = Depends(get_app_state)) -> list[HealthStatus]:
    """Get detailed health status of all system components"""
    try:
        health_monitor = app_state["health_monitor"]
        components = health_monitor.get_component_status()

        result = []
        for component_name, status in components.items():
            result.append(
                HealthStatus(
                    component=component_name,
                    status=status.get("status", "unknown"),
                    health_score=status.get("health_score", 0.0),
                    last_check=status.get("last_check", datetime.now()),
                    error_message=status.get("error_message"),
                    metrics=status.get("metrics", {}),
                )
            )

        return result

    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/alerts", response_model=list[AlertData])
async def get_system_alerts(
    level: Optional[str] = None,
    component: Optional[str] = None,
    limit: int = 100,
    app_state: dict[str, Any] = Depends(get_app_state),
) -> list[AlertData]:
    """Get system alerts"""
    try:
        health_monitor = app_state["health_monitor"]
        alerts = health_monitor.get_alerts(level=level, component=component, limit=limit)

        return [
            AlertData(
                id=alert.get("id"),
                level=alert.get("level"),
                component=alert.get("component"),
                message=alert.get("message"),
                timestamp=alert.get("timestamp"),
                acknowledged=alert.get("acknowledged", False),
                details=alert.get("details"),
            )
            for alert in alerts
        ]

    except Exception as e:
        logger.error(f"Failed to get system alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/alerts/{alert_id}/acknowledge", response_model=SuccessResponse)
async def acknowledge_alert(alert_id: int, app_state: dict[str, Any] = Depends(get_app_state)) -> SuccessResponse:
    """Acknowledge a system alert"""
    try:
        health_monitor = app_state["health_monitor"]
        success = health_monitor.acknowledge_alert(alert_id)

        if success:
            return SuccessResponse(success=True, message=f"Alert {alert_id} acknowledged", data={"alert_id": alert_id})
        raise HTTPException(status_code=404, detail="Alert not found")

    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/logs")
async def get_system_logs(
    level: Optional[str] = None,
    component: Optional[str] = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Get system logs"""
    try:
        # This would typically read from your logging system
        # For now, return a placeholder
        return {"logs": [], "total_count": 0, "filters": {"level": level, "component": component, "limit": limit}}

    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/metrics")
async def get_system_metrics(app_state: dict[str, Any] = Depends(get_app_state)) -> dict[str, Any]:
    """Get system performance metrics"""
    try:
        health_monitor = app_state["health_monitor"]

        # Get various system metrics
        return {
            "cpu_usage": health_monitor.get_cpu_usage(),
            "memory_usage": health_monitor.get_memory_usage(),
            "disk_usage": health_monitor.get_disk_usage(),
            "network_io": health_monitor.get_network_io(),
            "database_connections": health_monitor.get_database_connections(),
            "active_websockets": len(app_state.get("websocket_connections", [])),
            "background_tasks": len(app_state.get("background_tasks", [])),
            "timestamp": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/control", response_model=SuccessResponse)
async def system_control(
    request: SystemControlRequest, app_state: dict[str, Any] = Depends(get_app_state)
) -> SuccessResponse:
    """Execute system control commands"""
    try:
        action = request.action
        config: AppConfig = app_state["config"]

        if action == "clear_cache":
            # Clear system caches
            # Implementation depends on your caching system
            return SuccessResponse(success=True, message="System cache cleared", data={"action": action})

        if action == "reload_config":
            # Reload configuration
            from src.utils.config_loader import ConfigLoader

            config_loader = ConfigLoader()
            new_config = config_loader.reload_config()
            app_state["config"] = new_config

            return SuccessResponse(success=True, message="Configuration reloaded", data={"action": action})

        if action == "sync_instruments":
            # Sync instruments from broker
            from src.broker.instrument_manager import InstrumentManager
            from src.broker.rest_client import KiteRESTClient
            from src.database.instrument_repo import InstrumentRepository

            instrument_manager = InstrumentManager(
                KiteRESTClient(app_state["token_manager"], config.broker.api_key),
                InstrumentRepository(),
                config.trading,
            )
            await instrument_manager.sync_instruments()
            # Get instrument count from repository
            instrument_repo = InstrumentRepository()
            all_instruments = await instrument_repo.get_all_instruments()
            count = len(all_instruments)

            return SuccessResponse(
                success=True, message=f"Synced instruments. Total: {count}", data={"action": action, "count": count}
            )

        return SuccessResponse(success=False, message=f"Unknown action: {action}", data={"action": action})

    except Exception as e:
        logger.error(f"Failed to execute system control: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
