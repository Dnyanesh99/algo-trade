"""
FastAPI Backend for Algo Trading Control Panel
"""

import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.integrations import TradingSystemIntegration, get_integration
from src.api.models import (
    InstrumentData,
    SignalData,
    SystemStatus,
)
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and cleanup application lifecycle"""
    integration = await get_integration()
    app.state.integration = integration
    yield
    await integration.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Algo Trading Control Panel API",
    description="Backend API for managing algorithmic trading system",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Allow React/Vite dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (React/Vite build)
if os.path.exists("ui/dist"):
    app.mount("/", StaticFiles(directory="ui/dist", html=True), name="static")


class WebSocketManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")


websocket_manager = WebSocketManager()

# --- API Routes --- #


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}


@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status() -> SystemStatus:
    """Get current system status"""
    integration: TradingSystemIntegration = app.state.integration
    if not integration.system_state or not integration.config:
        raise HTTPException(status_code=503, detail="System not fully initialized")

    return SystemStatus(
        mode=integration.config.system.mode,
        version=integration.config.system.version,
        timezone=integration.config.system.timezone,
        uptime="N/A",
        status=integration.system_state.get_system_status(),
        health_score=0.0,  # Placeholder
        components={},
        memory_usage={},
        last_updated=datetime.now(),
    )


@app.get("/api/config", response_model=dict[str, Any])
async def get_configuration() -> dict[str, Any]:
    """Get complete system configuration"""
    integration: TradingSystemIntegration = app.state.integration
    if not integration.config:
        raise HTTPException(status_code=503, detail="Configuration not loaded")

    config_dict = integration.config.model_dump()
    # Remove sensitive data
    if "broker" in config_dict:
        config_dict["broker"].pop("api_key", None)
        config_dict["broker"].pop("api_secret", None)
    return config_dict


@app.get("/api/instruments", response_model=list[InstrumentData])
async def get_instruments() -> list[InstrumentData]:
    """Get all instruments"""
    integration: TradingSystemIntegration = app.state.integration
    instrument_repo = integration.repositories.get("instrument")
    if not instrument_repo:
        raise HTTPException(status_code=503, detail="Instrument repository not available")

    instruments = await instrument_repo.get_all_instruments()
    return [InstrumentData.model_validate(inst) for inst in instruments]


@app.get("/api/signals", response_model=list[SignalData])
async def get_signals(limit: int = 100) -> list[SignalData]:
    """Get recent signals"""
    integration: TradingSystemIntegration = app.state.integration
    signal_repo = integration.repositories.get("signal")
    if not signal_repo:
        raise HTTPException(status_code=503, detail="Signal repository not available")

    signals = await signal_repo.get_all_signals(limit=limit)
    return [SignalData.model_validate(sig) for sig in signals]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    # Load configuration to get the host
    from src.utils.config_loader import ConfigLoader

    config = ConfigLoader().get_config()
    host = config.api.host if hasattr(config, "api") and hasattr(config.api, "host") else "127.0.0.1"

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=8000,
        reload=True,
        reload_dirs=["src/api"],
    )
