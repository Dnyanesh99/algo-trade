import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.api.chart_routes import router as chart_router
from src.api.dependencies import set_websocket_manager
from src.api.main import app, websocket_manager
from src.api.routes.configuration import router as config_router
from src.api.routes.control import router as control_router
from src.api.routes.data import router as data_router
from src.api.routes.status import router as status_router
from src.api.routes.system import router as system_router
from src.utils.logger import logger

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Custom middleware for error handling and logging
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled error in {request.method} {request.url}: {e}")
            return JSONResponse(status_code=500, content={"error": "Internal server error", "message": str(e)})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        start_time = asyncio.get_event_loop().time()

        # Log request
        logger.info(f"{request.method} {request.url}")

        response = await call_next(request)

        # Log response
        process_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")

        return response


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail, "status_code": exc.status_code})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(status_code=422, content={"error": "Validation error", "details": exc.errors()})


# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Include routers
app.include_router(system_router)
app.include_router(config_router)
app.include_router(data_router)
app.include_router(chart_router)
app.include_router(status_router)
app.include_router(control_router)

# Setup dependencies
set_websocket_manager(websocket_manager)

# Serve React build files
if os.path.exists("ui/dist"):
    app.mount("/static", StaticFiles(directory="ui/dist/static"), name="static")

    @app.get("/")
    async def serve_frontend() -> FileResponse:
        return FileResponse("ui/dist/index.html")

    # Catch-all route for React Router
    @app.get("/{path:path}")
    async def catch_all(path: str) -> Any:
        # Don't serve index.html for API routes
        if path.startswith(("api/", "ws")):
            return JSONResponse({"error": "Not found"}, status_code=404)
        return FileResponse("ui/dist/index.html")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    return app


async def start_server() -> None:
    """Start the FastAPI server"""
    try:
        logger.info("Starting Algo Trading Control Panel API server...")

        # Load configuration to get the host
        from src.utils.config_loader import ConfigLoader

        config_loader = ConfigLoader()
        api_config = config_loader.get_config()
        host = api_config.api.host if api_config.api is not None else "127.0.0.1"

        # Configuration
        config = uvicorn.Config(
            app=create_app(),
            host=host,
            port=8000,
            log_level="info",
            reload=False,  # Disable reload in production
            access_log=True,
            loop="asyncio",
        )

        # Create and start server
        server = uvicorn.Server(config)
        await server.serve()

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


def main() -> None:
    """Main entry point"""
    try:
        # Run the server
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
