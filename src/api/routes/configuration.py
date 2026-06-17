"""
Configuration management routes
"""

import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.utils.config_loader import AppConfig, ConfigLoader, FeatureConfiguration
from src.utils.logger import logger

from ..dependencies import get_app_state, get_websocket_manager
from ..models import ConfigSection, ConfigUpdateRequest, IndicatorUpdateRequest, SuccessResponse

router = APIRouter(prefix="/api/config", tags=["configuration"])


@router.get("/", response_model=dict[str, Any])
async def get_full_configuration(app_state: dict[str, Any] = Depends(get_app_state)) -> dict[str, Any]:
    """Get complete system configuration"""
    try:
        config: AppConfig = app_state["config"]
        return config.model_dump()
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{section}", response_model=ConfigSection)
async def get_config_section(section: str, app_state: dict[str, Any] = Depends(get_app_state)) -> ConfigSection:
    """Get specific configuration section"""
    try:
        config: AppConfig = app_state["config"]

        if not hasattr(config, section):
            raise HTTPException(status_code=404, detail=f"Configuration section '{section}' not found")

        section_data = getattr(config, section)

        # Convert to dict if it's a Pydantic model
        if hasattr(section_data, "model_dump"):
            data = section_data.model_dump()
        elif hasattr(section_data, "dict"):
            data = section_data.dict()
        else:
            data = section_data

        return ConfigSection(section=section, data=data, last_modified=datetime.now())
    except Exception as e:
        logger.error(f"Failed to get config section {section}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/{section}", response_model=SuccessResponse)
async def update_config_section(
    section: str,
    request: ConfigUpdateRequest,
    app_state: dict[str, Any] = Depends(get_app_state),
    websocket_manager: Any = Depends(get_websocket_manager),
) -> SuccessResponse:
    """Update specific configuration section"""
    try:
        config: AppConfig = app_state["config"]

        if not hasattr(config, section):
            raise HTTPException(status_code=404, detail=f"Configuration section '{section}' not found")

        # Get current section
        section_obj = getattr(config, section)

        # Validate configuration if requested
        if request.validate_only:
            # Perform validation without saving
            try:
                # Create a temporary copy for validation
                temp_config = config.model_copy(deep=True)
                temp_section = getattr(temp_config, section)

                # Apply updates to temp config
                for key, value in request.data.items():
                    if hasattr(temp_section, key):
                        setattr(temp_section, key, value)

                # Validate the temp config
                temp_config.model_validate(temp_config)

                return SuccessResponse(
                    success=True,
                    message=f"Configuration section '{section}' validation passed",
                    data={"section": section, "valid": True},
                )
            except Exception as validation_error:
                return SuccessResponse(
                    success=False, message="Validation failed", data={"error": str(validation_error)}
                )

        # Update configuration
        old_values = {}
        for key, value in request.data.items():
            if hasattr(section_obj, key):
                old_values[key] = getattr(section_obj, key)
                setattr(section_obj, key, value)
            else:
                logger.warning(f"Unknown configuration key: {section}.{key}")

        # Save configuration using enhanced save method
        config_loader = ConfigLoader()

        # Create backup before saving
        try:
            backup_path = config_loader.backup_config()
            logger.info(f"Created configuration backup: {backup_path}")
        except Exception as backup_error:
            logger.warning(f"Failed to create backup: {backup_error}")

        # Save configuration
        config_loader.save_config(config)

        # Broadcast update to WebSocket clients
        await websocket_manager.broadcast(
            json.dumps(
                {
                    "type": "config_updated",
                    "section": section,
                    "data": request.data,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        )

        logger.info(f"Configuration section '{section}' updated")

        return SuccessResponse(
            success=True,
            message=f"Configuration section '{section}' updated successfully",
            data={"section": section, "updated_keys": list(request.data.keys()), "old_values": old_values},
        )

    except Exception as e:
        logger.error(f"Failed to update config section {section}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/export")
async def export_configuration(app_state: dict[str, Any] = Depends(get_app_state)) -> JSONResponse:
    """Export complete configuration as JSON"""
    try:
        config: AppConfig = app_state["config"]
        config_dict = config.model_dump()

        # Add metadata
        config_dict["_metadata"] = {
            "exported_at": datetime.now().isoformat(),
            "version": config.system.version,
            "export_type": "full_configuration",
        }

        return JSONResponse(
            content=config_dict,
            headers={
                "Content-Disposition": f"attachment; filename=config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            },
        )

    except Exception as e:
        logger.error(f"Failed to export configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/import", response_model=SuccessResponse)
async def import_configuration(
    file: UploadFile = File(...),
    app_state: dict[str, Any] = Depends(get_app_state),
    websocket_manager: Any = Depends(get_websocket_manager),
) -> SuccessResponse:
    """Import configuration from JSON file"""
    try:
        # Read file content
        content = await file.read()

        # Parse JSON
        try:
            config_data = json.loads(content.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}") from e

        # Remove metadata if present
        if "_metadata" in config_data:
            del config_data["_metadata"]

        # Validate configuration structure
        config_loader = ConfigLoader()

        try:
            # Basic validation - ensure required sections exist
            if "system" not in config_data:
                raise ValueError("System configuration is required")
        except Exception as validation_error:
            raise HTTPException(
                status_code=400, detail=f"Configuration validation failed: {validation_error}"
            ) from validation_error

        # Update current configuration
        current_config: AppConfig = app_state["config"]

        # Update each section
        updated_sections = []
        for section_name, section_data in config_data.items():
            if hasattr(current_config, section_name):
                section_obj = getattr(current_config, section_name)

                # Update section attributes
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

                updated_sections.append(section_name)

        # Save configuration
        config_loader.save_config(current_config)

        # Broadcast update to WebSocket clients
        await websocket_manager.broadcast(
            json.dumps(
                {"type": "config_imported", "sections": updated_sections, "timestamp": datetime.now().isoformat()}
            )
        )

        logger.info(f"Configuration imported from {file.filename}")

        return SuccessResponse(
            success=True,
            message="Configuration imported successfully",
            data={
                "filename": file.filename,
                "updated_sections": updated_sections,
                "total_sections": len(updated_sections),
            },
        )

    except Exception as e:
        logger.error(f"Failed to import configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{section}/{key}", response_model=SuccessResponse)
async def delete_config_key(
    section: str,
    key: str,
    app_state: dict[str, Any] = Depends(get_app_state),
    websocket_manager: Any = Depends(get_websocket_manager),
) -> SuccessResponse:
    """Delete a specific key from a configuration section."""
    try:
        config_loader = ConfigLoader()
        config: AppConfig = config_loader.get_config()

        if not hasattr(config, section):
            raise HTTPException(status_code=404, detail=f"Configuration section '{section}' not found")

        section_obj = getattr(config, section)
        if not hasattr(section_obj, key):
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found in section '{section}'")

        # Create a new dictionary without the specified key
        section_dict = section_obj.model_dump(mode="json")
        if key in section_dict:
            del section_dict[key]

        # Create a new model instance from the updated dictionary
        updated_section = type(section_obj)(**section_dict)

        # Update the main config object
        setattr(config, section, updated_section)

        # Save the updated configuration
        config_loader.save_config(config)

        # Broadcast update to WebSocket clients
        await websocket_manager.broadcast(
            json.dumps(
                {
                    "type": "config_deleted",
                    "section": section,
                    "key": key,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        )

        logger.info(f"Configuration key '{key}' deleted from section '{section}'")

        return SuccessResponse(
            success=True,
            message=f"Configuration key '{key}' deleted successfully",
            data={"section": section, "deleted_key": key},
        )

    except Exception as e:
        logger.error(f"Failed to delete config key {section}.{key}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/indicators", response_model=dict[str, Any])
async def get_indicators_config(app_state: dict[str, Any] = Depends(get_app_state)) -> dict[str, Any]:
    """Get technical indicators configuration"""
    try:
        config: AppConfig = app_state["config"]

        # Get indicators configuration from trading.features
        indicators_config: list[FeatureConfiguration] = config.trading.features.configurations

        # Convert to more UI-friendly format
        result: dict[str, Any] = {
            "indicators": [],
            "lookback_period": config.trading.features.lookback_period,
            "name_mapping": config.trading.features.name_mapping,
            "timeframes": config.trading.feature_timeframes,
        }

        for indicator in indicators_config:
            result["indicators"].append(
                {
                    "name": indicator.name,
                    "function": indicator.function,
                    "params": indicator.params,
                    "enabled": True,  # Default to enabled
                    "category": _get_indicator_category(indicator.function),
                }
            )

        return result

    except Exception as e:
        logger.error(f"Failed to get indicators config: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/indicators", response_model=SuccessResponse)
async def update_indicators_config(
    request: IndicatorUpdateRequest,
    app_state: dict[str, Any] = Depends(get_app_state),
    websocket_manager: Any = Depends(get_websocket_manager),
) -> SuccessResponse:
    """Update technical indicators configuration"""
    try:
        config: AppConfig = app_state["config"]

        # Update indicators configuration
        new_indicators: list[FeatureConfiguration] = []
        for indicator in request.indicators:
            new_indicator = FeatureConfiguration(
                name=indicator.name, function=indicator.function, params=indicator.params
            )
            new_indicators.append(new_indicator)

        config.trading.features.configurations = new_indicators

        # Save configuration using enhanced save method
        config_loader = ConfigLoader()

        # Create backup before saving
        try:
            backup_path = config_loader.backup_config()
            logger.info(f"Created configuration backup: {backup_path}")
        except Exception as backup_error:
            logger.warning(f"Failed to create backup: {backup_error}")

        # Save configuration
        config_loader.save_config(config)

        # Broadcast update to WebSocket clients
        await websocket_manager.broadcast(
            json.dumps(
                {"type": "indicators_updated", "count": len(new_indicators), "timestamp": datetime.now().isoformat()}
            )
        )

        logger.info(f"Updated {len(new_indicators)} indicators configuration")

        return SuccessResponse(
            success=True,
            message="Indicators configuration updated successfully",
            data={
                "indicators_count": len(new_indicators),
                "updated_indicators": [ind.name for ind in request.indicators],
            },
        )

    except Exception as e:
        logger.error(f"Failed to update indicators config: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/instruments", response_model=dict[str, Any])
async def get_instruments_config(app_state: dict[str, Any] = Depends(get_app_state)) -> dict[str, Any]:
    """Get trading instruments configuration"""
    try:
        config: AppConfig = app_state["config"]

        return {
            "instruments": config.trading.instruments,
            "aggregation_timeframes": config.trading.aggregation_timeframes,
            "feature_timeframes": config.trading.feature_timeframes,
            "labeling_timeframes": config.trading.labeling_timeframes,
        }

    except Exception as e:
        logger.error(f"Failed to get instruments config: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/instruments", response_model=SuccessResponse)
async def update_instruments_config(
    data: dict[str, Any],
    app_state: dict[str, Any] = Depends(get_app_state),
    websocket_manager: Any = Depends(get_websocket_manager),
) -> SuccessResponse:
    """Update trading instruments configuration"""
    try:
        config: AppConfig = app_state["config"]

        # Update instruments configuration
        if "instruments" in data:
            config.trading.instruments = data["instruments"]

        if "aggregation_timeframes" in data:
            config.trading.aggregation_timeframes = data["aggregation_timeframes"]

        if "feature_timeframes" in data:
            config.trading.feature_timeframes = data["feature_timeframes"]

        if "labeling_timeframes" in data:
            config.trading.labeling_timeframes = data["labeling_timeframes"]

        # Save configuration using enhanced save method
        config_loader = ConfigLoader()

        # Create backup before saving
        try:
            backup_path = config_loader.backup_config()
            logger.info(f"Created configuration backup: {backup_path}")
        except Exception as backup_error:
            logger.warning(f"Failed to create backup: {backup_error}")

        # Save configuration
        config_loader.save_config(config)

        # Broadcast update to WebSocket clients
        await websocket_manager.broadcast(
            json.dumps({"type": "instruments_config_updated", "timestamp": datetime.now().isoformat()})
        )

        logger.info("Instruments configuration updated")

        return SuccessResponse(
            success=True,
            message="Instruments configuration updated successfully",
            data={"updated_fields": list(data.keys())},
        )

    except Exception as e:
        logger.error(f"Failed to update instruments config: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/holidays", response_model=dict[str, Any])
async def get_holidays_config(app_state: dict[str, Any] = Depends(get_app_state)) -> dict[str, Any]:
    """Get market holidays configuration"""
    try:
        config: AppConfig = app_state["config"]

        if config.market_calendar is None:
            return {}

        return {
            "holidays": config.market_calendar.holidays,
            "market_open_time": config.market_calendar.market_open_time,
            "market_close_time": config.market_calendar.market_close_time,
            "muhurat_trading_sessions": config.market_calendar.muhurat_trading_sessions,
            "half_day_sessions": config.market_calendar.half_day_sessions,
        }

    except Exception as e:
        logger.error(f"Failed to get holidays config: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/holidays", response_model=SuccessResponse)
async def update_holidays_config(
    data: dict[str, Any],
    app_state: dict[str, Any] = Depends(get_app_state),
    websocket_manager: Any = Depends(get_websocket_manager),
) -> SuccessResponse:
    """Update market holidays configuration"""
    try:
        config: AppConfig = app_state["config"]

        if config.market_calendar is None:
            return SuccessResponse(success=False, message="Market calendar config not found", data={})

        # Update holidays configuration
        if "holidays" in data:
            config.market_calendar.holidays.clear()
            config.market_calendar.holidays.extend(data["holidays"])

        if "market_open_time" in data:
            config.market_calendar.market_open_time = data["market_open_time"]

        if "market_close_time" in data:
            config.market_calendar.market_close_time = data["market_close_time"]

        if "muhurat_trading_sessions" in data:
            config.market_calendar.muhurat_trading_sessions = data["muhurat_trading_sessions"]

        if "half_day_sessions" in data:
            config.market_calendar.half_day_sessions = data["half_day_sessions"]

        # Save configuration using enhanced save method
        config_loader = ConfigLoader()

        # Create backup before saving
        try:
            backup_path = config_loader.backup_config()
            logger.info(f"Created configuration backup: {backup_path}")
        except Exception as backup_error:
            logger.warning(f"Failed to create backup: {backup_error}")

        # Save configuration
        config_loader.save_config(config)

        # Broadcast update to WebSocket clients
        await websocket_manager.broadcast(
            json.dumps({"type": "holidays_config_updated", "timestamp": datetime.now().isoformat()})
        )

        logger.info("Holidays configuration updated")

        return SuccessResponse(
            success=True,
            message="Holidays configuration updated successfully",
            data={"updated_fields": list(data.keys())},
        )

    except Exception as e:
        logger.error(f"Failed to update holidays config: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _get_indicator_category(function_name: str) -> str:
    """Get category for an indicator function"""
    trend_indicators = ["MACD", "ADX", "SAR", "AROON", "HT_TRENDLINE"]
    momentum_indicators = ["RSI", "STOCH", "WILLR", "CMO", "BOP"]
    volatility_indicators = ["ATR", "BBANDS", "STDDEV"]
    volume_indicators = ["OBV", "ADOSC"]
    cycle_indicators = ["HT_DCPERIOD", "HT_PHASOR"]

    if function_name in trend_indicators:
        return "trend"
    if function_name in momentum_indicators:
        return "momentum"
    if function_name in volatility_indicators:
        return "volatility"
    if function_name in volume_indicators:
        return "volume"
    if function_name in cycle_indicators:
        return "cycle"
    return "other"
