from datetime import datetime, timedelta
from typing import Any, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

from src.database.chart_repo import ChartRepository
from src.database.instrument_repo import InstrumentRepository
from src.database.models import InstrumentRecord, OHLCVData
from src.utils.logger import LOGGER as logger

from .dependencies import get_app_state

router = APIRouter(prefix="/api/charts", tags=["charts"])


def get_chart_repo(app_state: dict[str, Any] = Depends(get_app_state)) -> ChartRepository:
    """Get chart repository from app state."""
    return cast("ChartRepository", app_state["repositories"]["chart"])


def get_instrument_repo(app_state: dict[str, Any] = Depends(get_app_state)) -> InstrumentRepository:
    """Get instrument repository from app state."""
    return cast("InstrumentRepository", app_state["repositories"]["instrument"])


class ChartDataRequest(BaseModel):
    """Request model for chart data."""

    instrument_id: int = Field(..., description="Instrument ID")
    timeframe: str = Field(..., description="Timeframe (1min, 5min, 15min, 60min)")
    start_time: Optional[datetime] = Field(None, description="Start time for data retrieval")
    end_time: Optional[datetime] = Field(None, description="End time for data retrieval")
    limit: Optional[int] = Field(1000, description="Maximum number of records to retrieve")


class ChartDataResponse(BaseModel):
    """Response model for chart data."""

    success: bool
    data: list[OHLCVData]
    total_records: int
    timeframe: str
    instrument_id: int
    start_time: Optional[datetime]
    end_time: Optional[datetime]


class ChartDataWithIndicatorsResponse(BaseModel):
    """Response model for chart data with indicators."""

    success: bool
    data: list[dict[str, Any]]
    total_records: int
    timeframe: str
    instrument_id: int
    start_time: Optional[datetime]
    end_time: Optional[datetime]


class InstrumentSearchResponse(BaseModel):
    """Response model for instrument search."""

    success: bool
    instruments: list[InstrumentRecord]
    total_found: int


class ChartSummaryResponse(BaseModel):
    """Response model for chart data summary."""

    success: bool
    summary: Optional[dict[str, Any]]


class AvailableTimeframesResponse(BaseModel):
    """Response model for available timeframes."""

    success: bool
    timeframes: list[str]
    instrument_id: int


@router.get("/instruments/search", response_model=InstrumentSearchResponse)
async def search_instruments(
    q: str = Query(..., description="Search term for instrument symbol or name"),
    limit: int = Query(50, description="Maximum number of results", ge=1, le=100),
    chart_repo: ChartRepository = Depends(get_chart_repo),
) -> InstrumentSearchResponse:
    """
    Search for instruments that can be used for charting.

    Args:
        q: Search term for instrument symbol or name
        limit: Maximum number of results to return

    Returns:
        List of matching instruments
    """
    try:
        instruments = await chart_repo.get_instruments_for_chart(q, limit)
        return InstrumentSearchResponse(success=True, instruments=instruments, total_found=len(instruments))
    except Exception as e:
        logger.error(f"Error searching instruments: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching instruments: {str(e)}") from e


@router.get("/instruments/{instrument_id}/timeframes", response_model=AvailableTimeframesResponse)
async def get_available_timeframes(
    instrument_id: int = Path(..., description="Instrument ID"),
    chart_repo: ChartRepository = Depends(get_chart_repo),
    instrument_repo: InstrumentRepository = Depends(get_instrument_repo),
) -> AvailableTimeframesResponse:
    """
    Get available timeframes for a specific instrument.

    Args:
        instrument_id: The instrument ID

    Returns:
        List of available timeframes
    """
    try:
        # Check if instrument exists
        instrument = await instrument_repo.get_instrument_by_id(instrument_id)
        if not instrument:
            raise HTTPException(status_code=404, detail="Instrument not found")

        timeframes = await chart_repo.get_available_timeframes(instrument_id)
        return AvailableTimeframesResponse(success=True, timeframes=timeframes, instrument_id=instrument_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting available timeframes for instrument {instrument_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting available timeframes: {str(e)}") from e


@router.get("/data/{instrument_id}/{timeframe}", response_model=ChartDataResponse)
async def get_chart_data(
    instrument_id: int = Path(..., description="Instrument ID"),
    timeframe: str = Path(..., description="Timeframe (1min, 5min, 15min, 60min)"),
    start_time: Optional[datetime] = Query(None, description="Start time for data retrieval"),
    end_time: Optional[datetime] = Query(None, description="End time for data retrieval"),
    limit: int = Query(1000, description="Maximum number of records", ge=1, le=5000),
    chart_repo: ChartRepository = Depends(get_chart_repo),
    instrument_repo: InstrumentRepository = Depends(get_instrument_repo),
) -> ChartDataResponse:
    """
    Get OHLCV chart data for a specific instrument and timeframe.

    Args:
        instrument_id: The instrument ID
        timeframe: The timeframe (1min, 5min, 15min, 60min)
        start_time: Start time for data retrieval (optional)
        end_time: End time for data retrieval (optional)
        limit: Maximum number of records to retrieve

    Returns:
        OHLCV chart data
    """
    try:
        # Validate timeframe
        valid_timeframes = ["1min", "5min", "15min", "60min"]
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=400, detail=f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}"
            )

        # Check if instrument exists
        instrument = await instrument_repo.get_instrument_by_id(instrument_id)
        if not instrument:
            raise HTTPException(status_code=404, detail="Instrument not found")

        # If no time range provided, get latest data
        if start_time is None or end_time is None:
            data = await chart_repo.get_chart_data_latest(instrument_id, timeframe, limit)
            # Reverse to get chronological order
            data.reverse()
        else:
            data = await chart_repo.get_chart_data(instrument_id, timeframe, start_time, end_time)

        return ChartDataResponse(
            success=True,
            data=data,
            total_records=len(data),
            timeframe=timeframe,
            instrument_id=instrument_id,
            start_time=start_time,
            end_time=end_time,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart data for instrument {instrument_id}, timeframe {timeframe}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting chart data: {str(e)}") from e


@router.get("/data/{instrument_id}/{timeframe}/with-indicators", response_model=ChartDataWithIndicatorsResponse)
async def get_chart_data_with_indicators(
    instrument_id: int = Path(..., description="Instrument ID"),
    timeframe: str = Path(..., description="Timeframe (1min, 5min, 15min, 60min)"),
    start_time: Optional[datetime] = Query(None, description="Start time for data retrieval"),
    end_time: Optional[datetime] = Query(None, description="End time for data retrieval"),
    chart_repo: ChartRepository = Depends(get_chart_repo),
    instrument_repo: InstrumentRepository = Depends(get_instrument_repo),
) -> ChartDataWithIndicatorsResponse:
    """
    Get OHLCV chart data with technical indicators.

    Args:
        instrument_id: The instrument ID
        timeframe: The timeframe (1min, 5min, 15min, 60min)
        start_time: Start time for data retrieval
        end_time: End time for data retrieval

    Returns:
        OHLCV chart data with technical indicators
    """
    try:
        # Validate timeframe
        valid_timeframes = ["1min", "5min", "15min", "60min"]
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=400, detail=f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}"
            )

        # Check if instrument exists
        instrument = await instrument_repo.get_instrument_by_id(instrument_id)
        if not instrument:
            raise HTTPException(status_code=404, detail="Instrument not found")

        # Default to last 30 days if no time range provided
        if start_time is None or end_time is None:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)

        data = await chart_repo.get_chart_data_with_indicators(instrument_id, timeframe, start_time, end_time)

        return ChartDataWithIndicatorsResponse(
            success=True,
            data=data,
            total_records=len(data),
            timeframe=timeframe,
            instrument_id=instrument_id,
            start_time=start_time,
            end_time=end_time,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting chart data with indicators for instrument {instrument_id}, timeframe {timeframe}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Error getting chart data with indicators: {str(e)}") from e


@router.get("/data/{instrument_id}/{timeframe}/realtime", response_model=ChartDataResponse)
async def get_realtime_chart_data(
    instrument_id: int = Path(..., description="Instrument ID"),
    timeframe: str = Path(..., description="Timeframe (1min, 5min, 15min, 60min)"),
    since_time: datetime = Query(..., description="Get data since this timestamp"),
    limit: int = Query(100, description="Maximum number of records", ge=1, le=1000),
    chart_repo: ChartRepository = Depends(get_chart_repo),
    instrument_repo: InstrumentRepository = Depends(get_instrument_repo),
) -> ChartDataResponse:
    """
    Get real-time chart data since a specific timestamp.

    Args:
        instrument_id: The instrument ID
        timeframe: The timeframe (1min, 5min, 15min, 60min)
        since_time: Timestamp to get data since
        limit: Maximum number of records to retrieve

    Returns:
        Real-time OHLCV chart data
    """
    try:
        # Validate timeframe
        valid_timeframes = ["1min", "5min", "15min", "60min"]
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=400, detail=f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}"
            )

        # Check if instrument exists
        instrument = await instrument_repo.get_instrument_by_id(instrument_id)
        if not instrument:
            raise HTTPException(status_code=404, detail="Instrument not found")

        data = await chart_repo.get_realtime_chart_data(instrument_id, timeframe, since_time, limit)

        return ChartDataResponse(
            success=True,
            data=data,
            total_records=len(data),
            timeframe=timeframe,
            instrument_id=instrument_id,
            start_time=since_time,
            end_time=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting realtime chart data for instrument {instrument_id}, timeframe {timeframe}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting realtime chart data: {str(e)}") from e


@router.get("/data/{instrument_id}/{timeframe}/summary", response_model=ChartSummaryResponse)
async def get_chart_data_summary(
    instrument_id: int = Path(..., description="Instrument ID"),
    timeframe: str = Path(..., description="Timeframe (1min, 5min, 15min, 60min)"),
    start_time: Optional[datetime] = Query(None, description="Start time for data retrieval"),
    end_time: Optional[datetime] = Query(None, description="End time for data retrieval"),
    chart_repo: ChartRepository = Depends(get_chart_repo),
    instrument_repo: InstrumentRepository = Depends(get_instrument_repo),
) -> ChartSummaryResponse:
    """
    Get summary statistics for chart data.

    Args:
        instrument_id: The instrument ID
        timeframe: The timeframe (1min, 5min, 15min, 60min)
        start_time: Start time for data retrieval
        end_time: End time for data retrieval

    Returns:
        Summary statistics for chart data
    """
    try:
        # Validate timeframe
        valid_timeframes = ["1min", "5min", "15min", "60min"]
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=400, detail=f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}"
            )

        # Check if instrument exists
        instrument = await instrument_repo.get_instrument_by_id(instrument_id)
        if not instrument:
            raise HTTPException(status_code=404, detail="Instrument not found")

        # Default to last 30 days if no time range provided
        if start_time is None or end_time is None:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)

        summary = await chart_repo.get_chart_data_summary(instrument_id, timeframe, start_time, end_time)

        return ChartSummaryResponse(success=True, summary=summary)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart data summary for instrument {instrument_id}, timeframe {timeframe}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting chart data summary: {str(e)}") from e


@router.post("/data", response_model=ChartDataResponse)
async def get_chart_data_post(
    request: ChartDataRequest,
    chart_repo: ChartRepository = Depends(get_chart_repo),
    instrument_repo: InstrumentRepository = Depends(get_instrument_repo),
) -> ChartDataResponse:
    """
    Get OHLCV chart data using POST request (for complex queries).

    Args:
        request: Chart data request parameters

    Returns:
        OHLCV chart data
    """
    try:
        # Validate timeframe
        valid_timeframes = ["1min", "5min", "15min", "60min"]
        if request.timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=400, detail=f"Invalid timeframe: {request.timeframe}. Must be one of {valid_timeframes}"
            )

        # Check if instrument exists
        instrument = await instrument_repo.get_instrument_by_id(request.instrument_id)
        if not instrument:
            raise HTTPException(status_code=404, detail="Instrument not found")

        # If no time range provided, get latest data
        if request.start_time is None or request.end_time is None:
            data = await chart_repo.get_chart_data_latest(
                request.instrument_id, request.timeframe, request.limit or 1000
            )
            # Reverse to get chronological order
            data.reverse()
        else:
            data = await chart_repo.get_chart_data(
                request.instrument_id, request.timeframe, request.start_time, request.end_time
            )

        return ChartDataResponse(
            success=True,
            data=data,
            total_records=len(data),
            timeframe=request.timeframe,
            instrument_id=request.instrument_id,
            start_time=request.start_time,
            end_time=request.end_time,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart data via POST: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting chart data: {str(e)}") from e
