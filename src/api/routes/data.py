"""
Data management routes for clearing tables
"""

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException

from src.utils.logger import logger

from ..dependencies import get_app_state
from ..models import SuccessResponse

if TYPE_CHECKING:
    from src.database.db_utils import DatabaseManager

router = APIRouter(prefix="/api/data", tags=["data"])


@router.post("/truncate", response_model=SuccessResponse)
async def truncate_tables(tables: list[str], app_state: dict[str, Any] = Depends(get_app_state)) -> SuccessResponse:
    """
    Truncate specified tables in the database.
    This is a destructive operation and should be used with caution.
    """
    allowed_tables_to_truncate = [
        "ohlcv_1min",
        "ohlcv_5min",
        "ohlcv_15min",
        "ohlcv_60min",
        "features",
        "engineered_features",
        "cross_asset_features",
        "labels",
        "signals",
        "feature_scores",
        "feature_selection_history",
    ]

    for table in tables:
        if table not in allowed_tables_to_truncate:
            raise HTTPException(status_code=400, detail=f"Table '{table}' is not allowed to be truncated.")

    try:
        db_manager: DatabaseManager = app_state["db_manager"]
        async with db_manager.transaction() as conn:
            for table in tables:
                logger.info(f"Truncating table: {table}")
                await conn.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")

        return SuccessResponse(
            success=True,
            message=f"Successfully truncated tables: {', '.join(tables)}",
            data={"truncated_tables": tables},
        )
    except Exception as e:
        logger.error(f"Failed to truncate tables: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
