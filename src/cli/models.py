"""Pydantic models for the CLI."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CLIStateRequest(BaseModel):
    """Model for CLI state management requests."""

    instrument: Optional[str] = Field(None, description="Trading symbol of the instrument (e.g., 'NIFTY_FUT').")
    step: Optional[Literal["historical", "aggregation", "features", "labeling"]] = Field(
        None, description="Specific processing step to target."
    )
