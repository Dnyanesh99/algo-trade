"""
Production-grade, API-driven CLI for managing the system's processing state.
This CLI is a client for the system's Control API.
"""

from typing import Annotated

import requests
import typer

# --- Configuration --- #
API_BASE_URL = "http://localhost:8000/api/v1/control"

app = typer.Typer(
    name="state-manager",
    help="A CLI for inspecting and managing the trading system's processing state via the Control API.",
)


# --- Helper Functions --- #
def handle_api_error(response: requests.Response) -> None:
    """A centralized handler for API errors."""
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        print(f"Error: API request failed with status {response.status_code}: {detail}")
        raise typer.Exit(code=1)


# --- CLI Commands --- #


@app.command()
def status() -> None:
    """Fetch the comprehensive, unified status of the entire system."""
    response = requests.get(f"{API_BASE_URL}/status", timeout=30)
    handle_api_error(response)
    print(response.json())


@app.command()
def reprocess(
    instrument: Annotated[str, typer.Option(help="Trading symbol (e.g., NIFTY_FUT) to reprocess.")],
    step: Annotated[str, typer.Option(help="Specific step to reprocess (e.g., historical_fetch).")],
) -> None:
    """Schedule a reprocessing job for a specific instrument and step."""
    params = {"instrument_symbol": instrument, "step": step}
    response = requests.post(f"{API_BASE_URL}/reprocess", params=params, timeout=30)
    handle_api_error(response)
    print(response.json()["message"])


@app.command()
def get_config() -> None:
    """Retrieve the current, live application configuration."""
    response = requests.get(f"{API_BASE_URL}/config", timeout=30)
    handle_api_error(response)
    print(response.json())


@app.command()
def set_config(
    section: Annotated[str, typer.Option(help="The configuration section (e.g., trading).")],
    key: Annotated[str, typer.Option(help="The configuration key (e.g., epsilon).")],
    value: Annotated[str, typer.Option(help="The new value for the key.")],
) -> None:
    """Update a configuration key at runtime."""
    params = {"section": section, "key": key, "value": value}
    response = requests.put(f"{API_BASE_URL}/config", params=params, timeout=30)
    handle_api_error(response)
    print(response.json()["message"])


if __name__ == "__main__":
    app()
