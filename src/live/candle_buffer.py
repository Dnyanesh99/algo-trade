import asyncio
import json
from collections.abc import Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from src.metrics import metrics_registry
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

config = ConfigLoader().get_config()


class CandleBuffer:
    """
    An in-memory buffer that constructs partial candles from the tick stream.
    Once a candle's time period is complete, it emits the completed OHLCV data.
    Supports multiple timeframes and persists state to avoid data loss on restart.
    """

    def __init__(
        self,
        on_candle_complete_callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
        timeframes: Optional[list[int]] = None,
        persistence_path: Optional[Path] = None,
    ):
        assert config.candle_buffer is not None
        self.on_candle_complete_callback = on_candle_complete_callback
        self.timeframes = timeframes or config.candle_buffer.timeframes
        self._partial_candles: dict[int, dict[int, dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self.persistence_path = persistence_path or Path(config.candle_buffer.persistence_path)
        self.persistence_interval = config.candle_buffer.persistence_interval_seconds
        self._persistence_task: Optional[asyncio.Task] = None
        self._load_persisted_state()
        logger.info(f"CandleBuffer initialized for timeframes: {self.timeframes}")

    async def start_persistence_task(self) -> None:
        """
        Starts a background task to periodically persist the partial candle state.
        """
        if self._persistence_task is None or self._persistence_task.done():
            self._persistence_task = asyncio.create_task(self._run_persistence_loop())
            logger.info("CandleBuffer persistence task started.")

    async def stop_persistence_task(self) -> None:
        """
        Stops the background persistence task gracefully.
        """
        if self._persistence_task:
            logger.info("Stopping CandleBuffer persistence task...")
            self._persistence_task.cancel()
            try:
                await self._persistence_task
            except asyncio.CancelledError:
                logger.info("CandleBuffer persistence task stopped successfully.")
            except Exception as e:
                logger.error(f"Error stopping CandleBuffer persistence task: {e}")
            finally:
                self._persistence_task = None

    async def _run_persistence_loop(self) -> None:
        """
        The main loop for the background persistence task.
        """
        while True:
            try:
                await asyncio.sleep(self.persistence_interval)
                self._persist_state()
            except asyncio.CancelledError:
                logger.info("CandleBuffer persistence loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in CandleBuffer persistence loop: {e}", exc_info=True)

    def _load_persisted_state(self) -> None:
        if self.persistence_path.exists():
            try:
                with open(self.persistence_path) as f:
                    persisted_data = json.load(f)
                    loaded_candles = {}
                    for tf_str, instruments_data in persisted_data.items():
                        try:
                            tf = int(tf_str)
                        except ValueError:
                            logger.warning(f"Invalid timeframe key in persisted data: {tf_str}. Skipping.")
                            continue

                        loaded_instruments = {}
                        for instrument_token_str, candle_data in instruments_data.items():
                            try:
                                instrument_token = int(instrument_token_str)
                            except ValueError:
                                logger.warning(
                                    f"Invalid instrument token key in persisted data: {instrument_token_str}. Skipping."
                                )
                                continue

                            if "start_time" in candle_data and isinstance(candle_data["start_time"], str):
                                try:
                                    candle_data["start_time"] = datetime.fromisoformat(candle_data["start_time"])
                                except ValueError:
                                    logger.warning(
                                        f"Invalid start_time format in persisted data for {instrument_token}: {candle_data['start_time']}. Skipping."
                                    )
                                    continue
                            else:
                                logger.warning(
                                    f"Missing or invalid start_time in persisted data for {instrument_token}. Skipping."
                                )
                                continue

                            if config.live_aggregator is None:
                                continue
                            required_candle_fields = config.live_aggregator.required_candle_fields
                            if not all(field in candle_data for field in required_candle_fields):
                                logger.warning(
                                    f"Persisted candle data for {instrument_token} is incomplete. Skipping: {candle_data}"
                                )
                                continue

                            loaded_instruments[instrument_token] = candle_data
                        loaded_candles[tf] = loaded_instruments

                    self._partial_candles = loaded_candles
                    logger.info(f"Loaded persisted partial candles from {self.persistence_path}")
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Error loading persisted candle state: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during state loading: {e}")

    def _persist_state(self) -> None:
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            serializable_candles = {}
            for tf, instruments in self._partial_candles.items():
                serializable_instruments = {}
                for instrument_token, candle_data in instruments.items():
                    serializable_candle_data = candle_data.copy()
                    if isinstance(serializable_candle_data.get("start_time"), datetime):
                        serializable_candle_data["start_time"] = serializable_candle_data["start_time"].isoformat()
                    serializable_instruments[instrument_token] = serializable_candle_data
                serializable_candles[tf] = serializable_instruments

            with open(self.persistence_path, "w") as f:
                json.dump(serializable_candles, f, indent=4)
            logger.debug(f"Persisted partial candles to {self.persistence_path}")
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error persisting candle state: {e}")

    async def process_tick(self, tick: dict[str, Any]) -> None:
        """
        Processes an incoming tick to update partial candles for all timeframes.
        Assumes tick contains OHLC data for the minute.
        """
        instrument_token = tick["instrument_token"]
        timestamp = tick["timestamp"]
        ltp = tick["last_traded_price"]
        ohlc = tick.get("ohlc", {})
        volume = tick.get("volume", 0)

        async with self._lock:
            for tf in self.timeframes:
                candle_start_time = self._get_candle_start_time(timestamp, tf)

                if tf not in self._partial_candles:
                    self._partial_candles[tf] = {}

                if instrument_token not in self._partial_candles[tf]:
                    self._partial_candles[tf][instrument_token] = {
                        "instrument_token": instrument_token,
                        "timeframe": tf,
                        "start_time": candle_start_time,
                        "open": ohlc.get("open", ltp),
                        "high": ohlc.get("high", ltp),
                        "low": ohlc.get("low", ltp),
                        "close": ltp,
                        "volume": volume,
                        "trades": 1,
                    }
                    logger.debug(f"Initialized new {tf}-min candle for {instrument_token} at {candle_start_time}")
                else:
                    current_candle = self._partial_candles[tf][instrument_token]

                    if candle_start_time == current_candle["start_time"]:
                        current_candle["high"] = max(current_candle["high"], ohlc.get("high", ltp))
                        current_candle["low"] = min(current_candle["low"], ohlc.get("low", ltp))
                        current_candle["close"] = ltp
                        current_candle["volume"] += volume
                        current_candle["trades"] += 1
                        logger.debug(f"Updated {tf}-min candle for {instrument_token} at {candle_start_time}")
                    else:
                        logger.info(
                            f"Completed {tf}-min candle for {instrument_token} at {current_candle['start_time']}. Emitting..."
                        )
                        metrics_registry.increment_counter(
                            "live_candle_formation_total",
                            {"instrument": str(instrument_token), "timeframe": f"{tf}min"},
                        )
                        await self.on_candle_complete_callback(current_candle)
                        # self._persist_state()  # Persistence handled by background task

                        self._partial_candles[tf][instrument_token] = {
                            "instrument_token": instrument_token,
                            "timeframe": tf,
                            "start_time": candle_start_time,
                            "open": ohlc.get("open", ltp),
                            "high": ohlc.get("high", ltp),
                            "low": ohlc.get("low", ltp),
                            "close": ltp,
                            "volume": volume,
                            "trades": 1,
                        }
                        logger.debug(f"Started new {tf}-min candle for {instrument_token} at {candle_start_time}")

    def _get_candle_start_time(self, timestamp: datetime, timeframe_minutes: int) -> datetime:
        """
        Calculates the start time of the candle for a given timestamp and timeframe.
        This ensures ticks are grouped into the correct fixed-interval candles.
        Example: For 15-min timeframe, 09:17:30 -> 09:15:00, 09:29:59 -> 09:15:00, 09:30:00 -> 09:30:00.
        """
        total_minutes = timestamp.hour * 60 + timestamp.minute
        start_minute = (total_minutes // timeframe_minutes) * timeframe_minutes
        return timestamp.replace(hour=start_minute // 60, minute=start_minute % 60, second=0, microsecond=0)

    async def flush_all_candles(self) -> None:
        async with self._lock:
            for tf in self.timeframes:
                if tf in self._partial_candles:
                    for instrument_token, candle_data in list(self._partial_candles[tf].items()):
                        logger.info(f"Flushing {tf}-min candle for {instrument_token} at {candle_data['start_time']}.")
                        await self.on_candle_complete_callback(candle_data)
                        del self._partial_candles[tf][instrument_token]
            self._persist_state()  # Ensure final state is persisted on flush
            logger.info("All partial candles flushed.")
