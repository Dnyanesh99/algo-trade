import asyncio
import json
from collections.abc import Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from src.metrics import metrics_registry
from src.utils.config_loader import ConfigLoader
from src.utils.live_pipeline_manager import LivePipelineManager
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
        live_pipeline_manager: LivePipelineManager,
        timeframes: Optional[list[int]] = None,
        persistence_path: Optional[Path] = None,
    ):
        if not callable(on_candle_complete_callback):
            raise TypeError("on_candle_complete_callback must be a callable coroutine function.")
        if not isinstance(live_pipeline_manager, LivePipelineManager):
            raise TypeError("live_pipeline_manager must be a valid LivePipelineManager instance.")

        if not config.candle_buffer:
            raise ValueError("Candle buffer configuration ('candle_buffer') is missing in config.yaml.")

        self.on_candle_complete_callback = on_candle_complete_callback
        self.live_pipeline_manager = live_pipeline_manager
        self.timeframes = timeframes or config.candle_buffer.timeframes
        self._partial_candles: dict[int, dict[int, dict[str, Any]]] = {tf: {} for tf in self.timeframes}
        self._lock = asyncio.Lock()

        self.persistence_path = persistence_path or Path(config.candle_buffer.persistence_path)
        self.persistence_interval = config.candle_buffer.persistence_interval_seconds
        self._persistence_task: Optional[asyncio.Task[None]] = None

        self._load_persisted_state()
        logger.info(
            f"CandleBuffer initialized for timeframes: {self.timeframes} with persistence path '{self.persistence_path}'."
        )

    async def start_persistence_task(self) -> None:
        """
        Starts a background task to periodically persist the partial candle state.
        """
        if self._persistence_task and not self._persistence_task.done():
            logger.warning("Attempted to start CandleBuffer persistence task, but it is already running.")
            return
        self._persistence_task = asyncio.create_task(self._run_persistence_loop())
        logger.info(
            f"CandleBuffer persistence task started. Will persist state every {self.persistence_interval} seconds."
        )

    async def stop_persistence_task(self) -> None:
        """
        Stops the background persistence task gracefully.
        """
        if not self._persistence_task or self._persistence_task.done():
            logger.info("CandleBuffer persistence task is not running.")
            return

        logger.info("Stopping CandleBuffer persistence task...")
        self._persistence_task.cancel()
        try:
            await self._persistence_task
        except asyncio.CancelledError:
            logger.info("CandleBuffer persistence task stopped successfully.")
        except Exception as e:
            logger.error(f"An error occurred while stopping the persistence task: {e}", exc_info=True)
        finally:
            self._persistence_task = None
            logger.info("CandleBuffer persistence task cleaned up.")

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
        if not self.persistence_path.exists():
            logger.info("No persisted state file found. Starting with a clean buffer.")
            return

        logger.info(f"Loading persisted partial candle state from {self.persistence_path}...")
        try:
            with open(self.persistence_path) as f:
                persisted_data = json.load(f)

            loaded_candles: dict[int, dict[int, dict[str, Any]]] = {tf: {} for tf in self.timeframes}
            for tf_str, instruments_data in persisted_data.items():
                try:
                    tf = int(tf_str)
                    if tf not in self.timeframes:
                        logger.warning(f"Persisted timeframe '{tf}' is not in the current configuration. Skipping.")
                        continue

                    for token_str, candle_data in instruments_data.items():
                        try:
                            token = int(token_str)
                            candle_data["start_time"] = datetime.fromisoformat(candle_data["start_time"])
                            # Further validation could be added here if needed
                            loaded_candles[tf][token] = candle_data
                        except (ValueError, TypeError, KeyError) as e:
                            logger.warning(f"Skipping invalid candle data record during load: {e}. Data: {candle_data}")
                            continue
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid timeframe data during load: {e}. Data: {instruments_data}")
                    continue

            self._partial_candles = loaded_candles
            logger.info(
                f"Successfully loaded {sum(len(v) for v in loaded_candles.values())} partial candles from state file."
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.error(
                f"Failed to load or parse persisted state file '{self.persistence_path}': {e}. Starting fresh."
            )
        except Exception as e:
            logger.critical(f"An unexpected error occurred during state loading: {e}", exc_info=True)

    def _persist_state(self) -> None:
        logger.debug("Persisting partial candle state...")
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
            logger.info(
                f"Successfully persisted {sum(len(v) for v in self._partial_candles.values())} partial candles to {self.persistence_path}."
            )
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error persisting candle state to '{self.persistence_path}': {e}", exc_info=True)

    async def process_tick(self, tick: dict[str, Any]) -> None:
        """
        Processes an incoming tick to update partial candles for all timeframes.
        Assumes tick contains OHLC data for the minute.
        """
        try:
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

                    current_candle = self._partial_candles[tf].get(instrument_token)

                    if not current_candle or candle_start_time > current_candle["start_time"]:
                        if current_candle:
                            logger.info(
                                f"Completed {tf}-min candle for {instrument_token} at {current_candle['start_time']}. Emitting..."
                            )
                            await self.on_candle_complete_callback(current_candle)
                            await self.live_pipeline_manager.record_candle_completion(
                                f"{tf}min", current_candle["start_time"]
                            )
                            metrics_registry.increment_counter(
                                "live_candle_formation_total",
                                {"instrument": str(instrument_token), "timeframe": f"{tf}min"},
                            )

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
                        logger.debug(f"Initialized new {tf}-min candle for {instrument_token} at {candle_start_time}.")

                    elif candle_start_time == current_candle["start_time"]:
                        current_candle["high"] = max(current_candle["high"], ohlc.get("high", ltp))
                        current_candle["low"] = min(current_candle["low"], ohlc.get("low", ltp))
                        current_candle["close"] = ltp
                        current_candle["volume"] += volume
                        current_candle["trades"] += 1
                        logger.debug(f"Updated {tf}-min candle for {instrument_token} at {candle_start_time}.")
        except Exception as e:
            logger.error(f"Error processing tick: {tick}. Error: {e}", exc_info=True)

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
