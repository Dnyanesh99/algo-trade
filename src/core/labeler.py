import asyncio
import logging
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Optional

import numpy as np
import pandas as pd
import psutil
from numba import jit, prange

from src.database.label_repo import LabelRepository
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

config = config_loader.get_config()
logging.basicConfig(level=getattr(logging, config.logging.level), format=config.logging.format)


class ExitReason(IntEnum):
    TIME_OUT = 0
    TP_HIT = 1
    SL_HIT = 2
    AMBIGUOUS = 3
    INSUFFICIENT_DATA = 4
    ERROR = 5


class Label(IntEnum):
    SELL = -1
    NEUTRAL = 0
    BUY = 1


@dataclass(frozen=True)
class BarrierConfig:
    atr_period: int = field(default_factory=lambda: config.trading.labeling.atr_period)
    tp_atr_multiplier: float = field(default_factory=lambda: config.trading.labeling.tp_atr_multiplier)
    sl_atr_multiplier: float = field(default_factory=lambda: config.trading.labeling.sl_atr_multiplier)
    max_holding_periods: int = field(default_factory=lambda: config.trading.labeling.max_holding_periods)
    min_bars_required: int = field(default_factory=lambda: config.trading.labeling.min_bars_required)
    atr_smoothing: str = field(default_factory=lambda: config.trading.labeling.atr_smoothing)
    epsilon: float = field(default_factory=lambda: config.trading.labeling.epsilon)


@dataclass
class LabelingStats:
    symbol: str
    total_bars: int
    labeled_bars: int
    label_distribution: dict[str, int]
    avg_return_by_label: dict[str, float]
    exit_reasons: dict[str, int]
    avg_holding_period: float
    processing_time_ms: float
    data_quality_score: float


class OptimizedTripleBarrierLabeler:
    """
    Production-grade, async-native Triple Barrier Labeler.
    """

    _atr_cache: dict[str, np.ndarray] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self, barrier_config: BarrierConfig, label_repo: LabelRepository, executor: Optional[ProcessPoolExecutor] = None
    ):
        self.config = barrier_config
        self.label_repo = label_repo
        self.label_stats: dict[str, LabelingStats] = {}
        self.stats_lock = threading.Lock()
        self.executor = executor or ProcessPoolExecutor(max_workers=psutil.cpu_count())

    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)  # type: ignore
    def _calculate_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        n = len(high)
        if n == 0:
            return np.array([], dtype=np.float64)
        tr = np.empty(n, dtype=np.float64)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, max(hc, lc))
        return tr  # type: ignore

    def _calculate_atr(self, df: pd.DataFrame) -> np.ndarray:
        tr = self._calculate_true_range(df["high"].values, df["low"].values, df["close"].values)
        if self.config.atr_smoothing == "ema":
            return pd.Series(tr).ewm(span=self.config.atr_period, adjust=False).mean().values  # type: ignore
        return pd.Series(tr).rolling(self.config.atr_period).mean().values  # type: ignore

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)  # type: ignore
    def _check_barriers_optimized(
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        close: np.ndarray,
        entry_prices: np.ndarray,
        tp_prices: np.ndarray,
        sl_prices: np.ndarray,
        max_periods: int,
        epsilon: float,
    ) -> tuple:
        n = len(entry_prices)
        labels = np.zeros(n, dtype=np.int8)
        exit_bar_offsets = np.zeros(n, dtype=np.int32)
        exit_prices = np.zeros(n, dtype=np.float64)
        exit_reasons = np.full(n, ExitReason.INSUFFICIENT_DATA.value, dtype=np.int8)
        mfe = np.zeros(n, dtype=np.float64)
        mae = np.zeros(n, dtype=np.float64)

        for i in prange(n):
            if np.isnan(tp_prices[i]):
                continue

            exit_reasons[i] = ExitReason.TIME_OUT.value
            for j in range(1, min(max_periods + 1, n - i)):
                bar_idx = i + j
                mfe[i] = max(mfe[i], (high[bar_idx] - entry_prices[i]) / entry_prices[i])
                mae[i] = min(mae[i], (low[bar_idx] - entry_prices[i]) / entry_prices[i])

                tp_hit = high[bar_idx] >= tp_prices[i]
                sl_hit = low[bar_idx] <= sl_prices[i]

                if tp_hit and sl_hit:
                    if abs(tp_prices[i] - open_[bar_idx]) < abs(sl_prices[i] - open_[bar_idx]):
                        labels[i], exit_reasons[i], exit_prices[i] = (
                            Label.BUY.value,
                            ExitReason.TP_HIT.value,
                            tp_prices[i],
                        )
                    else:
                        labels[i], exit_reasons[i], exit_prices[i] = (
                            Label.SELL.value,
                            ExitReason.SL_HIT.value,
                            sl_prices[i],
                        )
                    exit_bar_offsets[i] = j
                    break
                if tp_hit:
                    labels[i], exit_reasons[i], exit_prices[i] = Label.BUY.value, ExitReason.TP_HIT.value, tp_prices[i]
                    exit_bar_offsets[i] = j
                    break
                if sl_hit:
                    labels[i], exit_reasons[i], exit_prices[i] = Label.SELL.value, ExitReason.SL_HIT.value, sl_prices[i]
                    exit_bar_offsets[i] = j
                    break

            if exit_reasons[i] == ExitReason.TIME_OUT.value:
                exit_idx = min(i + max_periods, n - 1)
                exit_prices[i] = close[exit_idx]
                exit_bar_offsets[i] = max_periods
                final_return = (exit_prices[i] - entry_prices[i]) / (entry_prices[i] + epsilon)
                labels[i] = (
                    Label.BUY.value
                    if final_return > epsilon
                    else (Label.SELL.value if final_return < -epsilon else Label.NEUTRAL.value)
                )

        return labels, exit_bar_offsets, exit_prices, exit_reasons, mfe, mae

    def _prepare_labels_dataframe(self, instrument_id: int, df: pd.DataFrame, results: tuple) -> pd.DataFrame:
        labels, exit_offsets, exit_prices, exit_reasons, mfe, mae = results

        # Create a new DataFrame to avoid modifying the original
        result_df = pd.DataFrame(
            {
                "ts": df["ts"],
                "instrument_id": instrument_id,
                "label": labels,
                "exit_reason": [ExitReason(r).name for r in exit_reasons],
                "exit_bar_offset": exit_offsets,
                "entry_price": df["close"],
                "exit_price": exit_prices,
                "tp_price": df["close"] * (1 + self.config.tp_atr_multiplier * df["atr"] / df["close"]),
                "sl_price": df["close"] * (1 - self.config.sl_atr_multiplier * df["atr"] / df["close"]),
                "barrier_return": (exit_prices - df["close"]) / (df["close"] + self.config.epsilon),
                "max_favorable_excursion": mfe,
                "max_adverse_excursion": mae,
                "volatility_at_entry": df["atr"],
            }
        )

        # Filter out rows where labeling was not possible
        return result_df[result_df["exit_reason"] != "INSUFFICIENT_DATA"].copy()

    async def process_symbol(self, instrument_id: int, symbol: str, df: pd.DataFrame) -> None:
        start_time = datetime.now()
        loop = asyncio.get_running_loop()

        try:
            logger.info(f"Processing {symbol} (ID: {instrument_id}) with {len(df):,} bars.")

            if len(df) < self.config.min_bars_required:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} < {self.config.min_bars_required}")
                return

            df_sorted = df.sort_values("ts").reset_index(drop=True)
            df_sorted["atr"] = self._calculate_atr(df_sorted)

            entry_prices = df_sorted["close"].values
            tp_prices = entry_prices + self.config.tp_atr_multiplier * df_sorted["atr"].values
            sl_prices = entry_prices - self.config.sl_atr_multiplier * df_sorted["atr"].values

            # Offload the CPU-bound numba calculation to the process pool
            results = await loop.run_in_executor(
                self.executor,
                self._check_barriers_optimized,
                df_sorted["high"].values,
                df_sorted["low"].values,
                df_sorted["open"].values,
                df_sorted["close"].values,
                entry_prices,
                tp_prices,
                sl_prices,
                self.config.max_holding_periods,
                self.config.epsilon,
            )

            labeled_df = self._prepare_labels_dataframe(instrument_id, df_sorted, results)

            if not labeled_df.empty:
                await self.label_repo.insert_labels(instrument_id, labeled_df)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Completed and stored labels for {symbol} in {processing_time:.2f}ms.")

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}", exc_info=True)

    async def process_symbols_parallel(self, symbol_data: dict[int, dict[str, Any]]) -> None:
        """
        Processes multiple symbols in parallel using an async-native approach.

        Args:
            symbol_data: A dictionary where key is instrument_id and value is
                         a dict containing 'symbol' name and 'data' DataFrame.
        """
        tasks = [self.process_symbol(inst_id, info["symbol"], info["data"]) for inst_id, info in symbol_data.items()]
        await asyncio.gather(*tasks)
        logger.info(f"Parallel labeling process completed for {len(tasks)} symbols.")
