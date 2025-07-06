import asyncio
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
from src.database.models import LabelData
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

config = config_loader.get_config()


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
    use_dynamic_barriers: bool = field(default_factory=lambda: config.trading.labeling.use_dynamic_barriers)
    volatility_lookback: int = field(default_factory=lambda: config.trading.labeling.volatility_lookback)


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
    sharpe_ratio: float
    win_rate: float


class OptimizedTripleBarrierLabeler:
    """
    Production-grade Triple Barrier Labeler with advanced features:
    - Dynamic barrier adjustment based on volatility regime
    - Proper handling of overnight gaps
    - Advanced exit logic for ambiguous cases
    - Risk-reward ratio calculation
    - Performance metrics tracking
    """

    _atr_cache: dict[str, np.ndarray] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        barrier_config: BarrierConfig,
        label_repo: LabelRepository,
        executor: Optional[ProcessPoolExecutor] = None,
    ):
        self.config = barrier_config
        self.label_repo = label_repo
        self.label_stats: dict[str, LabelingStats] = {}
        self.stats_lock = threading.Lock()
        self.executor = executor or ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False))

        logger.info(
            f"TripleBarrierLabeler initialized with config: "
            f"TP={self.config.tp_atr_multiplier}x ATR, "
            f"SL={self.config.sl_atr_multiplier}x ATR, "
            f"Max holding={self.config.max_holding_periods} bars"
        )

    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def _calculate_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate True Range with proper handling of the first bar."""
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

        return tr

    def _calculate_atr(self, df: pd.DataFrame, symbol: str) -> np.ndarray:
        """Calculate ATR with a symbol-specific cache for performance."""
        # --- SUGGESTION IMPLEMENTED: More specific cache key ---
        cache_key = f"{symbol}_{df.index[0]}_{df.index[-1]}_{len(df)}"

        with self._cache_lock:
            if cache_key in self._atr_cache:
                return self._atr_cache[cache_key]

        tr = self._calculate_true_range(df["high"].values, df["low"].values, df["close"].values)

        if self.config.atr_smoothing == "ema":
            atr = pd.Series(tr).ewm(span=self.config.atr_period, adjust=False).mean().values
        else:
            atr = pd.Series(tr).rolling(window=self.config.atr_period).mean().values

        first_valid_idx = (~np.isnan(atr)).argmax()
        if first_valid_idx > 0:
            atr[:first_valid_idx] = np.full(first_valid_idx, atr[first_valid_idx])

        with self._cache_lock:
            self._atr_cache[cache_key] = atr
            if len(self._atr_cache) > 100:  # Limit cache size
                self._atr_cache.pop(next(iter(self._atr_cache)))

        return atr

    def _calculate_dynamic_barriers(self, df: pd.DataFrame, atr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate dynamic barriers based on volatility regime."""
        if not self.config.use_dynamic_barriers:
            tp_multiplier = np.full(len(df), self.config.tp_atr_multiplier)
            sl_multiplier = np.full(len(df), self.config.sl_atr_multiplier)
            return tp_multiplier, sl_multiplier

        volatility_rolling = pd.Series(atr).rolling(window=self.config.volatility_lookback, min_periods=1)
        vol_mean = volatility_rolling.mean()
        vol_std = volatility_rolling.std()

        vol_zscore = (atr - vol_mean) / (vol_std + 1e-10)

        tp_multiplier = self.config.tp_atr_multiplier * (1 + 0.2 * np.tanh(vol_zscore))
        sl_multiplier = self.config.sl_atr_multiplier * (1 + 0.1 * np.tanh(vol_zscore))

        return tp_multiplier.values, sl_multiplier.values

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
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
        """Optimized barrier checking with parallel processing."""
        n = len(entry_prices)
        labels = np.zeros(n, dtype=np.int8)
        exit_bar_offsets = np.zeros(n, dtype=np.int32)
        exit_prices = np.zeros(n, dtype=np.float64)
        exit_reasons = np.full(n, ExitReason.INSUFFICIENT_DATA, dtype=np.int8)
        mfe = np.zeros(n, dtype=np.float64)
        mae = np.zeros(n, dtype=np.float64)

        for i in prange(n):
            if np.isnan(tp_prices[i]) or np.isnan(sl_prices[i]):
                continue

            exit_reasons[i] = ExitReason.TIME_OUT

            for j in range(1, min(max_periods + 1, n - i)):
                bar_idx = i + j
                high_return = (high[bar_idx] - entry_prices[i]) / entry_prices[i]
                low_return = (low[bar_idx] - entry_prices[i]) / entry_prices[i]
                mfe[i] = max(mfe[i], high_return)
                mae[i] = min(mae[i], low_return)

                tp_hit = high[bar_idx] >= tp_prices[i]
                sl_hit = low[bar_idx] <= sl_prices[i]

                if tp_hit and sl_hit:
                    if abs(open_[bar_idx] - tp_prices[i]) < abs(open_[bar_idx] - sl_prices[i]):
                        labels[i], exit_reasons[i], exit_prices[i] = Label.BUY, ExitReason.TP_HIT, tp_prices[i]
                    else:
                        labels[i], exit_reasons[i], exit_prices[i] = Label.SELL, ExitReason.SL_HIT, sl_prices[i]
                    exit_bar_offsets[i] = j
                    break
                if tp_hit:
                    labels[i], exit_reasons[i], exit_prices[i] = Label.BUY, ExitReason.TP_HIT, tp_prices[i]
                    exit_bar_offsets[i] = j
                    break
                if sl_hit:
                    labels[i], exit_reasons[i], exit_prices[i] = Label.SELL, ExitReason.SL_HIT, sl_prices[i]
                    exit_bar_offsets[i] = j
                    break

            if exit_reasons[i] == ExitReason.TIME_OUT:
                exit_idx = min(i + max_periods, n - 1)
                exit_prices[i] = close[exit_idx]
                exit_bar_offsets[i] = exit_idx - i
                final_return = (exit_prices[i] - entry_prices[i]) / (entry_prices[i] + epsilon)
                if final_return > epsilon:
                    labels[i] = Label.BUY
                elif final_return < -epsilon:
                    labels[i] = Label.SELL
                else:
                    labels[i] = Label.NEUTRAL

        return labels, exit_bar_offsets, exit_prices, exit_reasons, mfe, mae

    def _prepare_labels_dataframe(
        self,
        instrument_id: int,
        df: pd.DataFrame,
        results: tuple,
        tp_multipliers: np.ndarray,
        sl_multipliers: np.ndarray,
    ) -> pd.DataFrame:
        """Prepare labels DataFrame with all required fields."""
        labels, exit_offsets, exit_prices, exit_reasons, mfe, mae = results
        tp_prices = df["close"].values * (1 + tp_multipliers * df["atr"].values / df["close"].values)
        sl_prices = df["close"].values * (1 - sl_multipliers * df["atr"].values / df["close"].values)
        barrier_returns = (exit_prices - df["close"].values) / (df["close"].values + self.config.epsilon)
        potential_profit = tp_prices - df["close"].values
        potential_loss = df["close"].values - sl_prices
        risk_reward_ratio = potential_profit / (potential_loss + 1e-10)

        result_df = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "instrument_id": instrument_id,
                "label": labels,
                "exit_reason": [ExitReason(r).name for r in exit_reasons],
                "exit_bar_offset": exit_offsets,
                "entry_price": df["close"].values,
                "exit_price": exit_prices,
                "tp_price": tp_prices,
                "sl_price": sl_prices,
                "barrier_return": barrier_returns,
                "max_favorable_excursion": mfe,
                "max_adverse_excursion": mae,
                "risk_reward_ratio": risk_reward_ratio,
                "volatility_at_entry": df["atr"].values,
                "tp_multiplier": tp_multipliers,
                "sl_multiplier": sl_multipliers,
            }
        )

        valid_mask = result_df["exit_reason"] != ExitReason.INSUFFICIENT_DATA.name
        result_df = result_df[valid_mask].copy()
        result_df["return_per_bar"] = result_df["barrier_return"] / result_df["exit_bar_offset"]
        result_df["efficiency"] = result_df["barrier_return"] / (result_df["max_favorable_excursion"] + 1e-10)

        return result_df

    async def process_symbol(
        self, instrument_id: int, symbol: str, df: pd.DataFrame, timeframe: str = "15min"
    ) -> Optional[LabelingStats]:
        """Process a symbol and generate labels with comprehensive statistics."""
        start_time = datetime.now()
        loop = asyncio.get_running_loop()

        try:
            logger.info(f"Processing {symbol} (ID: {instrument_id}) with {len(df):,} bars")
            if len(df) < self.config.min_bars_required:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} < {self.config.min_bars_required}")
                return None

            if "ts" in df.columns and "timestamp" not in df.columns:
                df["timestamp"] = df["ts"]
            df_sorted = df.sort_values("timestamp").reset_index(drop=True)

            # --- SUGGESTION IMPLEMENTED: Pass symbol to ATR calculation ---
            df_sorted["atr"] = self._calculate_atr(df_sorted, symbol)

            tp_multipliers, sl_multipliers = self._calculate_dynamic_barriers(df_sorted, df_sorted["atr"].values)
            entry_prices = df_sorted["close"].values
            tp_prices = entry_prices + tp_multipliers * df_sorted["atr"].values
            sl_prices = entry_prices - sl_multipliers * df_sorted["atr"].values

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

            labeled_df = self._prepare_labels_dataframe(
                instrument_id, df_sorted, results, tp_multipliers, sl_multipliers
            )
            if labeled_df.empty:
                logger.warning(f"No valid labels generated for {symbol}")
                return None

            stats = self._calculate_statistics(symbol, df_sorted, labeled_df, start_time)

            if not labeled_df.empty:
                labels_to_insert = [LabelData(**row) for row in labeled_df.to_dict("records")]
                await self.label_repo.insert_labels(instrument_id, timeframe, labels_to_insert)
                logger.info(f"Stored {len(labels_to_insert)} labels for {symbol} in {stats.processing_time_ms:.2f}ms")

            with self.stats_lock:
                self.label_stats[symbol] = stats
            return stats

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}", exc_info=True)
            return None

    def _calculate_statistics(
        self, symbol: str, original_df: pd.DataFrame, labeled_df: pd.DataFrame, start_time: datetime
    ) -> LabelingStats:
        """Calculate comprehensive labeling statistics."""
        label_counts = labeled_df["label"].value_counts().to_dict()
        label_distribution = {
            "BUY": label_counts.get(Label.BUY, 0),
            "NEUTRAL": label_counts.get(Label.NEUTRAL, 0),
            "SELL": label_counts.get(Label.SELL, 0),
        }

        avg_returns = labeled_df.groupby("label")["barrier_return"].mean().to_dict()
        avg_return_by_label = {
            "BUY": avg_returns.get(Label.BUY, 0.0),
            "NEUTRAL": avg_returns.get(Label.NEUTRAL, 0.0),
            "SELL": avg_returns.get(Label.SELL, 0.0),
        }

        exit_reason_counts = labeled_df["exit_reason"].value_counts().to_dict()
        avg_holding = labeled_df["exit_bar_offset"].mean()
        returns = labeled_df["barrier_return"]
        sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)

        # --- SUGGESTION IMPLEMENTED: Correct win rate calculation ---
        buy_trades = labeled_df[labeled_df["label"] == Label.BUY]
        sell_trades = labeled_df[labeled_df["label"] == Label.SELL]
        total_directional_trades = len(buy_trades) + len(sell_trades)
        win_rate = len(buy_trades) / max(1, total_directional_trades)

        data_quality = len(labeled_df) / len(original_df) * 100
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return LabelingStats(
            symbol=symbol,
            total_bars=len(original_df),
            labeled_bars=len(labeled_df),
            label_distribution=label_distribution,
            avg_return_by_label=avg_return_by_label,
            exit_reasons=exit_reason_counts,
            avg_holding_period=avg_holding,
            processing_time_ms=processing_time,
            data_quality_score=data_quality,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
        )

    async def process_symbols_parallel(self, symbol_data: dict[int, dict[str, Any]]) -> dict[str, LabelingStats]:
        """Process multiple symbols in parallel with progress tracking."""
        tasks = [
            self.process_symbol(inst_id, info["symbol"], info["data"], info.get("timeframe", "15min"))
            for inst_id, info in symbol_data.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        stats_dict = {}
        for i, (_inst_id, info) in enumerate(symbol_data.items()):
            if isinstance(results[i], LabelingStats):
                stats_dict[info["symbol"]] = results[i]
            elif isinstance(results[i], Exception):
                logger.error(f"Failed to process {info['symbol']}: {results[i]}")

        logger.info(f"Parallel labeling completed: {len(stats_dict)}/{len(symbol_data)} symbols processed successfully")
        return stats_dict

    async def shutdown(self):
        """--- SUGGESTION IMPLEMENTED: Gracefully shuts down the process pool executor. ---"""
        logger.info("Shutting down the ProcessPoolExecutor.")
        self.executor.shutdown(wait=True)
