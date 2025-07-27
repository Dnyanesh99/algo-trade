import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, time
from enum import IntEnum
from typing import Any, Optional

import numpy as np
import pandas as pd
import psutil
from numba import jit, prange
from numpy.typing import NDArray

from src.database.instrument_repo import InstrumentRepository
from src.database.label_repo import LabelRepository
from src.database.label_stats_repo import LabelingStatsRepository
from src.database.models import LabelData, LabelingStatsData
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

config = ConfigLoader().get_config()


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
    atr_period: int
    tp_atr_multiplier: float
    sl_atr_multiplier: float
    max_holding_periods: int
    min_bars_required: int
    atr_smoothing: str
    epsilon: float
    use_dynamic_barriers: bool
    use_dynamic_epsilon: bool
    volatility_lookback: int
    dynamic_barrier_tp_sensitivity: float
    dynamic_barrier_sl_sensitivity: float
    sample_weight_decay_factor: float
    dynamic_epsilon: Any
    path_dependent_timeout: Any

    @classmethod
    def from_config(cls, labeling_config: Any) -> "BarrierConfig":
        """Create BarrierConfig from Pydantic LabelingConfig."""
        return cls(
            atr_period=labeling_config.atr_period,
            tp_atr_multiplier=labeling_config.tp_atr_multiplier,
            sl_atr_multiplier=labeling_config.sl_atr_multiplier,
            max_holding_periods=labeling_config.max_holding_periods,
            min_bars_required=labeling_config.min_bars_required,
            atr_smoothing=labeling_config.atr_smoothing,
            epsilon=labeling_config.epsilon,
            use_dynamic_barriers=labeling_config.use_dynamic_barriers,
            use_dynamic_epsilon=labeling_config.use_dynamic_epsilon,
            volatility_lookback=labeling_config.volatility_lookback,
            dynamic_barrier_tp_sensitivity=labeling_config.dynamic_barrier_tp_sensitivity,
            dynamic_barrier_sl_sensitivity=labeling_config.dynamic_barrier_sl_sensitivity,
            sample_weight_decay_factor=labeling_config.sample_weight_decay_factor,
            dynamic_epsilon=labeling_config.dynamic_epsilon,
            path_dependent_timeout=labeling_config.path_dependent_timeout,
        )


@dataclass
class LabelingStats:
    symbol: str
    timeframe: str
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
    profit_factor: float


@dataclass(frozen=True)
class LabelingInputs:
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    open: NDArray[np.float64]
    close: NDArray[np.float64]
    entry_prices: NDArray[np.float64]
    tp_prices: NDArray[np.float64]
    sl_prices: NDArray[np.float64]
    max_periods: int
    epsilon_array: NDArray[np.float64]
    path_dependent_timeout_config: Any


@dataclass(frozen=True)
class LabelingResult:
    labels: NDArray[np.int8]
    exit_bar_offsets: NDArray[np.int32]
    exit_prices: NDArray[np.float64]
    exit_reasons: NDArray[np.int8]
    mfe: NDArray[np.float64]
    mae: NDArray[np.float64]
    path_adjusted_returns: NDArray[np.float64]  # Added: path-adjusted returns for transparency


class OptimizedTripleBarrierLabeler:
    """
    Production-grade Triple Barrier Labeler with advanced features:
    - Dynamic barrier adjustment based on volatility regime
    - Path-dependent timeout labeling (De Prado, AFML)
    - Lookahead bias correction for ATR calculation
    - Configuration-driven parameters for market specifics
    - Refactored for clarity, testability, and maintainability.
    """

    def __init__(
        self,
        barrier_config: BarrierConfig,
        label_repo: LabelRepository,
        instrument_repo: InstrumentRepository,
        stats_repo: LabelingStatsRepository,
        allowed_timeframes: list[int],
        executor: Optional[ProcessPoolExecutor] = None,
    ):
        if not isinstance(barrier_config, BarrierConfig):
            raise TypeError("barrier_config must be a valid BarrierConfig instance.")
        if not all(
            isinstance(repo, (LabelRepository, InstrumentRepository, LabelingStatsRepository))
            for repo in [label_repo, instrument_repo, stats_repo]
        ):
            raise TypeError("All repository arguments must be valid repository instances.")

        self.config = barrier_config
        self.allowed_timeframes = allowed_timeframes
        self.validate_config()
        self.label_repo = label_repo
        self.instrument_repo = instrument_repo
        self.stats_repo = stats_repo
        self.label_stats: dict[str, LabelingStats] = {}
        self.stats_lock = asyncio.Lock()
        self.executor = executor or ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False))

        logger.info(
            f"TripleBarrierLabeler initialized with config: "
            f"TP={self.config.tp_atr_multiplier}x ATR, "
            f"SL={self.config.sl_atr_multiplier}x ATR, "
            f"Max holding={self.config.max_holding_periods} bars, "
            f"Allowed timeframes: {self.allowed_timeframes}"
        )

    def validate_config(self) -> None:
        """Validate barrier configuration for production use."""
        logger.info("Validating barrier configuration...")
        if self.config.tp_atr_multiplier <= 0:
            raise ValueError("TP multiplier must be positive")
        if self.config.sl_atr_multiplier <= 0:
            raise ValueError("SL multiplier must be positive")
        if self.config.tp_atr_multiplier <= self.config.sl_atr_multiplier:
            raise RuntimeError(
                f"CRITICAL CONFIGURATION ERROR: TP multiplier ({self.config.tp_atr_multiplier}) <= SL multiplier ({self.config.sl_atr_multiplier}) - Invalid risk/reward ratio will compromise trading performance"
            )
        if self.config.max_holding_periods < 2:
            raise ValueError("Max holding periods must be at least 2")
        if self.config.atr_period < 2:
            raise ValueError("ATR period must be at least 2")
        if self.config.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if self.config.volatility_lookback < 1:
            raise ValueError("Volatility lookback must be at least 1")
        if not 0 < self.config.dynamic_barrier_tp_sensitivity < 1:
            raise ValueError("Dynamic barrier TP sensitivity must be between 0 and 1")
        if not 0 < self.config.dynamic_barrier_sl_sensitivity < 1:
            raise ValueError("Dynamic barrier SL sensitivity must be between 0 and 1")
        if not 0 <= self.config.sample_weight_decay_factor <= 1:
            raise ValueError("Sample weight decay factor must be between 0 and 1")
        logger.info("Barrier configuration validated successfully.")

    def _validate_timeframe(self, timeframe: str) -> None:
        """Validate that the requested timeframe is allowed in configuration."""
        logger.debug(f"Validating timeframe: {timeframe}")
        import re

        match = re.match(r"(\d+)min", timeframe.lower())
        if not match:
            raise ValueError(f"Invalid timeframe format: '{timeframe}'. Expected format like '15min'.")

        timeframe_minutes = int(match.group(1))
        if timeframe_minutes not in self.allowed_timeframes:
            raise ValueError(
                f"Timeframe '{timeframe}' ({timeframe_minutes} minutes) is not allowed. "
                f"Configured labeling_timeframes: {self.allowed_timeframes}."
            )

    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
    def _calculate_true_range(
        high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        n = len(high)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        tr = np.empty(n, dtype=np.float64)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, max(hc, lc))
        return tr

    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
    def _calculate_ema(values: NDArray[np.float64], period: int) -> NDArray[np.float64]:
        n = len(values)
        if n < period:
            return np.full(n, np.nan, dtype=np.float64)
        ema = np.empty(n, dtype=np.float64)
        ema[: period - 1] = np.nan
        ema[period - 1] = np.mean(values[:period])
        alpha = 2.0 / (period + 1)
        for i in range(period, n):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _calculate_atr(self, df: pd.DataFrame) -> NDArray[np.float64]:
        """Calculate ATR. Caching removed as it's ineffective across processes."""
        logger.debug(
            f"Calculating ATR with period {self.config.atr_period} and smoothing '{self.config.atr_smoothing}'."
        )
        tr = self._calculate_true_range(df["high"].values, df["low"].values, df["close"].values)
        if self.config.atr_smoothing == "ema":
            atr = self._calculate_ema(tr, self.config.atr_period)
        else:
            atr = pd.Series(tr).rolling(window=self.config.atr_period).mean().values

        # Forward-fill initial NaN values
        first_valid_idx = np.argmax(~np.isnan(atr))
        if first_valid_idx > 0:
            atr[:first_valid_idx] = atr[first_valid_idx]
        return atr

    def _calculate_dynamic_epsilon(self, df: pd.DataFrame, atr: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.config.use_dynamic_epsilon:
            return np.full(len(df), self.config.epsilon, dtype=np.float64)

        logger.debug("Calculating dynamic epsilon...")
        n = len(df)
        if n < self.config.volatility_lookback:
            logger.warning("Insufficient data for dynamic epsilon, returning default.")
            return np.full(n, self.config.epsilon, dtype=np.float64)

        cfg = self.config.dynamic_epsilon
        atr_series = pd.Series(atr, index=df.index)
        atr_rolling_mean = atr_series.rolling(window=self.config.volatility_lookback, min_periods=1).mean()

        with np.errstate(divide="ignore", invalid="ignore"):
            # Ensure aligned indices for Series operations
            division_result = atr_series.values / atr_rolling_mean.values
            atr_ratio = np.where(atr_rolling_mean.values > 0, division_result, 1.0)

        regime_conditions = [
            atr_ratio < cfg.low_volatility_threshold,
            atr_ratio < cfg.high_volatility_threshold,
            atr_ratio < cfg.extreme_volatility_threshold,
        ]
        regime_choices = [
            cfg.low_volatility_multiplier,
            cfg.normal_volatility_multiplier,
            cfg.high_volatility_multiplier,
        ]
        regime_multipliers = np.select(regime_conditions, regime_choices, default=cfg.extreme_volatility_multiplier)

        time_multipliers = np.ones(n, dtype=np.float64)
        timestamps = self._extract_timestamps(df)
        if timestamps is not None:
            ist_timestamps = (
                timestamps.tz_convert("Asia/Kolkata")
                if timestamps.tz
                else timestamps.tz_localize("UTC").tz_convert("Asia/Kolkata")
            )

            hours = ist_timestamps.hour.values
            minutes = ist_timestamps.minute.values
            day_of_week = ist_timestamps.dayofweek.values

            session_cfg = cfg.market_session_times
            # Map session names to multiplier names in config
            session_multiplier_map = {
                "opening": "market_open_multiplier",
                "pre_lunch": "pre_lunch_multiplier",
                "closing": "market_close_multiplier",
            }

            for session in ["opening", "pre_lunch", "closing"]:
                # session_cfg is a dict[str, MarketSessionTime]
                if session in session_cfg:
                    session_obj = session_cfg[session]
                    start_time = time.fromisoformat(session_obj.start)
                    end_time = time.fromisoformat(session_obj.end)
                else:
                    logger.warning(f"Session '{session}' not found in market_session_times config")
                    continue
                mask = (hours > start_time.hour) | ((hours == start_time.hour) & (minutes >= start_time.minute))
                mask &= (hours < end_time.hour) | ((hours == end_time.hour) & (minutes <= end_time.minute))

                # Use the correct multiplier name from config
                multiplier_name = session_multiplier_map[session]
                multiplier_value = getattr(cfg, multiplier_name)
                time_multipliers = np.where(mask, multiplier_value, time_multipliers)

            time_multipliers = np.where(day_of_week == 0, time_multipliers * cfg.monday_multiplier, time_multipliers)
            weekly_expiry_mask = day_of_week == cfg.weekly_expiry_day
            time_multipliers = np.where(weekly_expiry_mask, time_multipliers * cfg.friday_multiplier, time_multipliers)

            monthly_expiry_mask = self._detect_monthly_expiry(ist_timestamps, cfg.weekly_expiry_day)
            time_multipliers = np.where(
                monthly_expiry_mask, time_multipliers * cfg.expiry_volatility_multiplier, time_multipliers
            )

        atr_rolling_std = atr_series.rolling(window=self.config.volatility_lookback, min_periods=1).std().fillna(0)
        volatility_zscore = (atr_series.values - atr_rolling_mean.values) / (atr_rolling_std.values + 1e-10)
        abs_zscore = np.abs(volatility_zscore)
        zscore_conditions = [
            abs_zscore > cfg.extreme_zscore_threshold,
            abs_zscore > cfg.moderate_zscore_threshold,
            abs_zscore < cfg.stable_zscore_threshold,
        ]
        zscore_choices = [cfg.extreme_zscore_multiplier, cfg.moderate_zscore_multiplier, cfg.stable_zscore_multiplier]
        zscore_multipliers = np.select(zscore_conditions, zscore_choices, default=1.0)

        final_multipliers = regime_multipliers * time_multipliers * zscore_multipliers
        final_multipliers = np.clip(final_multipliers, cfg.min_epsilon_multiplier, cfg.max_epsilon_multiplier)
        return self.config.epsilon * final_multipliers

    def _extract_timestamps(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        for col in ["timestamp", "ts"]:
            if col in df.columns:
                ts = pd.to_datetime(df[col])
                return pd.DatetimeIndex(ts)
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index
        raise ValueError("No timestamp information found in DataFrame")

    def _detect_monthly_expiry(self, timestamps: pd.DatetimeIndex, weekly_expiry_day: int) -> NDArray[np.bool_]:
        if timestamps is None or len(timestamps) == 0:
            return np.zeros(0, dtype=bool)
        df = pd.DataFrame({"timestamp": timestamps})
        df["month_str"] = df["timestamp"].dt.strftime("%Y-%m")
        df["is_expiry_day"] = df["timestamp"].dt.dayofweek == weekly_expiry_day

        # Get the max expiry day for each month, then map back to full DataFrame
        monthly_expiry_dates = df[df["is_expiry_day"]].groupby("month_str")["timestamp"].max()
        df["monthly_expiry_date"] = df["month_str"].map(monthly_expiry_dates)

        return (df["timestamp"] == df["monthly_expiry_date"]).fillna(False).values

    def _calculate_dynamic_barriers(
        self, df: pd.DataFrame, atr: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self.config.use_dynamic_barriers:
            return (np.full(len(df), self.config.tp_atr_multiplier), np.full(len(df), self.config.sl_atr_multiplier))

        logger.debug("Calculating dynamic barriers...")
        valid_atr = np.where(np.isnan(atr), np.nanmedian(atr), atr)
        atr_series = pd.Series(valid_atr, index=df.index)
        vol_mean = atr_series.rolling(window=self.config.volatility_lookback, min_periods=1).mean()
        vol_std = atr_series.rolling(window=self.config.volatility_lookback, min_periods=1).std().fillna(0)
        vol_zscore = (atr_series.values - vol_mean.values) / (vol_std.values + 1e-10)

        tp_multiplier = self.config.tp_atr_multiplier * (
            1 + self.config.dynamic_barrier_tp_sensitivity * np.tanh(vol_zscore)
        )
        sl_multiplier = self.config.sl_atr_multiplier * (
            1 + self.config.dynamic_barrier_sl_sensitivity * np.tanh(vol_zscore)
        )
        return tp_multiplier, sl_multiplier

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)  # type: ignore[misc]
    def _check_barriers_optimized(
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        open_: NDArray[np.float64],
        close: NDArray[np.float64],
        entry_prices: NDArray[np.float64],
        tp_prices: NDArray[np.float64],
        sl_prices: NDArray[np.float64],
        max_periods: int,
        epsilon_array: NDArray[np.float64],
        path_dependent_timeout_enabled: bool,
        timeout_upper_thresh: float,
        timeout_lower_thresh: float,
    ) -> tuple[
        NDArray[np.int8],
        NDArray[np.int32],
        NDArray[np.float64],
        NDArray[np.int8],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],  # Added: path_adjusted_returns
    ]:
        n = len(entry_prices)
        labels = np.zeros(n, dtype=np.int8)
        exit_bar_offsets = np.zeros(n, dtype=np.int32)
        exit_prices = np.zeros(n, dtype=np.float64)
        exit_reasons = np.full(n, ExitReason.INSUFFICIENT_DATA, dtype=np.int8)
        mfe = np.zeros(n, dtype=np.float64)
        mae = np.zeros(n, dtype=np.float64)
        path_adjusted_returns = np.zeros(n, dtype=np.float64)  # Track path-adjusted returns

        for i in prange(n):
            if np.isnan(tp_prices[i]) or np.isnan(sl_prices[i]) or np.isnan(entry_prices[i]) or entry_prices[i] <= 0:
                continue
            if i + max_periods >= n:
                continue

            exit_reasons[i] = ExitReason.TIME_OUT
            for j in range(1, max_periods + 1):
                bar_idx = i + j
                if bar_idx >= n:
                    break

                high_return = (high[bar_idx] - entry_prices[i]) / entry_prices[i]
                low_return = (low[bar_idx] - entry_prices[i]) / entry_prices[i]
                mfe[i] = max(mfe[i], high_return)
                mae[i] = min(mae[i], low_return)

                tp_hit = high[bar_idx] >= tp_prices[i]
                sl_hit = low[bar_idx] <= sl_prices[i]

                if tp_hit and sl_hit:
                    open_to_tp = abs(open_[bar_idx] - tp_prices[i])
                    open_to_sl = abs(open_[bar_idx] - sl_prices[i])
                    if open_to_tp < open_to_sl:
                        labels[i], exit_reasons[i], exit_prices[i] = Label.BUY, ExitReason.TP_HIT, tp_prices[i]
                        # CRITICAL FIX: Set path_adjusted_return for TP_HIT cases
                        path_adjusted_returns[i] = (tp_prices[i] - entry_prices[i]) / entry_prices[i]
                    else:
                        labels[i], exit_reasons[i], exit_prices[i] = Label.SELL, ExitReason.SL_HIT, sl_prices[i]
                        # CRITICAL FIX: Set path_adjusted_return for SL_HIT cases
                        path_adjusted_returns[i] = (sl_prices[i] - entry_prices[i]) / entry_prices[i]
                    exit_bar_offsets[i] = j
                    break
                if tp_hit:
                    labels[i], exit_reasons[i], exit_prices[i] = Label.BUY, ExitReason.TP_HIT, tp_prices[i]
                    # CRITICAL FIX: Set path_adjusted_return for TP_HIT cases
                    path_adjusted_returns[i] = (tp_prices[i] - entry_prices[i]) / entry_prices[i]
                    exit_bar_offsets[i] = j
                    break
                if sl_hit:
                    labels[i], exit_reasons[i], exit_prices[i] = Label.SELL, ExitReason.SL_HIT, sl_prices[i]
                    # CRITICAL FIX: Set path_adjusted_return for SL_HIT cases
                    path_adjusted_returns[i] = (sl_prices[i] - entry_prices[i]) / entry_prices[i]
                    exit_bar_offsets[i] = j
                    break

            if exit_reasons[i] == ExitReason.TIME_OUT:
                exit_idx = min(i + max_periods, n - 1)
                exit_prices[i] = close[exit_idx]
                exit_bar_offsets[i] = exit_idx - i
                final_return = (exit_prices[i] - entry_prices[i]) / entry_prices[i]

                if path_dependent_timeout_enabled:
                    total_range = mfe[i] - mae[i]
                    if total_range > 1e-9:
                        # CORRECTED: Proper path-dependent normalization per De Prado AFML
                        # Normalize final_return position within [MAE, MFE] range to [0,1], then to [-1,+1]
                        relative_position = (final_return - mae[i]) / total_range
                        path_adjusted_return = (relative_position * 2.0) - 1.0
                    else:
                        # If no range (MAE == MFE), treat as neutral
                        path_adjusted_return = 0.0

                    path_adjusted_returns[i] = path_adjusted_return

                    if path_adjusted_return > timeout_upper_thresh:
                        labels[i] = Label.BUY
                    elif path_adjusted_return < timeout_lower_thresh:
                        labels[i] = Label.SELL
                    else:
                        labels[i] = Label.NEUTRAL
                else:
                    # Standard epsilon-based labeling for timeout cases
                    current_epsilon = epsilon_array[i]
                    path_adjusted_returns[i] = final_return  # Store actual return for consistency

                    if final_return > current_epsilon:
                        labels[i] = Label.BUY
                    elif final_return < -current_epsilon:
                        labels[i] = Label.SELL
                    else:
                        labels[i] = Label.NEUTRAL

        return labels, exit_bar_offsets, exit_prices, exit_reasons, mfe, mae, path_adjusted_returns

    def _prepare_labels_dataframe(
        self,
        instrument_id: int,
        df: pd.DataFrame,
        results: LabelingResult,
        entry_prices: NDArray[np.float64],
        tp_prices: NDArray[np.float64],
        sl_prices: NDArray[np.float64],
        tp_multipliers: NDArray[np.float64],
        sl_multipliers: NDArray[np.float64],
        timeframe: str,
    ) -> pd.DataFrame:
        logger.debug("Preparing final labeled DataFrame...")
        with np.errstate(divide="ignore", invalid="ignore"):
            barrier_returns = (results.exit_prices - entry_prices) / entry_prices
            potential_profit = tp_prices - entry_prices
            potential_loss = entry_prices - sl_prices
            risk_reward_ratio = potential_profit / (potential_loss + 1e-10)

        result_df = pd.DataFrame(
            {
                "ts": self._extract_timestamps(df),
                "timeframe": timeframe,
                "instrument_id": instrument_id,
                "label": results.labels,
                "exit_reason": [ExitReason(r).name for r in results.exit_reasons],
                "exit_bar_offset": results.exit_bar_offsets,
                "entry_price": entry_prices,
                "exit_price": results.exit_prices,
                "tp_price": tp_prices,
                "sl_price": sl_prices,
                "barrier_return": barrier_returns,
                "path_adjusted_return": results.path_adjusted_returns,  # Added: for consistency analysis
                "max_favorable_excursion": results.mfe,
                "max_adverse_excursion": results.mae,
                "risk_reward_ratio": risk_reward_ratio,
                "volatility_at_entry": df["atr"].values,
                "tp_multiplier": tp_multipliers,
                "sl_multiplier": sl_multipliers,
            }
        )

        valid_mask = (result_df["exit_reason"] != ExitReason.INSUFFICIENT_DATA.name) & (
            ~pd.isna(result_df["entry_price"])
        )
        result_df = result_df[valid_mask].copy()

        with np.errstate(divide="ignore", invalid="ignore"):
            result_df["return_per_bar"] = result_df["barrier_return"] / result_df["exit_bar_offset"].replace(0, 1)
            result_df["efficiency"] = result_df["barrier_return"] / (result_df["max_favorable_excursion"] + 1e-10)

        logger.info(f"Prepared {len(result_df)} valid labeled records.")
        return result_df

    async def _validate_and_prepare_data(
        self, df: pd.DataFrame, symbol: str, instrument_id: int
    ) -> tuple[bool, Optional[pd.DataFrame]]:
        logger.debug(f"Validating and preparing data for {symbol} (ID: {instrument_id}).")
        instrument = await self.instrument_repo.get_instrument_by_id(instrument_id)
        segment = (instrument.segment or "") if instrument else ""
        is_index = segment in config.broker.segment_types and config.broker.segment_types[0] in segment

        required_columns = {"open", "high", "low", "close", "timestamp"}
        if not required_columns.issubset(df.columns):
            raise RuntimeError(f"Missing required columns for {symbol}: {required_columns - set(df.columns)}")

        essential_cols = (
            ["open", "high", "low", "close", "timestamp"]
            if is_index
            else ["open", "high", "low", "close", "volume", "timestamp"]
        )
        df_essential = df[essential_cols].dropna()

        if len(df_essential) < self.config.min_bars_required:
            raise RuntimeError(
                f"Insufficient non-NaN data for {symbol}: {len(df_essential)} < {self.config.min_bars_required}."
            )

        # Use the cleaned data for all subsequent processing
        df_clean = df.dropna(subset=essential_cols)

        price_cols = ["open", "high", "low", "close"]
        if not np.isfinite(df_clean[price_cols].values).all() or (df_clean[price_cols] <= 0).any().any():
            raise RuntimeError(f"Non-finite or non-positive prices found for {symbol}.")

        df_sorted = df_clean.sort_values("timestamp").reset_index(drop=True)
        if df_sorted["timestamp"].duplicated().any():
            logger.warning(f"Duplicate timestamps found for {symbol}. Dropping duplicates.")
            df_sorted = df_sorted.drop_duplicates(subset=["timestamp"], keep="first")

        logger.info(f"Data validation and preparation successful for {symbol}.")
        return True, df_sorted

    def _prepare_labeling_inputs(
        self, df: pd.DataFrame
    ) -> tuple[LabelingInputs, NDArray[np.float64], NDArray[np.float64]]:
        logger.debug("Preparing inputs for barrier labeling...")
        df["atr"] = self._calculate_atr(df)
        entry_prices = df["open"].shift(-1).values

        # CRITICAL FIX: Prevent lookahead bias by using ATR from the previous bar.
        atr_at_entry = df["atr"].shift(1).values

        # Fix NaN in first position by using current ATR (acceptable for first bar)
        first_valid_atr_idx = np.argmax(~np.isnan(df["atr"].values))
        if np.isnan(atr_at_entry[0]) and first_valid_atr_idx < len(df):
            atr_at_entry[0] = df["atr"].values[first_valid_atr_idx]

        # Forward fill any remaining NaN values in atr_at_entry
        atr_series = pd.Series(atr_at_entry)
        atr_at_entry = atr_series.fillna(method="ffill").fillna(method="bfill").values

        tp_multipliers, sl_multipliers = self._calculate_dynamic_barriers(df, atr_at_entry)
        tp_prices = entry_prices + tp_multipliers * atr_at_entry
        sl_prices = entry_prices - sl_multipliers * atr_at_entry

        # Validate that we don't have NaN barriers (except for intentionally invalidated last entry)
        nan_tp_mask = np.isnan(tp_prices[:-1])  # Exclude last entry which is intentionally NaN
        nan_sl_mask = np.isnan(sl_prices[:-1])  # Exclude last entry which is intentionally NaN

        if nan_tp_mask.any() or nan_sl_mask.any():
            nan_indices = np.where(nan_tp_mask | nan_sl_mask)[0]
            logger.warning(f"Found NaN barrier prices at indices {nan_indices}. Setting to invalid.")
            # Mark these entries as invalid by setting entry_prices to NaN
            entry_prices[nan_indices] = np.nan
            tp_prices[nan_indices] = np.nan
            sl_prices[nan_indices] = np.nan

        dynamic_epsilon_array = self._calculate_dynamic_epsilon(df, atr_at_entry)
        epsilon_array = np.nan_to_num(dynamic_epsilon_array, nan=self.config.epsilon)

        # Invalidate last entry to prevent out-of-bounds access
        entry_prices[-1] = np.nan
        tp_prices[-1] = np.nan
        sl_prices[-1] = np.nan

        timeout_cfg = self.config.path_dependent_timeout
        inputs = LabelingInputs(
            high=df["high"].values,
            low=df["low"].values,
            open=df["open"].values,
            close=df["close"].values,
            entry_prices=entry_prices,
            tp_prices=tp_prices,
            sl_prices=sl_prices,
            max_periods=self.config.max_holding_periods,
            epsilon_array=epsilon_array,
            path_dependent_timeout_config={
                "enabled": timeout_cfg.enabled,
                "upper_thresh": timeout_cfg.upper_threshold,
                "lower_thresh": timeout_cfg.lower_threshold,
            },
        )
        logger.debug("Labeling inputs prepared successfully.")
        return inputs, tp_multipliers, sl_multipliers

    async def _run_barrier_check(self, inputs: LabelingInputs) -> LabelingResult:
        logger.debug("Running optimized barrier check in parallel...")
        loop = asyncio.get_running_loop()
        path_cfg = inputs.path_dependent_timeout_config

        results_tuple = await loop.run_in_executor(
            self.executor,
            self._check_barriers_optimized,
            inputs.high,
            inputs.low,
            inputs.open,
            inputs.close,
            inputs.entry_prices,
            inputs.tp_prices,
            inputs.sl_prices,
            inputs.max_periods,
            inputs.epsilon_array,
            path_cfg["enabled"],
            path_cfg["upper_thresh"],
            path_cfg["lower_thresh"],
        )
        logger.debug("Optimized barrier check completed.")
        return LabelingResult(*results_tuple)

    async def _calculate_and_store_results(
        self,
        instrument_id: int,
        symbol: str,
        timeframe: str,
        original_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        start_time: datetime,
    ) -> LabelingStats:
        logger.debug(f"Calculating and storing results for {symbol} ({timeframe}).")
        stats = self._calculate_statistics(symbol, original_df, labeled_df, start_time, timeframe)

        if not labeled_df.empty:
            labels_to_insert = [LabelData(**row) for row in labeled_df.to_dict("records")]
            await self.label_repo.insert_labels(instrument_id, labels_to_insert)
            logger.info(f"Stored {len(labels_to_insert)} labels for {symbol} in {stats.processing_time_ms:.2f}ms.")

            stats_data = LabelingStatsData(**stats.__dict__)
            await self.stats_repo.insert_stats(stats_data)
            logger.info(f"Stored labeling statistics for {symbol}.")

        async with self.stats_lock:
            self.label_stats[symbol] = stats
        return stats

    async def process_symbol(
        self, instrument_id: int, symbol: str, df: pd.DataFrame, timeframe: str = "15min"
    ) -> Optional[LabelingStats]:
        logger.info(f"Processing {symbol} (ID: {instrument_id}) with {len(df):,} bars for timeframe {timeframe}.")
        try:
            self._validate_timeframe(timeframe)
        except ValueError as e:
            logger.error(f"Cannot process {symbol}: {e}", exc_info=True)
            return None

        start_time = datetime.now()
        try:
            is_valid, df_sorted = await self._validate_and_prepare_data(df, symbol, instrument_id)
            if not is_valid or df_sorted is None:
                logger.error(f"Data validation failed for {symbol}, cannot proceed with labeling.")
                return None

            inputs, tp_multipliers, sl_multipliers = self._prepare_labeling_inputs(df_sorted)
            results = await self._run_barrier_check(inputs)

            labeled_df = self._prepare_labels_dataframe(
                instrument_id,
                df_sorted,
                results,
                inputs.entry_prices,
                inputs.tp_prices,
                inputs.sl_prices,
                tp_multipliers,
                sl_multipliers,
                timeframe,
            )

            if labeled_df.empty:
                logger.warning(
                    f"No valid labels were generated for {symbol}. This may be due to insufficient data or market conditions."
                )
                return None

            return await self._calculate_and_store_results(
                instrument_id, symbol, timeframe, df_sorted, labeled_df, start_time
            )

        except Exception as e:
            logger.error(f"A critical error occurred while processing {symbol}: {e}", exc_info=True)
            return None

    def _get_annualization_factor(self, timeframe: str) -> float:
        logger.debug(f"Calculating annualization factor for timeframe: {timeframe}")
        timeframe_upper = str(timeframe).upper().replace(" ", "")
        import re

        match = re.match(r"(\d+)([A-Z]+)", timeframe_upper)
        if not match:
            logger.error(f"Could not parse timeframe '{timeframe}'. Expected format like '15MIN', '1H', '1D'.")
            raise ValueError(f"Invalid timeframe format: {timeframe}")

        value = int(match.group(1))
        unit = match.group(2)

        MINUTES_PER_DAY = 375  # Typical for equity markets
        DAYS_PER_YEAR = 252

        if "MIN" in unit:
            return (MINUTES_PER_DAY / value) * DAYS_PER_YEAR
        if "H" in unit:
            return (MINUTES_PER_DAY / (value * 60)) * DAYS_PER_YEAR
        if "D" in unit:
            return DAYS_PER_YEAR / value
        if "W" in unit:
            return 52 / value
        if "M" in unit:
            return 12 / value

        logger.error(f"Unknown timeframe unit '{unit}' in timeframe '{timeframe}'.")
        raise ValueError(f"Unknown timeframe unit: {unit}")

    def _calculate_statistics(
        self, symbol: str, original_df: pd.DataFrame, labeled_df: pd.DataFrame, start_time: datetime, timeframe: str
    ) -> LabelingStats:
        logger.debug(f"Calculating statistics for {symbol} ({timeframe}).")
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

        annualization_factor = self._get_annualization_factor(timeframe)
        sharpe_ratio = np.sqrt(annualization_factor) * returns.mean() / (returns.std() + 1e-10)

        conclusive_trades = labeled_df[labeled_df["exit_reason"].isin([ExitReason.TP_HIT.name, ExitReason.SL_HIT.name])]
        win_rate = (
            len(conclusive_trades[conclusive_trades["exit_reason"] == ExitReason.TP_HIT.name]) / len(conclusive_trades)
            if not conclusive_trades.empty
            else 0.0
        )

        tp_returns = labeled_df[labeled_df["exit_reason"] == ExitReason.TP_HIT.name]["barrier_return"]
        sl_returns = labeled_df[labeled_df["exit_reason"] == ExitReason.SL_HIT.name]["barrier_return"]
        gross_profits = tp_returns[tp_returns > 0].sum()
        gross_losses = abs(sl_returns[sl_returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 1e-9 else float("inf")

        return LabelingStats(
            symbol=symbol,
            timeframe=timeframe,
            total_bars=len(original_df),
            labeled_bars=len(labeled_df),
            label_distribution=label_distribution,
            avg_return_by_label=avg_return_by_label,
            exit_reasons=exit_reason_counts,
            avg_holding_period=float(avg_holding),
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            data_quality_score=len(labeled_df) / len(original_df) * 100 if len(original_df) > 0 else 0,
            sharpe_ratio=float(sharpe_ratio),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
        )

    def calculate_sample_weights(self, labeled_df: pd.DataFrame) -> NDArray[np.float64]:
        logger.debug(f"Calculating sample weights for {len(labeled_df)} labels.")
        n = len(labeled_df)
        if n == 0:
            return np.array([])

        weights = np.ones(n)
        df = labeled_df.sort_values("ts").reset_index(drop=True)

        for i in range(n):
            start_i = i
            end_i = i + df.iloc[i]["exit_bar_offset"]
            total_overlap = 0.0
            overlap_count = 0
            for j in range(n):
                if i == j:
                    continue
                start_j = j
                end_j = j + df.iloc[j]["exit_bar_offset"]
                if start_i <= end_j and end_i >= start_j:
                    overlap_start = max(start_i, start_j)
                    overlap_end = min(end_i, end_j)
                    overlap_fraction = (overlap_end - overlap_start + 1) / (end_i - start_i + 1)
                    total_overlap += overlap_fraction
                    overlap_count += 1
            if overlap_count > 0:
                weights[i] = np.exp(-self.config.sample_weight_decay_factor * (total_overlap / overlap_count))

        normalized_weights = weights / weights.sum() * n
        logger.info("Sample weight calculation completed.")
        return normalized_weights

    async def process_symbols_parallel(self, symbol_data: dict[int, dict[str, Any]]) -> dict[str, LabelingStats]:
        logger.info(f"Processing {len(symbol_data)} symbols in parallel.")
        tasks = [
            self.process_symbol(inst_id, info["symbol"], info["data"], info.get("timeframe", "15min"))
            for inst_id, info in symbol_data.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        stats_dict: dict[str, LabelingStats] = {}
        for i, (_inst_id, info) in enumerate(symbol_data.items()):
            result_item = results[i]
            if isinstance(result_item, LabelingStats):
                stats_dict[info["symbol"]] = result_item
            elif isinstance(result_item, Exception):
                logger.error(f"An exception occurred while processing {info['symbol']}: {result_item}", exc_info=True)
            else:
                logger.warning(f"Processing for {info['symbol']} did not return statistics.")

        successful_count = len(stats_dict)
        total_count = len(symbol_data)
        logger.info(f"Parallel labeling completed: {successful_count}/{total_count} symbols processed successfully.")
        if successful_count < total_count:
            logger.error(f"{total_count - successful_count} symbols failed to process.")

        return stats_dict

    async def shutdown(self) -> None:
        logger.info("Shutting down the ProcessPoolExecutor.")
        self.executor.shutdown(wait=True)
