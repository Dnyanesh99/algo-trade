import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.labeler import BarrierConfig, ExitReason, Label, LabelingStats, OptimizedTripleBarrierLabeler
from src.database.instrument_repo import InstrumentRepository
from src.database.label_repo import LabelRepository


class MockConfigLoader:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {
            "trading": {
                "labeling": {
                    "atr_period": 14,
                    "tp_atr_multiplier": 2.0,
                    "sl_atr_multiplier": 1.5,
                    "atr_smoothing": "ema",
                    "max_holding_periods": 10,
                    "min_bars_required": 100,
                    "epsilon": 0.0005,
                    "use_dynamic_barriers": True,
                    "use_dynamic_epsilon": False,
                    "volatility_lookback": 20,
                    "atr_cache_size": 100,
                    "dynamic_barrier_tp_sensitivity": 0.2,
                    "dynamic_barrier_sl_sensitivity": 0.1,
                    "sample_weight_decay_factor": 0.5,
                    "ohlcv_data_limit_for_labeling": 5000,
                    "minimum_ohlcv_data_for_labeling": 200,
                    "dynamic_epsilon": {
                        "low_volatility_threshold": 0.8,
                        "high_volatility_threshold": 1.2,
                        "extreme_volatility_threshold": 1.5,
                        "low_volatility_multiplier": 0.5,
                        "normal_volatility_multiplier": 1.0,
                        "high_volatility_multiplier": 1.5,
                        "extreme_volatility_multiplier": 2.0,
                        "market_open_multiplier": 1.2,
                        "pre_lunch_multiplier": 0.9,
                        "market_close_multiplier": 1.3,
                        "monday_multiplier": 1.1,
                        "friday_multiplier": 1.1,
                        "weekly_expiry_day": 3,
                        "expiry_volatility_multiplier": 1.4,
                        "extreme_zscore_threshold": 2.0,
                        "moderate_zscore_threshold": 1.0,
                        "stable_zscore_threshold": 0.5,
                        "extreme_zscore_multiplier": 2.0,
                        "moderate_zscore_multiplier": 1.3,
                        "stable_zscore_multiplier": 0.8,
                        "min_epsilon_multiplier": 0.1,
                        "max_epsilon_multiplier": 5.0,
                    },
                }
            },
            "broker": {"segment_types": ["INDEX", "STOCK"]},
        }

    def load_config(self) -> dict[str, Any]:
        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        keys: list[str] = key.split(".")
        val: dict[str, Any] = self.config
        try:
            for k in keys:
                val = val[k]
            return val
        except KeyError:
            return default

    def get_config(self) -> MagicMock:
        mock_trading = MagicMock()
        mock_trading.labeling = MagicMock(
            atr_period=self.config["trading"]["labeling"]["atr_period"],
            tp_atr_multiplier=self.config["trading"]["labeling"]["tp_atr_multiplier"],
            sl_atr_multiplier=self.config["trading"]["labeling"]["sl_atr_multiplier"],
            atr_smoothing=self.config["trading"]["labeling"]["atr_smoothing"],
            max_holding_periods=self.config["trading"]["labeling"]["max_holding_periods"],
            min_bars_required=self.config["trading"]["labeling"]["min_bars_required"],
            epsilon=self.config["trading"]["labeling"]["epsilon"],
            use_dynamic_barriers=self.config["trading"]["labeling"]["use_dynamic_barriers"],
            use_dynamic_epsilon=self.config["trading"]["labeling"]["use_dynamic_epsilon"],
            volatility_lookback=self.config["trading"]["labeling"]["volatility_lookback"],
            atr_cache_size=self.config["trading"]["labeling"]["atr_cache_size"],
            dynamic_barrier_tp_sensitivity=self.config["trading"]["labeling"]["dynamic_barrier_tp_sensitivity"],
            dynamic_barrier_sl_sensitivity=self.config["trading"]["labeling"]["dynamic_barrier_sl_sensitivity"],
            sample_weight_decay_factor=self.config["trading"]["labeling"]["sample_weight_decay_factor"],
            dynamic_epsilon=self.config["trading"]["labeling"]["dynamic_epsilon"],
        )
        mock_config = MagicMock()
        mock_config.trading = mock_trading
        mock_broker = MagicMock()
        mock_broker.segment_types = self.config["broker"]["segment_types"]
        mock_config.broker = mock_broker
        return mock_config


@pytest.fixture
def mock_config_loader_instance() -> MockConfigLoader:
    return MockConfigLoader()


@pytest.fixture
def labeler_instance(
    mock_config_loader_instance: MockConfigLoader,
) -> OptimizedTripleBarrierLabeler:
    mock_label_repo = AsyncMock(spec=LabelRepository)
    mock_stats_repo = AsyncMock()
    mock_executor = MagicMock()
    mock_executor.submit = AsyncMock(return_value=asyncio.Future())
    mock_executor.shutdown = AsyncMock()
    (mock_executor.submit.return_value).set_result(
        (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    )

    with patch("src.utils.config_loader.ConfigLoader") as MockCL:
        MockCL.return_value.get_config.return_value = mock_config_loader_instance.get_config()

        labeling_config = mock_config_loader_instance.get_config().trading.labeling
        barrier_config = BarrierConfig(
            atr_period=labeling_config.atr_period,
            tp_atr_multiplier=labeling_config.tp_atr_multiplier,
            sl_atr_multiplier=labeling_config.sl_atr_multiplier,
            atr_smoothing=labeling_config.atr_smoothing,
            max_holding_periods=labeling_config.max_holding_periods,
            min_bars_required=labeling_config.min_bars_required,
            epsilon=labeling_config.epsilon,
            use_dynamic_barriers=labeling_config.use_dynamic_barriers,
            use_dynamic_epsilon=labeling_config.use_dynamic_epsilon,
            volatility_lookback=labeling_config.volatility_lookback,
            atr_cache_size=labeling_config.atr_cache_size,
            dynamic_barrier_tp_sensitivity=labeling_config.dynamic_barrier_tp_sensitivity,
            dynamic_barrier_sl_sensitivity=labeling_config.dynamic_barrier_sl_sensitivity,
            sample_weight_decay_factor=labeling_config.sample_weight_decay_factor,
            dynamic_epsilon=labeling_config.dynamic_epsilon,
        )

        mock_instrument_repo = MagicMock()
        return OptimizedTripleBarrierLabeler(barrier_config, mock_label_repo, mock_instrument_repo, mock_stats_repo, [15], mock_executor)


@pytest.fixture
def sample_ohlcv_data_for_labeling() -> pd.DataFrame:
    data_size: int = 200
    np.random.seed(42)
    dates: pd.DatetimeIndex = pd.date_range(start="2023-01-01", periods=data_size, freq="min")
    open_prices: np.ndarray = np.random.uniform(100, 110, data_size).cumsum() + 1000
    close_prices: np.ndarray = open_prices + np.random.uniform(-2, 2, data_size)
    high_prices: np.ndarray = np.maximum(open_prices, close_prices) + np.random.uniform(0, 1, data_size)
    low_prices: np.ndarray = np.minimum(open_prices, close_prices) - np.random.uniform(0, 1, data_size)
    volume: np.ndarray = np.random.randint(1000, 5000, data_size)

    df = pd.DataFrame(
        {
            "ts": dates,
            "timestamp": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        }
    )
    return df.set_index("ts")


@pytest.fixture
async def real_ohlcv_repository() -> object:
    """
    Fixture that provides a real OHLCV repository for database testing.
    """
    try:
        from src.database.ohlcv_repo import OHLCVRepository

        return OHLCVRepository()
    except Exception as e:
        pytest.skip(f"Could not create OHLCV repository: {e}")


@pytest.fixture
async def real_instrument_repository() -> object:
    """
    Fixture that provides a real instrument repository for database testing.
    """
    try:
        from src.database.instrument_repo import InstrumentRepository

        return InstrumentRepository()
    except Exception as e:
        pytest.skip(f"Could not create instrument repository: {e}")


@pytest.fixture
async def sample_instrument_data(real_instrument_repository) -> tuple[int, str]:
    """
    Fixture that fetches a sample instrument from the database for testing.
    Returns (instrument_id, symbol) tuple.
    """
    try:
        # Get available instruments
        instruments = await real_instrument_repository.get_all_instruments()

        if not instruments:
            pytest.skip("No instruments found in database")

        # Prefer liquid instruments like NIFTY, BANKNIFTY for testing
        preferred_symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]

        for instrument in instruments:
            if any(pref in instrument.symbol.upper() for pref in preferred_symbols):
                return instrument.id, instrument.symbol

        # If no preferred instrument found, use the first available
        return instruments[0].id, instruments[0].symbol

    except Exception as e:
        pytest.skip(f"Could not fetch instrument data: {e}")


@pytest.fixture
async def real_ohlcv_data_from_db(real_ohlcv_repository, sample_instrument_data) -> tuple[pd.DataFrame, int, str]:
    """
    Fixture that fetches real OHLCV data from database using the repository pattern.
    Returns (dataframe, instrument_id, symbol) tuple.
    """
    try:
        instrument_id, symbol = sample_instrument_data

        # Get recent OHLCV data for feature calculation/labeling
        ohlcv_data = await real_ohlcv_repository.get_ohlcv_data_for_features(
            instrument_id=instrument_id,
            timeframe=5,  # 5 minute timeframe
            limit=1000,  # Get last 1000 candles
        )

        if not ohlcv_data or len(ohlcv_data) < 200:
            # Try different timeframe if 5min doesn't have enough data
            ohlcv_data = await real_ohlcv_repository.get_ohlcv_data_for_features(
                instrument_id=instrument_id,
                timeframe=15,  # Try 15 minute timeframe
                limit=1000,
            )

        if not ohlcv_data or len(ohlcv_data) < 200:
            pytest.skip(f"Insufficient OHLCV data for instrument {instrument_id} ({symbol})")

        # Convert to DataFrame in the format expected by labeler
        df = pd.DataFrame(
            [
                {
                    "ts": row.timestamp,
                    "timestamp": row.timestamp,
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": int(row.volume),
                }
                for row in ohlcv_data
            ]
        )
        df = df.set_index("ts")

        return df, instrument_id, symbol

    except Exception as e:
        pytest.skip(f"Could not fetch real OHLCV data: {e}")


@pytest.fixture
async def historical_ohlcv_range_data(real_ohlcv_repository, sample_instrument_data) -> tuple[pd.DataFrame, int, str]:
    """
    Fixture that fetches historical OHLCV data for a specific date range.
    """
    try:
        from datetime import datetime, timedelta

        instrument_id, symbol = sample_instrument_data

        # Get data from last 30 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)

        ohlcv_data = await real_ohlcv_repository.get_ohlcv_data(
            instrument_id=instrument_id, timeframe="5min", start_time=start_time, end_time=end_time
        )

        if not ohlcv_data or len(ohlcv_data) < 100:
            pytest.skip(f"Insufficient historical data for instrument {instrument_id} ({symbol})")

        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {
                    "ts": row.timestamp,
                    "timestamp": row.timestamp,
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": int(row.volume),
                }
                for row in ohlcv_data
            ]
        )
        df = df.set_index("ts")

        return df, instrument_id, symbol

    except Exception as e:
        pytest.skip(f"Could not fetch historical range data: {e}")


def _generate_synthetic_ohlcv_data() -> pd.DataFrame:
    """Fallback synthetic data generator when real data is not available."""
    data_size = 500
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=data_size, freq="5min")

    # Generate more realistic price movements
    base_price = 18000  # NIFTY-like base price
    returns = np.random.normal(0, 0.001, data_size)  # 0.1% std returns
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from price series
    open_prices = prices.copy()
    close_prices = prices * (1 + np.random.normal(0, 0.0005, data_size))

    # High and low should respect open/close bounds
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.uniform(0, 0.002, data_size))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.uniform(0, 0.002, data_size))

    volume = np.random.randint(1000, 50000, data_size)

    df = pd.DataFrame(
        {
            "ts": dates,
            "timestamp": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        }
    )
    return df.set_index("ts")


# ============================================================================
# ORIGINAL TESTS (UPDATED)
# ============================================================================


@pytest.mark.asyncio
async def test_calculate_atr(
    labeler_instance: OptimizedTripleBarrierLabeler, sample_ohlcv_data_for_labeling: pd.DataFrame
) -> None:
    df: pd.DataFrame = sample_ohlcv_data_for_labeling.copy()
    atr_values: np.ndarray = labeler_instance._calculate_atr(df, "TEST_SYMBOL")
    assert "atr" not in df.columns  # Should not modify original df
    assert isinstance(atr_values, np.ndarray)
    assert len(atr_values) == len(df)
    assert not np.isnan(atr_values).all()  # ATR should be calculated for most rows
    assert np.all(atr_values[~np.isnan(atr_values)] >= 0)  # Ensure non-negative ATR values


@pytest.mark.asyncio
async def test_process_symbol_basic_tp(
    labeler_instance: OptimizedTripleBarrierLabeler, sample_ohlcv_data_for_labeling: pd.DataFrame
) -> None:
    df = sample_ohlcv_data_for_labeling.copy()
    # Manually set prices to trigger TP
    df["close"] = np.linspace(
        1000,
        1000 + 2 * df["close"].std() * labeler_instance.config.tp_atr_multiplier,
        len(df),
    )
    df["open"] = df["high"] = df["low"] = df["close"]

    # Mock the executor to return a result that simulates TP hit
    labels_mock = np.full(len(df), Label.NEUTRAL.value, dtype=np.int8)
    exit_bar_offsets_mock = np.zeros(len(df), dtype=np.int32)
    exit_prices_mock = np.zeros(len(df), dtype=np.float64)
    exit_reasons_mock = np.full(len(df), 0, dtype=np.int8)  # TIME_OUT

    # Simulate TP hit for the first entry
    labels_mock[0] = Label.BUY.value
    exit_bar_offsets_mock[0] = 1
    exit_prices_mock[0] = df["close"].iloc[1]
    exit_reasons_mock[0] = 1  # TP_HIT

    (labeler_instance.executor.submit.return_value).set_result(
        (labels_mock, exit_bar_offsets_mock, exit_prices_mock, exit_reasons_mock, np.zeros(len(df)), np.zeros(len(df)))
    )

    stats = await labeler_instance.process_symbol(1, "TEST_SYMBOL", df)

    assert stats is not None
    assert stats.labeled_bars > 0
    assert stats.label_distribution["BUY"] > 0
    assert stats.exit_reasons["TP_HIT"] > 0


@pytest.mark.asyncio
async def test_process_symbol_basic_sl(
    labeler_instance: OptimizedTripleBarrierLabeler, sample_ohlcv_data_for_labeling: pd.DataFrame
) -> None:
    df = sample_ohlcv_data_for_labeling.copy()
    # Manually set prices to trigger SL
    df["close"] = np.linspace(
        1000,
        1000 - 2 * df["close"].std() * labeler_instance.config.sl_atr_multiplier,
        len(df),
    )
    df["open"] = df["high"] = df["low"] = df["close"]

    # Mock the executor to return a result that simulates SL hit
    labels_mock = np.full(len(df), Label.NEUTRAL.value, dtype=np.int8)
    exit_bar_offsets_mock = np.zeros(len(df), dtype=np.int32)
    exit_prices_mock = np.zeros(len(df), dtype=np.float64)
    exit_reasons_mock = np.full(len(df), 0, dtype=np.int8)  # TIME_OUT

    # Simulate SL hit for the first entry
    labels_mock[0] = Label.SELL.value
    exit_bar_offsets_mock[0] = 1
    exit_prices_mock[0] = df["close"].iloc[1]
    exit_reasons_mock[0] = 2  # SL_HIT

    (labeler_instance.executor.submit.return_value).set_result(
        (labels_mock, exit_bar_offsets_mock, exit_prices_mock, exit_reasons_mock, np.zeros(len(df)), np.zeros(len(df)))
    )

    stats = await labeler_instance.process_symbol(1, "TEST_SYMBOL", df)

    assert stats is not None
    assert stats.labeled_bars > 0
    assert stats.label_distribution["SELL"] > 0
    assert stats.exit_reasons["SL_HIT"] > 0


@pytest.mark.asyncio
async def test_process_symbol_basic_time_out(
    labeler_instance: OptimizedTripleBarrierLabeler, sample_ohlcv_data_for_labeling: pd.DataFrame
) -> None:
    df = sample_ohlcv_data_for_labeling.copy()
    # Keep prices within a narrow range to trigger TIME_OUT
    df["close"] = np.full(len(df), 1000.0)
    df["open"] = df["high"] = df["low"] = df["close"]

    # Mock the executor to return a result that simulates TIME_OUT
    labels_mock = np.full(len(df), Label.NEUTRAL.value, dtype=np.int8)
    exit_bar_offsets_mock = np.full(len(df), labeler_instance.config.max_holding_periods, dtype=np.int32)
    exit_prices_mock = np.full(len(df), df["close"].iloc[-1], dtype=np.float64)
    exit_reasons_mock = np.full(len(df), 0, dtype=np.int8)  # TIME_OUT

    (labeler_instance.executor.submit.return_value).set_result(
        (labels_mock, exit_bar_offsets_mock, exit_prices_mock, exit_reasons_mock, np.zeros(len(df)), np.zeros(len(df)))
    )

    stats = await labeler_instance.process_symbol(1, "TEST_SYMBOL", df)

    assert stats is not None
    assert stats.labeled_bars > 0
    assert stats.label_distribution["NEUTRAL"] > 0
    assert stats.exit_reasons["TIME_OUT"] > 0


@pytest.mark.asyncio
async def test_process_symbol_insufficient_data(
    labeler_instance: OptimizedTripleBarrierLabeler,
) -> None:
    # Create a DataFrame with very few rows, less than min_bars_required
    data: dict[str, list[Any]] = {
        "ts": pd.to_datetime(["2023-01-01 09:15", "2023-01-01 09:16", "2023-01-01 09:17"]),
        "timestamp": pd.to_datetime(["2023-01-01 09:15", "2023-01-01 09:16", "2023-01-01 09:17"]),
        "open": [100, 101, 102],
        "high": [101, 102, 103],
        "low": [99, 100, 101],
        "close": [100, 101, 102],
        "volume": [100, 110, 120],
    }
    df: pd.DataFrame = pd.DataFrame(data).set_index("ts")

    stats = await labeler_instance.process_symbol(1, "TEST_SYMBOL", df)

    assert stats is None  # Should return None due to insufficient data


@pytest.mark.asyncio
async def test_process_symbol_dynamic_barriers(
    mock_config_loader_instance: MockConfigLoader, sample_ohlcv_data_for_labeling: pd.DataFrame
) -> None:
    # Create a new BarrierConfig with dynamic barriers enabled
    labeling_config = mock_config_loader_instance.get_config().trading.labeling
    dynamic_barrier_config = BarrierConfig(
        atr_period=labeling_config.atr_period,
        tp_atr_multiplier=labeling_config.tp_atr_multiplier,
        sl_atr_multiplier=labeling_config.sl_atr_multiplier,
        atr_smoothing=labeling_config.atr_smoothing,
        max_holding_periods=labeling_config.max_holding_periods,
        min_bars_required=labeling_config.min_bars_required,
        epsilon=labeling_config.epsilon,
        use_dynamic_barriers=True,
        use_dynamic_epsilon=labeling_config.use_dynamic_epsilon,
        volatility_lookback=labeling_config.volatility_lookback,
        atr_cache_size=labeling_config.atr_cache_size,
        dynamic_barrier_tp_sensitivity=labeling_config.dynamic_barrier_tp_sensitivity,
        dynamic_barrier_sl_sensitivity=labeling_config.dynamic_barrier_sl_sensitivity,
        sample_weight_decay_factor=labeling_config.sample_weight_decay_factor,
        dynamic_epsilon=labeling_config.dynamic_epsilon,
    )

    mock_label_repo = AsyncMock(spec=LabelRepository)
    mock_executor = MagicMock()
    mock_executor.submit = AsyncMock(return_value=asyncio.Future())
    mock_executor.shutdown = AsyncMock()
    (mock_executor.submit.return_value).set_result(
        (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.zeros(len(sample_ohlcv_data_for_labeling)),
            np.zeros(len(sample_ohlcv_data_for_labeling)),
        )
    )

    with patch("src.utils.config_loader.ConfigLoader") as MockCL:
        MockCL.return_value.get_config.return_value = mock_config_loader_instance.get_config()
        mock_instrument_repo = MagicMock()
        labeler_instance = OptimizedTripleBarrierLabeler(
            dynamic_barrier_config, mock_label_repo, mock_instrument_repo, mock_executor
        )

        df = sample_ohlcv_data_for_labeling.copy()

        # Mock the executor to return a result that simulates some labels
        labels_mock = np.full(len(df), Label.NEUTRAL.value, dtype=np.int8)
        exit_bar_offsets_mock = np.zeros(len(df), dtype=np.int32)
        exit_prices_mock = np.zeros(len(df), dtype=np.float64)
        exit_reasons_mock = np.full(len(df), 0, dtype=np.int8)

        (labeler_instance.executor.submit.return_value).set_result(
            (
                labels_mock,
                exit_bar_offsets_mock,
                exit_prices_mock,
                exit_reasons_mock,
                np.zeros(len(df)),
                np.zeros(len(df)),
            )
        )

        stats = await labeler_instance.process_symbol(1, "TEST_SYMBOL", df)

        assert stats is not None
        assert stats.labeled_bars > 0
        # This test primarily ensures the function runs without error when dynamic barriers are enabled
        # and that labels are generated. Specific validation of dynamic barrier logic would be complex
        # and require mocking internal volatility calculations.
        await labeler_instance.shutdown()


@pytest.mark.asyncio
async def test_process_symbol_multiple_hits(
    labeler_instance: OptimizedTripleBarrierLabeler, sample_ohlcv_data_for_labeling: pd.DataFrame
) -> None:
    df = sample_ohlcv_data_for_labeling.copy()
    # Create a scenario where TP and SL are very close, and time barrier is also near
    df["close"] = np.full(len(df), 1000.0)
    df["open"] = df["high"] = df["low"] = df["close"]

    # Mock the executor to return a result that simulates a TP hit first
    labels_mock = np.full(len(df), Label.NEUTRAL.value, dtype=np.int8)
    exit_bar_offsets_mock = np.zeros(len(df), dtype=np.int32)
    exit_prices_mock = np.zeros(len(df), dtype=np.float64)
    exit_reasons_mock = np.full(len(df), 0, dtype=np.int8)

    labels_mock[0] = Label.BUY.value
    exit_bar_offsets_mock[0] = 1
    exit_prices_mock[0] = 1000.0 + 10.0  # Simulate TP hit
    exit_reasons_mock[0] = 1  # TP_HIT

    (labeler_instance.executor.submit.return_value).set_result(
        (labels_mock, exit_bar_offsets_mock, exit_prices_mock, exit_reasons_mock, np.zeros(len(df)), np.zeros(len(df)))
    )

    stats = await labeler_instance.process_symbol(1, "TEST_SYMBOL", df)

    assert stats is not None
    assert stats.labeled_bars > 0
    assert stats.label_distribution["BUY"] > 0
    assert stats.exit_reasons["TP_HIT"] > 0


@pytest.mark.asyncio
async def test_process_symbols_parallel(
    labeler_instance: OptimizedTripleBarrierLabeler, sample_ohlcv_data_for_labeling: pd.DataFrame
) -> None:
    symbol_data = {
        1: {"symbol": "TEST_SYMBOL_1", "data": sample_ohlcv_data_for_labeling.copy(), "timeframe": "15min"},
        2: {"symbol": "TEST_SYMBOL_2", "data": sample_ohlcv_data_for_labeling.copy(), "timeframe": "15min"},
    }

    # Mock the executor to return a result for each symbol
    labels_mock = np.full(len(sample_ohlcv_data_for_labeling), Label.NEUTRAL.value, dtype=np.int8)
    exit_bar_offsets_mock = np.zeros(len(sample_ohlcv_data_for_labeling), dtype=np.int32)
    exit_prices_mock = np.zeros(len(sample_ohlcv_data_for_labeling), dtype=np.float64)
    exit_reasons_mock = np.full(len(sample_ohlcv_data_for_labeling), 0, dtype=np.int8)

    # Create a list of futures for asyncio.gather
    future1: asyncio.Future[Any] = asyncio.Future()
    future1.set_result(
        (
            labels_mock,
            exit_bar_offsets_mock,
            exit_prices_mock,
            exit_reasons_mock,
            np.zeros(len(sample_ohlcv_data_for_labeling)),
            np.zeros(len(sample_ohlcv_data_for_labeling)),
        )
    )
    future2: asyncio.Future[Any] = asyncio.Future()
    future2.set_result(
        (
            labels_mock,
            exit_bar_offsets_mock,
            exit_prices_mock,
            exit_reasons_mock,
            np.zeros(len(sample_ohlcv_data_for_labeling)),
            np.zeros(len(sample_ohlcv_data_for_labeling)),
        )
    )

    labeler_instance.executor.submit.side_effect = [future1, future2]

    stats_dict = await labeler_instance.process_symbols_parallel(symbol_data)

    assert len(stats_dict) == 2
    assert "TEST_SYMBOL_1" in stats_dict
    assert "TEST_SYMBOL_2" in stats_dict
    assert stats_dict["TEST_SYMBOL_1"].labeled_bars > 0
    assert stats_dict["TEST_SYMBOL_2"].labeled_bars > 0


@pytest.mark.asyncio
async def test_shutdown(labeler_instance: OptimizedTripleBarrierLabeler) -> None:
    await labeler_instance.shutdown()
    (labeler_instance.executor.shutdown).assert_called_once_with(wait=True)


# ============================================================================
# CORE NUMBA FUNCTION DIRECT TESTING (PRODUCTION-GRADE)
# ============================================================================


class TestCheckBarriersOptimized:
    """
    Directly tests the core Numba-JITed _check_barriers_optimized function.

    This is the most critical test class as it verifies the computational heart
    of the labeling system - the low-level barrier checking logic that runs
    in parallel with Numba JIT compilation.
    """

    def test_clear_tp_hit(self) -> None:
        """Test case where price hits the Take Profit barrier first."""
        # Setup: entry at 100, TP at 105, SL at 95
        # Price moves to 106 in the second bar (clear TP hit)
        high = np.array([101, 102, 106, 107], dtype=np.float64)
        low = np.array([99, 98, 101, 102], dtype=np.float64)
        open_ = np.array([100, 101, 103, 106], dtype=np.float64)
        close = np.array([101, 102, 105, 106], dtype=np.float64)
        entry_prices = np.array([100], dtype=np.float64)
        tp_prices = np.array([105], dtype=np.float64)
        sl_prices = np.array([95], dtype=np.float64)
        max_periods = 3
        epsilon_array = np.array([0.01], dtype=np.float64)  # 1% epsilon

        labels, offsets, exit_prices, reasons, mfe, mae = OptimizedTripleBarrierLabeler._check_barriers_optimized(
            high, low, open_, close, entry_prices, tp_prices, sl_prices, max_periods, epsilon_array
        )

        assert labels[0] == Label.BUY.value
        assert reasons[0] == ExitReason.TP_HIT.value
        assert offsets[0] == 2  # Hit happens at index 2
        assert exit_prices[0] == 105.0  # TP price
        assert mfe[0] > 0  # Should have positive MFE
        assert mae[0] <= 0  # MAE should be non-positive (or very small)

    def test_clear_sl_hit(self) -> None:
        """Test case where price hits the Stop Loss barrier first."""
        high = np.array([101, 102, 103, 104], dtype=np.float64)
        low = np.array([99, 98, 93, 92], dtype=np.float64)  # Clear SL hit at index 2
        open_ = np.array([100, 101, 102, 103], dtype=np.float64)
        close = np.array([101, 100, 94, 93], dtype=np.float64)
        entry_prices = np.array([100], dtype=np.float64)
        tp_prices = np.array([105], dtype=np.float64)
        sl_prices = np.array([95], dtype=np.float64)
        max_periods = 3
        epsilon_array = np.array([0.01], dtype=np.float64)

        labels, offsets, exit_prices, reasons, mfe, mae = OptimizedTripleBarrierLabeler._check_barriers_optimized(
            high, low, open_, close, entry_prices, tp_prices, sl_prices, max_periods, epsilon_array
        )

        assert labels[0] == Label.SELL.value
        assert reasons[0] == ExitReason.SL_HIT.value
        assert offsets[0] == 2  # Hit at index 2
        assert exit_prices[0] == 95.0  # SL price
        assert mfe[0] >= 0  # MFE should be non-negative (might be small positive)
        assert mae[0] < 0  # Should have negative MAE

    def test_timeout_neutral_label(self) -> None:
        """Test timeout where final return is within the epsilon band."""
        high = np.array([101, 102, 101, 102], dtype=np.float64)
        low = np.array([99, 99.5, 99, 99], dtype=np.float64)
        open_ = np.array([100, 100.5, 100, 101], dtype=np.float64)
        close = np.array([100.5, 100, 100.5, 100.2], dtype=np.float64)  # Final close is 100.2
        entry_prices = np.array([100], dtype=np.float64)
        tp_prices = np.array([105], dtype=np.float64)  # Far from reach
        sl_prices = np.array([95], dtype=np.float64)  # Far from reach
        max_periods = 3
        epsilon_array = np.array([0.01], dtype=np.float64)  # 1% = 1 point

        labels, offsets, exit_prices, reasons, mfe, mae = OptimizedTripleBarrierLabeler._check_barriers_optimized(
            high, low, open_, close, entry_prices, tp_prices, sl_prices, max_periods, epsilon_array
        )

        assert labels[0] == Label.NEUTRAL.value  # Return 0.2% < 1% epsilon
        assert reasons[0] == ExitReason.TIME_OUT.value
        assert offsets[0] == 3  # Max periods reached
        assert exit_prices[0] == 100.2  # Final close price

    def test_timeout_buy_label(self) -> None:
        """Test timeout where final return is above epsilon threshold."""
        high = np.array([101, 102, 103, 104], dtype=np.float64)
        low = np.array([99, 99.5, 101, 102], dtype=np.float64)
        open_ = np.array([100, 100.5, 102, 103], dtype=np.float64)
        close = np.array([100.5, 102, 103, 102], dtype=np.float64)  # Final close is 102
        entry_prices = np.array([100], dtype=np.float64)
        tp_prices = np.array([110], dtype=np.float64)  # Far TP, won't hit
        sl_prices = np.array([90], dtype=np.float64)  # Far SL, won't hit
        max_periods = 3
        epsilon_array = np.array([0.01], dtype=np.float64)  # 1%

        labels, offsets, exit_prices, reasons, mfe, mae = OptimizedTripleBarrierLabeler._check_barriers_optimized(
            high, low, open_, close, entry_prices, tp_prices, sl_prices, max_periods, epsilon_array
        )

        assert labels[0] == Label.BUY.value  # Return 2% > 1% epsilon
        assert reasons[0] == ExitReason.TIME_OUT.value
        assert offsets[0] == 3
        assert exit_prices[0] == 102.0

    def test_timeout_sell_label(self) -> None:
        """Test timeout where final return is below negative epsilon threshold."""
        high = np.array([101, 100, 99, 98], dtype=np.float64)
        low = np.array([99, 98, 97, 96], dtype=np.float64)
        open_ = np.array([100, 99, 98, 97], dtype=np.float64)
        close = np.array([99, 98, 97, 98], dtype=np.float64)  # Final close is 98
        entry_prices = np.array([100], dtype=np.float64)
        tp_prices = np.array([110], dtype=np.float64)  # Far TP
        sl_prices = np.array([90], dtype=np.float64)  # Far SL
        max_periods = 3
        epsilon_array = np.array([0.01], dtype=np.float64)  # 1%

        labels, offsets, exit_prices, reasons, mfe, mae = OptimizedTripleBarrierLabeler._check_barriers_optimized(
            high, low, open_, close, entry_prices, tp_prices, sl_prices, max_periods, epsilon_array
        )

        assert labels[0] == Label.SELL.value  # Return -2% < -1% epsilon
        assert reasons[0] == ExitReason.TIME_OUT.value
        assert offsets[0] == 3
        assert exit_prices[0] == 98.0

    def test_simultaneous_tp_sl_hit_tp_priority(self) -> None:
        """Test case where both TP and SL are hit in the same bar - TP should take priority."""
        # Bar where both barriers are crossed: low=93 (SL hit), high=107 (TP hit)
        high = np.array([101, 107], dtype=np.float64)  # TP hit in bar 1
        low = np.array([99, 93], dtype=np.float64)  # SL hit in bar 1
        open_ = np.array([100, 101], dtype=np.float64)
        close = np.array([101, 106], dtype=np.float64)  # Close near TP
        entry_prices = np.array([100], dtype=np.float64)
        tp_prices = np.array([105], dtype=np.float64)
        sl_prices = np.array([95], dtype=np.float64)
        max_periods = 3
        epsilon_array = np.array([0.01], dtype=np.float64)

        labels, offsets, exit_prices, reasons, mfe, mae = OptimizedTripleBarrierLabeler._check_barriers_optimized(
            high, low, open_, close, entry_prices, tp_prices, sl_prices, max_periods, epsilon_array
        )

        # TP should take priority in simultaneous hits
        assert labels[0] == Label.BUY.value
        assert reasons[0] == ExitReason.TP_HIT.value
        assert offsets[0] == 1  # Hit in first bar after entry
        assert exit_prices[0] == 105.0

    def test_multiple_entry_points(self) -> None:
        """Test multiple entry points with different outcomes."""
        # 6 bars total, 3 entry points
        high = np.array([101, 106, 103, 98, 107, 102], dtype=np.float64)
        low = np.array([99, 101, 97, 93, 102, 98], dtype=np.float64)
        open_ = np.array([100, 102, 102, 97, 104, 101], dtype=np.float64)
        close = np.array([101, 105, 98, 94, 106, 100], dtype=np.float64)

        # Entry points: bars 0, 1, 2 (indices 0, 1, 2)
        entry_prices = np.array([100, 102, 102], dtype=np.float64)
        tp_prices = np.array([105, 107, 107], dtype=np.float64)
        sl_prices = np.array([95, 97, 97], dtype=np.float64)
        max_periods = 3
        epsilon_array = np.array([0.01, 0.01, 0.01], dtype=np.float64)

        labels, offsets, exit_prices, reasons, mfe, mae = OptimizedTripleBarrierLabeler._check_barriers_optimized(
            high, low, open_, close, entry_prices, tp_prices, sl_prices, max_periods, epsilon_array
        )

        # First entry (bar 0): Should hit TP at bar 1 (high=106 > tp=105)
        assert labels[0] == Label.BUY.value
        assert reasons[0] == ExitReason.TP_HIT.value
        assert offsets[0] == 1

        # Second entry (bar 1): Should hit SL at bar 3 (low=93 < sl=97)
        assert labels[1] == Label.SELL.value
        assert reasons[1] == ExitReason.SL_HIT.value
        assert offsets[1] == 2  # 2 bars from entry at bar 1

        # Third entry (bar 2): Should hit TP at bar 4 (high=107 > tp=107)
        assert labels[2] == Label.BUY.value
        assert reasons[2] == ExitReason.TP_HIT.value
        assert offsets[2] == 2  # 2 bars from entry at bar 2

    def test_mfe_mae_calculations(self) -> None:
        """Test Maximum Favorable and Adverse Excursion calculations."""
        # Create scenario with clear MFE and MAE
        high = np.array([101, 108, 104, 102], dtype=np.float64)  # Peak at 108
        low = np.array([99, 102, 96, 100], dtype=np.float64)  # Trough at 96
        open_ = np.array([100, 103, 103, 101], dtype=np.float64)
        close = np.array([101, 107, 97, 101], dtype=np.float64)  # Final return ~1%
        entry_prices = np.array([100], dtype=np.float64)
        tp_prices = np.array([110], dtype=np.float64)  # Won't hit
        sl_prices = np.array([90], dtype=np.float64)  # Won't hit
        max_periods = 3
        epsilon_array = np.array([0.005], dtype=np.float64)  # 0.5% epsilon (so 1% return -> BUY)

        labels, offsets, exit_prices, reasons, mfe, mae = OptimizedTripleBarrierLabeler._check_barriers_optimized(
            high, low, open_, close, entry_prices, tp_prices, sl_prices, max_periods, epsilon_array
        )

        assert labels[0] == Label.BUY.value  # 1% > 0.5% epsilon
        assert reasons[0] == ExitReason.TIME_OUT.value

        # MFE should be around 8% (from 100 to 108)
        expected_mfe = (108 - 100) / 100
        assert abs(mfe[0] - expected_mfe) < 0.001

        # MAE should be around -4% (from 100 to 96)
        expected_mae = (96 - 100) / 100
        assert abs(mae[0] - expected_mae) < 0.001

    def test_edge_case_zero_max_periods(self) -> None:
        """Test edge case with zero max periods."""
        high = np.array([101], dtype=np.float64)
        low = np.array([99], dtype=np.float64)
        open_ = np.array([100], dtype=np.float64)
        close = np.array([101], dtype=np.float64)
        entry_prices = np.array([100], dtype=np.float64)
        tp_prices = np.array([105], dtype=np.float64)
        sl_prices = np.array([95], dtype=np.float64)
        max_periods = 0  # Zero holding period
        epsilon_array = np.array([0.01], dtype=np.float64)

        labels, offsets, exit_prices, reasons, mfe, mae = OptimizedTripleBarrierLabeler._check_barriers_optimized(
            high, low, open_, close, entry_prices, tp_prices, sl_prices, max_periods, epsilon_array
        )

        # Should immediately timeout with entry price as exit price
        assert reasons[0] == ExitReason.TIME_OUT.value
        assert offsets[0] == 0
        assert exit_prices[0] == 100.0  # Entry price

    def test_dynamic_epsilon_array(self) -> None:
        """Test with varying epsilon values across different entry points."""
        high = np.array([101, 102, 103], dtype=np.float64)
        low = np.array([99, 98, 97], dtype=np.float64)
        open_ = np.array([100, 101, 102], dtype=np.float64)
        close = np.array([100.5, 101.5, 102.5], dtype=np.float64)  # All 0.5% returns

        # Two entry points with different epsilon thresholds
        entry_prices = np.array([100, 101], dtype=np.float64)
        tp_prices = np.array([110, 111], dtype=np.float64)  # Won't hit
        sl_prices = np.array([90, 91], dtype=np.float64)  # Won't hit
        max_periods = 1
        # First entry: 0.3% epsilon (0.5% > 0.3% -> BUY)
        # Second entry: 0.7% epsilon (0.5% < 0.7% -> NEUTRAL)
        epsilon_array = np.array([0.003, 0.007], dtype=np.float64)

        labels, offsets, exit_prices, reasons, mfe, mae = OptimizedTripleBarrierLabeler._check_barriers_optimized(
            high, low, open_, close, entry_prices, tp_prices, sl_prices, max_periods, epsilon_array
        )

        # First entry: 0.5% return > 0.3% epsilon -> BUY
        assert labels[0] == Label.BUY.value
        assert reasons[0] == ExitReason.TIME_OUT.value

        # Second entry: 0.5% return < 0.7% epsilon -> NEUTRAL
        assert labels[1] == Label.NEUTRAL.value
        assert reasons[1] == ExitReason.TIME_OUT.value


# ============================================================================
# PRODUCTION-GRADE TEST CLASSES (FROM COMPREHENSIVE FILE)
# ============================================================================


@pytest.mark.asyncio
class TestBarrierConfigValidation:
    """Test configuration validation for production deployment."""

    async def test_valid_config_passes_validation(self, mock_config_loader_instance: MockConfigLoader) -> None:
        config = mock_config_loader_instance.get_config().trading.labeling
        barrier_config = BarrierConfig(
            atr_period=config.atr_period,
            tp_atr_multiplier=config.tp_atr_multiplier,
            sl_atr_multiplier=config.sl_atr_multiplier,
            atr_smoothing=config.atr_smoothing,
            max_holding_periods=config.max_holding_periods,
            min_bars_required=config.min_bars_required,
            epsilon=config.epsilon,
            use_dynamic_barriers=config.use_dynamic_barriers,
            use_dynamic_epsilon=config.use_dynamic_epsilon,
            volatility_lookback=config.volatility_lookback,
            atr_cache_size=config.atr_cache_size,
            dynamic_barrier_tp_sensitivity=config.dynamic_barrier_tp_sensitivity,
            dynamic_barrier_sl_sensitivity=config.dynamic_barrier_sl_sensitivity,
            sample_weight_decay_factor=config.sample_weight_decay_factor,
            dynamic_epsilon=config.dynamic_epsilon,
        )

        mock_label_repo = AsyncMock(spec=LabelRepository)
        mock_instrument_repo = AsyncMock(spec=InstrumentRepository)
        mock_stats_repo = AsyncMock()

        with patch("src.utils.config_loader.ConfigLoader") as MockCL:
            MockCL.return_value.get_config.return_value = mock_config_loader_instance.get_config()

            labeler = OptimizedTripleBarrierLabeler(barrier_config, mock_label_repo, mock_instrument_repo, mock_stats_repo, [15])
            assert labeler.config == barrier_config
            await labeler.shutdown()

    async def test_invalid_tp_multiplier_raises_error(self) -> None:
        invalid_config = BarrierConfig(
            atr_period=14,
            tp_atr_multiplier=-1.0,  # Invalid: negative
            sl_atr_multiplier=1.5,
            atr_smoothing="ema",
            max_holding_periods=10,
            min_bars_required=100,
            epsilon=0.0005,
            use_dynamic_barriers=False,
            use_dynamic_epsilon=False,
            volatility_lookback=20,
            atr_cache_size=100,
            dynamic_barrier_tp_sensitivity=0.2,
            dynamic_barrier_sl_sensitivity=0.1,
            sample_weight_decay_factor=0.5,
            dynamic_epsilon={},
        )

        mock_label_repo = AsyncMock(spec=LabelRepository)
        mock_instrument_repo = AsyncMock(spec=InstrumentRepository)
        mock_stats_repo = AsyncMock()

        with pytest.raises(ValueError, match="TP multiplier must be positive"):
            OptimizedTripleBarrierLabeler(invalid_config, mock_label_repo, mock_instrument_repo, mock_stats_repo, [15])

    async def test_invalid_sl_multiplier_raises_error(self) -> None:
        invalid_config = BarrierConfig(
            atr_period=14,
            tp_atr_multiplier=2.0,
            sl_atr_multiplier=0.0,  # Invalid: zero
            atr_smoothing="ema",
            max_holding_periods=10,
            min_bars_required=100,
            epsilon=0.0005,
            use_dynamic_barriers=False,
            use_dynamic_epsilon=False,
            volatility_lookback=20,
            atr_cache_size=100,
            dynamic_barrier_tp_sensitivity=0.2,
            dynamic_barrier_sl_sensitivity=0.1,
            sample_weight_decay_factor=0.5,
            dynamic_epsilon={},
        )

        mock_label_repo = AsyncMock(spec=LabelRepository)
        mock_instrument_repo = AsyncMock(spec=InstrumentRepository)
        mock_stats_repo = AsyncMock()

        with pytest.raises(ValueError, match="SL multiplier must be positive"):
            OptimizedTripleBarrierLabeler(invalid_config, mock_label_repo, mock_instrument_repo, mock_stats_repo, [15])

    async def test_invalid_max_holding_periods_raises_error(self) -> None:
        invalid_config = BarrierConfig(
            atr_period=14,
            tp_atr_multiplier=2.0,
            sl_atr_multiplier=1.5,
            atr_smoothing="ema",
            max_holding_periods=1,  # Invalid: too small
            min_bars_required=100,
            epsilon=0.0005,
            use_dynamic_barriers=False,
            use_dynamic_epsilon=False,
            volatility_lookback=20,
            atr_cache_size=100,
            dynamic_barrier_tp_sensitivity=0.2,
            dynamic_barrier_sl_sensitivity=0.1,
            sample_weight_decay_factor=0.5,
            dynamic_epsilon={},
        )

        mock_label_repo = AsyncMock(spec=LabelRepository)
        mock_instrument_repo = AsyncMock(spec=InstrumentRepository)
        mock_stats_repo = AsyncMock()

        with pytest.raises(ValueError, match="Max holding periods must be at least 2"):
            OptimizedTripleBarrierLabeler(invalid_config, mock_label_repo, mock_instrument_repo, mock_stats_repo, [15])

    async def test_invalid_atr_period_raises_error(self) -> None:
        invalid_config = BarrierConfig(
            atr_period=1,  # Invalid: too small
            tp_atr_multiplier=2.0,
            sl_atr_multiplier=1.5,
            atr_smoothing="ema",
            max_holding_periods=10,
            min_bars_required=100,
            epsilon=0.0005,
            use_dynamic_barriers=False,
            use_dynamic_epsilon=False,
            volatility_lookback=20,
            atr_cache_size=100,
            dynamic_barrier_tp_sensitivity=0.2,
            dynamic_barrier_sl_sensitivity=0.1,
            sample_weight_decay_factor=0.5,
            dynamic_epsilon={},
        )

        mock_label_repo = AsyncMock(spec=LabelRepository)
        mock_instrument_repo = AsyncMock(spec=InstrumentRepository)
        mock_stats_repo = AsyncMock()

        with pytest.raises(ValueError, match="ATR period must be at least 2"):
            OptimizedTripleBarrierLabeler(invalid_config, mock_label_repo, mock_instrument_repo, mock_stats_repo, [15])

    async def test_invalid_epsilon_raises_error(self) -> None:
        invalid_config = BarrierConfig(
            atr_period=14,
            tp_atr_multiplier=2.0,
            sl_atr_multiplier=1.5,
            atr_smoothing="ema",
            max_holding_periods=10,
            min_bars_required=100,
            epsilon=-0.001,  # Invalid: negative
            use_dynamic_barriers=False,
            use_dynamic_epsilon=False,
            volatility_lookback=20,
            atr_cache_size=100,
            dynamic_barrier_tp_sensitivity=0.2,
            dynamic_barrier_sl_sensitivity=0.1,
            sample_weight_decay_factor=0.5,
            dynamic_epsilon={},
        )

        mock_label_repo = AsyncMock(spec=LabelRepository)
        mock_instrument_repo = AsyncMock(spec=InstrumentRepository)
        mock_stats_repo = AsyncMock()

        with pytest.raises(ValueError, match="Epsilon must be positive"):
            OptimizedTripleBarrierLabeler(invalid_config, mock_label_repo, mock_instrument_repo, mock_stats_repo, [15])

    async def test_invalid_dynamic_barrier_sensitivity_raises_error(self) -> None:
        invalid_config = BarrierConfig(
            atr_period=14,
            tp_atr_multiplier=2.0,
            sl_atr_multiplier=1.5,
            atr_smoothing="ema",
            max_holding_periods=10,
            min_bars_required=100,
            epsilon=0.0005,
            use_dynamic_barriers=False,
            use_dynamic_epsilon=False,
            volatility_lookback=20,
            atr_cache_size=100,
            dynamic_barrier_tp_sensitivity=1.5,  # Invalid: > 1
            dynamic_barrier_sl_sensitivity=0.1,
            sample_weight_decay_factor=0.5,
            dynamic_epsilon={},
        )

        mock_label_repo = AsyncMock(spec=LabelRepository)
        mock_instrument_repo = AsyncMock(spec=InstrumentRepository)
        mock_stats_repo = AsyncMock()

        with pytest.raises(ValueError, match="Dynamic barrier TP sensitivity must be between 0 and 1"):
            OptimizedTripleBarrierLabeler(invalid_config, mock_label_repo, mock_instrument_repo, mock_stats_repo, [15])


@pytest.mark.asyncio
class TestATRCalculation:
    """Test ATR calculation edge cases and performance."""

    async def test_atr_calculation_with_gaps(self, labeler_instance: OptimizedTripleBarrierLabeler) -> None:
        data_size = 50
        dates = pd.date_range(start="2023-01-01", periods=data_size, freq="min")
        prices = np.concatenate(
            [
                np.linspace(100, 105, 20),  # Normal price movement
                np.linspace(120, 125, 30),  # Gap up
            ]
        )

        df = pd.DataFrame(
            {
                "ts": dates,
                "open": prices,
                "high": prices + np.random.uniform(0, 1, data_size),
                "low": prices - np.random.uniform(0, 1, data_size),
                "close": prices + np.random.uniform(-0.5, 0.5, data_size),
                "volume": np.random.randint(1000, 5000, data_size),
            }
        )
        df = df.set_index("ts")

        atr_values = labeler_instance._calculate_atr(df, "TEST_GAP")

        assert len(atr_values) == len(df)
        assert not np.isnan(atr_values).all()
        # ATR should be higher around the gap
        gap_atr = atr_values[20:25].mean()
        normal_atr = atr_values[5:15].mean()
        assert gap_atr > normal_atr

    async def test_atr_calculation_with_zeros(self, labeler_instance: OptimizedTripleBarrierLabeler) -> None:
        data_size = 30
        dates = pd.date_range(start="2023-01-01", periods=data_size, freq="min")

        df = pd.DataFrame(
            {
                "ts": dates,
                "open": [100.0] * data_size,
                "high": [100.0] * data_size,
                "low": [100.0] * data_size,
                "close": [100.0] * data_size,
                "volume": [1000] * data_size,
            }
        )
        df = df.set_index("ts")

        atr_values = labeler_instance._calculate_atr(df, "TEST_ZERO")

        assert len(atr_values) == len(df)
        assert not np.isnan(atr_values).all()
        assert np.all(atr_values >= 0)

    async def test_atr_caching(self, labeler_instance: OptimizedTripleBarrierLabeler) -> None:
        data_size = 100
        dates = pd.date_range(start="2023-01-01", periods=data_size, freq="min")

        df = pd.DataFrame(
            {
                "ts": dates,
                "open": np.random.uniform(100, 110, data_size),
                "high": np.random.uniform(105, 115, data_size),
                "low": np.random.uniform(95, 105, data_size),
                "close": np.random.uniform(100, 110, data_size),
                "volume": np.random.randint(1000, 5000, data_size),
            }
        )
        df = df.set_index("ts")

        # First calculation
        atr1 = labeler_instance._calculate_atr(df, "TEST_CACHE")

        # Second calculation should use cache
        atr2 = labeler_instance._calculate_atr(df, "TEST_CACHE")

        np.testing.assert_array_equal(atr1, atr2)


@pytest.mark.asyncio
class TestDynamicEpsilon:
    """Test dynamic epsilon calculation."""

    async def test_static_epsilon_when_disabled(self, mock_config_loader_instance: MockConfigLoader) -> None:
        config = mock_config_loader_instance.get_config().trading.labeling
        barrier_config = BarrierConfig(
            atr_period=config.atr_period,
            tp_atr_multiplier=config.tp_atr_multiplier,
            sl_atr_multiplier=config.sl_atr_multiplier,
            atr_smoothing=config.atr_smoothing,
            max_holding_periods=config.max_holding_periods,
            min_bars_required=config.min_bars_required,
            epsilon=config.epsilon,
            use_dynamic_barriers=config.use_dynamic_barriers,
            use_dynamic_epsilon=False,  # Disabled
            volatility_lookback=config.volatility_lookback,
            atr_cache_size=config.atr_cache_size,
            dynamic_barrier_tp_sensitivity=config.dynamic_barrier_tp_sensitivity,
            dynamic_barrier_sl_sensitivity=config.dynamic_barrier_sl_sensitivity,
            sample_weight_decay_factor=config.sample_weight_decay_factor,
            dynamic_epsilon=config.dynamic_epsilon,
        )

        mock_label_repo = AsyncMock(spec=LabelRepository)
        mock_instrument_repo = AsyncMock(spec=InstrumentRepository)
        mock_stats_repo = AsyncMock()

        with patch("src.utils.config_loader.ConfigLoader") as MockCL:
            MockCL.return_value.get_config.return_value = mock_config_loader_instance.get_config()

            labeler = OptimizedTripleBarrierLabeler(barrier_config, mock_label_repo, mock_instrument_repo, mock_stats_repo, [15])

            data_size = 100
            dates = pd.date_range(start="2023-01-01", periods=data_size, freq="min")
            df = pd.DataFrame(
                {
                    "ts": dates,
                    "open": np.random.uniform(100, 110, data_size),
                    "high": np.random.uniform(105, 115, data_size),
                    "low": np.random.uniform(95, 105, data_size),
                    "close": np.random.uniform(100, 110, data_size),
                }
            )
            df = df.set_index("ts")

            atr = np.random.uniform(1, 5, data_size)
            epsilon_array = labeler._calculate_dynamic_epsilon(df, atr)

            expected = np.full(data_size, barrier_config.epsilon)
            np.testing.assert_array_equal(epsilon_array, expected)

            await labeler.shutdown()

    async def test_dynamic_epsilon_with_insufficient_data(self, mock_config_loader_instance: MockConfigLoader) -> None:
        config = mock_config_loader_instance.get_config().trading.labeling
        barrier_config = BarrierConfig(
            atr_period=config.atr_period,
            tp_atr_multiplier=config.tp_atr_multiplier,
            sl_atr_multiplier=config.sl_atr_multiplier,
            atr_smoothing=config.atr_smoothing,
            max_holding_periods=config.max_holding_periods,
            min_bars_required=config.min_bars_required,
            epsilon=config.epsilon,
            use_dynamic_barriers=config.use_dynamic_barriers,
            use_dynamic_epsilon=True,  # Enabled
            volatility_lookback=config.volatility_lookback,
            atr_cache_size=config.atr_cache_size,
            dynamic_barrier_tp_sensitivity=config.dynamic_barrier_tp_sensitivity,
            dynamic_barrier_sl_sensitivity=config.dynamic_barrier_sl_sensitivity,
            sample_weight_decay_factor=config.sample_weight_decay_factor,
            dynamic_epsilon=config.dynamic_epsilon,
        )

        mock_label_repo = AsyncMock(spec=LabelRepository)
        mock_instrument_repo = AsyncMock(spec=InstrumentRepository)
        mock_stats_repo = AsyncMock()

        with patch("src.utils.config_loader.ConfigLoader") as MockCL:
            MockCL.return_value.get_config.return_value = mock_config_loader_instance.get_config()

            labeler = OptimizedTripleBarrierLabeler(barrier_config, mock_label_repo, mock_instrument_repo, mock_stats_repo, [15])

            # Create insufficient data (less than volatility_lookback)
            data_size = 10  # Less than volatility_lookback (20)
            dates = pd.date_range(start="2023-01-01", periods=data_size, freq="min")
            df = pd.DataFrame(
                {
                    "ts": dates,
                    "open": np.random.uniform(100, 110, data_size),
                    "high": np.random.uniform(105, 115, data_size),
                    "low": np.random.uniform(95, 105, data_size),
                    "close": np.random.uniform(100, 110, data_size),
                }
            )
            df = df.set_index("ts")

            atr = np.random.uniform(1, 5, data_size)
            epsilon_array = labeler._calculate_dynamic_epsilon(df, atr)

            expected = np.full(data_size, barrier_config.epsilon)
            np.testing.assert_array_equal(epsilon_array, expected)

            await labeler.shutdown()

    @pytest.mark.parametrize(
        "timestamp_str, expected_multiplier_key",
        [
            ("2023-10-23 09:15:00", "market_open_multiplier"),  # Monday, 9:15 AM
            ("2023-10-23 12:00:00", "pre_lunch_multiplier"),  # Monday, 12:00 PM
            ("2023-10-27 15:15:00", "market_close_multiplier"),  # Friday, 3:15 PM
            ("2023-10-26 10:00:00", "normal_volatility_multiplier"),  # Thursday (weekly expiry), normal time
            ("2023-10-30 09:30:00", "monday_multiplier"),  # Monday morning normal
            ("2023-11-03 14:30:00", "friday_multiplier"),  # Friday afternoon normal
        ],
    )
    async def test_dynamic_epsilon_time_based_multipliers(
        self, mock_config_loader_instance: MockConfigLoader, timestamp_str: str, expected_multiplier_key: str
    ) -> None:
        """Test that dynamic epsilon applies correct time-based multipliers."""
        config = mock_config_loader_instance.get_config().trading.labeling
        barrier_config = BarrierConfig(
            atr_period=config.atr_period,
            tp_atr_multiplier=config.tp_atr_multiplier,
            sl_atr_multiplier=config.sl_atr_multiplier,
            atr_smoothing=config.atr_smoothing,
            max_holding_periods=config.max_holding_periods,
            min_bars_required=config.min_bars_required,
            epsilon=config.epsilon,
            use_dynamic_barriers=config.use_dynamic_barriers,
            use_dynamic_epsilon=True,  # Enable dynamic epsilon
            volatility_lookback=config.volatility_lookback,
            atr_cache_size=config.atr_cache_size,
            dynamic_barrier_tp_sensitivity=config.dynamic_barrier_tp_sensitivity,
            dynamic_barrier_sl_sensitivity=config.dynamic_barrier_sl_sensitivity,
            sample_weight_decay_factor=config.sample_weight_decay_factor,
            dynamic_epsilon=config.dynamic_epsilon,
        )

        mock_label_repo = AsyncMock(spec=LabelRepository)
        mock_instrument_repo = AsyncMock(spec=InstrumentRepository)
        mock_stats_repo = AsyncMock()

        with patch("src.utils.config_loader.ConfigLoader") as MockCL:
            MockCL.return_value.get_config.return_value = mock_config_loader_instance.get_config()

            labeler = OptimizedTripleBarrierLabeler(barrier_config, mock_label_repo, mock_instrument_repo, mock_stats_repo, [15])

            cfg = labeler.config.dynamic_epsilon
            base_epsilon = labeler.config.epsilon
            data_size = 50

            # Create timestamps centered around the specific test time
            dates = pd.date_range(end=timestamp_str, periods=data_size, freq="min", tz="Asia/Kolkata")
            df = pd.DataFrame({"timestamp": dates}, index=dates)
            atr = np.full(data_size, 1.0)  # Constant ATR for simplicity

            # Mock volatility to be 'normal' to isolate the time multiplier
            with patch.object(pd.Series, "rolling") as mock_rolling:
                mock_window = MagicMock()
                mock_window.mean.return_value = pd.Series(np.full(data_size, 1.0))  # atr_ratio = 1.0
                mock_window.std.return_value = pd.Series(np.full(data_size, 0.1))
                mock_rolling.return_value = mock_window

                epsilon_array = labeler._calculate_dynamic_epsilon(df, atr)

                # The last value in the array corresponds to our target timestamp
                expected_multiplier = cfg[expected_multiplier_key]
                expected_epsilon = base_epsilon * expected_multiplier

                # Allow for small floating point differences
                assert abs(epsilon_array[-1] - expected_epsilon) < 1e-6

            await labeler.shutdown()

    @pytest.mark.parametrize(
        "atr_ratio, expected_volatility_key",
        [
            (0.7, "low_volatility_multiplier"),  # Low volatility
            (1.0, "normal_volatility_multiplier"),  # Normal volatility
            (1.3, "high_volatility_multiplier"),  # High volatility
            (1.6, "extreme_volatility_multiplier"),  # Extreme volatility
        ],
    )
    async def test_dynamic_epsilon_volatility_based_multipliers(
        self, mock_config_loader_instance: MockConfigLoader, atr_ratio: float, expected_volatility_key: str
    ) -> None:
        """Test that dynamic epsilon applies correct volatility-based multipliers."""
        config = mock_config_loader_instance.get_config().trading.labeling
        barrier_config = BarrierConfig(
            atr_period=config.atr_period,
            tp_atr_multiplier=config.tp_atr_multiplier,
            sl_atr_multiplier=config.sl_atr_multiplier,
            atr_smoothing=config.atr_smoothing,
            max_holding_periods=config.max_holding_periods,
            min_bars_required=config.min_bars_required,
            epsilon=config.epsilon,
            use_dynamic_barriers=config.use_dynamic_barriers,
            use_dynamic_epsilon=True,
            volatility_lookback=config.volatility_lookback,
            atr_cache_size=config.atr_cache_size,
            dynamic_barrier_tp_sensitivity=config.dynamic_barrier_tp_sensitivity,
            dynamic_barrier_sl_sensitivity=config.dynamic_barrier_sl_sensitivity,
            sample_weight_decay_factor=config.sample_weight_decay_factor,
            dynamic_epsilon=config.dynamic_epsilon,
        )

        mock_label_repo = AsyncMock(spec=LabelRepository)
        mock_instrument_repo = AsyncMock(spec=InstrumentRepository)
        mock_stats_repo = AsyncMock()

        with patch("src.utils.config_loader.ConfigLoader") as MockCL:
            MockCL.return_value.get_config.return_value = mock_config_loader_instance.get_config()

            labeler = OptimizedTripleBarrierLabeler(barrier_config, mock_label_repo, mock_instrument_repo, mock_stats_repo, [15])

            cfg = labeler.config.dynamic_epsilon
            base_epsilon = labeler.config.epsilon
            data_size = 50

            # Use a neutral time (Tuesday 11 AM) to isolate volatility effects
            dates = pd.date_range(start="2023-10-24 11:00:00", periods=data_size, freq="min", tz="Asia/Kolkata")
            df = pd.DataFrame({"timestamp": dates}, index=dates)
            atr = np.full(data_size, 1.0)

            # Mock specific volatility regime
            with patch.object(pd.Series, "rolling") as mock_rolling:
                mock_window = MagicMock()
                mock_window.mean.return_value = pd.Series(np.full(data_size, atr_ratio))  # Set specific ATR ratio
                mock_window.std.return_value = pd.Series(np.full(data_size, 0.1))
                mock_rolling.return_value = mock_window

                epsilon_array = labeler._calculate_dynamic_epsilon(df, atr)

                expected_multiplier = cfg[expected_volatility_key]
                expected_epsilon = base_epsilon * expected_multiplier

                assert abs(epsilon_array[-1] - expected_epsilon) < 1e-6

            await labeler.shutdown()

    async def test_dynamic_epsilon_combined_multipliers(self, mock_config_loader_instance: MockConfigLoader) -> None:
        """Test that dynamic epsilon correctly combines volatility and time multipliers."""
        config = mock_config_loader_instance.get_config().trading.labeling
        barrier_config = BarrierConfig(
            atr_period=config.atr_period,
            tp_atr_multiplier=config.tp_atr_multiplier,
            sl_atr_multiplier=config.sl_atr_multiplier,
            atr_smoothing=config.atr_smoothing,
            max_holding_periods=config.max_holding_periods,
            min_bars_required=config.min_bars_required,
            epsilon=config.epsilon,
            use_dynamic_barriers=config.use_dynamic_barriers,
            use_dynamic_epsilon=True,
            volatility_lookback=config.volatility_lookback,
            atr_cache_size=config.atr_cache_size,
            dynamic_barrier_tp_sensitivity=config.dynamic_barrier_tp_sensitivity,
            dynamic_barrier_sl_sensitivity=config.dynamic_barrier_sl_sensitivity,
            sample_weight_decay_factor=config.sample_weight_decay_factor,
            dynamic_epsilon=config.dynamic_epsilon,
        )

        mock_label_repo = AsyncMock(spec=LabelRepository)
        mock_instrument_repo = AsyncMock(spec=InstrumentRepository)
        mock_stats_repo = AsyncMock()

        with patch("src.utils.config_loader.ConfigLoader") as MockCL:
            MockCL.return_value.get_config.return_value = mock_config_loader_instance.get_config()

            labeler = OptimizedTripleBarrierLabeler(barrier_config, mock_label_repo, mock_instrument_repo, mock_stats_repo, [15])

            cfg = labeler.config.dynamic_epsilon
            base_epsilon = labeler.config.epsilon
            data_size = 50

            # Market open on Monday (both market_open_multiplier and monday_multiplier should apply)
            dates = pd.date_range(end="2023-10-23 09:15:00", periods=data_size, freq="min", tz="Asia/Kolkata")
            df = pd.DataFrame({"timestamp": dates}, index=dates)
            atr = np.full(data_size, 1.0)

            # High volatility regime
            atr_ratio = 1.3  # High volatility
            with patch.object(pd.Series, "rolling") as mock_rolling:
                mock_window = MagicMock()
                mock_window.mean.return_value = pd.Series(np.full(data_size, atr_ratio))
                mock_window.std.return_value = pd.Series(np.full(data_size, 0.1))
                mock_rolling.return_value = mock_window

                epsilon_array = labeler._calculate_dynamic_epsilon(df, atr)

                # Should combine high_volatility_multiplier, market_open_multiplier, and monday_multiplier
                expected_multiplier = (
                    cfg["high_volatility_multiplier"] * cfg["market_open_multiplier"] * cfg["monday_multiplier"]
                )
                expected_epsilon = base_epsilon * expected_multiplier

                assert abs(epsilon_array[-1] - expected_epsilon) < 1e-6

            await labeler.shutdown()


@pytest.mark.asyncio
class TestSampleWeightCalculation:
    """Test sample weight calculation based on label uniqueness."""

    async def test_calculate_sample_weights_basic(self, labeler_instance: OptimizedTripleBarrierLabeler) -> None:
        data = {
            "ts": pd.date_range("2023-01-01", periods=10, freq="min"),
            "timeframe": ["15min"] * 10,
            "instrument_id": [1] * 10,
            "label": [1, -1, 0, 1, -1, 0, 1, -1, 0, 1],
            "exit_reason": ["TP_HIT"] * 10,
            "exit_bar_offset": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            "entry_price": [100.0] * 10,
            "exit_price": [101.0] * 10,
            "tp_price": [102.0] * 10,
            "sl_price": [98.0] * 10,
            "barrier_return": [0.01] * 10,
        }
        labeled_df = pd.DataFrame(data)

        weights = labeler_instance.calculate_sample_weights(labeled_df)

        assert len(weights) == len(labeled_df)
        assert np.all(weights > 0)
        assert abs(weights.sum() - len(labeled_df)) < 0.1

    async def test_calculate_sample_weights_overlapping(self, labeler_instance: OptimizedTripleBarrierLabeler) -> None:
        data = {
            "ts": pd.date_range("2023-01-01", periods=5, freq="min"),
            "timeframe": ["15min"] * 5,
            "instrument_id": [1] * 5,
            "label": [1, 1, 1, 1, 1],
            "exit_reason": ["TP_HIT"] * 5,
            "exit_bar_offset": [4, 3, 2, 1, 1],  # Overlapping positions
            "entry_price": [100.0] * 5,
            "exit_price": [101.0] * 5,
            "tp_price": [102.0] * 5,
            "sl_price": [98.0] * 5,
            "barrier_return": [0.01] * 5,
        }
        labeled_df = pd.DataFrame(data)

        weights = labeler_instance.calculate_sample_weights(labeled_df)

        assert weights[0] < 1.0  # First position overlaps with others
        assert np.all(weights > 0)


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and edge cases."""

    async def test_process_symbol_with_missing_columns(self, labeler_instance: OptimizedTripleBarrierLabeler) -> None:
        df = pd.DataFrame(
            {
                "ts": pd.date_range("2023-01-01", periods=10, freq="min"),
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                # Missing 'low', 'close', 'timestamp'
            }
        )

        with patch.object(labeler_instance.instrument_repo, "get_instrument_by_id", return_value=None):
            result = await labeler_instance.process_symbol(1, "TEST_MISSING", df)

        assert result is None

    async def test_process_symbol_with_non_finite_prices(self, labeler_instance: OptimizedTripleBarrierLabeler) -> None:
        df = pd.DataFrame(
            {
                "ts": pd.date_range("2023-01-01", periods=10, freq="min"),
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="min"),
                "open": [100.0, np.inf, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.5] * 10,
                "volume": [1000] * 10,
            }
        )

        with patch.object(labeler_instance.instrument_repo, "get_instrument_by_id", return_value=None):
            result = await labeler_instance.process_symbol(1, "TEST_INF", df)

        assert result is None

    async def test_process_symbol_with_negative_prices(self, labeler_instance: OptimizedTripleBarrierLabeler) -> None:
        df = pd.DataFrame(
            {
                "ts": pd.date_range("2023-01-01", periods=10, freq="min"),
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="min"),
                "open": [-1.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.5] * 10,
                "volume": [1000] * 10,
            }
        )

        with patch.object(labeler_instance.instrument_repo, "get_instrument_by_id", return_value=None):
            result = await labeler_instance.process_symbol(1, "TEST_NEG", df)

        assert result is None

    async def test_process_symbol_with_duplicate_timestamps(
        self, labeler_instance: OptimizedTripleBarrierLabeler
    ) -> None:
        duplicate_time = pd.Timestamp("2023-01-01 09:15:00")
        df = pd.DataFrame(
            {
                "ts": [duplicate_time] * 5 + list(pd.date_range("2023-01-01 09:16:00", periods=200, freq="min")),
                "timestamp": [duplicate_time] * 5 + list(pd.date_range("2023-01-01 09:16:00", periods=200, freq="min")),
                "open": np.random.uniform(100, 110, 205),
                "high": np.random.uniform(105, 115, 205),
                "low": np.random.uniform(95, 105, 205),
                "close": np.random.uniform(100, 110, 205),
                "volume": np.random.randint(1000, 5000, 205),
            }
        )

        # Mock instrument repo to return a stock instrument
        mock_instrument = MagicMock()
        mock_instrument.segment = "STOCK"

        with (
            patch.object(labeler_instance.instrument_repo, "get_instrument_by_id", return_value=mock_instrument),
            patch.object(labeler_instance.label_repo, "insert_labels", return_value=None),
        ):
            result = await labeler_instance.process_symbol(1, "TEST_DUP", df)

        # Should handle duplicates and still process
        assert result is not None
        assert result.symbol == "TEST_DUP"


@pytest.mark.asyncio
class TestStatisticsCalculation:
    """Test comprehensive statistics calculation."""

    async def test_calculate_statistics_comprehensive(self, labeler_instance: OptimizedTripleBarrierLabeler) -> None:
        original_df = pd.DataFrame(
            {
                "ts": pd.date_range("2023-01-01", periods=100, freq="min"),
                "open": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(105, 115, 100),
                "low": np.random.uniform(95, 105, 100),
                "close": np.random.uniform(100, 110, 100),
            }
        )

        labeled_df = pd.DataFrame(
            {
                "ts": pd.date_range("2023-01-01", periods=80, freq="min"),
                "timeframe": ["15min"] * 80,
                "instrument_id": [1] * 80,
                "label": [1] * 30 + [-1] * 25 + [0] * 25,
                "exit_reason": [ExitReason.TP_HIT.name] * 30
                + [ExitReason.SL_HIT.name] * 25
                + [ExitReason.TIME_OUT.name] * 25,
                "exit_bar_offset": np.random.randint(1, 5, 80),
                "barrier_return": [0.02] * 30 + [-0.015] * 25 + [0.001] * 25,
                "entry_price": [100.0] * 80,
                "exit_price": [102.0] * 30 + [98.5] * 25 + [100.1] * 25,
            }
        )

        start_time = datetime.now()

        stats = labeler_instance._calculate_statistics("TEST", original_df, labeled_df, start_time, "15min")

        assert isinstance(stats, LabelingStats)
        assert stats.symbol == "TEST"
        assert stats.total_bars == 100
        assert stats.labeled_bars == 80
        assert stats.label_distribution["BUY"] == 30
        assert stats.label_distribution["SELL"] == 25
        assert stats.label_distribution["NEUTRAL"] == 25
        assert stats.win_rate > 0
        assert stats.profit_factor > 0
        assert stats.data_quality_score == 80.0

    async def test_get_annualization_factor(self, labeler_instance: OptimizedTripleBarrierLabeler) -> None:
        assert labeler_instance._get_annualization_factor("1MIN") > labeler_instance._get_annualization_factor("5MIN")
        assert labeler_instance._get_annualization_factor("5MIN") > labeler_instance._get_annualization_factor("15MIN")
        assert labeler_instance._get_annualization_factor("15MIN") > labeler_instance._get_annualization_factor("1H")
        assert labeler_instance._get_annualization_factor("1H") > labeler_instance._get_annualization_factor("1D")

        default_factor = labeler_instance._get_annualization_factor("INVALID")
        assert default_factor == 252


@pytest.mark.asyncio
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""

    @pytest.mark.slow
    async def test_large_dataset_performance(self, labeler_instance: OptimizedTripleBarrierLabeler) -> None:
        # Test with large dataset
        data_size = 10000
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=data_size, freq="min")

        df = pd.DataFrame(
            {
                "ts": dates,
                "timestamp": dates,
                "open": np.random.uniform(100, 110, data_size).cumsum() / 100 + 1000,
                "high": lambda x: x["open"] + np.random.uniform(0, 2, data_size),
                "low": lambda x: x["open"] - np.random.uniform(0, 2, data_size),
                "close": lambda x: x["open"] + np.random.uniform(-1, 1, data_size),
                "volume": np.random.randint(1000, 50000, data_size),
            }
        )

        # Properly calculate derived columns
        df["high"] = df["open"] + np.random.uniform(0, 2, data_size)
        df["low"] = df["open"] - np.random.uniform(0, 2, data_size)
        df["close"] = df["open"] + np.random.uniform(-1, 1, data_size)

        # Mock instrument repo
        mock_instrument = MagicMock()
        mock_instrument.segment = "STOCK"

        start_time = datetime.now()

        with (
            patch.object(labeler_instance.instrument_repo, "get_instrument_by_id", return_value=mock_instrument),
            patch.object(labeler_instance.label_repo, "insert_labels", return_value=None),
        ):
            result = await labeler_instance.process_symbol(1, "LARGE_TEST", df)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        assert result is not None
        assert result.labeled_bars > 0
        # Should process large dataset in reasonable time (< 30 seconds)
        assert processing_time < 30.0

    async def test_empty_dataframe(self, labeler_instance: OptimizedTripleBarrierLabeler) -> None:
        df = pd.DataFrame()

        result = await labeler_instance.process_symbol(1, "EMPTY", df)
        assert result is None

    async def test_single_row_dataframe(self, labeler_instance: OptimizedTripleBarrierLabeler) -> None:
        df = pd.DataFrame(
            {
                "ts": [pd.Timestamp("2023-01-01 09:15:00")],
                "timestamp": [pd.Timestamp("2023-01-01 09:15:00")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            }
        )

        result = await labeler_instance.process_symbol(1, "SINGLE", df)
        assert result is None  # Insufficient data

    async def test_process_symbols_parallel_with_exceptions(
        self, labeler_instance: OptimizedTripleBarrierLabeler
    ) -> None:
        # Create test data with some symbols that will fail
        symbol_data = {
            1: {
                "symbol": "GOOD_SYMBOL",
                "data": pd.DataFrame(
                    {
                        "ts": pd.date_range("2023-01-01", periods=200, freq="min"),
                        "timestamp": pd.date_range("2023-01-01", periods=200, freq="min"),
                        "open": np.random.uniform(100, 110, 200),
                        "high": np.random.uniform(105, 115, 200),
                        "low": np.random.uniform(95, 105, 200),
                        "close": np.random.uniform(100, 110, 200),
                        "volume": np.random.randint(1000, 5000, 200),
                    }
                ),
                "timeframe": "15min",
            },
            2: {
                "symbol": "BAD_SYMBOL",
                "data": pd.DataFrame(
                    {  # Missing columns
                        "ts": pd.date_range("2023-01-01", periods=10, freq="min"),
                        "open": [100.0] * 10,
                    }
                ),
                "timeframe": "15min",
            },
        }

        # Mock instrument repo for good symbol
        mock_instrument = MagicMock()
        mock_instrument.segment = "STOCK"

        with (
            patch.object(labeler_instance.instrument_repo, "get_instrument_by_id", return_value=mock_instrument),
            patch.object(labeler_instance.label_repo, "insert_labels", return_value=None),
        ):
            stats_dict = await labeler_instance.process_symbols_parallel(symbol_data)

        # Should handle good symbol, skip bad one
        assert "GOOD_SYMBOL" in stats_dict
        assert "BAD_SYMBOL" not in stats_dict
        assert len(stats_dict) == 1


# ============================================================================
# REAL DATABASE DATA TESTING (PRODUCTION-GRADE)
# ============================================================================


@pytest.mark.asyncio
class TestLabelerWithRealData:
    """
    Test labeler functionality using real database data.
    This provides the most realistic testing scenarios with actual market patterns.
    """

    async def test_labeler_with_real_database_data(
        self,
        labeler_instance: OptimizedTripleBarrierLabeler,
        real_ohlcv_data_from_db: tuple[pd.DataFrame, int, str]
    ) -> None:
        """
        Test labeler with real OHLCV data from the database.
        This is the ultimate integration test using actual market data.
        """
        df, instrument_id, symbol = real_ohlcv_data_from_db

        print(f"\nTesting with real data for {symbol} (ID: {instrument_id})")
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        # Validate data quality before testing
        assert len(df) >= 200, f"Need at least 200 data points, got {len(df)}"
        assert not df.isnull().any().any(), "Real data should not contain NaN values"
        assert (df['high'] >= df['low']).all(), "High prices should be >= Low prices"
        assert (df['high'] >= df['open']).all(), "High prices should be >= Open prices"
        assert (df['high'] >= df['close']).all(), "High prices should be >= Close prices"
        assert (df['low'] <= df['open']).all(), "Low prices should be <= Open prices"
        assert (df['low'] <= df['close']).all(), "Low prices should be <= Close prices"

        # Test labeling on real data
        try:
            stats = await labeler_instance.process_symbol(instrument_id, symbol, df)

            # Validate results
            assert stats is not None, "Labeler should return statistics"
            assert stats.labeled_bars >= 0, "Should have non-negative labeled bars"
            assert stats.total_bars > 0, "Should have processed some bars"

            # With real market data, we expect some labels
            if stats.labeled_bars > 0:
                # Validate label distribution
                total_labels = stats.buy_labels + stats.sell_labels + stats.neutral_labels
                assert total_labels == stats.labeled_bars, "Label counts should sum to total labeled bars"

                # Real market data should produce reasonable label distribution
                buy_ratio = stats.buy_labels / stats.labeled_bars if stats.labeled_bars > 0 else 0
                sell_ratio = stats.sell_labels / stats.labeled_bars if stats.labeled_bars > 0 else 0
                neutral_ratio = stats.neutral_labels / stats.labeled_bars if stats.labeled_bars > 0 else 0

                assert 0 <= buy_ratio <= 1, f"Buy ratio should be 0-1, got {buy_ratio}"
                assert 0 <= sell_ratio <= 1, f"Sell ratio should be 0-1, got {sell_ratio}"
                assert 0 <= neutral_ratio <= 1, f"Neutral ratio should be 0-1, got {neutral_ratio}"

                # Real market data typically produces mixed labels (not all one type)
                label_types = sum([1 for x in [buy_ratio, sell_ratio, neutral_ratio] if x > 0])
                assert label_types >= 2, f"Expected mixed labels from real data, got only {label_types} types"

                print("Labeling Results:")
                print(f"  Total bars processed: {stats.total_bars}")
                print(f"  Labeled bars: {stats.labeled_bars}")
                print(f"  Buy labels: {stats.buy_labels} ({buy_ratio:.1%})")
                print(f"  Sell labels: {stats.sell_labels} ({sell_ratio:.1%})")
                print(f"  Neutral labels: {stats.neutral_labels} ({neutral_ratio:.1%})")
                print(f"  Average holding period: {stats.avg_holding_period:.2f}")

        except Exception as e:
            pytest.fail(f"Labeler failed on real market data for {symbol}: {e}")

    async def test_real_data_atr_calculation(
        self,
        labeler_instance: OptimizedTripleBarrierLabeler,
        real_ohlcv_data_from_db: tuple[pd.DataFrame, int, str]
    ) -> None:
        """
        Test ATR calculation with real market data to ensure realistic values.
        """
        df, instrument_id, symbol = real_ohlcv_data_from_db

        # Calculate ATR using the labeler's method
        atr_values = labeler_instance._calculate_atr(df, symbol)

        # Validate ATR results
        assert len(atr_values) == len(df), "ATR array should match data length"
        assert not np.isnan(atr_values[-100:]).any(), "Recent ATR values should not be NaN"
        assert (atr_values[-100:] > 0).all(), "ATR values should be positive"

        # Real market ATR should be reasonable relative to prices
        recent_atr = np.mean(atr_values[-20:])  # Last 20 ATR values
        recent_close = np.mean(df['close'].tail(20))
        atr_percentage = recent_atr / recent_close

        # ATR should typically be 0.1% to 10% of price for most instruments
        assert 0.001 <= atr_percentage <= 0.1, f"ATR percentage {atr_percentage:.3%} seems unrealistic for {symbol}"

        print(f"ATR Analysis for {symbol}:")
        print(f"  Average recent ATR: {recent_atr:.2f}")
        print(f"  Average recent close: {recent_close:.2f}")
        print(f"  ATR as % of price: {atr_percentage:.3%}")

    async def test_real_data_feature_engineering(
        self,
        labeler_instance: OptimizedTripleBarrierLabeler,
        real_ohlcv_data_from_db: tuple[pd.DataFrame, int, str]
    ) -> None:
        """
        Test that the labeler works correctly with the full feature engineering pipeline on real data.
        """
        df, instrument_id, symbol = real_ohlcv_data_from_db

        # Test that barrier calculation works with real volatility patterns
        atr_values = labeler_instance._calculate_atr(df, symbol)

        # Calculate barriers
        tp_barriers, sl_barriers = labeler_instance._calculate_dynamic_barriers(atr_values)

        # Validate barrier calculations
        assert len(tp_barriers) == len(df), "TP barriers should match data length"
        assert len(sl_barriers) == len(df), "SL barriers should match data length"
        assert (tp_barriers > 1).all(), "TP multipliers should be > 1"
        assert (sl_barriers > 0).all() and (sl_barriers < 1).all(), "SL multipliers should be 0-1"

        # Test epsilon calculation with real market timing
        epsilon_array = labeler_instance._calculate_dynamic_epsilon(df, atr_values)
        assert len(epsilon_array) == len(df), "Epsilon array should match data length"
        assert (epsilon_array > 0).all(), "Epsilon values should be positive"

        print(f"Feature Engineering Results for {symbol}:")
        print(f"  ATR range: {np.min(atr_values[-100:]):.2f} - {np.max(atr_values[-100:]):.2f}")
        print(f"  TP multiplier range: {np.min(tp_barriers[-100:]):.3f} - {np.max(tp_barriers[-100:]):.3f}")
        print(f"  SL multiplier range: {np.min(sl_barriers[-100:]):.3f} - {np.max(sl_barriers[-100:]):.3f}")
        print(f"  Epsilon range: {np.min(epsilon_array[-100:]):.5f} - {np.max(epsilon_array[-100:]):.5f}")


@pytest.mark.asyncio
class TestLabelerDatabaseIntegration:
    """
    Test labeler integration with database repositories and data persistence.
    """

    async def test_label_persistence_integration(
        self,
        labeler_instance: OptimizedTripleBarrierLabeler,
        real_ohlcv_data_from_db: tuple[pd.DataFrame, int, str]
    ) -> None:
        """
        Test that labels are correctly structured for database persistence.
        """
        df, instrument_id, symbol = real_ohlcv_data_from_db

        # Take a smaller subset for focused testing
        test_df = df.tail(300).copy()

        # Mock the label repository to capture what would be saved
        mock_label_repo = labeler_instance.label_repo

        # Process the data
        stats = await labeler_instance.process_symbol(instrument_id, symbol, test_df)

        if stats and stats.labeled_bars > 0:
            # Verify that save_labels was called
            mock_label_repo.save_labels.assert_called()

            # Get the labels that would be saved
            call_args = mock_label_repo.save_labels.call_args[0][0]

            # Validate label structure
            assert isinstance(call_args, list), "Labels should be a list"
            assert len(call_args) == stats.labeled_bars, "Should save all labeled bars"

            # Validate each label has required database fields
            sample_label = call_args[0]
            required_attrs = ['instrument_id', 'timestamp', 'label', 'exit_reason', 'exit_price']
            for attr in required_attrs:
                assert hasattr(sample_label, attr), f"Label missing required attribute: {attr}"

            # Validate data types and ranges
            for label in call_args[:5]:  # Check first 5 labels
                assert isinstance(label.instrument_id, int)
                assert label.instrument_id == instrument_id
                assert label.label in [-1, 0, 1]  # Valid label values
                assert label.exit_price > 0  # Price should be positive

            print(f"Database Integration Test for {symbol}:")
            print(f"  Labels to be saved: {len(call_args)}")
            print(f"  Sample label: ID={sample_label.instrument_id}, Label={sample_label.label}")


class TestIntegrationScenarios:
    """Test realistic trading scenarios."""

    async def test_realistic_trading_scenario(
        self, labeler_instance: OptimizedTripleBarrierLabeler, sample_ohlcv_data_for_labeling: pd.DataFrame
    ) -> None:
        df = sample_ohlcv_data_for_labeling.copy()

        # Create realistic price movements
        df["close"] = np.linspace(1000, 1050, len(df))  # Uptrend
        df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
        df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(0, 2, len(df))
        df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(0, 2, len(df))

        # Mock the executor to return realistic results
        labels_mock = np.random.choice([-1, 0, 1], len(df), p=[0.3, 0.4, 0.3])
        exit_bar_offsets_mock = np.random.randint(1, 6, len(df))
        exit_prices_mock = df["close"].values + np.random.uniform(-1, 1, len(df))
        exit_reasons_mock = np.random.choice([0, 1, 2], len(df), p=[0.4, 0.3, 0.3])
        mfe_mock = np.random.uniform(0, 0.05, len(df))
        mae_mock = np.random.uniform(-0.05, 0, len(df))

        (labeler_instance.executor.submit.return_value).set_result(
            (labels_mock, exit_bar_offsets_mock, exit_prices_mock, exit_reasons_mock, mfe_mock, mae_mock)
        )

        # Mock instrument repo
        mock_instrument = MagicMock()
        mock_instrument.segment = "STOCK"

        with (
            patch.object(labeler_instance.instrument_repo, "get_instrument_by_id", return_value=mock_instrument),
            patch.object(labeler_instance.label_repo, "insert_labels", return_value=None),
        ):
            stats = await labeler_instance.process_symbol(1, "REALISTIC", df)

        assert stats is not None
        assert stats.labeled_bars > 0
        assert stats.symbol == "REALISTIC"
        assert "BUY" in stats.label_distribution
        assert "SELL" in stats.label_distribution
        assert "NEUTRAL" in stats.label_distribution
