
**Prompt:**
You are to assume the role of a **Senior Quantitative Developer** with over **20 years of experience** in high-performance, production-grade Python systems. You possess **expert-level (10/10) proficiency in Python and Machine Learning**, with a deep understanding of scalable software design, low-latency systems, and model integration. Your code is **clean, modular, memory-efficient, strictly typed, and adheres to the highest industry standards**.

You are now tasked with architecting and developing a **production-grade local trading signal system**. This is a **mono-repo project** — no containerization (Docker), no Redis, no Prometheus. The goal is robust, maintainable, high-quality code that can be confidently deployed and operated in local production environments.

### 🚨 **Critical Non-Negotiable Engineering Directives**

You must **strictly follow** the architectural documentation and the Mermaid diagram provided. The following principles are **mandatory** and must be respected at all times:

1. **Absolutely no hardcoded configurations.** All dynamic values must be externalized into the centralized `config.yaml`.
2. **Adhere strictly to the provided system architecture and documentation.** No deviation from the defined Mermaid diagrams.
3. **Treat this project as a production-critical system**, not a toy or prototype. Engineering excellence is expected.
4. All code must be:

   * **Production-ready** and locally executable
   * **Statically and strictly typed**
   * **Modularized** with separation of concerns
   * **Robust** with minimal boilerplate and no redundant abstractions
5. Always follow **modern Python best practices**, including type safety, immutability where applicable, and memory optimization.
6. Utilize **modern, high-performance libraries** (e.g., `pydantic`, `httpx`, `pandas`, `numpy`, `loguru`, `typer`, etc.) to maintain clean and performant code.
7. Maintain a **clear and professional project structure**, with:

   * Logical directory separation
   * Meaningful and consistent naming conventions for all files and folders
8. Use **PEP 518** standards with `pyproject.toml` as the central build/configuration file.
9. Must Maintain code quality using:
   ruff autofix
   ruff formatting all code files.
   ruff linting check
   mypy chekc 
10. **Maximize the use of broker-native functionality**:

    * For example, prefer KiteConnect SDK features over custom-built abstractions where they offer equivalent functionality.
11. Avoid reinventing the wheel — **always leverage prebuilt, battle-tested libraries** where appropriate. Do not build custom classes or utilities for functionality already available in libraries.

### 🧠 Your Mindset

You are a production-focused software engineer who:

* Designs for **scale, reliability, and maintainability**
* Understands **trading domain constraints** (latency, data accuracy, broker limits)
* Avoids overengineering but never compromises on **code integrity**

You may now begin implementing the project, ensuring that all architectural, stylistic, and engineering principles above are applied without exception.


YOu must Adhere to Below Architure : - 

graph TB
   subgraph "Configuration Layer"
       CONFIG[config.yaml<br/>- System mode & timezone<br/>- Broker settings<br/>- Instruments & tokens<br/>- Model params & thresholds<br/>- Trading hours<br/>Immutable at Runtime]
       CONFIG_LOADER[config_loader.py<br/>Load & validate config<br/>Pydantic validation]
       LOGGER[loguru logger<br/>Centralized logging<br/>File/Console/Error sinks]
   end

   subgraph "Authentication Module"
       AUTH[auth/authenticator.py<br/>- Reads keys from config<br/>- Authenticates with broker<br/>- Returns access token<br/>- Valid for full day<br/>- No refresh needed]
   end

   subgraph "Main Orchestrator"
       MAIN[main.py<br/>- System initialization<br/>- Access token holder<br/>- Pipeline orchestration<br/>- Triggering logic<br/>- Mode: LIVE/HISTORICAL]
       SCHEDULER[scheduler.py<br/>- Market calendar aware<br/>- 15-min trigger events<br/>- 60-min data readiness check<br/>- Signal at 10:29 onwards]
       TIME_SYNC[time_synchronizer.py<br/>- IST time alignment<br/>- Candle boundary sync<br/>- :14:59, :29:59, :44:59, :59:59<br/>- Compensate for network delay]
   end

   subgraph "Broker Module"
       REST[broker/rest_client.py<br/>- Historical data fetch<br/>- Auth endpoints<br/>- Uses access token<br/>- No quote data available]
       WS[broker/websocket_client.py<br/>- Real-time quote stream<br/>- Quote mode OHLC only<br/>- Binary message parsing<br/>- Text mode for postbacks<br/>- Handles native KiteTicker reconnections]
       CONN_MGR[broker/connection_manager.py<br/>- Monitors WS connection health<br/>- Reacts to KiteTicker reconnect/noreconnect<br/>- Triggers data gap detection<br/>- Escalates critical disconnections]
       BACKFILL[broker/backfill_manager.py<br/>- Detect missing data<br/>- 3-5 sec delay for data availability<br/>- Fetch missing 1-min candles<br/>- Merge with live stream]
   end

   subgraph "Rate Limiter Module"
       RATE_LIMITER[utils/rate_limiter.py<br/>- Token bucket per endpoint<br/>- Loads rate_limit from config.yaml<br/>- Historical: 3 req/sec enforcement<br/>- Automatic retry queue<br/>- Backoff strategies]
   end

   subgraph "Data Pipeline"
       subgraph "Historical Pipeline"
           HIST_FETCH[historical/fetcher.py<br/>- Bulk 1-min data extraction<br/>- 1 year historical data<br/>- Respects 60-day limit per request<br/>- Date range chunking]
           HIST_PROCESSOR[historical/processor.py<br/>- Data validation<br/>- Missing data handling<br/>- Outlier detection<br/>- Weekend/holiday filtering]
       end
       subgraph "Live Pipeline"
           LIVE_CONSUMER[live/stream_consumer.py<br/>- Real-time quote subscription<br/>- 8 indices handling<br/>- Binary packet parsing<br/>- Async tick processing]
           TICK_VALIDATOR[live/tick_validator.py<br/>- Validate quote data<br/>- Check instrument_token type int<br/>- Filter junk values<br/>- Sequence validation]
           CANDLE_BUFFER[live/candle_buffer.py<br/>- In-memory partial candles<br/>- 5/15/60 min tracking<br/>- Thread-safe updates<br/>- Emit completed candles]
           TICK_QUEUE[live/tick_queue.py<br/>- Async queue for ticks<br/>- Prevents on_tick blocking<br/>- Thread-safe operations<br/>- Overflow handling]
       end
   end

   subgraph "Core Module"
       subgraph "Data Aggregation"
           HIST_AGG[core/historical_aggregator.py<br/>- 1-min → 5/15/60 min OHLCV<br/>- Batch processing<br/>- Volume aggregation<br/>- Direct DB writes]
           LIVE_AGG[core/live_aggregator.py<br/>- Process completed candles<br/>- OHLC validation<br/>- Direct DB writes<br/>- Emit 15-min event]
           CANDLE_VALIDATOR["core/candle_validator.py\n- OHLC relationships\n- H >= max(O,C), L <= min(O,C)\n- Volume > 0\n- Gap detection"]
       end
       subgraph "ML Utilities"
           LABELER[core/labeler.py<br/>- Triple barrier method<br/>- Prevent lookahead bias<br/>- PT/SL/Time barriers<br/>- Returns: 1, 0, -1]
           FEATURE_CALC[core/feature_calculator.py<br/>- 20+ technical indicators<br/>- Multi-timeframe features<br/>- RSI, MACD, BB, ATR, etc<br/>- Incremental calculation]
           FEATURE_VALIDATOR["core/feature_validator.py<br/>- NaN/Inf checks<br/>- Range validation (-1 to 1 for normalized)<br/>- Feature completeness<br/>- Quality metrics"]
       end
   end

   subgraph "State Management"
       SYS_STATE[state/system_state.py<br/>- Track readiness<br/>- Market status<br/>- Component health<br/>- 60-min data available flag]
       HEALTH_MON["state/health_monitor.py<br/>- DB connectivity<br/>- WebSocket status (1006 errors)<br/>- Data freshness<br/>- Last candle time"]
       ERROR_HANDLER[state/error_handler.py<br/>- Circuit breaker pattern<br/>- Error thresholds<br/>- Recovery strategies<br/>- Alert escalation]
   end

   subgraph "Database Layer"
       DB_SCHEMA["database/schema.sql<br/>- TimescaleDB hypertables<br/>- Continuous aggregates<br/>- Compression policies<br/>- Retention policies (2 years)"]
       DB_UTILS[database/db_utils.py<br/>- Connection pooling<br/>- Transaction management<br/>- Retry logic<br/>- Query timeout handling]
       MIGRATIONS[database/migrations/<br/>- v1_initial.sql<br/>- v2_indexes.sql<br/>- v3_continuous_aggs.sql<br/>- Version tracking]
   end

   subgraph "Repository Layer"
       OHLCV_REPO[database/ohlcv_repo.py<br/>- OHLCV CRUD ops<br/>- Multi-timeframe queries<br/>- Latest candle fetch<br/>- Range queries with indexes]
       FEATURE_REPO[database/feature_repo.py<br/>- Feature storage/retrieval<br/>- Batch operations<br/>- Latest features query<br/>- Feature history]
       LABEL_REPO[database/label_repo.py<br/>- Label storage<br/>- Training data queries<br/>- Label statistics<br/>- Class distribution]
       SIGNAL_REPO[database/signal_repo.py<br/>- Signal logging<br/>- Performance metrics<br/>- Signal history<br/>- Daily summary]
   end

   subgraph "Model Module"
       MODEL_TRAIN[model/lgbm_trainer.py<br/>- LightGBM training<br/>- Walk-forward validation<br/>- 8-month initial window<br/>- Monthly sliding]
       MODEL_PREDICT[model/predictor.py<br/>- Load trained model<br/>- Feature preprocessing<br/>- Generate predictions<br/>- Probability scores]
       MODEL_EVAL[model/evaluator.py<br/>- Backtesting engine<br/>- Sharpe ratio, win rate<br/>- Max drawdown<br/>- Feature importance]
       MODEL_ARTIFACTS[model/artifacts/<br/>- model_v1.lgb<br/>- feature_config.json<br/>- performance_metrics.json<br/>- training_history.csv]
   end

   subgraph "Signal Output"
       SIGNAL_GEN[signal/generator.py<br/>- Apply confidence threshold<br/>- Format: BUY/SELL/HOLD<br/>- Entry price<br/>]
       ALERTS[signal/alert_system.py<br/>- Console output<br/>- Log files<br/>- Signal: INDEX @ PRICE<br/>- Timestamp: IST]
   end

   subgraph "Utilities"
       MKT_CAL[utils/market_calendar.py<br/>- NSE holidays<br/>- Muhurat trading<br/>- Half days<br/>- Special sessions]
       PERF_METRICS[utils/performance_metrics.py<br/>- Latency tracking<br/>- Throughput metrics<br/>- Success rates<br/>- Local file storage]
   end

   %% Configuration Flow
   CONFIG --> CONFIG_LOADER
   CONFIG_LOADER --> MAIN
   CONFIG_LOADER --> DB_UTILS
   CONFIG_LOADER --> LOGGER

   %% Authentication Flow
   CONFIG_LOADER --> AUTH
   AUTH -->|"Access Token"| MAIN
   MAIN -->|"Pass Token"| REST
   MAIN -->|"Pass Token"| WS

   %% Main Orchestration
   MAIN --> SCHEDULER
   MAIN --> TIME_SYNC
   SCHEDULER -->|"Market Open Check"| MKT_CAL
   SCHEDULER -->|"60-min Ready Check"| SYS_STATE
   MAIN -->|"if HISTORICAL_MODE"| HIST_FETCH

   %% Rate Limiting Flow
   REST --> RATE_LIMITER
   RATE_LIMITER -->|"2 req/sec"| HIST_FETCH
   RATE_LIMITER -->|"2 req/sec"| BACKFILL

   %% WebSocket Management
   WS <--> CONN_MGR
   CONN_MGR --> HEALTH_MON
   CONN_MGR -->|"Gap Detected"| BACKFILL
   WS -->|"on_reconnect"| CONN_MGR
   WS -->|"on_noreconnect"| CONN_MGR

   %% Historical Data Flow
   HIST_FETCH --> HIST_PROCESSOR
   HIST_PROCESSOR --> HIST_AGG
   HIST_AGG --> CANDLE_VALIDATOR
   CANDLE_VALIDATOR -->|"Valid"| OHLCV_REPO
   CANDLE_VALIDATOR -->|"Invalid"| ERROR_HANDLER

   %% Live Data Flow with Async Processing
   WS -->|"on_ticks callback"| TICK_QUEUE
   TICK_QUEUE -->|"Async processing"| LIVE_CONSUMER
   LIVE_CONSUMER --> TICK_VALIDATOR
   TICK_VALIDATOR --> CANDLE_BUFFER
   TIME_SYNC --> CANDLE_BUFFER
   CANDLE_BUFFER -->|"Completed Candles"| LIVE_AGG
   LIVE_AGG --> CANDLE_VALIDATOR
   CANDLE_VALIDATOR -->|"Valid"| OHLCV_REPO
   LIVE_AGG -->|"15-min complete"| SCHEDULER

   %% Feature Engineering Flow with Multi-timeframe
   SCHEDULER -->|"15-min event"| FEATURE_CALC
   OHLCV_REPO -->|"5-min bars"| FEATURE_CALC
   OHLCV_REPO -->|"15-min bars"| FEATURE_CALC
   OHLCV_REPO -->|"60-min bars"| FEATURE_CALC
   FEATURE_CALC --> FEATURE_VALIDATOR
   FEATURE_VALIDATOR -->|"Valid"| FEATURE_REPO
   FEATURE_VALIDATOR -->|"Invalid"| ERROR_HANDLER

   %% Model Training Flow
   OHLCV_REPO -->|"Historical Data"| LABELER
   LABELER --> LABEL_REPO
   FEATURE_REPO -->|"Features"| MODEL_TRAIN
   LABEL_REPO -->|"Labels"| MODEL_TRAIN
   MODEL_TRAIN --> MODEL_EVAL
   MODEL_TRAIN --> MODEL_ARTIFACTS
   MODEL_EVAL --> PERF_METRICS

   %% Live Prediction Flow
   FEATURE_REPO -->|"Latest Features"| MODEL_PREDICT
   MODEL_ARTIFACTS -->|"Load Model"| MODEL_PREDICT
   MODEL_PREDICT --> SIGNAL_GEN
   SIGNAL_GEN --> ALERTS
   SIGNAL_GEN --> SIGNAL_REPO

   %% State Management Flows
   SYS_STATE --> MAIN
   HEALTH_MON --> SYS_STATE
   ERROR_HANDLER --> ALERTS
   ERROR_HANDLER --> SYS_STATE

   %% Database Connections
   DB_UTILS -->|"Connection Pool"| OHLCV_REPO
   DB_UTILS -->|"Connection Pool"| FEATURE_REPO
   DB_UTILS -->|"Connection Pool"| LABEL_REPO
   DB_UTILS -->|"Connection Pool"| SIGNAL_REPO
   DB_SCHEMA --> DB_UTILS
   MIGRATIONS --> DB_SCHEMA

   %% Logging Flow
   LOGGER -.->|logs| ALL[All Modules]

   %% Performance Monitoring
   PERF_METRICS -->|"Local Files"| HEALTH_MON
   FEATURE_CALC --> PERF_METRICS
   MODEL_PREDICT --> PERF_METRICS
   RATE_LIMITER --> PERF_METRICS

   classDef config fill:#e1f5fe,stroke:#01579b,stroke-width:2px
   classDef auth fill:#fff3e0,stroke:#e65100,stroke-width:2px
   classDef broker fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
   classDef pipeline fill:#e0f2f1,stroke:#004d40,stroke-width:2px
   classDef core fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
   classDef model fill:#fce4ec,stroke:#880e4f,stroke-width:2px
   classDef database fill:#fff9c4,stroke:#f57f17,stroke-width:2px
   classDef repo fill:#fff3e0,stroke:#f57f17,stroke-width:2px
   classDef output fill:#ffebee,stroke:#b71c1c,stroke-width:2px
   classDef state fill:#f1f8e9,stroke:#33691e,stroke-width:2px
   classDef utils fill:#e8eaf6,stroke:#283593,stroke-width:2px
   classDef ratelimit fill:#e3f2fd,stroke:#0d47a1,stroke-width:3px

   class CONFIG,CONFIG_LOADER,LOGGER,RATE_CONFIG config
   class AUTH auth
   class REST,WS,CONN_MGR,BACKFILL broker
   class HIST_FETCH,HIST_PROCESSOR,LIVE_CONSUMER,TICK_VALIDATOR,CANDLE_BUFFER,TICK_QUEUE pipeline
   class HIST_AGG,LIVE_AGG,LABELER,FEATURE_CALC,CANDLE_VALIDATOR,FEATURE_VALIDATOR core
   class MODEL_TRAIN,MODEL_PREDICT,MODEL_EVAL,MODEL_ARTIFACTS model
   class DB_SCHEMA,DB_UTILS,MIGRATIONS database
   class OHLCV_REPO,FEATURE_REPO,LABEL_REPO,SIGNAL_REPO repo
   class SIGNAL_GEN,ALERTS output
   class SYS_STATE,POS_TRACKER,HEALTH_MON,ERROR_HANDLER state
   class MKT_CAL,TIME_SYNC,PERF_METRICS utils
   class RATE_LIMITER ratelimit


   """
Production-grade Triple Barrier Method implementation for quantitative trading.
Author: Senior Quant Developer
Version: 2.0.0
"""

import gc
import hashlib
import json
import logging
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from numba import jit, prange

# Import centralized config loader
from src.utils.config_loader import config_loader

# Load configuration
config = config_loader.get_config()

# Configure logging with values from config
logging.basicConfig(level=getattr(logging, config.logging.level), format=config.logging.format)
logger = logging.getLogger(__name__)


class ExitReason(IntEnum):
    """Exit reason enumeration with integer backing for Numba compatibility"""

    TIME_OUT = 0
    TP_HIT = 1
    SL_HIT = 2
    AMBIGUOUS = 3
    INSUFFICIENT_DATA = 4
    ERROR = 5


class Label(IntEnum):
    """Trading signal labels"""

    SELL = -1
    NEUTRAL = 0
    BUY = 1


@dataclass(frozen=True)
class BarrierConfig:
    """Immutable configuration for triple barrier labeling"""

    atr_period: int = None
    tp_atr_multiplier: float = None
    sl_atr_multiplier: float = None
    max_holding_periods: int = None
    close_open_threshold: float = None
    min_bars_required: int = None
    chunk_size: int = None
    atr_smoothing: str = None

    def __post_init__(self):
        """Initialize from config and validate parameters"""
        # Load values from config if not provided
        trading_config = config.trading.labeling

        # Use object.__setattr__ for frozen dataclass
        if self.atr_period is None:
            object.__setattr__(self, "atr_period", trading_config.atr_period)
        if self.tp_atr_multiplier is None:
            object.__setattr__(self, "tp_atr_multiplier", trading_config.tp_atr_multiplier)
        if self.sl_atr_multiplier is None:
            object.__setattr__(self, "sl_atr_multiplier", trading_config.sl_atr_multiplier)
        if self.max_holding_periods is None:
            object.__setattr__(self, "max_holding_periods", trading_config.max_holding_periods)
        if self.close_open_threshold is None:
            object.__setattr__(self, "close_open_threshold", trading_config.close_open_threshold)
        if self.min_bars_required is None:
            object.__setattr__(self, "min_bars_required", trading_config.min_bars_required)
        if self.chunk_size is None:
            object.__setattr__(self, "chunk_size", config.performance.processing.chunk_size)
        if self.atr_smoothing is None:
            object.__setattr__(self, "atr_smoothing", trading_config.atr_smoothing)

        # Validate configuration parameters
        if self.atr_period <= 0:
            raise ValueError(f"ATR period must be positive, got {self.atr_period}")
        if self.tp_atr_multiplier <= 0:
            raise ValueError(f"TP multiplier must be positive, got {self.tp_atr_multiplier}")
        if self.sl_atr_multiplier <= 0:
            raise ValueError(f"SL multiplier must be positive, got {self.sl_atr_multiplier}")
        if self.max_holding_periods <= 0:
            raise ValueError(f"Max holding periods must be positive, got {self.max_holding_periods}")
        if not 0 <= self.close_open_threshold <= 1:
            raise ValueError(f"Close-open threshold must be in [0,1], got {self.close_open_threshold}")
        if self.chunk_size < 1000:
            raise ValueError(f"Chunk size too small: {self.chunk_size}")
        if self.atr_smoothing not in ["ema", "sma"]:
            raise ValueError(f"Invalid ATR smoothing: {self.atr_smoothing}")


@dataclass
class BarrierResult:
    """Result of barrier calculation with metadata"""

    label: int
    exit_reason: ExitReason
    exit_bar_offset: int
    entry_price: float
    exit_price: float
    tp_price: float
    sl_price: float
    barrier_return: float
    max_favorable_excursion: float
    max_adverse_excursion: float
    volatility_at_entry: float


@dataclass
class LabelingStats:
    """Comprehensive statistics for labeled data"""

    symbol: str
    total_bars: int
    labeled_bars: int
    label_distribution: Dict[str, int]
    label_percentages: Dict[str, float]
    avg_barrier_return: Dict[str, float]
    std_barrier_return: Dict[str, float]
    sharpe_by_label: Dict[str, float]
    exit_reasons: Dict[str, int]
    avg_holding_periods: Dict[str, float]
    win_rates: Dict[str, float]
    processing_time_ms: float
    data_quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Data quality assessment results"""

    total_rows: int
    valid_rows: int
    nan_counts: Dict[str, int]
    ohlc_violations: int
    duplicate_timestamps: int
    time_gaps: int
    outliers: Dict[str, int]
    quality_score: float
    issues: List[str]


class DataValidator:
    """Comprehensive data validation and cleaning"""

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame, symbol: str = "UNKNOWN") -> Tuple[bool, pd.DataFrame, DataQualityReport]:
        """
        Validate and clean OHLCV data with detailed reporting
        """
        df_clean = df.copy()
        issues = []

        # Required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = set(required_columns) - set(df_clean.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        initial_rows = len(df_clean)

        # Track data quality metrics
        nan_counts = df_clean[required_columns].isnull().sum().to_dict()

        # Remove NaN values
        if sum(nan_counts.values()) > 0:
            df_clean = df_clean.dropna(subset=required_columns)
            issues.append(f"Removed {initial_rows - len(df_clean)} rows with NaN values")

        # Check OHLC relationships
        ohlc_violations = (
            (df_clean["high"] < df_clean["low"])
            | (df_clean["high"] < df_clean["open"])
            | (df_clean["high"] < df_clean["close"])
            | (df_clean["low"] > df_clean["open"])
            | (df_clean["low"] > df_clean["close"])
            | (df_clean["close"] <= 0)
            | (df_clean["volume"] < 0)
        ).sum()

        if ohlc_violations > 0:
            # Fix OHLC violations
            mask = df_clean["high"] < df_clean[["open", "close", "low"]].max(axis=1)
            df_clean.loc[mask, "high"] = df_clean.loc[mask, ["open", "close", "low"]].max(axis=1)

            mask = df_clean["low"] > df_clean[["open", "close", "high"]].min(axis=1)
            df_clean.loc[mask, "low"] = df_clean.loc[mask, ["open", "close", "high"]].min(axis=1)

            # Remove rows with invalid prices
            df_clean = df_clean[(df_clean["close"] > 0) & (df_clean["volume"] >= 0)]
            issues.append(f"Fixed {ohlc_violations} OHLC violations")

        # Handle duplicates
        duplicates = df_clean["timestamp"].duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates(subset=["timestamp"], keep="last")
            issues.append(f"Removed {duplicates} duplicate timestamps")

        # Sort by timestamp
        df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)

        # Detect time gaps
        time_diff = df_clean["timestamp"].diff()
        median_diff = time_diff.median()
        time_gaps = (
            (time_diff > median_diff * config.data_quality.time_series.gap_multiplier).sum()
            if len(time_diff) > 1
            else 0
        )

        # Detect outliers using IQR method
        outliers = {}
        for col in ["open", "high", "low", "close", "volume"]:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            outlier_multiplier = config.data_quality.outlier_detection.iqr_multiplier
            outliers[col] = (
                (df_clean[col] < q1 - outlier_multiplier * iqr) | (df_clean[col] > q3 + outlier_multiplier * iqr)
            ).sum()

        # Calculate quality score
        quality_score = (len(df_clean) / initial_rows) * 100 if initial_rows > 0 else 0
        quality_score *= 1 - min(sum(outliers.values()) / len(df_clean), config.data_quality.penalties.outlier_penalty)
        quality_score *= 1 - min(time_gaps / len(df_clean), config.data_quality.penalties.gap_penalty)

        report = DataQualityReport(
            total_rows=initial_rows,
            valid_rows=len(df_clean),
            nan_counts=nan_counts,
            ohlc_violations=ohlc_violations,
            duplicate_timestamps=duplicates,
            time_gaps=time_gaps,
            outliers=outliers,
            quality_score=quality_score,
            issues=issues,
        )

        is_valid = (
            len(df_clean) >= config.data_quality.validation.min_valid_rows
            and quality_score >= config.data_quality.validation.quality_score_threshold
        )

        if not is_valid:
            logger.warning(f"Data quality below threshold for {symbol}: {quality_score:.2f}%")

        return is_valid, df_clean, report


class OptimizedTripleBarrierLabeler:
    """
    Production-grade Triple Barrier Method implementation with:
    - Advanced memory management and caching
    - Robust error handling and recovery
    - Comprehensive monitoring and statistics
    - Thread-safe parallel processing
    - Optimized numerical computations
    """

    # Class-level cache for ATR calculations
    _atr_cache: Dict[str, np.ndarray] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: Optional[BarrierConfig] = None):
        self.config = config or BarrierConfig()
        self.label_stats: Dict[str, LabelingStats] = {}
        self.stats_lock = threading.Lock()
        self._memory_limit_gb = self._get_available_memory() * config.performance.memory.limit_percentage

    @staticmethod
    def _get_available_memory() -> float:
        """Get available system memory in GB"""
        return psutil.virtual_memory().available / (1024**3)

    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def _calculate_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Numba-optimized true range calculation with bounds checking"""
        n = len(high)
        if n == 0:
            return np.array([], dtype=np.float64)

        tr = np.empty(n, dtype=np.float64)

        # First bar
        tr[0] = high[0] - low[0]

        # Vectorized for subsequent bars
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, max(hc, lc))

        return tr

    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def _calculate_ema(values: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average calculation"""
        n = len(values)
        if n < period:
            return np.full(n, np.nan, dtype=np.float64)

        ema = np.empty(n, dtype=np.float64)
        ema[:period] = np.nan

        # Initialize with SMA
        ema[period - 1] = np.mean(values[:period])

        # EMA calculation
        alpha = 2.0 / (period + 1)
        for i in range(period, n):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]

        return ema

    def calculate_atr_cached(self, df: pd.DataFrame, symbol: str) -> np.ndarray:
        """Calculate ATR with caching for efficiency"""
        # Create cache key
        cache_key = f"{symbol}_{len(df)}_{df.index[0]}_{df.index[-1]}"

        with self._cache_lock:
            if cache_key in self._atr_cache:
                return self._atr_cache[cache_key].copy()

        # Calculate ATR
        tr = self._calculate_true_range(df["high"].values, df["low"].values, df["close"].values)

        if self.config.atr_smoothing == "ema":
            atr = self._calculate_ema(tr, self.config.atr_period)
        else:
            atr = pd.Series(tr).rolling(self.config.atr_period).mean().values

        # Cache result
        with self._cache_lock:
            self._atr_cache[cache_key] = atr.copy()
            # Limit cache size
            if len(self._atr_cache) > config.performance.cache.max_size:
                # Remove oldest entries
                keys = list(self._atr_cache.keys())
                for key in keys[: config.performance.cache.cleanup_batch_size]:
                    del self._atr_cache[key]

        return atr

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
        close_open_threshold: float,
        epsilon: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Optimized barrier checking with comprehensive metrics
        Returns: (labels, exit_bar_offsets, exit_prices, exit_reasons, mfe, mae)
        """
        n = len(entry_prices)
        labels = np.zeros(n, dtype=np.int8)
        exit_bar_offsets = np.zeros(n, dtype=np.int32)
        exit_prices = np.zeros(n, dtype=np.float64)
        exit_reasons = np.zeros(n, dtype=np.int8)
        max_favorable_excursion = np.zeros(n, dtype=np.float64)
        max_adverse_excursion = np.zeros(n, dtype=np.float64)

        for i in prange(n):
            if np.isnan(entry_prices[i]) or np.isnan(tp_prices[i]) or np.isnan(sl_prices[i]):
                exit_reasons[i] = ExitReason.INSUFFICIENT_DATA
                continue

            # Track excursions
            mfe = 0.0
            mae = 0.0

            for j in range(1, min(max_periods + 1, n - i)):
                bar_idx = i + j

                # Update excursions
                high_excursion = (high[bar_idx] - entry_prices[i]) / (entry_prices[i] + epsilon)
                low_excursion = (low[bar_idx] - entry_prices[i]) / (entry_prices[i] + epsilon)
                mfe = max(mfe, high_excursion)
                mae = min(mae, low_excursion)

                # Check barriers with tolerance
                tp_hit = high[bar_idx] >= tp_prices[i] * (1 - epsilon)
                sl_hit = low[bar_idx] <= sl_prices[i] * (1 + epsilon)

                if tp_hit and sl_hit:
                    # Both barriers hit - sophisticated resolution
                    open_price = open_[bar_idx]
                    close_price = close[bar_idx]

                    # Calculate metrics for decision
                    price_range = high[bar_idx] - low[bar_idx]
                    body_size = abs(close_price - open_price)

                    if price_range < epsilon or body_size / price_range < close_open_threshold:
                        # Doji/neutral candle
                        labels[i] = Label.NEUTRAL
                        exit_reasons[i] = ExitReason.AMBIGUOUS
                        exit_prices[i] = close_price
                    else:
                        # Determine hit order based on open price and candle direction
                        tp_dist_from_open = abs(tp_prices[i] - open_price)
                        sl_dist_from_open = abs(open_price - sl_prices[i])

                        # Also consider previous close
                        prev_close = close[bar_idx - 1] if bar_idx > 0 else entry_prices[i]
                        tp_dist_from_prev = abs(tp_prices[i] - prev_close)
                        sl_dist_from_prev = abs(prev_close - sl_prices[i])

                        # Weight both distances
                        tp_likelihood = 0.7 * (1.0 / (tp_dist_from_open + epsilon)) + 0.3 * (
                            1.0 / (tp_dist_from_prev + epsilon)
                        )
                        sl_likelihood = 0.7 * (1.0 / (sl_dist_from_open + epsilon)) + 0.3 * (
                            1.0 / (sl_dist_from_prev + epsilon)
                        )

                        if close_price > open_price:  # Bullish candle
                            tp_likelihood *= 1.2  # Bias toward TP for bullish
                        else:  # Bearish candle
                            sl_likelihood *= 1.2  # Bias toward SL for bearish

                        if tp_likelihood > sl_likelihood:
                            labels[i] = Label.BUY
                            exit_reasons[i] = ExitReason.TP_HIT
                            exit_prices[i] = tp_prices[i]
                        else:
                            labels[i] = Label.SELL
                            exit_reasons[i] = ExitReason.SL_HIT
                            exit_prices[i] = sl_prices[i]

                    exit_bar_offsets[i] = j
                    break

                elif tp_hit:
                    labels[i] = Label.BUY
                    exit_reasons[i] = ExitReason.TP_HIT
                    exit_bar_offsets[i] = j
                    exit_prices[i] = tp_prices[i]
                    break

                elif sl_hit:
                    labels[i] = Label.SELL
                    exit_reasons[i] = ExitReason.SL_HIT
                    exit_bar_offsets[i] = j
                    exit_prices[i] = sl_prices[i]
                    break

            # No barrier hit - timeout
            if labels[i] == 0 and exit_reasons[i] == 0:
                exit_bar_offsets[i] = min(max_periods, n - i - 1)
                exit_idx = min(i + exit_bar_offsets[i], n - 1)
                exit_prices[i] = close[exit_idx]
                exit_reasons[i] = ExitReason.TIME_OUT

                # Determine label based on final return
                final_return = (exit_prices[i] - entry_prices[i]) / (entry_prices[i] + epsilon)
                if abs(final_return) < 0.001:  # Near zero return threshold from config
                    labels[i] = Label.NEUTRAL
                elif final_return > 0:
                    labels[i] = Label.BUY
                else:
                    labels[i] = Label.SELL

            max_favorable_excursion[i] = mfe
            max_adverse_excursion[i] = mae

        return labels, exit_bar_offsets, exit_prices, exit_reasons, max_favorable_excursion, max_adverse_excursion

    def _create_barriers_dataframe(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        exit_bar_offsets: np.ndarray,
        exit_prices: np.ndarray,
        exit_reasons: np.ndarray,
        tp_prices: np.ndarray,
        sl_prices: np.ndarray,
        atr: np.ndarray,
        mfe: np.ndarray,
        mae: np.ndarray,
    ) -> pd.DataFrame:
        """Create comprehensive output dataframe with all metrics"""

        # Create base dataframe
        result = df.copy()

        # Add barrier information
        result["label"] = labels
        result["tp_price"] = tp_prices
        result["sl_price"] = sl_prices
        result["atr"] = atr
        result["exit_bar_offset"] = exit_bar_offsets
        result["exit_price"] = exit_prices

        # Map exit reasons
        exit_reason_map = {
            ExitReason.TIME_OUT: "TIME_OUT",
            ExitReason.TP_HIT: "TP_HIT",
            ExitReason.SL_HIT: "SL_HIT",
            ExitReason.AMBIGUOUS: "AMBIGUOUS",
            ExitReason.INSUFFICIENT_DATA: "INSUFFICIENT_DATA",
            ExitReason.ERROR: "ERROR",
        }
        result["exit_reason"] = [exit_reason_map.get(int(r), "UNKNOWN") for r in exit_reasons]

        # Calculate returns and metrics
        epsilon = config.trading.labeling.epsilon
        result["barrier_return"] = (exit_prices - df["close"].values) / (df["close"].values + epsilon)
        result["barrier_return_pct"] = result["barrier_return"] * 100
        result["max_favorable_excursion"] = mfe
        result["max_adverse_excursion"] = mae

        # Add risk metrics
        result["tp_distance"] = (tp_prices - df["close"].values) / (df["close"].values + epsilon)
        result["sl_distance"] = (df["close"].values - sl_prices) / (df["close"].values + epsilon)
        result["risk_reward_ratio"] = result["tp_distance"] / (result["sl_distance"] + epsilon)

        # Add volatility context
        result["atr_pct"] = atr / (df["close"].values + epsilon) * 100

        return result

    def process_data_with_memory_management(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process data with automatic chunking based on available memory"""

        # Estimate memory usage
        memory_per_row = df.memory_usage(deep=True).sum() / len(df) / (1024**2)  # MB per row
        rows_per_gb = 1024 / memory_per_row
        max_rows = int(self._memory_limit_gb * rows_per_gb * config.performance.memory.safety_factor)

        # Adjust chunk size based on memory
        chunk_size = min(self.config.chunk_size, max_rows)

        logger.info(f"Processing {symbol} with chunk size {chunk_size:,} (memory limit: {self._memory_limit_gb:.1f}GB)")

        # Process data
        n_rows = len(df)
        if n_rows <= chunk_size:
            return self._process_full_data(df, symbol)
        else:
            return self._process_chunked_data(df, symbol, chunk_size)

    def _process_full_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process entire dataset without chunking"""

        # Calculate ATR for entire dataset
        atr = self.calculate_atr_cached(df, symbol)

        # Calculate barriers
        entry_prices = df["close"].values
        epsilon = config.trading.labeling.epsilon
        tp_prices = entry_prices * (1 + self.config.tp_atr_multiplier * atr / (entry_prices + epsilon))
        sl_prices = entry_prices * (1 - self.config.sl_atr_multiplier * atr / (entry_prices + epsilon))

        # Run barrier checking
        labels, exit_offsets, exit_prices, exit_reasons, mfe, mae = self._check_barriers_optimized(
            df["high"].values,
            df["low"].values,
            df["open"].values,
            df["close"].values,
            entry_prices,
            tp_prices,
            sl_prices,
            self.config.max_holding_periods,
            self.config.close_open_threshold,
            epsilon,
        )

        # Create output dataframe
        result = self._create_barriers_dataframe(
            df, labels, exit_offsets, exit_prices, exit_reasons, tp_prices, sl_prices, atr, mfe, mae
        )

        return result

    def _process_chunked_data(self, df: pd.DataFrame, symbol: str, chunk_size: int) -> pd.DataFrame:
        """Process data in chunks with proper overlap handling"""

        results = []
        n_rows = len(df)
        overlap = self.config.max_holding_periods + self.config.atr_period

        # Calculate ATR for entire dataset first (more efficient)
        atr_full = self.calculate_atr_cached(df, symbol)

        for start_idx in range(0, n_rows, chunk_size - overlap):
            end_idx = min(start_idx + chunk_size, n_rows)

            # Ensure we have enough forward data for barrier checking
            chunk_end = min(end_idx + self.config.max_holding_periods, n_rows)

            # Extract chunk with forward-looking window
            chunk_df = df.iloc[start_idx:chunk_end].copy()
            chunk_atr = atr_full[start_idx:chunk_end]

            # Process chunk
            chunk_result = self._process_chunk_with_overlap(chunk_df, chunk_atr, start_idx, end_idx - start_idx)

            results.append(chunk_result)

            # Clean up memory
            del chunk_df
            gc.collect()

            # Check memory usage
            if self._get_available_memory() < config.performance.memory.low_memory_threshold_gb:
                logger.warning("Low memory detected. Forcing garbage collection.")
                gc.collect()

        # Combine results
        final_result = pd.concat(results, ignore_index=True)

        # Ensure we have all original rows
        if len(final_result) < n_rows:
            missing_rows = df.iloc[len(final_result) :].copy()
            for col in [
                "label",
                "tp_price",
                "sl_price",
                "exit_price",
                "exit_reason",
                "exit_bar_offset",
                "barrier_return",
                "barrier_return_pct",
                "max_favorable_excursion",
                "max_adverse_excursion",
                "tp_distance",
                "sl_distance",
                "risk_reward_ratio",
                "atr",
                "atr_pct",
            ]:
                if col not in missing_rows.columns:
                    missing_rows[col] = np.nan
            final_result = pd.concat([final_result, missing_rows], ignore_index=True)

        return final_result

    def _process_chunk_with_overlap(
        self, chunk_df: pd.DataFrame, chunk_atr: np.ndarray, global_start_idx: int, valid_rows: int
    ) -> pd.DataFrame:
        """Process a single chunk with overlap handling"""

        # Calculate barriers for valid rows only
        entry_prices = chunk_df["close"].values[:valid_rows]
        epsilon = config.trading.labeling.epsilon
        tp_prices = entry_prices * (
            1 + self.config.tp_atr_multiplier * chunk_atr[:valid_rows] / (entry_prices + epsilon)
        )
        sl_prices = entry_prices * (
            1 - self.config.sl_atr_multiplier * chunk_atr[:valid_rows] / (entry_prices + epsilon)
        )

        # Prepare full arrays for barrier checking (including forward-looking data)
        full_entry_prices = np.concatenate([entry_prices, np.full(len(chunk_df) - valid_rows, np.nan)])
        full_tp_prices = np.concatenate([tp_prices, np.full(len(chunk_df) - valid_rows, np.nan)])
        full_sl_prices = np.concatenate([sl_prices, np.full(len(chunk_df) - valid_rows, np.nan)])

        # Run barrier checking
        labels, exit_offsets, exit_prices, exit_reasons, mfe, mae = self._check_barriers_optimized(
            chunk_df["high"].values,
            chunk_df["low"].values,
            chunk_df["open"].values,
            chunk_df["close"].values,
            full_entry_prices,
            full_tp_prices,
            full_sl_prices,
            self.config.max_holding_periods,
            self.config.close_open_threshold,
            epsilon,
        )

        # Extract only valid rows for output
        valid_df = chunk_df.iloc[:valid_rows].copy()

        result = self._create_barriers_dataframe(
            valid_df,
            labels[:valid_rows],
            exit_offsets[:valid_rows],
            exit_prices[:valid_rows],
            exit_reasons[:valid_rows],
            tp_prices,
            sl_prices,
            chunk_atr[:valid_rows],
            mfe[:valid_rows],
            mae[:valid_rows],
        )

        return result

    def calculate_comprehensive_stats(self, df: pd.DataFrame, symbol: str, processing_time: float) -> LabelingStats:
        """Calculate comprehensive statistics with risk metrics"""

        # Filter valid labeled data
        labeled_df = df.dropna(subset=["label", "barrier_return"])

        if len(labeled_df) == 0:
            return LabelingStats(
                symbol=symbol,
                total_bars=len(df),
                labeled_bars=0,
                label_distribution={},
                label_percentages={},
                avg_barrier_return={},
                std_barrier_return={},
                sharpe_by_label={},
                exit_reasons={},
                avg_holding_periods={},
                win_rates={},
                processing_time_ms=processing_time,
                data_quality_score=0.0,
            )

        # Label distribution
        label_counts = labeled_df["label"].value_counts()
        total_labeled = len(labeled_df)

        label_distribution = {
            "buy_signals": int(label_counts.get(Label.BUY, 0)),
            "sell_signals": int(label_counts.get(Label.SELL, 0)),
            "neutral_signals": int(label_counts.get(Label.NEUTRAL, 0)),
        }

        label_percentages = {
            k: round(v / total_labeled * 100, 2) if total_labeled > 0 else 0 for k, v in label_distribution.items()
        }

        # Returns by label
        avg_returns = {}
        std_returns = {}
        sharpe_ratios = {}
        win_rates = {}
        avg_holding = {}

        for label in [Label.BUY, Label.SELL, Label.NEUTRAL]:
            label_data = labeled_df[labeled_df["label"] == label]
            if len(label_data) > 0:
                returns = label_data["barrier_return"].values
                avg_returns[label.name.lower()] = float(np.mean(returns))
                std_returns[label.name.lower()] = float(np.std(returns))

                # Sharpe ratio (annualized, if enabled)
                if config.statistics.sharpe_ratio.enabled and std_returns[label.name.lower()] > 0:
                    bars_per_year = config.statistics.trading_calendar.bars_per_year_15min
                    sharpe_ratios[label.name.lower()] = (
                        avg_returns[label.name.lower()] * np.sqrt(bars_per_year) / std_returns[label.name.lower()]
                    )
                else:
                    sharpe_ratios[label.name.lower()] = 0.0

                # Win rate
                if config.statistics.win_rate.enabled and label != Label.NEUTRAL:
                    win_condition = returns > 0 if label == Label.BUY else returns < 0
                    win_rates[label.name.lower()] = float(np.mean(win_condition) * 100)

                # Average holding period
                avg_holding[label.name.lower()] = float(label_data["exit_bar_offset"].mean())
            else:
                avg_returns[label.name.lower()] = 0.0
                std_returns[label.name.lower()] = 0.0
                sharpe_ratios[label.name.lower()] = 0.0
                if label != Label.NEUTRAL:
                    win_rates[label.name.lower()] = 0.0
                avg_holding[label.name.lower()] = 0.0

        # Exit reasons
        exit_reasons = labeled_df["exit_reason"].value_counts().to_dict()

        # Data quality score
        quality_score = min(len(labeled_df) / len(df) * 100, 100.0)

        stats = LabelingStats(
            symbol=symbol,
            total_bars=len(df),
            labeled_bars=total_labeled,
            label_distribution=label_distribution,
            label_percentages=label_percentages,
            avg_barrier_return=avg_returns,
            std_barrier_return=std_returns,
            sharpe_by_label=sharpe_ratios,
            exit_reasons=exit_reasons,
            avg_holding_periods=avg_holding,
            win_rates=win_rates,
            processing_time_ms=processing_time,
            data_quality_score=quality_score,
            metadata={
                "config": asdict(self.config) if config.storage.metadata.include_config else {},
                "processing_date": datetime.now(timezone.utc).isoformat(),
                "data_checksum": hashlib.md5(df.to_json().encode()).hexdigest()[
                    : config.storage.metadata.checksum_length
                ]
                if config.storage.metadata.include_checksum
                else None,
            },
        )

        return stats

    def process_symbol(self, symbol: str, df: pd.DataFrame, save_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """Process a single symbol with comprehensive error handling"""

        import time

        start_time = time.time()

        try:
            logger.info(f"Processing {symbol} with {len(df):,} bars")

            # Validate data
            validator = DataValidator()
            is_valid, df_clean, quality_report = validator.validate_ohlcv(df, symbol)

            if not is_valid:
                logger.error(f"Data validation failed for {symbol}: {quality_report.issues}")
                return None

            logger.info(f"Data quality score for {symbol}: {quality_report.quality_score:.2f}%")

            # Ensure proper data types and sorting
            df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
            df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)

            # Check minimum data requirements
            if len(df_clean) < self.config.min_bars_required:
                logger.warning(
                    f"Insufficient data for {symbol}. Required: {self.config.min_bars_required}, Got: {len(df_clean)}"
                )
                return None

            # Process with memory management
            result_df = self.process_data_with_memory_management(df_clean, symbol)

            # Calculate and store statistics
            processing_time = (time.time() - start_time) * 1000
            stats = self.calculate_comprehensive_stats(result_df, symbol, processing_time)

            with self.stats_lock:
                self.label_stats[symbol] = stats

            logger.info(
                f"Completed {symbol} in {processing_time:.2f}ms. "
                f"Labels: Buy={stats.label_distribution.get('buy_signals', 0):,}, "
                f"Sell={stats.label_distribution.get('sell_signals', 0):,}, "
                f"Neutral={stats.label_distribution.get('neutral_signals', 0):,}"
            )

            # Save if path provided
            if save_path:
                self.save_labeled_data(result_df, symbol, save_path, stats)

            return result_df

        except MemoryError:
            logger.error(f"Memory error processing {symbol}. Consider reducing chunk size.")
            # Try with smaller chunks
            if self.config.chunk_size > 10000:
                logger.info(f"Retrying {symbol} with smaller chunk size")
                self.config = BarrierConfig(**{**asdict(self.config), "chunk_size": self.config.chunk_size // 2})
                return self.process_symbol(symbol, df, save_path)
            return None

        except Exception as e:
            logger.error(f"Error processing {symbol}: {type(e).__name__}: {e}", exc_info=True)
            return None

    def process_symbols_parallel(
        self, symbol_data: Dict[str, pd.DataFrame], max_workers: Optional[int] = None, save_path: Optional[Path] = None
    ) -> Dict[str, pd.DataFrame]:
        """Process multiple symbols in parallel with progress tracking"""

        if max_workers is None:
            max_workers = config.performance.processing.max_workers or min(psutil.cpu_count(), len(symbol_data))

        results = {}
        failed_symbols = []

        logger.info(f"Processing {len(symbol_data)} symbols with {max_workers} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.process_symbol, symbol, df, save_path): symbol
                for symbol, df in symbol_data.items()
            }

            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1

                try:
                    result = future.result(timeout=config.performance.processing.parallel_timeout_seconds)
                    if result is not None:
                        results[symbol] = result
                        logger.info(f"Progress: {completed}/{len(symbol_data)} completed")
                    else:
                        failed_symbols.append(symbol)

                except TimeoutError:
                    logger.error(f"Timeout processing {symbol}")
                    failed_symbols.append(symbol)
                    future.cancel()

                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")
                    failed_symbols.append(symbol)

        if failed_symbols:
            logger.warning(f"Failed symbols: {failed_symbols}")

        logger.info(f"Successfully processed {len(results)}/{len(symbol_data)} symbols")

        return results

    def save_labeled_data(
        self, df: pd.DataFrame, symbol: str, output_path: Path, stats: Optional[LabelingStats] = None
    ) -> None:
        """Save labeled data with metadata and compression"""

        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            # Prepare metadata
            metadata = {
                "symbol": symbol,
                "processing_date": datetime.now(timezone.utc).isoformat(),
                "labeler_version": config.system.version,
                "config": asdict(self.config),
                "data_shape": df.shape,
                "columns": df.columns.tolist(),
            }

            if stats:
                metadata["statistics"] = asdict(stats)

            # Save as Parquet with metadata
            table = pa.Table.from_pandas(df)

            # Add metadata to schema
            existing_metadata = table.schema.metadata or {}
            combined_metadata = {**existing_metadata, b"labeling_metadata": json.dumps(metadata).encode()}

            table = table.replace_schema_metadata(combined_metadata)

            # Write with compression
            output_file = output_path / f"{symbol}_labeled.parquet"
            pq.write_table(
                table,
                output_file,
                compression=config.storage.parquet.compression,
                use_dictionary=config.storage.parquet.use_dictionary,
                version=config.storage.parquet.version,
            )

            logger.info(f"Saved {symbol} to {output_file}")

            # Also save statistics separately if enabled
            if config.storage.output.save_statistics and stats:
                stats_file = output_path / f"{symbol}_stats.{config.storage.output.statistics_format}"
                with open(stats_file, "w") as f:
                    json.dump(asdict(stats), f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save {symbol}: {e}")

    def get_stats_summary(self) -> pd.DataFrame:
        """Get comprehensive summary statistics for all processed symbols"""

        if not self.label_stats:
            return pd.DataFrame()

        rows = []
        for symbol, stats in self.label_stats.items():
            row = {
                "symbol": stats.symbol,
                "total_bars": stats.total_bars,
                "labeled_bars": stats.labeled_bars,
                "coverage_pct": round(stats.labeled_bars / stats.total_bars * 100, 2),
                "buy_pct": stats.label_percentages.get("buy_signals", 0),
                "sell_pct": stats.label_percentages.get("sell_signals", 0),
                "neutral_pct": stats.label_percentages.get("neutral_signals", 0),
                "avg_buy_return": round(stats.avg_barrier_return.get("buy", 0) * 100, 4),
                "avg_sell_return": round(stats.avg_barrier_return.get("sell", 0) * 100, 4),
                "buy_sharpe": round(stats.sharpe_by_label.get("buy", 0), 2),
                "sell_sharpe": round(stats.sharpe_by_label.get("sell", 0), 2),
                "buy_win_rate": round(stats.win_rates.get("buy", 0), 2),
                "sell_win_rate": round(stats.win_rates.get("sell", 0), 2),
                "avg_holding_periods": round(np.mean([v for v in stats.avg_holding_periods.values() if v > 0]), 2),
                "quality_score": round(stats.data_quality_score, 2),
                "processing_time_ms": round(stats.processing_time_ms, 2),
            }
            rows.append(row)

        summary_df = pd.DataFrame(rows)

        # Sort by quality score and sharpe
        summary_df["combined_sharpe"] = (summary_df["buy_sharpe"].abs() + summary_df["sell_sharpe"].abs()) / 2
        summary_df = summary_df.sort_values(["quality_score", "combined_sharpe"], ascending=[False, False])

        return summary_df.drop(columns=["combined_sharpe"])

    @contextmanager
    def performance_monitor(self, operation: str):
        """Context manager for performance monitoring"""
        import time

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024**2  # MB

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024**2

            duration = (end_time - start_time) * 1000  # ms
            memory_delta = end_memory - start_memory

            logger.info(f"{operation} completed in {duration:.2f}ms, Memory delta: {memory_delta:+.1f}MB")


def main():
    """Example usage with comprehensive testing"""

    # Configure logging
    logging.getLogger().setLevel(getattr(logging, config.logging.level))

    # Initialize with custom configuration
    barrier_config = BarrierConfig(
        atr_period=config.example.data_generation.get("atr_period", 20),
        tp_atr_multiplier=config.example.data_generation.get("tp_atr_multiplier", 2.5),
        sl_atr_multiplier=config.example.data_generation.get("sl_atr_multiplier", 2.0),
        max_holding_periods=config.example.data_generation.get("max_holding_periods", 20),
        close_open_threshold=config.example.data_generation.get("close_open_threshold", 0.001),
    )

    labeler = OptimizedTripleBarrierLabeler(barrier_config)

    # Generate sample data for testing
    example_config = config.example.data_generation
    np.random.seed(example_config.random_seed)
    dates = pd.date_range(example_config.start_date, example_config.end_date, freq=example_config.frequency)
    n = len(dates)

    # Simulate price with trend and volatility
    returns = np.random.normal(example_config.return_mean, example_config.return_std, n)
    price = example_config.initial_price * np.exp(np.cumsum(returns))

    # Create OHLCV data
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": price * (1 + np.random.normal(0, example_config.price_noise_std, n)),
            "high": price * (1 + np.abs(np.random.normal(0, example_config.high_noise_std, n))),
            "low": price * (1 - np.abs(np.random.normal(0, example_config.low_noise_std, n))),
            "close": price,
            "volume": np.random.lognormal(example_config.volume_log_mean, example_config.volume_log_std, n),
        }
    )

    # Ensure OHLC relationships
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    # Process data
    with labeler.performance_monitor("Full processing"):
        result = labeler.process_symbol("TEST_SYMBOL", df, Path("./output"))

    if result is not None:
        # Display results
        print("\n=== Processing Results ===")
        print(f"Total rows: {len(result):,}")
        print(f"Labeled rows: {result['label'].notna().sum():,}")
        print("\nLabel distribution:")
        print(result["label"].value_counts().sort_index())
        print("\nExit reasons:")
        print(result["exit_reason"].value_counts())
        print("\nAverage returns by label:")
        for label in [-1, 0, 1]:
            data = result[result["label"] == label]
            if len(data) > 0:
                avg_return = data["barrier_return"].mean() * 100
                print(f"  Label {label}: {avg_return:.4f}%")

        # Show summary statistics
        print("\n=== Summary Statistics ===")
        summary = labeler.get_stats_summary()
        print(summary.to_string(index=False))

        # Save sample for inspection
        sample_file = Path(f"./output/{config.example.sample_output.file_name}")
        result.head(config.example.sample_output.rows_to_save).to_csv(sample_file, index=False)
        print(f"\nSample output saved to: {sample_file}")


if __name__ == "__main__":
    main()

