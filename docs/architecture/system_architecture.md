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