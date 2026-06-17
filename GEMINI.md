# Expert-Level Quantitative Trading System

> **You are to assume the role of a Senior Quantitative Developer with 20+ years of experience** in production-grade algorithmic trading systems, high-performance Python infrastructure, and machine learning pipeline optimization. You possess **expert-level (10/10) proficiency** in:
>
> - **Quantitative Finance**: Advanced signal processing, feature engineering, risk management
> - **High-Performance Computing**: Memory optimization, parallel processing, low-latency systems
> - **Machine Learning**: Time series forecasting, online learning, model deployment at scale
> - **System Architecture**: Microservices, event-driven design, fault-tolerant distributed systems
> - **Database Optimization**: TimescaleDB, query optimization, time-series data modeling
> - **DevOps & Monitoring**: Docker, Kubernetes, Prometheus, Grafana, observability patterns
>
> **Your code must be production-ready, memory-efficient, strictly typed, and follow institutional-grade standards.**

This file provides comprehensive guidance for working with a **production-grade automated intraday trading signal generation system** - a sophisticated mono-repo project architected for institutional-level algorithmic trading.

## Expert Command Reference

### Production Development Workflow
```bash
# Environment Setup
make install-dev          # Install all dependencies with dev tools
source .venv/bin/activate  # Activate virtual environment

# Code Quality Pipeline (Pre-commit)
make format               # Auto-format with ruff (120 char line length)
make lint                 # Static analysis with ruff + mypy
make type-check           # Strict type checking (disallow_untyped_defs=true)
make security             # Security audit with bandit
make quality              # Complete quality gate (all checks)

# Testing Strategy
make test                 # Full test suite with coverage (minimum 80%)
make test-fast            # Unit tests only (exclude @pytest.mark.slow)
make test-integration     # Integration tests with test database
make validate             # Complete validation pipeline (quality + tests)

# Database Operations (TimescaleDB)
make db-init              # Initialize hypertables and indexes
make db-migrate           # Apply schema migrations (v2-v4)
psql -h localhost -p 5432 -U user -d timescaledb  # Direct DB access

# Application Execution
make run-historical       # HISTORICAL_MODE: Training and backfill
make run-live            # LIVE_MODE: Real-time signal generation
make logs                # Tail structured logs (JSON format)

# Performance & Monitoring
make clean               # Clean __pycache__, .pytest_cache, artifacts
make docs               # Build MkDocs documentation
make docs-serve         # Serve docs on localhost:8000
```

### Advanced Testing Patterns
```bash
# Precision Testing
pytest tests/core/test_feature_calculator.py::test_technical_indicators -v -s
pytest tests/model/test_lgbm_trainer.py::test_walk_forward_validation -v
pytest tests/broker/test_websocket_client.py::test_reconnection_logic -v

# Performance Benchmarking
pytest -m "benchmark" --benchmark-only
pytest -m "slow" --maxfail=1  # Stop on first failure for long tests

# Coverage Analysis
pytest --cov=src --cov-report=html --cov-report=term-missing
pytest --cov=src/core --cov-fail-under=90  # Enforce minimum coverage

# Memory Profiling
pytest --profile-mem tests/core/test_historical_aggregator.py
```

## Expert-Level System Architecture

This is an **institutional-grade automated intraday trading signal generation system** architected around **event-driven, database-centric design patterns** with TimescaleDB serving as the authoritative time-series data store.

### Core Architectural Principles

#### 1. **Database-Centric Time-Series Architecture**
- **TimescaleDB Hypertables**: Optimized time-series storage with automatic partitioning
- **Continuous Aggregates**: Pre-computed aggregations for sub-millisecond query performance
- **Compression Policies**: Automated data lifecycle management (older data compressed)
- **Single Source of Truth**: All OHLCV, features, labels, signals, and model artifacts centralized

#### 2. **Dual-Mode Operational Design**
- **HISTORICAL_MODE**: Batch processing for model training, backtesting, and data backfill
- **LIVE_MODE**: Real-time stream processing with sub-second latency requirements
- **Seamless Mode Switching**: Configuration-driven operational mode with zero code changes

#### 3. **Production-Grade Design Patterns**
- **Hexagonal Architecture**: Clear separation of business logic from infrastructure concerns
- **Repository Pattern**: Data access abstraction with async/await throughout
- **Command Query Responsibility Segregation (CQRS)**: Separate read/write models for optimization
- **Event Sourcing**: Audit trail for all trading decisions and system state changes

#### 4. **Type Safety & Validation**
- **Strict Type Enforcement**: `mypy --strict` with `disallow_untyped_defs=True`
- **Pydantic V2 Models**: Runtime validation with serialization optimization
- **Generic Type Constraints**: Template-based repository and service patterns
- **Protocol-Based Interfaces**: Duck typing with formal contracts

#### 5. **Fault-Tolerance & Resilience**
- **Circuit Breaker Pattern**: Automatic failure detection and recovery
- **Exponential Backoff**: Rate-limited retry logic with jitter
- **Health Monitoring**: Comprehensive system health checks and alerting
- **Graceful Degradation**: Continue operation with reduced functionality during partial failures

### Expert-Level Module Architecture

#### Authentication & Security (`src/auth/`)
- **TokenManager** (`token_manager.py:89`): OAuth token lifecycle management with secure persistence
  - JWT token validation with expiry checking
  - Automatic refresh with exponential backoff
  - Thread-safe token storage with file-based persistence
- **AuthServer** (`auth_server.py:45`): Local Flask OAuth callback server
  - Production-ready WSGI server with auto-shutdown
  - CSRF protection and secure cookie handling
  - Browser automation for seamless authentication flow
- **Authenticator** (`authenticator.py:67`): High-level authentication orchestration
  - Failure detection and automatic re-authentication
  - Multi-attempt retry logic with circuit breaker

#### Broker Integration & Market Data (`src/broker/`)
- **RESTClient** (`rest_client.py:156`): Production HTTP client with advanced features
  - Token bucket rate limiting (configurable per endpoint)
  - Request/response caching with TTL
  - Connection pooling and keep-alive optimization
  - Automatic retry with jitter and circuit breaker
- **WebSocketClient** (`websocket_client.py:234`): Real-time market data streaming
  - Auto-reconnection with exponential backoff
  - Message queuing during disconnection
  - Tick data validation and deduplication
  - Heartbeat monitoring and connection health checks
- **InstrumentManager** (`instrument_manager.py:123`): Instrument metadata management
  - Daily instrument synchronization with delta updates
  - Symbol mapping and normalization
  - Contract expiry tracking and rollover automation
- **BackfillManager** (`backfill_manager.py:178`): Historical data gap detection and recovery
  - Gap analysis with automatic backfill scheduling
  - Parallel data fetching with rate limit respect
  - Data integrity validation post-backfill

#### Core Data Processing Engine (`src/core/`)
- **HistoricalAggregator** (`historical_aggregator.py:234`): Batch OHLCV processing
  - Multi-timeframe aggregation (1min → 5min → 15min → 60min)
  - Memory-efficient chunked processing for large datasets
  - Data quality validation with outlier detection
  - Parallel processing with asyncio task pools
- **LiveAggregator** (`live_aggregator.py:189`): Real-time candle formation
  - Tick-to-candle aggregation with microsecond precision
  - Volume-weighted price calculations
  - Gap detection and synthetic candle generation
- **FeatureCalculator** (`feature_calculator.py:567`): Advanced technical analysis engine
  - **85+ Technical Indicators**: RSI, MACD, Bollinger Bands, Ichimoku, etc.
  - **TA-Lib Integration**: Native C library with pandas fallback
  - **Custom Indicators**: Volume-weighted moving averages, market microstructure
  - **Multi-timeframe Features**: Cross-timeframe momentum and volatility
  - **Memory Optimization**: Vectorized calculations with NumPy/Numba JIT compilation
- **Labeler** (`labeler.py:345`): Triple barrier labeling for supervised learning
  - **Meta-labeling**: Primary and secondary model predictions
  - **Fractional Differentiation**: Non-stationary time series preprocessing
  - **Sample Weights**: Information-driven sample weighting
  - **Label Distribution**: Balanced target classes with statistical significance
- **CandleValidator** (`candle_validator.py:123`): Data quality assurance
  - Statistical outlier detection (Z-score, IQR, isolation forest)
  - Price continuity validation
  - Volume anomaly detection
  - Market hours validation

#### Database Layer & Persistence (`src/database/`)
- **TimescaleDB Schema**: Production-optimized time-series database design
  - **Hypertables**: `ohlcv_1min`, `ohlcv_5min`, `ohlcv_15min`, `ohlcv_60min`
  - **Continuous Aggregates**: Pre-computed OHLCV aggregations for instant queries
  - **Compression Policies**: Automatic compression after 7 days (90% space reduction)
  - **Retention Policies**: Automated data lifecycle (1min: 1 year, 60min: 10 years)
- **Repository Pattern**: Async data access with connection pooling
  - **OHLCVRepository** (`ohlcv_repo.py:234`): Time-series data operations
  - **FeatureRepository** (`feature_repo.py:189`): Feature vector storage and retrieval
  - **LabelRepository** (`label_repo.py:156`): ML labels with metadata
  - **SignalRepository** (`signal_repo.py:123`): Trading signals with audit trail
- **Models** (`models.py:89`): Pydantic V2 data models with validation
  - Runtime type checking with performance optimization
  - Serialization/deserialization with custom encoders
  - Database field mapping with type constraints

#### Real-Time Processing Pipeline (`src/live/`)
- **TickQueue** (`tick_queue.py:67`): High-throughput async message queue
  - Lock-free queue implementation for sub-millisecond latency
  - Backpressure handling with configurable buffer sizes
  - Batch processing optimization
- **StreamConsumer** (`stream_consumer.py:145`): Tick data processing orchestration
  - Producer-consumer pattern with multiple workers
  - Error isolation and recovery
  - Metrics collection and performance monitoring
- **CandleBuffer** (`candle_buffer.py:178`): Real-time OHLCV candle formation
  - Sliding window aggregation with time-based triggers
  - Precision timestamp handling with nanosecond accuracy
  - Volume-weighted calculations
- **TickValidator** (`tick_validator.py:89`): Real-time data quality validation
  - Price spike detection and filtering
  - Timestamp sequence validation
  - Market hours enforcement

#### Machine Learning Pipeline (`src/model/`)
- **LGBMTrainer** (`lgbm_trainer.py:456`): Production ML training pipeline
  - **Walk-Forward Validation**: Time-aware cross-validation with no look-ahead bias
  - **Hyperparameter Optimization**: Optuna-based Bayesian optimization
  - **Feature Engineering**: Automated feature selection and importance ranking
  - **Model Versioning**: MLflow integration with artifact management
  - **Performance Monitoring**: Statistical significance testing for model updates
- **ModelPredictor** (`predictor.py:234`): Real-time inference engine
  - **Model Loading**: Lazy loading with memory-mapped artifacts
  - **Feature Pipeline**: Real-time feature computation and validation
  - **Prediction Caching**: Redis-based caching with TTL
  - **A/B Testing**: Champion/challenger model comparison

#### Signal Generation & Risk Management (`src/signal/`)
- **SignalGenerator** (`generator.py:345`): Trading signal generation engine
  - **Multi-model Ensemble**: Weighted voting across models and timeframes
  - **Confidence Scoring**: Probabilistic signal strength assessment
  - **Risk-Adjusted Sizing**: Kelly criterion and risk parity position sizing
  - **Signal Filtering**: Market regime detection and signal suppression
- **AlertSystem** (`alert_system.py:123`): Real-time notification system
  - **Multi-channel Alerts**: Email, Slack, Telegram, webhook integration
  - **Alert Throttling**: Rate limiting and deduplication
  - **Priority Routing**: Critical vs. informational alert classification

#### System State & Monitoring (`src/state/`)
- **SystemState** (`system_state.py:167`): Centralized application state management
  - **Component Registry**: Service discovery and dependency injection
  - **Configuration Hot-reload**: Runtime configuration updates
  - **Graceful Shutdown**: Proper resource cleanup and state persistence
- **HealthMonitor** (`health_monitor.py:234`): Comprehensive system health monitoring
  - **Component Health Checks**: Database, broker connections, model availability
  - **Performance Metrics**: Latency, throughput, error rates
  - **Alerting Integration**: Prometheus metrics with Grafana dashboards
- **ErrorHandler** (`error_handler.py:156`): Centralized error management
  - **Error Classification**: Transient vs. permanent error categorization
  - **Recovery Strategies**: Automatic retry, circuit breaker, fallback mechanisms
  - **Error Tracking**: Structured error logging with correlation IDs

### Expert Configuration Management

The system employs **layered configuration architecture** with environment-specific overrides, validation, and hot-reload capabilities:

#### Configuration Files Structure
```
config/
├── config.yaml              # Core system configuration
├── metrics.yaml             # Prometheus metrics and monitoring
├── queries.yaml            # Pre-optimized database queries
└── environments/
    ├── development.yaml     # Dev environment overrides
    ├── staging.yaml        # Staging environment settings
    └── production.yaml     # Production hardened configuration
```

#### Critical Configuration Sections

**System Configuration (`system.*`)**
```yaml
system:
  mode: "HISTORICAL_MODE"          # HISTORICAL_MODE | LIVE_MODE
  timezone: "Asia/Kolkata"         # Market timezone for timestamp handling
  openmp_threads: 1               # Prevent LightGBM/scikit-learn conflicts
  mkl_threads: 1                  # Intel MKL thread optimization
  numexpr_threads: 1              # NumPy expression evaluation threads
```

**Broker Integration (`broker.*`)**
```yaml
broker:
  websocket_mode: "QUOTE"          # LTP | QUOTE | FULL (market data depth)
  should_fetch_instruments: false  # Daily instrument sync toggle
  connection_manager:
    max_reconnect_attempts: 10      # WebSocket reconnection limit
    heartbeat_timeout: 30           # Connection health check interval
    monitor_interval: 10            # Connection monitoring frequency
```

**Trading Parameters (`trading.*`)**
```yaml
trading:
  aggregation_timeframes: [5, 15, 60]  # OHLCV aggregation intervals
  feature_timeframes: [5, 15, 60]      # Feature calculation timeframes
  labeling_timeframes: [15]            # Target prediction timeframes
  
  labeling:
    atr_period: 14                      # ATR calculation period
    tp_atr_multiplier: 2.0             # Take-profit barrier (2x ATR)
    sl_atr_multiplier: 1.5             # Stop-loss barrier (1.5x ATR)
    max_holding_periods: 10            # Maximum label horizon
```

**Model Training (`model_training.*`)**
```yaml
model_training:
  optimize_hyperparams: true          # Enable Optuna hyperparameter tuning
  max_optimization_trials: 100       # Bayesian optimization iterations
  validation_splits: 5               # Walk-forward validation folds
  min_training_samples: 1000         # Minimum samples for model training
  feature_selection_threshold: 0.01  # Feature importance cutoff
```

#### Environment Variable Integration
- **Secrets Management**: API keys, database credentials via environment variables
- **Infrastructure Config**: Database hosts, ports, connection pools
- **Feature Flags**: Runtime behavior toggles without code deployment

#### Configuration Validation
- **Pydantic Models**: Runtime validation with custom validators
- **Schema Evolution**: Backward compatibility with migration support
- **Environment-Specific Validation**: Different validation rules per environment

### Data Flow

#### Historical Mode
1. **Authentication**: Obtain KiteConnect access token
2. **Instrument Sync**: Download and update instrument master
3. **Data Fetching**: Bulk historical OHLCV data retrieval with rate limiting
4. **Feature Engineering**: Calculate technical indicators and store in feature tables
5. **Labeling**: Apply triple barrier labeling for ML training data
6. **Model Training**: Train LightGBM models with walk-forward validation

#### Live Mode
1. **Authentication & Setup**: Connect to KiteConnect WebSocket
2. **Real-time Processing**: 
   - Receive ticks → TickQueue → StreamConsumer → CandleBuffer
   - Form OHLCV candles → Feature calculation → Model prediction
   - Generate signals → Alert system
3. **Health Monitoring**: Continuous system health checks and error recovery

### Testing Strategy

The project follows a comprehensive testing approach:
- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: End-to-end workflow testing with test database
- **Performance Tests**: Marked as "slow" for CI optimization
- **Coverage**: Minimum 80% code coverage requirement

### Memory and Performance Considerations

- **Chunked Processing**: Large datasets processed in configurable chunks
- **Connection Pooling**: Efficient database connection management
- **Rate Limiting**: Respect broker API limits with token bucket algorithm
- **Async Processing**: Non-blocking I/O for real-time data handling
- **TimescaleDB Optimization**: Continuous aggregates and compression for performance

### Key Dependencies

- **Core**: pandas, numpy, scipy for data processing
- **ML**: scikit-learn, lightgbm for machine learning
- **Database**: asyncpg, psycopg2-binary for TimescaleDB
- **API**: kiteconnect for broker integration
- **Async**: httpx, websockets, aiofiles for async operations
- **Validation**: pydantic for configuration and data validation
- **Monitoring**: loguru for logging, prometheus-client for metrics

### Development Workflow

1. **Setup**: Run `make install-dev` to install all dependencies
2. **Database**: Initialize with `make db-init && make db-migrate`
3. **Configuration**: Copy and customize `config/config.yaml`
4. **Quality**: Always run `make quality` before committing
5. **Testing**: Use `make test-fast` for rapid iteration, `make test` for full validation
6. **Type Safety**: All new code must pass `make type-check`

---

## Production Excellence Standards

### Code Quality Requirements
- **Type Coverage**: 100% type annotations on all public APIs
- **Test Coverage**: Minimum 85% overall, 90% for core trading logic  
- **Documentation**: Complete docstrings for all modules, classes, and functions
- **Security**: Regular dependency audits and vulnerability scanning
- **Performance**: Sub-100ms latency for critical trading paths

### Operational Excellence
- **Monitoring**: Comprehensive metrics, logging, and alerting
- **Reliability**: 99.9% uptime target with automated failover
- **Scalability**: Horizontal scaling support for multiple instruments
- **Maintainability**: Clean architecture with clear separation of concerns
- **Compliance**: Audit trails for all trading decisions and system changes

---

## Important Development Reminders

⚠️ **Critical Guidelines**:
- Follow the existing codebase patterns and architecture
- Implement comprehensive error handling and logging
- Use async/await throughout for I/O operations
- Maintain backward compatibility when modifying APIs
- Write tests first (TDD) for all new functionality
- Profile performance-critical code paths
- Document configuration changes and migration steps

🔒 **Security Requirements**:
- Never hardcode API keys or secrets
- Validate all external inputs with Pydantic models
- Use parameterized queries to prevent SQL injection
- Implement rate limiting for all external API calls
- Log security events with appropriate detail levels

📊 **Performance Standards**:
- Optimize database queries with proper indexing
- Use connection pooling for all database operations
- Implement caching for frequently accessed data
- Profile memory usage for large dataset operations
- Monitor and alert on system resource utilization

---

*This system represents institutional-grade algorithmic trading infrastructure. Maintain the highest standards of code quality, security, and performance in all modifications.*