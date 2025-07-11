# CLAUDE.md

You are to assume the role of a **Senior Quantitative Developer** with over **20 years of experience** in high-performance, production-grade Python systems. You possess **expert-level (10/10) proficiency in Python and Machine Learning**, with a deep understanding of scalable software design, low-latency systems, and model integration. Your code is **clean, modular, memory-efficient, strictly typed, and adheres to the highest industry standards**.

You are now tasked with architecting and developing a **production-grade local trading signal system**. This is a **mono-repo project**

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Commands
```bash
# Install dependencies
make install-dev

# Code quality
make format        # Format code with ruff
make lint         # Run linting checks
make type-check   # Run mypy type checking
make security     # Run bandit security checks
make quality      # Run all quality checks (lint + type-check + security)

# Testing
make test         # Run all tests with coverage
make test-fast    # Run only fast tests (exclude slow markers)
make test-integration  # Run integration tests only
make validate     # Full validation (quality + tests)

# Database operations
make db-init      # Initialize TimescaleDB schema
make db-migrate   # Run database migrations

# Application
make run-historical  # Run in historical mode (training/backfill)
make run-live       # Run in live trading mode
make logs          # View application logs

# Development utilities
make clean        # Clean build artifacts and cache
make docs         # Build documentation
make docs-serve   # Serve docs locally
```

### Single Test Execution
```bash
# Run specific test file
pytest tests/auth/test_authenticator.py -v

# Run specific test function
pytest tests/broker/test_rest_client.py::test_fetch_historical_data -v

# Run tests with specific marker
pytest -m "not slow" -v
pytest -m "integration" -v
```

## Architecture Overview

This is a production-grade automated intraday trading signal generation system built around a **database-centric architecture** with TimescaleDB as the single source of truth.

### Core Design Principles
- **Database-Centric**: TimescaleDB stores all OHLCV data, features, labels, and signals
- **Dual-Mode Operation**: HISTORICAL_MODE for training/backfill, LIVE_MODE for real-time signals
- **Modular Architecture**: Clean separation between data ingestion, processing, ML, and signal generation
- **Type Safety**: Strict typing with Pydantic models and mypy validation
- **Error Resilience**: Circuit breaker patterns, graceful degradation, comprehensive error handling

### Module Structure

#### Authentication (`src/auth/`)
- **TokenManager**: Manages Zerodha KiteConnect access tokens with persistence
- **AuthServer**: Local Flask server for OAuth authentication flow
- **Authenticator**: Handles login process and token refresh

#### Broker Integration (`src/broker/`)
- **RESTClient**: Rate-limited HTTP client for historical data and API calls
- **WebSocketClient**: Real-time tick data streaming with auto-reconnection
- **InstrumentManager**: Manages instrument metadata and synchronization
- **ConnectionManager**: Handles connection lifecycle and health monitoring
- **BackfillManager**: Manages historical data gaps and recovery

#### Data Processing (`src/core/`)
- **HistoricalAggregator**: Processes historical OHLCV data from REST API
- **LiveAggregator**: Real-time candle formation from tick data
- **FeatureCalculator**: Technical indicator computation (RSI, MACD, Bollinger, etc.)
- **Labeler**: Triple barrier labeling methodology for ML training
- **CandleValidator**: Data quality validation and outlier detection

#### Database Layer (`src/database/`)
- **TimescaleDB Schema**: Hypertables for OHLCV data (1min, 5min, 15min, 60min)
- **Repositories**: Data access layer for instruments, OHLCV, features, labels, signals
- **Models**: Pydantic models for data validation
- **Migrations**: Schema evolution scripts

#### Live Processing (`src/live/`)
- **TickQueue**: Async queue for incoming market ticks
- **StreamConsumer**: Processes tick queue and validates data
- **CandleBuffer**: Aggregates ticks into OHLCV candles
- **TickValidator**: Validates tick data quality and filters anomalies

#### Machine Learning (`src/model/`)
- **LGBMTrainer**: LightGBM model training with walk-forward validation
- **ModelPredictor**: Real-time prediction serving with model artifacts
- **Artifacts**: Model storage and versioning

#### Signal Generation (`src/signal/`)
- **SignalGenerator**: Converts predictions to BUY/SELL/HOLD signals with confidence scores
- **AlertSystem**: Notification system for generated signals

#### State Management (`src/state/`)
- **SystemState**: Central state management for application components
- **HealthMonitor**: System health checks and monitoring
- **ErrorHandler**: Centralized error handling and recovery strategies

### Configuration Management

The system uses YAML-based configuration with Pydantic validation:

- **config/config.yaml**: Main system configuration
- **config/rate_limit_config.yaml**: API rate limiting rules
- **config/metrics.yaml**: Monitoring and metrics configuration

Key configuration sections:
- `system.mode`: Controls HISTORICAL_MODE vs LIVE_MODE operation
- `broker`: Zerodha KiteConnect API credentials and settings
- `database`: TimescaleDB connection parameters
- `trading`: Instrument selection and trading parameters
- `model`: ML model hyperparameters and training settings

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

### Deployment Considerations

- **Environment**: Python 3.9+ with TimescaleDB cluster
- **Authentication**: Secure credential management for production
- **Monitoring**: Health checks, metrics collection, and alerting
- **Backup**: Database backups and model artifact versioning
- **Scaling**: Horizontal scaling considerations for multiple instruments