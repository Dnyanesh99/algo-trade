# Automated Intraday Trading Signal Generation System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

A **production-grade, local trading signal system** designed to generate intraday trading signals for Indian indices (NIFTY 50, BANK NIFTY, etc.) on a 15-minute timeframe. Built by a Senior Quantitative Developer with 20+ years of experience, this system emphasizes **clean, modular, memory-efficient, strictly typed code** that adheres to the highest industry standards.

### ⚠️ Important Notice

**This system is for signal generation only and does NOT execute trades or manage portfolios.** It is designed to boost trading confidence through data-driven insights.

## 🏗️ Architecture

The system follows a **database-centric architecture** with TimescaleDB as the single source of truth, implementing a robust modular design:

### Core Components

- **Configuration Layer**: Centralized YAML-based configuration with Pydantic validation
- **Authentication Module**: Zerodha KiteConnect API integration
- **Broker Module**: REST client, WebSocket client, connection management, and backfill handling
- **Data Pipeline**: Separate historical and live data processing workflows
- **Core Processing**: Data aggregation, validation, feature engineering, and labeling
- **Database Layer**: TimescaleDB with hypertables, compression, and optimized queries
- **State Management**: System health monitoring, error handling, and position tracking
- **Model Module**: LightGBM training with walk-forward validation
- **Signal Generation**: Confidence-based signal output with alert system

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- TimescaleDB (PostgreSQL with TimescaleDB extension)
- TA-Lib library
- Zerodha KiteConnect API credentials

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd algo-trade
   ```

2. **Install dependencies**
   ```bash
   make install-dev
   ```

3. **Configure the system**
   ```bash
   cp config/config.yaml.example config/config.yaml
   # Edit config/config.yaml with your settings
   ```

4. **Set up database**
   ```bash
   make db-init
   make db-migrate
   ```

5. **Run the system**
   ```bash
   # Historical mode (for training)
   make run-historical
   
   # Live mode (for signal generation)
   make run-live
   ```

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

```yaml
system:
  version: "2.0.0"
  mode: "HISTORICAL_MODE"  # or "LIVE_MODE"
  timezone: "Asia/Kolkata"

broker:
  api_key: "YOUR_API_KEY"
  api_secret: "YOUR_API_SECRET"
  redirect_url: "http://localhost:3000"

database:
  host: "localhost"
  port: 5432
  dbname: "timescaledb"
  user: "user"
  password: "password"
```

### Rate Limits (`config/rate_limit_config.yaml`)

```yaml
api_rate_limits:
  historical_data:
    limit: 2  # requests per second
    interval: 1
  websocket_subscriptions:
    limit: 3000  # instruments
    interval: 1
```

## 🛠️ Development

### Code Quality

This project maintains the highest code quality standards:

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Security checks
make security

# Run all quality checks
make quality
```

### Testing

```bash
# Run all tests
make test

# Run fast tests only
make test-fast

# Run integration tests
make test-integration
```

### Database Operations

```bash
# Initialize database schema
make db-init

# Run migrations
make db-migrate

# View logs
make logs
```

## 📊 System Features

### ✅ Production-Ready Features

- **Robust Error Handling**: Circuit breaker patterns and graceful degradation
- **Memory Management**: Intelligent chunking and garbage collection
- **Rate Limiting**: Respect broker API limits with token bucket algorithm
- **Data Quality**: Comprehensive validation and outlier detection
- **Performance Monitoring**: Built-in metrics and profiling
- **Logging**: Structured logging with Loguru
- **Type Safety**: Strict typing with mypy validation
- **Configuration Validation**: Pydantic-based configuration management

### 📈 Trading Features

- **Multi-timeframe Analysis**: 5min, 15min, and 60min data processing
- **Technical Indicators**: 20+ indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- **Machine Learning**: LightGBM with walk-forward validation
- **Triple Barrier Labeling**: Advanced labeling methodology
- **Signal Generation**: Confidence-based BUY/SELL/HOLD signals
- **Market Calendar**: NSE holiday and special session handling

### 🗄️ Database Features

- **TimescaleDB Integration**: Optimized for time-series data
- **Hypertables**: Automatic partitioning and compression
- **Continuous Aggregates**: Pre-calculated views for performance
- **Data Retention**: Configurable retention policies
- **Connection Pooling**: Efficient database connections

## 📁 Project Structure

```
algo-trade/
├── src/
│   ├── auth/              # Authentication (KiteConnect)
│   ├── broker/            # Broker integration (REST, WebSocket)
│   ├── core/              # Core processing (aggregation, features)
│   ├── database/          # Database layer (repos, migrations)
│   ├── historical/        # Historical data pipeline
│   ├── live/              # Live data pipeline
│   ├── model/             # Machine learning models
│   ├── signal/            # Signal generation and alerts
│   ├── state/             # State management
│   └── utils/             # Utilities (config, logging, etc.)
├── config/                # Configuration files
├── tests/                 # Test suite
├── docs/                  # Documentation
└── logs/                  # Application logs
```

## 🔧 Architecture Principles

### Non-Negotiable Engineering Directives

1. **No hardcoded configurations** - All values externalized to YAML
2. **Strict architectural adherence** - Follow the provided Mermaid diagrams
3. **Production-critical quality** - Not a toy or prototype
4. **Type safety** - Statically and strictly typed
5. **Modular design** - Separation of concerns
6. **Modern Python practices** - PEP 518 standards
7. **High-performance libraries** - Pydantic, httpx, pandas, numpy, loguru
8. **Broker-native functionality** - Leverage KiteConnect SDK features

## 📊 Monitoring and Metrics

The system includes comprehensive monitoring:

- **System Health**: Database connectivity, WebSocket status, data freshness
- **Performance Metrics**: Latency tracking, throughput, success rates
- **Model Performance**: Accuracy, Sharpe ratio, win rate, maximum drawdown
- **Signal Statistics**: Daily summaries, confidence distributions

## 🔒 Security

- **API Key Management**: Secure credential handling
- **Input Validation**: Comprehensive data validation
- **Error Sanitization**: Prevent information leakage
- **Security Scanning**: Bandit integration for vulnerability detection

## 🚀 Deployment

### Prerequisites for Production

1. **Infrastructure**: 
   - TimescaleDB cluster
   - Python 3.9+ environment
   - Sufficient memory (8GB+ recommended)
   - Stable internet connection

2. **Monitoring**:
   - Log aggregation system
   - Performance monitoring
   - Alert notifications

3. **Backup Strategy**:
   - Database backups
   - Model artifact versioning
   - Configuration management

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the coding standards
4. Run quality checks (`make quality`)
5. Run tests (`make test`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended to provide investment advice. Trading in financial markets involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## 🤝 Support

For support, please create an issue in the GitHub repository or contact the development team.

---

**Built with ❤️ by Senior Quantitative Developers for the Trading Community**# algo-trade
