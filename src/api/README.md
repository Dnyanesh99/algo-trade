# Algo Trading Control Panel API

This directory contains the FastAPI backend that integrates with the existing Python trading application to provide a comprehensive control panel interface.

## Architecture Overview

The API backend serves as a bridge between the React UI and the core trading system, providing:

- **RESTful API endpoints** for system control and data access
- **WebSocket connections** for real-time updates
- **Configuration management** for all system parameters
- **Integration layer** that connects with existing trading components
- **Authentication and authorization** for secure access

## Key Components

### 1. FastAPI Application (`main.py`)
The core FastAPI application with:
- Application lifecycle management
- CORS configuration
- WebSocket endpoint for real-time updates
- Static file serving for React UI
- Error handling and logging

### 2. API Routes (`routes/`)
- **System Routes** (`routes/system.py`): System control, monitoring, health checks
- **Configuration Routes** (`routes/configuration.py`): Configuration management, import/export
- **Additional routes** for instruments, signals, models, etc.

### 3. Integration Layer (`integrations.py`)
Connects the API with the existing trading system:
- System state management
- Component lifecycle control
- Background task coordination
- Event broadcasting

### 4. Dependencies (`dependencies.py`)
FastAPI dependency injection for:
- Authentication and authorization
- Database connections
- Validation helpers
- Rate limiting
- Error handling

### 5. Data Models (`models.py`)
Pydantic models for:
- Request/response validation
- API documentation
- Type safety
- Data serialization

## API Endpoints

### System Control
- `GET /api/system/status` - Get system status
- `POST /api/system/mode` - Change system mode
- `POST /api/system/start` - Start trading system
- `POST /api/system/stop` - Stop trading system
- `POST /api/system/restart` - Restart trading system

### Configuration Management
- `GET /api/config` - Get complete configuration
- `GET /api/config/{section}` - Get specific configuration section
- `PUT /api/config/{section}` - Update configuration section
- `POST /api/config/export` - Export configuration
- `POST /api/config/import` - Import configuration

### Instruments
- `GET /api/instruments` - Get all instruments
- `POST /api/instruments` - Create new instrument
- `PUT /api/instruments/{id}` - Update instrument
- `DELETE /api/instruments/{id}` - Delete instrument

### Signals
- `GET /api/signals` - Get recent signals
- `GET /api/signals/{id}` - Get specific signal
- `POST /api/signals/filter` - Filter signals

### Models
- `GET /api/models` - Get model metrics
- `POST /api/models/train` - Start model training
- `POST /api/models/deploy` - Deploy model

### Authentication
- `GET /api/auth/status` - Get authentication status
- `POST /api/auth/login` - Initiate login flow

### Health & Monitoring
- `GET /api/health` - Health check
- `GET /api/health/components` - Component health
- `GET /api/metrics` - System metrics

## WebSocket Endpoints

### Real-time Updates (`/ws`)
- System status changes
- Configuration updates
- Signal generation
- Model training progress
- Health alerts

## Integration with Trading System

The API integrates with the existing trading system through:

### 1. Configuration Integration
- Loads configuration from `config/config.yaml`
- Validates configuration changes
- Persists configuration updates
- Broadcasts changes to all components

### 2. Database Integration
- Uses existing repository pattern
- Shares database connections
- Maintains data consistency
- Provides efficient queries

### 3. Component Integration
- Manages component lifecycle
- Coordinates background tasks
- Handles inter-component communication
- Provides unified system state

### 4. Authentication Integration
- Uses existing TokenManager
- Handles OAuth flow
- Maintains session state
- Provides secure access

## Development Setup

### 1. Install Dependencies
```bash
pip install fastapi uvicorn websockets
```

### 2. Environment Setup
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=trading_db
export DB_USER=trading_user
export DB_PASSWORD=trading_password
```

### 3. Start API Server
```bash
python start_api.py
```

### 4. Access Control Panel
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws

## Configuration

The API uses the same configuration system as the main application:
- YAML-based configuration
- Environment variable overrides
- Pydantic validation
- Hot reloading support

## Security

### Authentication
- Token-based authentication
- Role-based access control
- Session management
- Secure credential storage

### API Security
- CORS protection
- Request validation
- Rate limiting
- Error handling

## Monitoring

### Health Checks
- Component health monitoring
- Database connectivity
- Broker connection status
- System resource usage

### Metrics
- Prometheus metrics integration
- Custom application metrics
- Performance monitoring
- Error tracking

## Error Handling

### Exception Handling
- Graceful error responses
- Detailed error logging
- User-friendly error messages
- Recovery mechanisms

### Validation
- Request/response validation
- Configuration validation
- Data integrity checks
- Business logic validation

## Testing

### Unit Tests
- API endpoint tests
- Integration tests
- Mock external dependencies
- Test data fixtures

### Integration Tests
- End-to-end API tests
- Database integration tests
- WebSocket communication tests
- System integration tests

## Deployment

### Production Setup
```bash
# Using Gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker

# Using Docker
docker build -t algo-trading-api .
docker run -p 8000:8000 algo-trading-api
```

### Environment Variables
```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=trading_user
DB_PASSWORD=trading_password

# Broker API
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret

# Application
LOG_LEVEL=INFO
ENVIRONMENT=production
```

## File Structure

```
src/api/
├── __init__.py
├── main.py              # FastAPI application
├── server.py            # Server startup script
├── integrations.py      # Integration layer
├── dependencies.py      # FastAPI dependencies
├── models.py           # Pydantic models
├── routes/
│   ├── __init__.py
│   ├── system.py       # System control routes
│   ├── configuration.py # Configuration routes
│   ├── instruments.py   # Instrument routes
│   ├── signals.py      # Signal routes
│   ├── models.py       # Model routes
│   └── auth.py         # Authentication routes
└── README.md           # This file
```

## Contributing

1. Follow the existing code structure
2. Add proper error handling
3. Include comprehensive tests
4. Update documentation
5. Validate with existing components

## Support

For issues and questions:
1. Check the logs in `logs/trading_system.log`
2. Review the API documentation at `/docs`
3. Validate configuration with `/api/config`
4. Check system health at `/api/health`