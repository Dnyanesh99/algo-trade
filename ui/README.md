# Algo Trading Control Panel

A modern, TradingView-style control panel for managing your algorithmic trading system.

## Features

- **System Dashboard** - Real-time monitoring and system health
- **Configuration Management** - Dynamic editing of all system parameters
- **Technical Indicators** - TradingView-style indicator configuration
- **Instruments Management** - CSV import/export for trading instruments
- **ML Model Control** - Model training and deployment management
- **Market Calendar** - Holiday and trading session management
- **Monitoring Integration** - Quick access to Grafana and Prometheus

## Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS + Radix UI components
- **State Management**: Zustand
- **Data Fetching**: React Query
- **Charts**: Recharts
- **Theme**: Dark/Light mode support

## Quick Start

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

3. **Build for production**:
   ```bash
   npm run build
   ```

## Configuration

The UI connects to your Python backend on port 8000. Update the proxy configuration in `vite.config.ts` if needed.

## Architecture

### Pages
- `/dashboard` - System overview and quick actions
- `/configuration` - Complete config.yaml management
- `/indicators` - Technical indicators setup
- `/instruments` - Trading instruments management
- `/models` - ML model training and deployment
- `/calendar` - Market calendar and holidays
- `/monitoring` - System monitoring overview

### Components
- **UI Components**: Reusable Shadcn/ui components
- **Layout**: Sidebar navigation and header
- **Theme**: Dark/light mode toggle
- **File Upload**: CSV import/export functionality

## Integration

The UI is designed to work with your existing:
- **Python Backend**: FastAPI endpoints
- **TimescaleDB**: Database operations
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

## Development

### File Structure
```
ui/
├── src/
│   ├── components/     # Reusable UI components
│   ├── pages/          # Main application pages
│   ├── types/          # TypeScript type definitions
│   ├── hooks/          # Custom React hooks
│   ├── lib/            # Utility functions
│   └── styles/         # Global styles
├── public/             # Static assets
└── package.json        # Dependencies and scripts
```

### Key Features
- **Type Safety**: Full TypeScript coverage
- **Responsive Design**: Mobile-first approach
- **Real-time Updates**: WebSocket ready
- **File Management**: CSV import/export
- **Theme Support**: Dark/light modes
- **Accessibility**: ARIA compliant

## Contributing

1. Follow the existing code style
2. Add TypeScript types for all new components
3. Test responsive design
4. Update documentation as needed