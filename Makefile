# Makefile for Automated Intraday Trading Signal Generation System
.PHONY: help install install-dev test lint format type-check security clean docs run

# Default target
help: ## Show this help message
	@echo "Production-grade Automated Intraday Trading Signal Generation System"
	@echo "=================================================================="
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev,test,docs]"

test: ## Run tests
	pytest -v --cov=src --cov-report=term-missing --cov-report=html

test-fast: ## Run fast tests only
	pytest -v -m "not slow" --cov=src

test-integration: ## Run integration tests
	pytest -v -m "integration" tests/

lint: ## Run linting
	ruff check src main.py
	ruff format --check src main.py

format: ## Format code
	ruff format src main.py
	ruff check --fix src main.py

type-check: ## Run type checking
	mypy src

security: ## Run security checks
	bandit -r src

clean: ## Clean build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

docs: ## Build documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

quality: lint format type-check security ## Run all quality checks

validate: quality test ## Full validation (quality + tests)

# Database operations
db-init: ## Initialize database schema
	psql -h localhost -p 5432 -U user -d timescaledb -f src/database/schema.sql

db-migrate: ## Run database migrations
	psql -h localhost -p 5432 -U user -d timescaledb -f src/database/migrations/v2_indexes.sql
	psql -h localhost -p 5432 -U user -d timescaledb -f src/database/migrations/v3_continuous_aggs.sql

# Application operations
run-historical: ## Run in historical mode
	python main.py

run-live: ## Run in live mode (ensure config is set to LIVE_MODE)
	python main.py

# Development utilities
logs: ## View application logs
	tail -f logs/app.log

build: ## Build distribution
	python -m build

pre-commit: format quality test-fast ## Run pre-commit checks

# CI/CD targets
ci: install-dev quality test ## CI pipeline

release: clean validate build ## Prepare release