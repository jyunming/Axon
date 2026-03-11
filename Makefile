.PHONY: help install install-dev test lint format type-check clean run-api run-ui docker-build docker-run

help:  ## Show this help message
	@echo "Axon - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and dependencies
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=axon --cov-report=term-missing --cov-report=html

lint:  ## Run linting checks
	ruff check src/ tests/
	black --check src/ tests/

format:  ## Format code with black and ruff
	black src/ tests/
	ruff check --fix src/ tests/

type-check:  ## Run type checking
	mypy src/axon/ --ignore-missing-imports

clean:  ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

run-cli:  ## Run the interactive REPL CLI (local, no Docker needed)
	axon

run-api:  ## Run the FastAPI server
	axon-api

run-ui:  ## Run the Streamlit UI
	axon-ui

run-all:  ## Run API + UI together (local, no Docker)
	@echo "Starting API on :8000 and UI on :8501 ..."
	axon-api & axon-ui

docker-build:  ## Build Docker image
	docker build -t axon:latest .

docker-run:  ## Run Docker containers with docker-compose
	docker-compose up -d

docker-down:  ## Stop Docker containers
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f

all: format lint type-check test  ## Run all checks

ci: lint type-check test  ## Run CI checks (no auto-formatting)
