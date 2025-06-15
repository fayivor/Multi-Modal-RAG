# Multi-Modal RAG System Makefile

.PHONY: help install install-dev test test-cov lint format type-check security clean build run docker-build docker-run docker-stop docs

# Default target
help:
	@echo "Multi-Modal RAG System - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  run          Run the application locally"
	@echo "  clean        Clean up temporary files"
	@echo ""
	@echo "Code Quality:"
	@echo "  format       Format code with black and isort"
	@echo "  lint         Run linting with flake8"
	@echo "  type-check   Run type checking with mypy"
	@echo "  security     Run security checks with bandit"
	@echo "  check-all    Run all code quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  test-fast    Run tests in parallel"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run with Docker Compose"
	@echo "  docker-stop  Stop Docker containers"
	@echo "  docker-logs  View Docker logs"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Generate documentation"
	@echo "  docs-serve   Serve documentation locally"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# Development
run:
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage

# Code Quality
format:
	black src/ tests/ examples/
	isort src/ tests/ examples/

lint:
	flake8 src/ tests/ examples/

type-check:
	mypy src/

security:
	bandit -r src/

check-all: format lint type-check security
	@echo "All code quality checks completed"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -n auto

# Docker
docker-build:
	docker build -t multi-modal-rag:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v
	docker system prune -f

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080

# Database
db-init:
	python scripts/init_db.py

db-migrate:
	alembic upgrade head

db-reset:
	python scripts/reset_db.py

# Monitoring
metrics:
	curl http://localhost:8000/metrics

health:
	curl http://localhost:8000/health

# Load Testing
load-test:
	locust -f tests/load/locustfile.py --host=http://localhost:8000

# Deployment
deploy-staging:
	@echo "Deploying to staging environment..."
	# Add staging deployment commands here

deploy-prod:
	@echo "Deploying to production environment..."
	# Add production deployment commands here

# Backup
backup:
	python scripts/backup.py

restore:
	python scripts/restore.py

# Development utilities
shell:
	python -c "from src.api.main import app; import IPython; IPython.embed()"

requirements:
	pip-compile requirements.in
	pip-compile requirements-dev.in

update-deps:
	pip-compile --upgrade requirements.in
	pip-compile --upgrade requirements-dev.in

# CI/CD helpers
ci-install:
	pip install -r requirements.txt -r requirements-dev.txt

ci-test:
	pytest tests/ --cov=src --cov-report=xml --junitxml=test-results.xml

ci-lint:
	flake8 src/ tests/ --format=junit-xml --output-file=lint-results.xml

ci-security:
	bandit -r src/ -f json -o security-results.json

# Performance
profile:
	python -m cProfile -o profile.stats scripts/profile_app.py

benchmark:
	python scripts/benchmark.py
