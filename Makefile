.PHONY: setup test lint fmt api cli clean install dev

# Setup and installation
setup: install
	@echo "✅ Project setup complete"

install:
	pip install -r requirements.txt
	pip install -e .

dev: install
	pip install -r requirements-dev.txt

# Development commands
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

cli:
	python -m cli.app run --agent strategist --input "Hello from Council CLI"

council:
	python -m cli.app council --agents "strategist,architect" --input "Design a scalable AI platform"

# Testing and quality
test:
	pytest -v tests/

test-watch:
	pytest-watch --runner "pytest -v"

lint:
	ruff check .

fmt:
	ruff format .

check: lint test
	@echo "✅ All checks passed"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

# Docker (optional)
docker-build:
	docker build -t ai-council-agent .

docker-run:
	docker run -p 8000:8000 --env-file .env ai-council-agent

# Graph visualization (if graphviz installed)
graph:
	python -c "from council.graph import CouncilGraph; g = CouncilGraph(); g.visualize()"

# Help
help:
	@echo "Available commands:"
	@echo "  setup     - Install dependencies and setup project"
	@echo "  api       - Start FastAPI server"
	@echo "  cli       - Run single agent via CLI"
	@echo "  council   - Run multi-agent council via CLI"
	@echo "  test      - Run test suite"
	@echo "  lint      - Run linting"
	@echo "  fmt       - Format code"
	@echo "  check     - Run lint and tests"
	@echo "  clean     - Clean up generated files"
	@echo "  help      - Show this help message"