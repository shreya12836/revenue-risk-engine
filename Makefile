.PHONY: install test lint format run-api run-dashboard clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov=dashboard --cov-report=term-missing

lint:
	flake8 src/ tests/ dashboard/ --max-line-length=100
	mypy src/ dashboard/ --ignore-missing-imports

format:
	black src/ tests/ dashboard/

run-api:
	uvicorn src.api.main:app --reload --port 8000

run-dashboard:
	streamlit run dashboard/app.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
