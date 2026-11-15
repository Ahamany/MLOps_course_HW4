.PHONY: install lint test coverage train evaluate

install:
	uv sync --locked

lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .

test:
	uv run --locked pytest

coverage:
	uv run --locked pytest --cov=ml_pipeline --cov-branch --cov-report=term-missing --cov-report=html

train:
	uv run --locked ml-pipeline train --output artifacts/demo

evaluate:
	uv run --locked ml-pipeline evaluate --checkpoint artifacts/demo/model.pt --seed 987
