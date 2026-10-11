UV ?= uv

.DEFAULT_GOAL := check
.PHONY: check test lint format-check format

check: lint format-check test

test:
	$(UV) run --locked python -m pytest

lint:
	$(UV) run --locked ruff check .

format-check:
	$(UV) run --locked ruff format --check .

format:
	$(UV) run --locked ruff format .
