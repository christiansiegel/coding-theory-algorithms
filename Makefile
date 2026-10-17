UV ?= uv
PYTHONS ?= 3.10 3.11 3.12 3.13 3.14

.DEFAULT_GOAL := check
.PHONY: check test lint format-check format compatibility

check: lint format-check test

test:
	$(UV) run --locked python -m pytest

lint:
	$(UV) run --locked ruff check .

format-check:
	$(UV) run --locked ruff format --check .

format:
	$(UV) run --locked ruff format .

compatibility:
	$(UV) run --locked python scripts/check_compatibility.py --uv "$(UV)" $(PYTHONS)
