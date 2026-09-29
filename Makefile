.PHONY: test lint reference bench check api-diff

test:
	uv run pytest

lint:
	uv run ruff check .

reference:
	uv run python examples/reference_case/run_reference.py

bench:
	uv run python examples/benchmarks/run_benchmarks.py

check: lint test reference bench

# Public API breaks since the latest release tag, read before writing the
# CHANGELOG. A release step, not part of check: griffe exits 1 on a break.
api-diff:
	uvx griffe check tsdive -s src --against $$(git describe --tags --abbrev=0 --match 'v*')
