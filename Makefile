.PHONY: test lint reference bench check

test:
	uv run pytest

lint:
	uv run ruff check .

reference:
	uv run python examples/reference_case/run_reference.py

bench:
	uv run python examples/benchmarks/run_benchmarks.py

check: lint test reference bench
