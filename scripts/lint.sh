#!/bin/bash
set -e

echo "--- Running Ruff (Linting & Formatting) ---"
uv run ruff check src --fix --exclude "*.ipynb"
uv run ruff format src --exclude "*.ipynb"

echo "--- Running MyPy ---"
uv run mypy src

echo "--- Running PyLint ---"
uv run pylint src

echo "--- All checks passed! ---"
