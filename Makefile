.PHONY: help install install-dev format lint test data-format data-validate clean

PYTHON ?= python3

help:
	@echo "Available targets:"
	@echo "  install         Install production dependencies"
	@echo "  install-dev     Install development dependencies"
	@echo "  format          Format codebase using ruff"
	@echo "  lint            Lint codebase using ruff"
	@echo "  test            Run unit tests with pytest"
	@echo "  data-format     Convert raw chat datasets to Gemini SFT format"
	@echo "  data-validate   Validate processed Gemini datasets"
	@echo "  clean           Clean build artifacts, cache files, and test outputs"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

format:
	ruff format src/ tests/

lint:
	ruff check src/ tests/

test:
	pytest -v tests/

data-format:
	$(PYTHON) -m src.data.formatter --input data/raw/sft_train_samples.jsonl --output data/processed/train_gemini.jsonl
	$(PYTHON) -m src.data.formatter --input data/raw/sft_val_samples.jsonl --output data/processed/val_gemini.jsonl

data-validate:
	$(PYTHON) -m src.data.validator --file data/processed/train_gemini.jsonl
	$(PYTHON) -m src.data.validator --file data/processed/val_gemini.jsonl

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .coverage htmlcov
