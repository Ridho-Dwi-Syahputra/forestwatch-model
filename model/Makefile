# ForestWatch Papua - Makefile
# Catatan: di Windows PowerShell, jalankan via `make <target>` jika GNU Make terinstal,
# atau salin perintah yang relevan langsung ke terminal.

.PHONY: help setup setup-min test test-cov lint dummy validate clean

help:
	@echo "ForestWatch Papua - perintah yang tersedia:"
	@echo "  make setup       - install package + semua optional deps (gee, gis, ml, dev)"
	@echo "  make setup-min   - install package + dev deps saja (tanpa gee/gis/ml)"
	@echo "  make test        - jalankan pytest"
	@echo "  make test-cov    - jalankan pytest dengan coverage report"
	@echo "  make lint        - jalankan ruff check"
	@echo "  make dummy       - generate 7 file dummy ke outputs/dummy/"
	@echo "  make validate    - validasi outputs/dummy/ terhadap skema PRD"
	@echo "  make clean       - hapus build artifacts dan cache"

setup:
	pip install -e ".[all]"

setup-min:
	pip install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov=forestwatch --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests scripts

dummy:
	python scripts/generate_dummy_data.py --out outputs/dummy --n-polygons 60 --seed 42

validate:
	python scripts/validate_outputs.py --dir outputs/dummy

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
