"""Konfigurasi backend — baca dari .env atau environment variable."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Root project backend
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent

# Folder data (7 file kontrak dari Orang 1)
DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BACKEND_ROOT / "data"))).resolve()

# Folder static (untuk serve PNG via /static)
STATIC_DIR: Path = BACKEND_ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)

# CORS origins — pisahkan dengan koma di .env
_cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
CORS_ORIGINS: list[str] = [o.strip() for o in _cors_raw.split(",") if o.strip()]

# Nama file 7 kontrak (PRD §B.1)
FILES = {
    "landcover_t1_png":    "landcover_2021.png",
    "landcover_t1_bounds": "landcover_2021_bounds.json",
    "landcover_t2_png":    "landcover_2025.png",
    "landcover_t2_bounds": "landcover_2025_bounds.json",
    "deforestation":       "deforestation.geojson",
    "statistics":          "statistics.json",
    "legend":              "legend.json",
    "metrics":             "metrics.json",
    "model_onnx":          "model.onnx",
    "model_card":          "model_card.md",
}

VALID_YEARS = {2021, 2025}
VALID_DOWNLOAD_FILES = {"geojson", "metrics", "legend"}
