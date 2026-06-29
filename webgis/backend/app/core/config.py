"""Konfigurasi backend — baca dari .env atau environment variable."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Root project backend
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent

# Folder data (7 file kontrak dari Orang 1) -- sumber sync ke database, lihat app/db/sync_static.py
DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BACKEND_ROOT / "data"))).resolve()

# === Database (MySQL) -- dashboard data + log/hasil "Analisis Wilayah Custom" ===
# Default ala XAMPP/Laragon lokal (root, tanpa password) -- sesuaikan di .env ke kredensial nyata.
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "forestwatch_papua")
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
# URL server TANPA nama DB -- dipakai init_db() utk CREATE DATABASE IF NOT EXISTS sekali di awal.
DATABASE_SERVER_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/?charset=utf8mb4"

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
}

VALID_YEARS = {2021, 2025}

# === Konfigurasi fitur "Analisis Wilayah Custom" (POST /api/analyze) ===
# Auth GEE pakai Service Account (server, tanpa interaksi manusia) -- BUKAN ee.Authenticate().
GEE_PROJECT = os.getenv("GEE_PROJECT", "forestwatch-papua-unand")
GEE_SERVICE_ACCOUNT_EMAIL = os.getenv("GEE_SERVICE_ACCOUNT_EMAIL", "")
GEE_SERVICE_ACCOUNT_KEY_PATH = os.getenv("GEE_SERVICE_ACCOUNT_KEY_PATH", "")

# Checkpoint model (.pt) hasil training Orang 1 -- disalin/symlink ke backend/data/
MODEL_CHECKPOINT_PATH: Path = Path(
    os.getenv("MODEL_CHECKPOINT_PATH", str(DATA_DIR / "best_model.pt"))
).resolve()
MODEL_ARCHITECTURE = os.getenv("MODEL_ARCHITECTURE", "unet_scse")
MODEL_ENCODER_NAME = os.getenv("MODEL_ENCODER_NAME", "resnet50")

# Folder sementara untuk komposit GEE + mask hasil inferensi (dibersihkan tiap request)
ANALYZE_TMP_DIR: Path = BACKEND_ROOT / ".analyze_tmp"
ANALYZE_TMP_DIR.mkdir(exist_ok=True)

# Batas ukuran AOI custom. 12 km dipilih karena batas byte getDownloadURL GEE (~48 MiB):
# komposit 6 band float32 @ 10 m -> side_px^2 * 6 * 4 <= ~48 MiB -> sisi <= ~14 km. Ambil 12
# km sebagai margin aman; AOI lebih besar akan ditolak (HTTP 400) dengan pesan jelas.
ANALYZE_MAX_SIDE_KM = float(os.getenv("ANALYZE_MAX_SIDE_KM", "12"))
ANALYZE_SCALE_M = 10
