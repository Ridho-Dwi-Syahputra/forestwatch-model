"""ForestWatch Papua — Backend FastAPI.

Menserve 7 file kontrak (PRD §B.1), kini disinkronkan ke database MySQL (lihat app/db/),
sebagai REST API untuk frontend WebGIS. Tidak melakukan inferensi model untuk endpoint
pre-computed -- hanya POST /api/analyze yang menjalankan pipeline live.

Jalankan:
    uvicorn app.main:app --reload

Docs:
    http://localhost:8000/docs
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import gee_client, model_singleton
from app.core.config import CORS_ORIGINS
from app.db.session import SessionLocal, init_db
from app.db.sync_static import sync_all
from app.routers import analyze, deforestation, download, health, landcover, legend, statistics

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ForestWatch Papua API",
    description=(
        "REST API untuk dashboard WebGIS deteksi deforestasi Papua. "
        "Data bersumber dari model ResNet50-U-Net trained on Sentinel-2."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — izinkan frontend React (localhost dev + URL produksi)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    # Non-fatal: kalau MySQL/GEE/model belum dikonfigurasi, endpoint pre-computed tetap jalan
    # selama database sebelumnya sudah pernah ke-sync -- cuma POST /api/analyze yang akan
    # respond 503 sampai GEE/model env var-nya diisi.
    try:
        init_db()
        db = SessionLocal()
        try:
            counts = sync_all(db)
            logger.info("Sync database dari DATA_DIR selesai: %s", counts)
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gagal init/sync database (%s) -- cek koneksi MySQL di .env.", exc)

    gee_client.init_ee_service_account()
    model_singleton.load_model()

# Register semua router di bawah prefix /api
PREFIX = "/api"
app.include_router(health.router,        prefix=PREFIX, tags=["Health"])
app.include_router(legend.router,        prefix=PREFIX, tags=["Legend"])
app.include_router(landcover.router,     prefix=PREFIX, tags=["Landcover"])
app.include_router(deforestation.router, prefix=PREFIX, tags=["Deforestation"])
app.include_router(statistics.router,    prefix=PREFIX, tags=["Statistics"])
app.include_router(download.router,      prefix=PREFIX, tags=["Download"])
app.include_router(analyze.router,       prefix=PREFIX, tags=["Analyze"])


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "ForestWatch Papua API",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": [
            "GET /api/health",
            "GET /api/legend",
            "GET /api/landcover/{year}",
            "GET /api/landcover/{year}/image",
            "GET /api/deforestation",
            "GET /api/statistics",
            "GET /api/statistics/per-province",
            "GET /api/statistics/per-transition",
            "GET /api/statistics/summary",
            "GET /api/download/{file_type}",
            "GET /api/download/deforestation/csv",
            "POST /api/analyze",
        ],
    }
