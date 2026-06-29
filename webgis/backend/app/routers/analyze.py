"""POST /api/analyze -- analisis deforestasi LIVE untuk wilayah custom yang digambar user.

Beda dari endpoint lain (yang semuanya baca file pre-computed): endpoint ini menjalankan
pipeline penuh on-demand -- tarik komposit Sentinel-2 dari GEE (2 tahun), jalankan inferensi
model segmentasi, lalu deteksi transisi Hutan->{5 kelas target}. Karena itu jauh lebih lambat
(detik s/d menit) dan satu-satunya endpoint yang butuh GEE + model di-load -- lihat
``app/core/gee_client.py`` dan ``app/core/model_singleton.py``.
"""

from __future__ import annotations

import base64
import io
import math
import shutil
import time
import uuid
from pathlib import Path

import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core import gee_client, model_singleton
from app.core.config import ANALYZE_MAX_SIDE_KM, ANALYZE_SCALE_M, ANALYZE_TMP_DIR
from app.core.forestwatch_bridge import (
    FORESTWATCH_AVAILABLE,
    FORESTWATCH_IMPORT_ERROR,
    detect_transitions_from_arrays,
    infer_tile,
    mask_to_rgb,
    s2_composite,
    summarize_geojson_transitions,
)
from app.db.analyze_persistence import log_analyze_request, save_analyze_result
from app.db.session import get_db
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse

router = APIRouter()

_DEG_TO_KM = 111.0


def _mask_to_png_data_url(mask) -> str:
    """Colorize mask kelas (H,W uint8) -> PNG RGBA -> data URL base64 utk preview di frontend."""
    from PIL import Image  # noqa: PLC0415

    rgba = mask_to_rgb(mask)  # (H, W, 4) uint8, pakai PALETTE_RGB resmi
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _aoi_side_km(aoi: list[float]) -> tuple[float, float]:
    lon_min, lat_min, lon_max, lat_max = aoi
    mid_lat = (lat_min + lat_max) / 2
    width_km = (lon_max - lon_min) * _DEG_TO_KM * math.cos(math.radians(mid_lat))
    height_km = (lat_max - lat_min) * _DEG_TO_KM
    return width_km, height_km


def _download_ee_geotiff(image, region, scale: int, out_path: Path) -> None:
    url = image.getDownloadURL({"region": region, "scale": scale, "format": "GEO_TIFF"})
    resp = requests.get(url, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Gagal unduh komposit GEE (HTTP {resp.status_code}): {resp.text[:200]}")
    out_path.write_bytes(resp.content)


@router.post("/analyze", response_model=AnalyzeResponse, tags=["Analyze"])
def analyze(req: AnalyzeRequest, db: Session = Depends(get_db)):
    start = time.monotonic()
    try:
        response = _run_analysis(req)
    except HTTPException as exc:
        log_analyze_request(
            db, req, status="error", http_status_code=exc.status_code,
            error_message=str(exc.detail), duration_ms=int((time.monotonic() - start) * 1000),
        )
        raise
    except Exception as exc:  # noqa: BLE001 -- tetap 500 default FastAPI, cuma dicatat dulu
        log_analyze_request(
            db, req, status="error", http_status_code=500,
            error_message=str(exc), duration_ms=int((time.monotonic() - start) * 1000),
        )
        raise

    save_analyze_result(db, req, response, duration_ms=int((time.monotonic() - start) * 1000))
    return response


def _run_analysis(req: AnalyzeRequest) -> AnalyzeResponse:
    if not FORESTWATCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Paket forestwatch tidak tersedia di server ini: {FORESTWATCH_IMPORT_ERROR}",
        )
    if not gee_client.is_ready() and not gee_client.init_ee_service_account():
        raise HTTPException(
            status_code=503,
            detail="Earth Engine belum siap -- cek GEE_SERVICE_ACCOUNT_EMAIL / "
            "GEE_SERVICE_ACCOUNT_KEY_PATH di .env backend.",
        )
    if not model_singleton.is_ready() and not model_singleton.load_model():
        raise HTTPException(
            status_code=503,
            detail=f"Model belum siap: {model_singleton.get_load_error()}",
        )

    width_km, height_km = _aoi_side_km(req.aoi)
    if width_km > ANALYZE_MAX_SIDE_KM or height_km > ANALYZE_MAX_SIDE_KM:
        raise HTTPException(
            status_code=400,
            detail=(
                f"AOI terlalu besar ({width_km:.1f} x {height_km:.1f} km). "
                f"Maksimum {ANALYZE_MAX_SIDE_KM:.0f} x {ANALYZE_MAX_SIDE_KM:.0f} km per "
                "analisis -- gambar kotak yang lebih kecil."
            ),
        )

    import ee  # noqa: PLC0415 -- hanya diimpor setelah GEE pasti siap
    import rasterio  # noqa: PLC0415

    work_dir = ANALYZE_TMP_DIR / uuid.uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        aoi_geom = ee.Geometry.Rectangle(req.aoi)

        img_t1 = s2_composite(req.year_t1, aoi_geom)
        img_t2 = s2_composite(req.year_t2, aoi_geom)

        tif_t1, tif_t2 = work_dir / "t1.tif", work_dir / "t2.tif"
        try:
            _download_ee_geotiff(img_t1, aoi_geom, ANALYZE_SCALE_M, tif_t1)
            _download_ee_geotiff(img_t2, aoi_geom, ANALYZE_SCALE_M, tif_t2)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail=f"Gagal mengambil citra dari Earth Engine: {exc}",
            ) from exc

        mask_t1_path = infer_tile(tif_t1, work_dir / "mask_t1.tif", model_singleton.get_model())
        mask_t2_path = infer_tile(tif_t2, work_dir / "mask_t2.tif", model_singleton.get_model())

        with rasterio.open(mask_t1_path) as src1, rasterio.open(mask_t2_path) as src2:
            mask_t1 = src1.read(1)
            mask_t2 = src2.read(1)
            transform = src1.transform
            bounds = src1.bounds

        features = detect_transitions_from_arrays(
            mask_t1, mask_t2, transform,
            min_area_ha=req.min_area_ha,
            period_from=req.year_t1,
            period_to=req.year_t2,
        )
        fc = {"type": "FeatureCollection", "features": features, "total": len(features)}
        stats = summarize_geojson_transitions(fc)

        # Preview "sebelum vs sesudah": klasifikasi model di AOI (bukan foto satelit tahun tsb).
        landcover_t1_png = _mask_to_png_data_url(mask_t1)
        landcover_t2_png = _mask_to_png_data_url(mask_t2)

        return AnalyzeResponse(
            deforestation=fc,
            statistics=stats,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            year_t1=req.year_t1,
            year_t2=req.year_t2,
            landcover_t1_png=landcover_t1_png,
            landcover_t2_png=landcover_t2_png,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
