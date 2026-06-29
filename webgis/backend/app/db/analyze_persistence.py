"""Persistensi log + hasil POST /api/analyze -- TIDAK bergantung GEE/torch/rasterio (cuma ORM +
schema Pydantic), supaya bisa di-unit-test terpisah tanpa boot pipeline inferensi penuh.

Kedua fungsi menelan semua exception (log warning + rollback, TAK PERNAH raise): kegagalan
nulis log/hasil tidak boleh merusak response live yang sudah berhasil dihitung.
"""

from __future__ import annotations

import base64
import json
import logging

from sqlalchemy.orm import Session

from app.db.orm import AnalyzeRequestLog, AnalyzeResult
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)


def _decode_png_data_url(data_url: str | None) -> bytes | None:
    """'data:image/png;base64,XXXX' -> bytes mentah. None kalau data_url None/rusak."""
    if not data_url:
        return None
    try:
        _, b64 = data_url.split(",", 1)
        return base64.b64decode(b64)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gagal decode PNG data URL utk persist analyze result: %s", exc)
        return None


def log_analyze_request(
    db: Session,
    req: AnalyzeRequest,
    *,
    status: str,
    http_status_code: int | None,
    error_message: str | None = None,
    duration_ms: int | None = None,
) -> None:
    """Best-effort insert 1 baris log (jalur GAGAL -- tanpa AnalyzeResult)."""
    try:
        db.add(
            AnalyzeRequestLog(
                aoi_lon_min=req.aoi[0], aoi_lat_min=req.aoi[1],
                aoi_lon_max=req.aoi[2], aoi_lat_max=req.aoi[3],
                year_t1=req.year_t1, year_t2=req.year_t2, min_area_ha=req.min_area_ha,
                status=status, http_status_code=http_status_code,
                error_message=error_message, duration_ms=duration_ms,
            )
        )
        db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gagal mencatat analyze_request_log: %s", exc)
        db.rollback()


def save_analyze_result(
    db: Session,
    req: AnalyzeRequest,
    response: AnalyzeResponse,
    duration_ms: int | None,
) -> None:
    """Best-effort insert log (status=success) + result, satu transaksi (jalur SUKSES)."""
    try:
        log_row = AnalyzeRequestLog(
            aoi_lon_min=req.aoi[0], aoi_lat_min=req.aoi[1],
            aoi_lon_max=req.aoi[2], aoi_lat_max=req.aoi[3],
            year_t1=req.year_t1, year_t2=req.year_t2, min_area_ha=req.min_area_ha,
            status="success", http_status_code=200, duration_ms=duration_ms,
        )
        db.add(log_row)
        db.flush()  # butuh log_row.id sebelum bikin FK di result_row

        db.add(
            AnalyzeResult(
                request_log_id=log_row.id,
                deforestation_geojson=json.dumps(response.deforestation),
                statistics_json=json.dumps(response.statistics),
                bounds_south=response.bounds[0][0], bounds_west=response.bounds[0][1],
                bounds_north=response.bounds[1][0], bounds_east=response.bounds[1][1],
                landcover_t1_png=_decode_png_data_url(response.landcover_t1_png),
                landcover_t2_png=_decode_png_data_url(response.landcover_t2_png),
            )
        )
        db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gagal menyimpan analyze_result: %s", exc)
        db.rollback()
