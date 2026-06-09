"""Endpoint landcover PNG + bounds untuk L.imageOverlay Leaflet."""

import json
import shutil
from fastapi import APIRouter, HTTPException, Request
from app.core.config import DATA_DIR, FILES, STATIC_DIR, VALID_YEARS

router = APIRouter()


def _ensure_png_in_static(year: int) -> str:
    """Copy PNG ke folder static agar bisa diakses via /static/."""
    src = DATA_DIR / FILES[f"landcover_t{'1' if year == 2021 else '2'}_png"]
    dst = STATIC_DIR / src.name
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"landcover_{year}.png tidak ditemukan")
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
    return src.name


@router.get("/landcover/{year}")
def get_landcover(year: int, request: Request):
    if year not in VALID_YEARS:
        raise HTTPException(
            status_code=400,
            detail=f"Year harus salah satu dari {sorted(VALID_YEARS)}"
        )

    key = "t1" if year == 2021 else "t2"
    bounds_path = DATA_DIR / FILES[f"landcover_{key}_bounds"]
    if not bounds_path.exists():
        raise HTTPException(status_code=404, detail=f"bounds JSON untuk {year} tidak ditemukan")

    bounds_data = json.loads(bounds_path.read_text(encoding="utf-8"))
    filename = _ensure_png_in_static(year)

    # URL image: /static/{filename}
    base_url = str(request.base_url).rstrip("/")
    image_url = f"{base_url}/static/{filename}"

    return {
        "year": year,
        "image_url": image_url,
        "bounds": bounds_data["bounds"],
        "crs": bounds_data.get("crs", "EPSG:4326"),
    }
