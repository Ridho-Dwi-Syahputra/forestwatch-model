"""Endpoint landcover PNG (BLOB di DB) + bounds untuk L.imageOverlay Leaflet."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from app.core.config import VALID_YEARS
from app.db.orm import Landcover
from app.db.session import get_db

router = APIRouter()


def _get_landcover_row(year: int, db: Session) -> Landcover:
    if year not in VALID_YEARS:
        raise HTTPException(
            status_code=400, detail=f"Year harus salah satu dari {sorted(VALID_YEARS)}"
        )
    row = db.get(Landcover, year)
    if row is None:
        raise HTTPException(status_code=404, detail=f"landcover {year} tidak ditemukan di database")
    return row


@router.get("/landcover/{year}")
def get_landcover(year: int, request: Request, db: Session = Depends(get_db)):
    row = _get_landcover_row(year, db)
    base_url = str(request.base_url).rstrip("/")
    image_url = f"{base_url}/api/landcover/{year}/image"

    return {
        "year": year,
        "image_url": image_url,
        "bounds": [[row.bounds_south, row.bounds_west], [row.bounds_north, row.bounds_east]],
        "crs": row.crs,
    }


@router.get("/landcover/{year}/image")
def get_landcover_image(year: int, db: Session = Depends(get_db)):
    row = _get_landcover_row(year, db)
    return Response(content=row.png_bytes, media_type="image/png")
