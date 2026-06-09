"""Endpoint GeoJSON deforestasi dengan filter server-side."""

import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from app.core.config import DATA_DIR, FILES

router = APIRouter()

VALID_TRANSITIONS = {
    "hutan_ke_lahan_terbuka",
    "hutan_ke_sawit",
    "hutan_ke_pertanian_lain",
    "hutan_ke_terbakar",
}


def _load_geojson() -> dict:
    path = DATA_DIR / FILES["deforestation"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="deforestation.geojson tidak ditemukan")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/deforestation")
def get_deforestation(
    transition_type: Optional[str] = Query(None, description="Filter jenis transisi"),
    province: Optional[str] = Query(None, description="Filter nama provinsi"),
    min_area_ha: Optional[float] = Query(None, ge=0, description="Luas minimum (ha)"),
):
    if transition_type and transition_type not in VALID_TRANSITIONS:
        raise HTTPException(
            status_code=400,
            detail=f"transition_type tidak valid. Pilihan: {sorted(VALID_TRANSITIONS)}"
        )

    fc = _load_geojson()
    features = fc.get("features", [])

    # Filter
    if transition_type:
        features = [
            f for f in features
            if f.get("properties", {}).get("transition_type") == transition_type
        ]
    if province:
        features = [
            f for f in features
            if f.get("properties", {}).get("province", "").lower() == province.lower()
        ]
    if min_area_ha is not None:
        features = [
            f for f in features
            if float(f.get("properties", {}).get("area_ha", 0)) >= min_area_ha
        ]

    return {
        "type": "FeatureCollection",
        "name": fc.get("name", "deforestation"),
        "crs": fc.get("crs"),
        "features": features,
        "total": len(features),
    }
