"""Endpoint GeoJSON deforestasi dengan filter server-side (SQL WHERE, bukan list-comprehension)."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.orm import DeforestationFeature
from app.db.session import get_db

router = APIRouter()

VALID_TRANSITIONS = {
    "hutan_ke_lahan_terbuka",
    "hutan_ke_sawit",
    "hutan_ke_pertanian_lain",
    "hutan_ke_terbakar",
}

# Selalu konstan di dataset ini -- bukan tabel utk 2 string statis.
_FEATURE_COLLECTION_NAME = "deforestation"
_FEATURE_COLLECTION_CRS = {"type": "name", "properties": {"name": "EPSG:4326"}}


def feature_to_dict(row: DeforestationFeature) -> dict:
    return {
        "type": "Feature",
        "geometry": json.loads(row.geometry_json),
        "properties": {
            "id": row.id,
            "transition_type": row.transition_type,
            "area_ha": row.area_ha,
            "period_from": row.period_from,
            "period_to": row.period_to,
            "province": row.province,
            "kawasan_status": row.kawasan_status,
        },
    }


def query_deforestation_features(
    db: Session,
    *,
    transition_type: str | None = None,
    province: str | None = None,
    min_area_ha: float | None = None,
) -> list[DeforestationFeature]:
    """Reused oleh /api/download/deforestation/csv -- jangan duplikasi query."""
    query = db.query(DeforestationFeature)
    if transition_type:
        query = query.filter(DeforestationFeature.transition_type == transition_type)
    if province:
        query = query.filter(func.lower(DeforestationFeature.province) == province.lower())
    if min_area_ha is not None:
        query = query.filter(DeforestationFeature.area_ha >= min_area_ha)
    return query.all()


@router.get("/deforestation")
def get_deforestation(
    transition_type: Optional[str] = Query(None, description="Filter jenis transisi"),
    province: Optional[str] = Query(None, description="Filter nama provinsi"),
    min_area_ha: Optional[float] = Query(None, ge=0, description="Luas minimum (ha)"),
    db: Session = Depends(get_db),
):
    if transition_type and transition_type not in VALID_TRANSITIONS:
        raise HTTPException(
            status_code=400,
            detail=f"transition_type tidak valid. Pilihan: {sorted(VALID_TRANSITIONS)}",
        )

    rows = query_deforestation_features(
        db, transition_type=transition_type, province=province, min_area_ha=min_area_ha
    )
    features = [feature_to_dict(r) for r in rows]

    return {
        "type": "FeatureCollection",
        "name": _FEATURE_COLLECTION_NAME,
        "crs": _FEATURE_COLLECTION_CRS,
        "features": features,
        "total": len(features),
    }
