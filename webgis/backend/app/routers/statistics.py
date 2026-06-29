"""Endpoint statistics — full, per-province, per-transition."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.orm import ProvinceStat, Statistics
from app.db.session import get_db

router = APIRouter()


def load_stats(db: Session) -> dict:
    """Susun ulang dict identik bentuk statistics.json lama. Reused oleh download.py."""
    stats = db.get(Statistics, 1)
    if stats is None:
        raise HTTPException(status_code=404, detail="Data statistics belum tersedia di database")

    provinces = db.query(ProvinceStat).all()

    return {
        "period_from": stats.period_from,
        "period_to": stats.period_to,
        "total_deforestation_ha": stats.total_deforestation_ha,
        "n_hotspots": stats.n_hotspots,
        "per_transition_ha": json.loads(stats.per_transition_ha_json),
        "per_province": [
            {"province": p.province, "deforestation_ha": p.deforestation_ha} for p in provinces
        ],
        "per_class_area_ha": json.loads(stats.per_class_area_ha_json),
        "model_metrics": json.loads(stats.model_metrics_json),
    }


@router.get("/statistics")
def get_statistics(db: Session = Depends(get_db)):
    return load_stats(db)


@router.get("/statistics/per-province")
def get_per_province(db: Session = Depends(get_db)):
    stats = load_stats(db)
    return {"data": stats.get("per_province", [])}


@router.get("/statistics/per-transition")
def get_per_transition(db: Session = Depends(get_db)):
    stats = load_stats(db)
    return {"data": stats.get("per_transition_ha", {})}


@router.get("/statistics/summary")
def get_summary(db: Session = Depends(get_db)):
    """Ringkasan cepat untuk header/card statistik."""
    stats = load_stats(db)
    return {
        "period_from": stats.get("period_from"),
        "period_to": stats.get("period_to"),
        "total_deforestation_ha": stats.get("total_deforestation_ha"),
        "n_hotspots": stats.get("n_hotspots"),
        "mean_iou": stats.get("model_metrics", {}).get("mean_iou"),
        "overall_accuracy": stats.get("model_metrics", {}).get("overall_accuracy"),
    }
