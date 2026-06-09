"""Endpoint statistics — full, per-province, per-transition."""

import json
from fastapi import APIRouter, HTTPException
from app.core.config import DATA_DIR, FILES

router = APIRouter()


def _load_stats() -> dict:
    path = DATA_DIR / FILES["statistics"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="statistics.json tidak ditemukan")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/statistics")
def get_statistics():
    return _load_stats()


@router.get("/statistics/per-province")
def get_per_province():
    stats = _load_stats()
    return {"data": stats.get("per_province", [])}


@router.get("/statistics/per-transition")
def get_per_transition():
    stats = _load_stats()
    return {"data": stats.get("per_transition_ha", {})}


@router.get("/statistics/summary")
def get_summary():
    """Ringkasan cepat untuk header/card statistik."""
    stats = _load_stats()
    return {
        "period_from": stats.get("period_from"),
        "period_to": stats.get("period_to"),
        "total_deforestation_ha": stats.get("total_deforestation_ha"),
        "n_hotspots": stats.get("n_hotspots"),
        "mean_iou": stats.get("model_metrics", {}).get("mean_iou"),
        "overall_accuracy": stats.get("model_metrics", {}).get("overall_accuracy"),
    }
