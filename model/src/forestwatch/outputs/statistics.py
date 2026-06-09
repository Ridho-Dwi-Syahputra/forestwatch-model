"""Build ``statistics.json`` & ``metrics.json`` — PRD §B.1.2."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from forestwatch.constants import CLASS_NAMES, PROVINCES, TRANSITION_MAP
from forestwatch.utils.io import save_json


def summarize_geojson_transitions(
    feature_collection: dict[str, Any],
    *,
    transition_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Agregat feature deforestation.geojson → ringkasan transisi & provinsi.

    Returns:
        Dict berisi ``total_deforestation_ha``, ``n_hotspots``,
        ``per_transition_ha``, ``per_province``.
    """
    if transition_keys is None:
        transition_keys = tuple(TRANSITION_MAP.values())

    features = feature_collection.get("features", [])
    per_transition: dict[str, float] = {k: 0.0 for k in transition_keys}
    per_province: dict[str, float] = defaultdict(float)
    total_ha = 0.0
    for feat in features:
        props = feat.get("properties", {})
        ttype = props.get("transition_type")
        area_ha = float(props.get("area_ha", 0.0))
        province = props.get("province")
        if ttype in per_transition:
            per_transition[ttype] += area_ha
        total_ha += area_ha
        if province:
            per_province[province] += area_ha

    per_transition = {k: round(v, 1) for k, v in per_transition.items()}
    per_province_list = [
        {"province": p, "deforestation_ha": round(per_province.get(p, 0.0), 1)}
        for p in PROVINCES
    ]
    # Tambahkan provinsi yang tidak ada di list canonical (jika ada di data)
    for p, v in per_province.items():
        if p not in PROVINCES:
            per_province_list.append({"province": p, "deforestation_ha": round(v, 1)})

    return {
        "total_deforestation_ha": round(total_ha, 1),
        "n_hotspots": len(features),
        "per_transition_ha": per_transition,
        "per_province": per_province_list,
    }


def build_statistics_json(
    *,
    period_from: int,
    period_to: int,
    deforestation_geojson: dict[str, Any],
    per_class_area_ha: dict[str, float] | None = None,
    model_metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    out_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build dict ``statistics.json`` sesuai skema PRD §B.1.2.

    Args:
        period_from, period_to: Tahun T1 dan T2.
        deforestation_geojson: FeatureCollection dari ``inference.change_detection``.
        per_class_area_ha: Dict ``{nama_kelas: ha}`` dari mask T2.
        model_metrics: Dict dengan ``overall_accuracy``, ``mean_iou``, ``per_class``.
        extra: Field opsional tambahan (akan di-merge di root).
        out_path: Bila di-set, simpan ke file.

    Returns:
        Dict statistics lengkap.
    """
    summary = summarize_geojson_transitions(deforestation_geojson)

    if per_class_area_ha is None:
        per_class_area_ha = {name: 0.0 for name in CLASS_NAMES}

    if model_metrics is None:
        model_metrics = {
            "overall_accuracy": 0.0,
            "mean_iou": 0.0,
            "per_class": [
                {"class": name, "iou": 0.0, "f1": 0.0} for name in CLASS_NAMES
            ],
        }

    stats: dict[str, Any] = {
        "period_from": int(period_from),
        "period_to": int(period_to),
        "total_deforestation_ha": summary["total_deforestation_ha"],
        "n_hotspots": summary["n_hotspots"],
        "per_transition_ha": summary["per_transition_ha"],
        "per_province": summary["per_province"],
        "per_class_area_ha": per_class_area_ha,
        "model_metrics": model_metrics,
    }
    if extra:
        stats.update(extra)
    if out_path is not None:
        save_json(stats, out_path)
    return stats


def save_metrics_json(
    metrics: dict[str, Any],
    out_path: str | os.PathLike[str],
) -> Path:
    """Simpan dict metric (dari ``training.metrics.metric_summary``) ke file."""
    return save_json(metrics, out_path)
