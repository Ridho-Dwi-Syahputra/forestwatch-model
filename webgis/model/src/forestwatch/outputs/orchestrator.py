"""Orkestrasi: dari folder mask + metrik → 7 file kontrak siap kirim ke Orang 2.

Adalah entry point yang dipanggil dari notebook 04 atau ``scripts/generate_outputs.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from forestwatch.constants import OUTPUT_FILES
from forestwatch.inference.change_detection import detect_transitions
from forestwatch.outputs.landcover_png import (
    compute_per_class_area_ha_from_geotiff,
    mosaic_masks_to_png,
)
from forestwatch.outputs.legend import build_legend_json
from forestwatch.outputs.model_card import render_model_card
from forestwatch.outputs.statistics import build_statistics_json, save_metrics_json
from forestwatch.utils.io import load_json
from forestwatch.utils.logging import get_logger

_logger = get_logger("forestwatch.outputs.orchestrator")


def generate_all_outputs(
    *,
    mask_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    period_from: int = 2021,
    period_to: int = 2025,
    metrics: dict[str, Any] | None = None,
    model_card_kwargs: dict[str, Any] | None = None,
    mask_t1_prefix: str = "mask_t1_",
    mask_t2_prefix: str = "mask_t2_",
    min_area_ha: float = 0.5,
    onnx_src: str | os.PathLike[str] | None = None,
) -> dict[str, Path]:
    """Generate 7 file kontrak (PRD §B.1) ke ``out_dir``.

    Workflow:
      1. Mosaic mask T2 → ``landcover_2025.png`` + bounds
      2. Mosaic mask T1 → ``landcover_2021.png`` + bounds
      3. Deteksi 4 transisi → ``deforestation.geojson``
      4. Hitung per_class_area_ha dari T2
      5. Build ``statistics.json`` (gabung dengan ``metrics``)
      6. Build ``legend.json``
      7. Tulis ``metrics.json``
      8. Tulis ``model_card.md``
      9. Copy ``model.onnx`` jika diberikan

    Returns:
        Dict ``{output_key: path}`` untuk semua file yang dihasilkan.
    """
    import shutil  # noqa: PLC0415

    mask_dir_p = Path(mask_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1 & 2. Mosaic PNG + bounds
    t2_files = sorted(mask_dir_p.glob(f"{mask_t2_prefix}*.tif"))
    t1_files = sorted(mask_dir_p.glob(f"{mask_t1_prefix}*.tif"))
    if not t2_files:
        raise FileNotFoundError(f"Tidak ada '{mask_t2_prefix}*.tif' di {mask_dir_p}.")
    if not t1_files:
        raise FileNotFoundError(f"Tidak ada '{mask_t1_prefix}*.tif' di {mask_dir_p}.")

    paths["landcover_t2_png"], paths["landcover_t2_bounds"] = mosaic_masks_to_png(
        t2_files,
        out_dir_p / OUTPUT_FILES["landcover_t2_png"],
        out_dir_p / OUTPUT_FILES["landcover_t2_bounds"],
    )
    paths["landcover_t1_png"], paths["landcover_t1_bounds"] = mosaic_masks_to_png(
        t1_files,
        out_dir_p / OUTPUT_FILES["landcover_t1_png"],
        out_dir_p / OUTPUT_FILES["landcover_t1_bounds"],
    )

    # 3. Change detection → GeoJSON (asumsi t1 & t2 di folder yang sama dengan prefix berbeda)
    geojson_path = out_dir_p / OUTPUT_FILES["deforestation_geojson"]
    fc = detect_transitions(
        t1_dir=mask_dir_p,
        t2_dir=mask_dir_p,
        out_geojson=geojson_path,
        t1_prefix=mask_t1_prefix,
        t2_prefix=mask_t2_prefix,
        period_from=period_from,
        period_to=period_to,
        min_area_ha=min_area_ha,
    )
    paths["deforestation_geojson"] = geojson_path

    # 4. Per-class area dari merged T2 (mosaic ulang via rasterio sederhana untuk hitung)
    per_class_area = _aggregate_per_class_area(t2_files)

    # 5. Statistics
    stats_path = out_dir_p / OUTPUT_FILES["statistics_json"]
    build_statistics_json(
        period_from=period_from,
        period_to=period_to,
        deforestation_geojson=fc,
        per_class_area_ha=per_class_area,
        model_metrics=metrics,
        out_path=stats_path,
    )
    paths["statistics_json"] = stats_path

    # 6. Legend
    legend_path = out_dir_p / OUTPUT_FILES["legend_json"]
    build_legend_json(out_path=legend_path)
    paths["legend_json"] = legend_path

    # 7. Metrics
    if metrics is not None:
        metrics_path = out_dir_p / OUTPUT_FILES["metrics_json"]
        save_metrics_json(metrics, metrics_path)
        paths["metrics_json"] = metrics_path

    # 8. Model card
    card_kwargs = dict(model_card_kwargs or {})
    card_kwargs.setdefault("metrics", metrics or {})
    card_path = out_dir_p / OUTPUT_FILES["model_card"]
    render_model_card(card_path, **card_kwargs)
    paths["model_card"] = card_path

    # 9. ONNX (copy bila diberikan)
    if onnx_src is not None:
        onnx_dst = out_dir_p / OUTPUT_FILES["model_onnx"]
        shutil.copy2(onnx_src, onnx_dst)
        paths["model_onnx"] = onnx_dst

    _logger.info("Generated %d file di %s: %s", len(paths), out_dir_p, list(paths.keys()))
    return paths


def _aggregate_per_class_area(t2_files: list[Path]) -> dict[str, float]:
    """Sum per-class area dari banyak ubin mask T2."""
    from collections import defaultdict  # noqa: PLC0415

    agg: dict[str, float] = defaultdict(float)
    for f in t2_files:
        per = compute_per_class_area_ha_from_geotiff(f)
        for name, ha in per.items():
            agg[name] += ha
    return {k: round(v, 1) for k, v in agg.items()}


def generate_outputs_from_geojson(
    *,
    deforestation_geojson_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    period_from: int = 2021,
    period_to: int = 2025,
    per_class_area_ha: dict[str, float] | None = None,
    metrics: dict[str, Any] | None = None,
    model_card_kwargs: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Alternatif: rebuild statistics + legend + model_card dari GeoJSON yang sudah ada.

    Cocok jika user sudah punya ``deforestation.geojson`` (mis. dummy generator).
    """
    fc = load_json(deforestation_geojson_path)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    stats_path = out_dir_p / OUTPUT_FILES["statistics_json"]
    build_statistics_json(
        period_from=period_from,
        period_to=period_to,
        deforestation_geojson=fc,
        per_class_area_ha=per_class_area_ha,
        model_metrics=metrics,
        out_path=stats_path,
    )
    paths["statistics_json"] = stats_path
    legend_path = out_dir_p / OUTPUT_FILES["legend_json"]
    build_legend_json(out_path=legend_path)
    paths["legend_json"] = legend_path

    card_kwargs = dict(model_card_kwargs or {})
    card_kwargs.setdefault("metrics", metrics or {})
    paths["model_card"] = render_model_card(
        out_dir_p / OUTPUT_FILES["model_card"], **card_kwargs
    )
    if metrics is not None:
        paths["metrics_json"] = save_metrics_json(
            metrics, out_dir_p / OUTPUT_FILES["metrics_json"]
        )
    return paths
