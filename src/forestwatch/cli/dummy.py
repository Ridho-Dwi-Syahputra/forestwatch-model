"""CLI: generate 7 file dummy schema-valid untuk Orang 2 (WebGIS).

Penggunaan:

    fw-dummy --out outputs/dummy --n-polygons 60 --seed 42
    python -m forestwatch.cli.dummy --out outputs/dummy

Output bisa langsung di-copy ke ``docs/webgis/backend/data/`` agar Orang 2
develop UI sebelum data asli dari Orang 1 siap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from forestwatch.constants import (
    CLASS_NAMES,
    CRS_DEFAULT,
    OUTPUT_FILES,
    PALETTE_RGB,
    PAPUA_BBOX,
    PROVINCES,
    TRANSITION_MAP,
)
from forestwatch.outputs.landcover_png import save_mask_array_as_png
from forestwatch.outputs.legend import build_legend_json
from forestwatch.outputs.model_card import render_model_card
from forestwatch.outputs.statistics import build_statistics_json, save_metrics_json
from forestwatch.utils.geo import bbox_to_polygon_coords
from forestwatch.utils.io import save_geojson


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fw-dummy",
        description="Generate 7 file dummy schema-valid untuk Orang 2 (WebGIS).",
    )
    p.add_argument("--out", "-o", required=True, help="Folder output.")
    p.add_argument("--n-polygons", "-n", type=int, default=60, help="Jumlah polygon deforestasi.")
    p.add_argument("--seed", "-s", type=int, default=42, help="Random seed.")
    p.add_argument("--period-from", type=int, default=2021)
    p.add_argument("--period-to", type=int, default=2025)
    p.add_argument(
        "--png-size", type=int, default=512, help="Sisi PNG mock landcover (default 512×512)."
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        print("ERROR: butuh numpy. Install: pip install numpy", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)

    # ---- 1 & 2. Buat dummy landcover PNG (gradien acak per cell) ----
    bbox = PAPUA_BBOX
    mask_t1 = _make_dummy_landcover(args.png_size, rng)
    mask_t2 = _erode_some_forest_to_simulate_change(mask_t1, rng, fraction=0.06)

    save_mask_array_as_png(
        mask_t2,
        out_dir / OUTPUT_FILES["landcover_t2_png"],
        bbox=bbox,
        out_bounds_json=out_dir / OUTPUT_FILES["landcover_t2_bounds"],
    )
    save_mask_array_as_png(
        mask_t1,
        out_dir / OUTPUT_FILES["landcover_t1_png"],
        bbox=bbox,
        out_bounds_json=out_dir / OUTPUT_FILES["landcover_t1_bounds"],
    )

    # ---- 3. Dummy deforestation.geojson ----
    fc = _make_dummy_geojson(
        n=args.n_polygons,
        bbox=bbox,
        rng=rng,
        period_from=args.period_from,
        period_to=args.period_to,
    )
    save_geojson(fc, out_dir / OUTPUT_FILES["deforestation_geojson"])

    # ---- 4. Per-class area dari mask T2 (numpy bincount) ----
    pixel_area_ha = _approx_pixel_area_ha(bbox, mask_t2.shape)
    per_class_area_ha = _per_class_area_from_mask(mask_t2, pixel_area_ha)

    # ---- 5. Dummy model metrics (realistic per Master Plan target) ----
    from forestwatch.training.metrics import cohen_kappa  # noqa: PLC0415

    _cm = _dummy_confusion_matrix(rng)
    dummy_metrics = {
        "overall_accuracy": 0.86,
        "mean_iou": 0.71,
        "kappa": round(cohen_kappa(np.array(_cm)), 4),
        "per_class": [
            {"class": "Perairan", "iou": 0.91, "f1": 0.95},
            {"class": "Hutan", "iou": 0.94, "f1": 0.97},
            {"class": "Deforestasi", "iou": 0.66, "f1": 0.79},
            {"class": "Sawit", "iou": 0.62, "f1": 0.76},
            {"class": "Pertanian Lain", "iou": 0.71, "f1": 0.83},
            {"class": "Lahan Terbakar", "iou": 0.55, "f1": 0.71},
        ],
        "confusion_matrix": _cm,
    }

    # ---- 6. Statistics ----
    build_statistics_json(
        period_from=args.period_from,
        period_to=args.period_to,
        deforestation_geojson=fc,
        per_class_area_ha=per_class_area_ha,
        model_metrics=dummy_metrics,
        out_path=out_dir / OUTPUT_FILES["statistics_json"],
    )

    # ---- 7. Legend ----
    build_legend_json(out_path=out_dir / OUTPUT_FILES["legend_json"])

    # ---- 8. Metrics ----
    save_metrics_json(dummy_metrics, out_dir / OUTPUT_FILES["metrics_json"])

    # ---- 9. Model card ----
    render_model_card(
        out_dir / OUTPUT_FILES["model_card"],
        metrics=dummy_metrics,
        n_parameters=32_523_846,
        epochs=50,
        batch_size=8,
        extra_sections={
            "Catatan Dummy": (
                "File ini adalah **dummy generator output** dari "
                "`forestwatch.cli.dummy` — angka belum mencerminkan training nyata."
            )
        },
    )

    print(
        json.dumps(
            {
                "ok": True,
                "out_dir": str(out_dir.resolve()),
                "files": sorted(p.name for p in out_dir.iterdir()),
                "n_features": len(fc["features"]),
            },
            indent=2,
        )
    )
    return 0


# ============================================================================
# Internals
# ============================================================================


def _make_dummy_landcover(size: int, rng: "Any") -> "Any":
    """Buat mask dummy 6 kelas: dominan hutan dengan patch acak kelas lain."""
    import numpy as np  # noqa: PLC0415

    H = W = int(size)
    mask = np.ones((H, W), dtype=np.uint8)  # 1 = Hutan (default)

    # Tabur patch sederhana dengan jumlah piksel kira-kira proporsional
    n_blobs_per_class = {
        0: 6,    # Perairan
        2: 25,   # Deforestasi
        3: 12,   # Sawit
        4: 30,   # Pertanian Lain
        5: 4,    # Lahan Terbakar
    }
    for cls, n_blobs in n_blobs_per_class.items():
        for _ in range(n_blobs):
            cy = int(rng.integers(0, H))
            cx = int(rng.integers(0, W))
            r = int(rng.integers(4, max(5, H // 30)))
            yy, xx = np.ogrid[:H, :W]
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
            mask[disk] = cls
    return mask


def _erode_some_forest_to_simulate_change(
    mask_t1: "Any", rng: "Any", *, fraction: float = 0.05
) -> "Any":
    """Buat mask T2 dengan sebagian Hutan dirubah → Sawit/Pertanian/Lahan Terbuka."""
    import numpy as np  # noqa: PLC0415

    mask = mask_t1.copy()
    forest = mask == 1
    n_forest = int(forest.sum())
    n_change = int(fraction * n_forest)
    if n_change == 0:
        return mask
    fx, fy = np.where(forest)
    indices = rng.choice(len(fx), size=n_change, replace=False)
    target_classes = rng.choice([2, 3, 4, 5], size=n_change, p=[0.25, 0.4, 0.3, 0.05])
    mask[fx[indices], fy[indices]] = target_classes.astype(np.uint8)
    return mask


def _make_dummy_geojson(
    *,
    n: int,
    bbox: tuple[float, float, float, float],
    rng: "Any",
    period_from: int,
    period_to: int,
) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415

    min_lon, min_lat, max_lon, max_lat = bbox
    transition_names = list(TRANSITION_MAP.values())
    # Bobot transisi: sawit dominan
    weights = np.array([0.25, 0.45, 0.20, 0.10])
    weights = weights / weights.sum()
    province_weights = np.array(
        [
            0.08,  # Papua
            0.50,  # Papua Selatan (food estate Merauke)
            0.15,  # Papua Tengah
            0.05,  # Papua Pegunungan
            0.12,  # Papua Barat
            0.10,  # Papua Barat Daya
        ]
    )
    province_weights = province_weights / province_weights.sum()

    features: list[dict[str, Any]] = []
    for i in range(n):
        # Centroid bias: Papua Selatan (sekitar Merauke -8.5, 140.4)
        if rng.random() < 0.5:
            cy = -8.5 + float(rng.normal(0, 0.4))
            cx = 140.4 + float(rng.normal(0, 0.4))
        else:
            cx = float(rng.uniform(min_lon + 0.5, max_lon - 0.5))
            cy = float(rng.uniform(min_lat + 0.5, max_lat - 0.5))
        # Polygon kotak ~0.005°–0.04° di sekitar centroid (≈55m–4.5km)
        size = float(rng.uniform(0.005, 0.04))
        bbox_pol = (cx - size, cy - size, cx + size, cy + size)
        coords = bbox_to_polygon_coords(*bbox_pol)
        ttype = transition_names[int(rng.choice(len(transition_names), p=weights))]
        province = PROVINCES[int(rng.choice(len(PROVINCES), p=province_weights))]
        area_ha = round((2 * size * 111_000) ** 2 / 10_000, 2)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": coords},
                "properties": {
                    "id": f"DF-{i:05d}",
                    "transition_type": ttype,
                    "area_ha": area_ha,
                    "period_from": period_from,
                    "period_to": period_to,
                    "province": province,
                    "kawasan_status": _kawasan_status(province, ttype),
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "name": f"deforestation_{period_from}_{period_to}",
        "crs": {"type": "name", "properties": {"name": CRS_DEFAULT}},
        "features": features,
    }


def _kawasan_status(province: str, ttype: str) -> str:
    if province == "Papua Selatan" and ttype == "hutan_ke_pertanian_lain":
        return "APL / Food Estate Merauke"
    if ttype == "hutan_ke_sawit":
        return "HGU / Perkebunan"
    if ttype == "hutan_ke_terbakar":
        return "Areal terdegradasi"
    return "Areal Penggunaan Lain"


def _approx_pixel_area_ha(
    bbox: tuple[float, float, float, float], shape: tuple[int, int]
) -> float:
    """Aproksimasi luas 1 piksel mask dummy dalam hektar."""
    min_lon, min_lat, max_lon, max_lat = bbox
    H, W = shape
    width_m = (max_lon - min_lon) * 111_000.0
    height_m = (max_lat - min_lat) * 111_000.0
    return (width_m / W) * (height_m / H) / 10_000.0


def _per_class_area_from_mask(mask: "Any", pixel_area_ha: float) -> dict[str, float]:
    import numpy as np  # noqa: PLC0415

    u, c = np.unique(mask, return_counts=True)
    by_id = dict(zip(u.tolist(), c.tolist(), strict=False))
    out: dict[str, float] = {}
    for cls in range(len(CLASS_NAMES)):
        out[CLASS_NAMES[cls]] = round(by_id.get(cls, 0) * pixel_area_ha, 1)
    return out


def _dummy_confusion_matrix(rng: "Any") -> list[list[int]]:
    """Generate confusion matrix dummy 6×6 dengan diagonal kuat."""
    import numpy as np  # noqa: PLC0415

    cm = np.zeros((6, 6), dtype=int)
    diagonal_ranges = [(900, 1000), (4500, 5000), (200, 400), (180, 360), (350, 500), (50, 120)]
    for i, (lo, hi) in enumerate(diagonal_ranges):
        cm[i, i] = int(rng.integers(lo, hi))
        # noise off-diagonal
        for j in range(6):
            if i == j:
                continue
            cm[i, j] = int(rng.integers(0, max(1, cm[i, i] // 20)))
    return cm.tolist()


# Re-export agar palette tetap konsisten (untuk introspection)
__all__ = ["main", "build_parser"]


# Reference (no-op) — pastikan palette tetap ter-import
_ = PALETTE_RGB


if __name__ == "__main__":
    sys.exit(main())
