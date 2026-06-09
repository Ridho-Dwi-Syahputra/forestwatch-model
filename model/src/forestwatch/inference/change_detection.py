"""Deteksi perubahan 4 transisi: Hutan → {Lahan Terbuka, Sawit, Pertanian Lain, Tambang}.

Sumber: PRD §A.5 Cell 9 (blok change detection).

Output: GeoJSON FeatureCollection sesuai skema PRD §B.1.1.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from forestwatch.constants import (
    CRS_DEFAULT,
    SOURCE_CLASS_FOREST,
    TRANSITION_MAP,
)
from forestwatch.utils.geo import area_ha_from_polygon
from forestwatch.utils.io import save_geojson
from forestwatch.utils.logging import get_logger

_logger = get_logger("forestwatch.inference.change")


def detect_transitions_from_arrays(
    mask_t1: "object",
    mask_t2: "object",
    transform: "object",
    *,
    source_class: int = SOURCE_CLASS_FOREST,
    transitions: dict[int, str] | None = None,
    min_area_ha: float = 0.5,
    period_from: int = 2021,
    period_to: int = 2025,
    province: str | None = None,
    id_offset: int = 0,
) -> list[dict[str, Any]]:
    """Deteksi transisi pada sepasang numpy mask (T1, T2) + transform rasterio.

    Args:
        mask_t1, mask_t2: Numpy ``(H, W)`` uint8 mask hasil inferensi.
        transform: Affine transform rasterio (untuk geometri output).
        source_class: Kelas asal transisi (default 1 = Hutan).
        transitions: Map ``{target_class_id: transition_name}``.
        min_area_ha: Threshold luas minimum (buang polygon kecil = noise).
        period_from, period_to: Tahun T1 dan T2 (untuk properties).
        province: Nama provinsi (jika diketahui, untuk properties).
        id_offset: Mulai numbering ID dari nilai ini.

    Returns:
        List feature dict (langsung bisa di-append ke FeatureCollection).
    """
    try:
        import numpy as np  # noqa: PLC0415
        from rasterio.features import shapes  # noqa: PLC0415
        from shapely.geometry import mapping, shape  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Butuh numpy + rasterio + shapely. Install: pip install -e \".[gis]\""
        ) from e

    transitions = transitions or TRANSITION_MAP
    was_source = mask_t1 == source_class

    features: list[dict[str, Any]] = []
    counter = id_offset
    for target_class, transition_name in transitions.items():
        changed = was_source & (mask_t2 == target_class)
        if not changed.any():
            continue
        changed_u8 = changed.astype("uint8")
        for geom, _ in shapes(changed_u8, mask=changed_u8 == 1, transform=transform):
            poly = shape(geom)
            # Sentroid lat untuk koreksi area
            lat_hint = float(poly.centroid.y)
            area = area_ha_from_polygon(poly, latitude_hint=lat_hint)
            if area < min_area_ha:
                continue
            properties: dict[str, Any] = {
                "id": f"DF-{counter:05d}",
                "transition_type": transition_name,
                "area_ha": round(float(area), 2),
                "period_from": int(period_from),
                "period_to": int(period_to),
            }
            if province is not None:
                properties["province"] = province
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": properties,
                }
            )
            counter += 1
    return features


def detect_transitions(
    t1_dir: str | os.PathLike[str],
    t2_dir: str | os.PathLike[str],
    out_geojson: str | os.PathLike[str],
    *,
    t1_prefix: str = "mask_t1_",
    t2_prefix: str = "mask_t2_",
    source_class: int = SOURCE_CLASS_FOREST,
    transitions: dict[int, str] | None = None,
    min_area_ha: float = 0.5,
    period_from: int = 2021,
    period_to: int = 2025,
) -> dict[str, Any]:
    """Iterasi semua ubin T1 ↔ T2, hasilkan GeoJSON deteksi perubahan.

    Konvensi nama: file ``{t1_dir}/{t1_prefix}{X}.tif`` harus berpasangan dengan
    ``{t2_dir}/{t2_prefix}{X}.tif`` (substring ``{X}`` sama).

    Returns:
        FeatureCollection dict yang juga disimpan ke ``out_geojson``.
    """
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("Butuh rasterio. Install: pip install -e \".[gis]\"") from e

    t1_dir_p = Path(t1_dir)
    t2_dir_p = Path(t2_dir)
    t1_files = sorted(t1_dir_p.glob(f"{t1_prefix}*.tif"))
    if not t1_files:
        raise FileNotFoundError(f"Tidak ada '{t1_prefix}*.tif' di {t1_dir_p}.")

    all_features: list[dict[str, Any]] = []
    transitions = transitions or TRANSITION_MAP
    counter = 0

    for t1_path in tqdm(t1_files, desc="Change detection"):
        # Konstruksi pasangan T2: ganti prefix
        base = t1_path.name.replace(t1_prefix, "", 1)
        t2_path = t2_dir_p / f"{t2_prefix}{base}"
        if not t2_path.exists():
            _logger.warning("Pasangan T2 tidak ditemukan untuk %s, di-skip.", t1_path.name)
            continue

        with rasterio.open(t1_path) as s1, rasterio.open(t2_path) as s2:
            m1 = s1.read(1)
            m2 = s2.read(1)
            transform = s1.transform

        features = detect_transitions_from_arrays(
            m1,
            m2,
            transform,
            source_class=source_class,
            transitions=transitions,
            min_area_ha=min_area_ha,
            period_from=period_from,
            period_to=period_to,
            id_offset=counter,
        )
        all_features.extend(features)
        counter += len(features)

    fc: dict[str, Any] = {
        "type": "FeatureCollection",
        "name": f"deforestation_{period_from}_{period_to}",
        "crs": {"type": "name", "properties": {"name": CRS_DEFAULT}},
        "features": all_features,
    }
    save_geojson(fc, out_geojson)
    _logger.info("GeoJSON deteksi perubahan disimpan: %s (%d feature)", out_geojson, len(all_features))
    return fc
