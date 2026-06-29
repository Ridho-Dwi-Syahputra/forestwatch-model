"""Deteksi perubahan 4 transisi: Hutan → {Lahan Terbuka, Sawit, Pertanian Lain, Tambang}.

Sumber: PRD §A.5 Cell 9 (blok change detection).

Output: GeoJSON FeatureCollection sesuai skema PRD §B.1.1.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

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


def _tile_index(filename_after_prefix: str) -> str:
    """``'00-0000000000-0000000000.tif'`` -> ``'00'``; ``'05.tif'`` -> ``'05'``.

    GEE men-shard 1 tile jadi BEBERAPA GeoTIFF kalau ukurannya melewati limit
    ``Export.image.toDrive`` (suffix ``-{x_offset}-{y_offset}`` ditambah otomatis oleh GEE,
    bukan dari kode kita) -- shard ini diidentifikasi lewat angka tile di depan nama file.
    """
    return filename_after_prefix.split("-")[0].split(".")[0]


def _group_shards_by_tile(files: list[Path], prefix: str) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for f in files:
        idx = _tile_index(f.name.replace(prefix, "", 1))
        groups.setdefault(idx, []).append(f)
    return groups


def _union_bounds(bounds_list: list["object"]) -> tuple[float, float, float, float]:
    return (
        min(b.left for b in bounds_list),
        min(b.bottom for b in bounds_list),
        max(b.right for b in bounds_list),
        max(b.top for b in bounds_list),
    )


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
    """Iterasi semua tile T1 ↔ T2, hasilkan GeoJSON deteksi perubahan.

    Konvensi nama: file ``{t1_dir}/{t1_prefix}{X}.tif`` berpasangan dengan
    ``{t2_dir}/{t2_prefix}{X}.tif`` lewat nomor tile di depan ``{X}`` (lihat
    ``_tile_index``) -- BUKAN ``{X}`` penuh, karena GEE bisa men-shard 1 tile logis jadi
    beberapa file dengan batas piksel yang BISA BEDA antar export job (mis. job dengan band
    lebih banyak di-shard lebih kecil). Tiap tile logis di-mosaic dulu (shard T1 & shard T2
    SENDIRI-SENDIRI, dipaksa ``bounds``+``res`` yang SAMA) sebelum dibandingkan -- supaya
    array yang di-diff PASTI selaras piksel, bukan asumsi shape sama
    (lihat riwayat bug nyata: ``ValueError: operands could not be broadcast ... (13568,13568)
    (12544,12544)`` saat T1=2021 [6 band, shard 13568px] dipasangkan langsung dgn T2=2025
    [6+label band, shard 12544px]).

    Returns:
        FeatureCollection dict yang juga disimpan ke ``out_geojson``.
    """
    try:
        import rasterio  # noqa: PLC0415
        from rasterio.merge import merge  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("Butuh rasterio. Install: pip install -e \".[gis]\"") from e

    t1_dir_p = Path(t1_dir)
    t2_dir_p = Path(t2_dir)
    t1_files = sorted(t1_dir_p.glob(f"{t1_prefix}*.tif"))
    t2_files = sorted(t2_dir_p.glob(f"{t2_prefix}*.tif"))
    if not t1_files:
        raise FileNotFoundError(f"Tidak ada '{t1_prefix}*.tif' di {t1_dir_p}.")
    if not t2_files:
        raise FileNotFoundError(f"Tidak ada '{t2_prefix}*.tif' di {t2_dir_p}.")

    t1_groups = _group_shards_by_tile(t1_files, t1_prefix)
    t2_groups = _group_shards_by_tile(t2_files, t2_prefix)

    common_tiles = sorted(set(t1_groups) & set(t2_groups))
    missing = sorted(set(t1_groups) - set(t2_groups))
    for idx in missing:
        _logger.warning("Tile %s tak ada pasangan T2, di-skip.", idx)

    all_features: list[dict[str, Any]] = []
    transitions = transitions or TRANSITION_MAP
    counter = 0

    for tile_idx in tqdm(common_tiles, desc="Change detection"):
        srcs1 = [rasterio.open(f) for f in t1_groups[tile_idx]]
        srcs2 = [rasterio.open(f) for f in t2_groups[tile_idx]]
        try:
            bounds = _union_bounds([s.bounds for s in srcs1] + [s.bounds for s in srcs2])
            res = srcs1[0].res
            mosaic1, transform = merge(srcs1, bounds=bounds, res=res)
            mosaic2, _ = merge(srcs2, bounds=bounds, res=res)
        finally:
            for s in srcs1 + srcs2:
                s.close()

        features = detect_transitions_from_arrays(
            mosaic1[0],
            mosaic2[0],
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
