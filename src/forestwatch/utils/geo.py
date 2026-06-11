"""Geometri helper — konversi area, bounds, dll.

Konvensi:
- Koordinat: longitude pertama (EPSG:4326).
- Bounds Leaflet: ``[[south, west], [north, east]]`` (latitude pertama).
"""

from __future__ import annotations

import math
from typing import Any

from forestwatch.constants import DEG_TO_M, SQM_PER_HA


def area_ha_from_polygon(polygon: Any, *, latitude_hint: float | None = None) -> float:
    """Hitung luas polygon Shapely dalam hektar (EPSG:4326 → meter approx).

    Aproksimasi yang sama dengan PRD §A.5 Cell 9::

        area_ha = polygon.area * (111_000 ** 2) / 10_000

    Bila ``latitude_hint`` diberikan, koreksi lintang diterapkan
    (1° lon ≈ 111 km × cos(lat)). Untuk presisi tinggi reproject ke UTM dahulu.
    """
    deg_sq = float(polygon.area)
    if latitude_hint is None:
        return deg_sq * (DEG_TO_M**2) / SQM_PER_HA
    lat_rad = math.radians(latitude_hint)
    m_per_deg_lon = DEG_TO_M * math.cos(lat_rad)
    m_per_deg_lat = DEG_TO_M
    return deg_sq * m_per_deg_lon * m_per_deg_lat / SQM_PER_HA


def bounds_to_leaflet(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> list[list[float]]:
    """Konversi (minLon, minLat, maxLon, maxLat) → ``[[south, west], [north, east]]``.

    Format ini adalah yang dipakai ``L.imageOverlay(url, bounds)``.
    """
    return [[float(min_lat), float(min_lon)], [float(max_lat), float(max_lon)]]


def bounds_to_leaflet_from_transform(
    height: int, width: int, transform: Any
) -> list[list[float]]:
    """Hitung bounds Leaflet dari rasterio ``Affine`` transform + shape.

    Lazy import ``rasterio.transform`` agar fungsi tidak fail tanpa rasterio.
    """
    from rasterio.transform import array_bounds  # noqa: PLC0415

    minx, miny, maxx, maxy = array_bounds(height, width, transform)
    return bounds_to_leaflet(minx, miny, maxx, maxy)


def bbox_to_polygon_coords(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> list[list[list[float]]]:
    """BBox → koordinat polygon GeoJSON (ring tunggal, tertutup)."""
    return [
        [
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]
    ]


def pixel_area_ha(scale_m: float = 10.0) -> float:
    """Luas 1 piksel dalam hektar untuk resolusi ``scale_m`` meter."""
    return (scale_m * scale_m) / SQM_PER_HA
