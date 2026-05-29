"""Test utils.geo."""

from __future__ import annotations

import math

from shapely.geometry import Polygon

from forestwatch.utils.geo import (
    area_ha_from_polygon,
    bbox_to_polygon_coords,
    bounds_to_leaflet,
    pixel_area_ha,
)


def test_bounds_to_leaflet():
    # bbox: minLon=100, minLat=-5, maxLon=110, maxLat=5
    bounds = bounds_to_leaflet(100, -5, 110, 5)
    assert bounds == [[-5.0, 100.0], [5.0, 110.0]]
    # Format Leaflet: latitude pertama
    assert bounds[0][0] < bounds[1][0]


def test_area_ha_from_polygon_simple_square():
    # 0.01 degrees square di sekitar equator approx (111m)^2 per degree
    poly = Polygon([(100, 0), (100.01, 0), (100.01, 0.01), (100, 0.01)])
    area_ha = area_ha_from_polygon(poly)
    # Exact: (0.01 * 111000)^2 / 10000 = 1230.1 ha approx
    expected = (0.01 * 111_000) ** 2 / 10_000
    assert math.isclose(area_ha, expected, rel_tol=1e-6)


def test_area_ha_with_latitude_correction():
    poly = Polygon([(100, 0), (100.01, 0), (100.01, 0.01), (100, 0.01)])
    # Pada equator, cos(0) = 1 -> sama dengan default
    a_equator = area_ha_from_polygon(poly, latitude_hint=0.0)
    a_default = area_ha_from_polygon(poly)
    assert math.isclose(a_equator, a_default, rel_tol=1e-6)

    # Pada lintang lebih tinggi, area lebih kecil (cos < 1)
    a_high = area_ha_from_polygon(poly, latitude_hint=60.0)
    assert a_high < a_default
    assert math.isclose(a_high, a_default * math.cos(math.radians(60)), rel_tol=1e-6)


def test_bbox_to_polygon_coords_closed_ring():
    coords = bbox_to_polygon_coords(100, -5, 110, 5)
    assert len(coords) == 1  # single outer ring
    ring = coords[0]
    assert len(ring) == 5  # 4 corners + closure
    assert ring[0] == ring[-1]  # closed
    # Min/max corners
    assert ring[0] == [100, -5]
    assert [110, -5] in ring
    assert [110, 5] in ring


def test_pixel_area_ha_10m():
    assert math.isclose(pixel_area_ha(10), 0.01, rel_tol=1e-9)


def test_pixel_area_ha_30m():
    # 30m * 30m = 900 m^2 = 0.09 ha
    assert math.isclose(pixel_area_ha(30), 0.09, rel_tol=1e-9)
