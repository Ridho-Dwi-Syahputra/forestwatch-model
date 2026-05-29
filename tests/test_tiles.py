"""Test tiles (versi pure-Python tanpa GEE)."""

from __future__ import annotations

import math

from forestwatch.gee.tiles import make_tiles_from_bbox


def test_make_tiles_count():
    tiles = make_tiles_from_bbox((0, 0, 10, 10), nx=3, ny=4)
    assert len(tiles) == 12


def test_tiles_cover_bbox():
    bbox = (0.0, 0.0, 10.0, 10.0)
    tiles = make_tiles_from_bbox(bbox, nx=2, ny=2)
    # tiap tile harus dalam bbox
    for x0, y0, x1, y1 in tiles:
        assert bbox[0] <= x0 < x1 <= bbox[2]
        assert bbox[1] <= y0 < y1 <= bbox[3]
    # Luas total = luas bbox
    total = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in tiles)
    expected = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    assert math.isclose(total, expected, rel_tol=1e-9)


def test_papua_36_tiles():
    from forestwatch.constants import PAPUA_BBOX

    tiles = make_tiles_from_bbox(PAPUA_BBOX, nx=6, ny=6)
    assert len(tiles) == 36
