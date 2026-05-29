"""Test landcover PNG rendering."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from forestwatch.outputs.landcover_png import (
    compute_per_class_area_ha,
    mask_to_rgb,
    save_mask_array_as_png,
)


def test_mask_to_rgb_shape_and_colors():
    mask = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=np.uint8,
    )
    rgb = mask_to_rgb(mask)
    assert rgb.shape == (2, 3, 4)  # RGBA
    assert rgb.dtype == np.uint8
    # Cek warna kelas 0 (perairan biru)
    assert tuple(rgb[0, 0, :3]) == (42, 111, 219)
    # Cek alpha = 255 untuk piksel valid
    assert rgb[0, 0, 3] == 255


def test_mask_to_rgb_background_for_out_of_range():
    mask = np.full((3, 3), 99, dtype=np.uint8)
    rgb = mask_to_rgb(mask)
    # Default background transparan
    assert tuple(rgb[0, 0]) == (0, 0, 0, 0)


def test_save_mask_array_as_png(tmp_path):
    mask = np.array([[1, 2], [3, 0]], dtype=np.uint8)
    png_path, bounds_path = save_mask_array_as_png(
        mask,
        tmp_path / "lc.png",
        bbox=(100, -5, 110, 5),
        out_bounds_json=tmp_path / "lc_bounds.json",
    )
    assert png_path.exists()
    assert bounds_path.exists()

    # Verifikasi PNG bisa di-load
    img = Image.open(png_path)
    assert img.mode == "RGBA"
    assert img.size == (2, 2)

    # Verifikasi bounds format Leaflet
    bounds = json.loads(bounds_path.read_text())
    assert bounds["crs"] == "EPSG:4326"
    assert bounds["bounds"] == [[-5.0, 100.0], [5.0, 110.0]]


def test_save_without_bounds_does_not_create_sidecar(tmp_path):
    mask = np.ones((2, 2), dtype=np.uint8)
    png_path, bounds_path = save_mask_array_as_png(
        mask, tmp_path / "lc.png", bbox=None, out_bounds_json=None
    )
    assert png_path.exists()
    assert bounds_path is None


def test_compute_per_class_area_ha():
    # 10x10 mask, semua hutan
    mask = np.ones((10, 10), dtype=np.uint8)
    areas = compute_per_class_area_ha(mask, pixel_area_ha=0.01)
    # 100 piksel * 0.01 ha = 1.0 ha untuk hutan
    assert areas["Hutan"] == 1.0
    assert areas["Perairan"] == 0.0
    # Semua nama kelas hadir
    assert len(areas) == 6
