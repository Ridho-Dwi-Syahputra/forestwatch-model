"""Test landcover PNG rendering."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

rasterio = pytest.importorskip("rasterio")

from forestwatch.outputs.landcover_png import (
    compute_per_class_area_ha,
    mask_to_rgb,
    mosaic_masks_to_png,
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


def _write_fake_mask_tile(path, data, transform, crs="EPSG:4326"):
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype=str(data.dtype), crs=crs, transform=transform,
    ) as dst:
        dst.write(data, 1)


def _make_2x2_tile_grid(tmp_path, tile_px=2, res=0.01):
    """4 tile bertetangga (grid 2x2), masing-masing tile_px x tile_px piksel."""
    from rasterio.transform import from_origin

    tile_files = []
    for i in range(2):
        for j in range(2):
            transform = from_origin(
                100 + i * tile_px * res, 0 - j * tile_px * res, res, res
            )
            data = np.full((tile_px, tile_px), i * 2 + j, dtype="uint8")
            p = tmp_path / f"mask_{i}_{j}.tif"
            _write_fake_mask_tile(p, data, transform)
            tile_files.append(p)
    return tile_files


def test_mosaic_masks_to_png_tanpa_downsample_saat_kecil(tmp_path):
    tile_files = _make_2x2_tile_grid(tmp_path)

    png_path, bounds_path = mosaic_masks_to_png(
        tile_files, tmp_path / "lc.png", tmp_path / "lc_bounds.json", max_dim_px=4096
    )
    assert png_path.exists() and bounds_path.exists()
    img = Image.open(png_path)
    # grid 2x2 tile @ 2x2 px -> mosaic native 4x4 px, jauh di bawah max_dim_px -> tak disusutkan
    assert img.size == (4, 4)


def test_mosaic_masks_to_png_downsample_saat_terlalu_besar(tmp_path):
    """Reproduksi inti bug OOM: mosaic native >> max_dim_px harus disusutkan, BUKAN
    direkonstruksi penuh di memori (lihat docstring mosaic_masks_to_png)."""
    tile_files = _make_2x2_tile_grid(tmp_path)

    png_path, _ = mosaic_masks_to_png(
        tile_files, tmp_path / "lc2.png", tmp_path / "lc2_bounds.json", max_dim_px=2
    )
    img = Image.open(png_path)
    # mosaic native 4x4 px, max_dim_px=2 -> dimensi terbesar hasil harus disusutkan mendekati 2
    assert max(img.size) <= 3


def test_compute_per_class_area_ha():
    # 10x10 mask, semua hutan
    mask = np.ones((10, 10), dtype=np.uint8)
    areas = compute_per_class_area_ha(mask, pixel_area_ha=0.01)
    # 100 piksel * 0.01 ha = 1.0 ha untuk hutan
    assert areas["Hutan"] == 1.0
    assert areas["Perairan"] == 0.0
    # Semua nama kelas hadir
    assert len(areas) == 7
