"""Test deteksi perubahan (versi numpy + versi file/tile)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from forestwatch.inference.change_detection import detect_transitions, detect_transitions_from_arrays

rasterio = pytest.importorskip("rasterio")


def test_detects_all_transitions(sample_mask_t1, sample_mask_t2, sample_transform):
    features = detect_transitions_from_arrays(
        sample_mask_t1,
        sample_mask_t2,
        sample_transform,
        min_area_ha=0.0,  # tidak filter area di test
    )
    # Setidaknya ada feature dari masing-masing 5 transisi (termasuk Permukiman)
    ttypes = {f["properties"]["transition_type"] for f in features}
    assert "hutan_ke_lahan_terbuka" in ttypes
    assert "hutan_ke_sawit" in ttypes
    assert "hutan_ke_pertanian_lain" in ttypes
    assert "hutan_ke_tambang" in ttypes
    assert "hutan_ke_permukiman" in ttypes


def test_skips_when_no_change():
    mask = np.ones((10, 10), dtype=np.uint8)  # all forest
    from rasterio.transform import from_bounds

    transform = from_bounds(0, 0, 1, 1, 10, 10)
    features = detect_transitions_from_arrays(mask, mask, transform, min_area_ha=0.0)
    assert features == []


def test_only_source_class_forest_counted():
    """Perubahan non-forest -> X tidak dihitung."""
    t1 = np.full((10, 10), 4, dtype=np.uint8)  # all pertanian lain
    t2 = np.full((10, 10), 3, dtype=np.uint8)  # all sawit
    from rasterio.transform import from_bounds

    transform = from_bounds(0, 0, 1, 1, 10, 10)
    features = detect_transitions_from_arrays(t1, t2, transform, min_area_ha=0.0)
    # Tidak ada feature - bukan dari Hutan
    assert features == []


def test_min_area_filter():
    """Polygon < min_area_ha harus dibuang."""
    t1 = np.ones((4, 4), dtype=np.uint8)
    t2 = np.ones((4, 4), dtype=np.uint8)
    t2[0, 0] = 3  # 1 pixel berubah -> sawit
    from rasterio.transform import from_bounds

    # Bbox kecil: 1 pixel = (0.0001 * 111km)^2 = ~123 m^2 = 0.0123 ha
    transform = from_bounds(140, -8, 140.0004, -7.9996, 4, 4)

    # Tanpa filter
    features_no_filter = detect_transitions_from_arrays(t1, t2, transform, min_area_ha=0.0)
    assert len(features_no_filter) == 1

    # Dengan filter besar
    features_filtered = detect_transitions_from_arrays(t1, t2, transform, min_area_ha=10.0)
    assert features_filtered == []


def test_feature_schema_matches_prd(sample_mask_t1, sample_mask_t2, sample_transform):
    features = detect_transitions_from_arrays(
        sample_mask_t1,
        sample_mask_t2,
        sample_transform,
        min_area_ha=0.0,
        period_from=2021,
        period_to=2024,
        province="Papua Selatan",
    )
    assert features  # ada minimal 1
    for feat in features:
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] in ("Polygon", "MultiPolygon")
        p = feat["properties"]
        assert p["id"].startswith("DF-")
        assert p["transition_type"].startswith("hutan_ke_")
        assert isinstance(p["area_ha"], float)
        assert p["area_ha"] > 0
        assert p["period_from"] == 2021
        assert p["period_to"] == 2024
        assert p["province"] == "Papua Selatan"


def test_unique_ids(sample_mask_t1, sample_mask_t2, sample_transform):
    features = detect_transitions_from_arrays(
        sample_mask_t1, sample_mask_t2, sample_transform, min_area_ha=0.0
    )
    ids = [f["properties"]["id"] for f in features]
    assert len(ids) == len(set(ids))


def _write_mask_tif(path, data, transform, crs="EPSG:4326"):
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype=str(data.dtype), crs=crs, transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_detect_transitions_pasangan_shard_beda_ukuran(tmp_path):
    """Reproduksi bug nyata: tile T1 & T2 logis SAMA tapi shard GEE-nya beda ukuran
    (mis. T1=2021 [6 band] di-shard GEE jadi 13568px, T2=2025 [6+label band] jadi 12544px)
    -- detect_transitions() dulu nge-crash ValueError shape mismatch krn baca .read(1) langsung
    tanpa selaraskan grid. Sekarang harus mosaic-merge per tile logis dgn bounds+res sama dulu,
    BUKAN crash."""
    from rasterio.transform import from_origin

    t1_dir, t2_dir = tmp_path / "t1", tmp_path / "t2"
    t1_dir.mkdir()
    t2_dir.mkdir()

    res = 0.01
    # T1 tile "00": 1 shard, 5x5 px, full forest -- cakupan LEBIH BESAR dari shard T2.
    t1_mask = np.ones((5, 5), dtype="uint8")
    _write_mask_tif(
        t1_dir / "mask_papua_t1_tile_00-0000000000-0000000000.tif",
        t1_mask, from_origin(100.0, -4.95, res, res),
    )
    # T2 tile "00": 1 shard, 4x4 px (LEBIH KECIL -- shard GEE beda batas dari T1), dgn
    # perubahan Hutan->Sawit (kelas 3) di sebagian areanya.
    t2_mask = np.ones((4, 4), dtype="uint8")
    t2_mask[0:2, 0:2] = 3
    _write_mask_tif(
        t2_dir / "mask_papua_t2_tile_00-0000000000-0000000000.tif",
        t2_mask, from_origin(100.0, -4.96, res, res),
    )

    out_geojson = tmp_path / "deforestation_test.geojson"
    fc = detect_transitions(
        t1_dir=t1_dir, t2_dir=t2_dir, out_geojson=out_geojson,
        t1_prefix="mask_papua_t1_tile_", t2_prefix="mask_papua_t2_tile_",
        min_area_ha=0.0,
    )

    assert out_geojson.exists()
    ttypes = {f["properties"]["transition_type"] for f in fc["features"]}
    assert "hutan_ke_sawit" in ttypes


def test_detect_transitions_tile_tanpa_pasangan_di_skip(tmp_path):
    """Tile yg cuma ada di T1 (tak ada file T2 dgn indeks sama) harus di-skip dgn warning,
    bukan crash -- bukti yg dites: `features` kosong tetap valid (proses selesai normal)."""
    from rasterio.transform import from_origin

    t1_dir, t2_dir = tmp_path / "t1", tmp_path / "t2"
    t1_dir.mkdir()
    t2_dir.mkdir()

    res = 0.01
    t1_mask = np.ones((3, 3), dtype="uint8")
    _write_mask_tif(
        t1_dir / "mask_papua_t1_tile_00-0000000000-0000000000.tif",
        t1_mask, from_origin(100.0, -4.95, res, res),
    )
    # T2 sengaja punya tile index BEDA ("01", bukan "00") -- tak ada pasangan utk tile "00".
    t2_mask = np.ones((3, 3), dtype="uint8")
    _write_mask_tif(
        t2_dir / "mask_papua_t2_tile_01-0000000000-0000000000.tif",
        t2_mask, from_origin(100.0, -4.95, res, res),
    )

    out_geojson = tmp_path / "deforestation_test.geojson"
    fc = detect_transitions(
        t1_dir=t1_dir, t2_dir=t2_dir, out_geojson=out_geojson,
        t1_prefix="mask_papua_t1_tile_", t2_prefix="mask_papua_t2_tile_",
        min_area_ha=0.0,
    )
    assert fc["features"] == []
    assert json.loads(out_geojson.read_text())["features"] == []
