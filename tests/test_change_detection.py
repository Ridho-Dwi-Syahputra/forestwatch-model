"""Test deteksi perubahan (versi numpy)."""

from __future__ import annotations

import numpy as np

from forestwatch.inference.change_detection import detect_transitions_from_arrays


def test_detects_all_four_transitions(sample_mask_t1, sample_mask_t2, sample_transform):
    features = detect_transitions_from_arrays(
        sample_mask_t1,
        sample_mask_t2,
        sample_transform,
        min_area_ha=0.0,  # tidak filter area di test
    )
    # Setidaknya ada feature dari masing-masing 4 transisi
    ttypes = {f["properties"]["transition_type"] for f in features}
    assert "hutan_ke_lahan_terbuka" in ttypes
    assert "hutan_ke_sawit" in ttypes
    assert "hutan_ke_pertanian_lain" in ttypes
    assert "hutan_ke_tambang" in ttypes


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
