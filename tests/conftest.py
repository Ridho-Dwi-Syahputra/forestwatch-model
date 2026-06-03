"""Pytest fixtures - sample masks & feature collections untuk testing."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_mask_t1():
    """Mask T1 (2021) - sebagian besar hutan."""
    mask = np.ones((100, 100), dtype=np.uint8)  # 1 = Hutan
    mask[10:20, 10:20] = 0  # Perairan
    mask[50:60, 50:60] = 4  # Pertanian Lain (sudah ada di T1)
    return mask


@pytest.fixture
def sample_mask_t2(sample_mask_t1):
    """Mask T2 (2024) - sebagian hutan berubah jadi sawit/pertanian/terbuka/tambang/permukiman."""
    mask = sample_mask_t1.copy()
    mask[30:35, 30:40] = 3  # Hutan -> Sawit
    mask[70:80, 20:30] = 4  # Hutan -> Pertanian Lain
    mask[40:45, 70:80] = 2  # Hutan -> Lahan Terbuka
    mask[80:82, 80:82] = 5  # Hutan -> Tambang
    mask[85:90, 10:15] = 6  # Hutan -> Permukiman
    return mask


@pytest.fixture
def sample_transform():
    """Affine transform untuk patch 100x100 di Papua (around Merauke)."""
    from rasterio.transform import from_bounds

    return from_bounds(140.0, -8.5, 140.1, -8.4, 100, 100)


@pytest.fixture
def sample_feature_collection():
    """Sample FeatureCollection dengan 5 feature lintas 4 transisi."""
    return {
        "type": "FeatureCollection",
        "name": "deforestation_2021_2024",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [140.40, -8.31],
                            [140.45, -8.31],
                            [140.45, -8.28],
                            [140.40, -8.28],
                            [140.40, -8.31],
                        ]
                    ],
                },
                "properties": {
                    "id": "DF-00001",
                    "transition_type": "hutan_ke_sawit",
                    "area_ha": 12.4,
                    "period_from": 2021,
                    "period_to": 2024,
                    "province": "Papua Selatan",
                    "kawasan_status": "HGU / Perkebunan",
                },
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [140.50, -8.20],
                            [140.52, -8.20],
                            [140.52, -8.18],
                            [140.50, -8.18],
                            [140.50, -8.20],
                        ]
                    ],
                },
                "properties": {
                    "id": "DF-00002",
                    "transition_type": "hutan_ke_pertanian_lain",
                    "area_ha": 5.1,
                    "period_from": 2021,
                    "period_to": 2024,
                    "province": "Papua Selatan",
                    "kawasan_status": "APL / Food Estate Merauke",
                },
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [137.10, -3.50],
                            [137.12, -3.50],
                            [137.12, -3.48],
                            [137.10, -3.48],
                            [137.10, -3.50],
                        ]
                    ],
                },
                "properties": {
                    "id": "DF-00003",
                    "transition_type": "hutan_ke_lahan_terbuka",
                    "area_ha": 2.8,
                    "period_from": 2021,
                    "period_to": 2024,
                    "province": "Papua Tengah",
                },
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [135.50, -1.50],
                            [135.51, -1.50],
                            [135.51, -1.49],
                            [135.50, -1.49],
                            [135.50, -1.50],
                        ]
                    ],
                },
                "properties": {
                    "id": "DF-00004",
                    "transition_type": "hutan_ke_tambang",
                    "area_ha": 1.0,
                    "period_from": 2021,
                    "period_to": 2024,
                    "province": "Papua Barat",
                },
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [138.0, -2.0],
                            [138.04, -2.0],
                            [138.04, -1.96],
                            [138.0, -1.96],
                            [138.0, -2.0],
                        ]
                    ],
                },
                "properties": {
                    "id": "DF-00005",
                    "transition_type": "hutan_ke_sawit",
                    "area_ha": 20.5,
                    "period_from": 2021,
                    "period_to": 2024,
                    "province": "Papua",
                },
            },
        ],
    }


@pytest.fixture
def sample_metrics():
    return {
        "overall_accuracy": 0.86,
        "mean_iou": 0.71,
        "per_class": [
            {"class": "Perairan", "iou": 0.91, "f1": 0.95},
            {"class": "Hutan", "iou": 0.94, "f1": 0.97},
            {"class": "Lahan Terbuka", "iou": 0.66, "f1": 0.79},
            {"class": "Sawit", "iou": 0.62, "f1": 0.76},
            {"class": "Pertanian Lain", "iou": 0.71, "f1": 0.83},
            {"class": "Tambang", "iou": 0.55, "f1": 0.71},
            {"class": "Permukiman", "iou": 0.58, "f1": 0.73},
        ],
        "confusion_matrix": [
            [100, 1, 0, 0, 1, 0, 0],
            [2, 5000, 50, 30, 20, 5, 8],
            [0, 40, 200, 10, 5, 2, 1],
            [0, 30, 8, 180, 10, 0, 0],
            [1, 20, 5, 12, 350, 1, 3],
            [0, 5, 2, 1, 2, 60, 0],
            [0, 8, 1, 0, 3, 0, 55],
        ],
    }
