"""Test schema validator + end-to-end dummy generator → validate."""

from __future__ import annotations

import json

import pytest

from forestwatch.validation.schema import (
    ValidationError,
    validate_bounds_json,
    validate_deforestation_geojson,
    validate_legend_json,
    validate_outputs_dir,
    validate_statistics_json,
)


def test_legend_valid():
    legend = [
        {"id": 0, "name": "Perairan", "color": "#2A6FDB"},
        {"id": 1, "name": "Hutan", "color": "#0B3D0B"},
        {"id": 2, "name": "Deforestasi", "color": "#E03B24"},
        {"id": 3, "name": "Sawit", "color": "#F97316"},
        {"id": 4, "name": "Pertanian Lain", "color": "#E9C46A"},
        {"id": 5, "name": "Tambang", "color": "#8E24AA"},
    ]
    report = validate_legend_json(legend)
    assert report.ok


def test_legend_missing_field():
    bad = [{"id": 0, "name": "X"}]  # missing color
    report = validate_legend_json(bad)
    assert not report.ok
    assert any("color" in e for e in report.errors)


def test_legend_duplicate_id():
    bad = [
        {"id": 0, "name": "A", "color": "#000000"},
        {"id": 0, "name": "B", "color": "#FFFFFF"},
    ] + [{"id": i, "name": f"X{i}", "color": "#123456"} for i in range(1, 5)]
    report = validate_legend_json(bad)
    assert not report.ok
    assert any("duplikat" in e for e in report.errors)


def test_statistics_missing_top_level():
    bad = {"period_from": 2021}  # banyak yang missing
    report = validate_statistics_json(bad)
    assert not report.ok
    assert len(report.errors) >= 5


def test_geojson_invalid_transition_type():
    bad = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
                "properties": {
                    "id": "DF-00001",
                    "transition_type": "invalid_type",
                    "area_ha": 1.0,
                    "period_from": 2021,
                    "period_to": 2024,
                },
            }
        ],
    }
    report = validate_deforestation_geojson(bad)
    assert not report.ok
    assert any("transition_type" in e for e in report.errors)


def test_bounds_json_valid():
    bounds = {"bounds": [[-5.0, 100.0], [5.0, 110.0]], "crs": "EPSG:4326"}
    report = validate_bounds_json(bounds)
    assert report.ok


def test_bounds_json_invalid_shape():
    bad = {"bounds": [[-5.0, 100.0]]}  # only 1 corner
    report = validate_bounds_json(bad)
    assert not report.ok


# ============================================================================
# End-to-end: dummy generator -> validator
# ============================================================================


def test_dummy_generator_produces_valid_outputs(tmp_path):
    """E2E test: jalankan dummy generator, lalu validate. Harus pass tanpa error."""
    from forestwatch.cli.dummy import main as dummy_main

    out_dir = tmp_path / "dummy"
    rc = dummy_main(
        [
            "--out", str(out_dir),
            "--n-polygons", "20",
            "--seed", "123",
            "--png-size", "64",  # kecil untuk speed
        ]
    )
    assert rc == 0

    # Validasi
    report = validate_outputs_dir(out_dir)
    print(report.render())
    assert report.ok, f"Validator gagal:\n{report.render()}"


def test_validate_dir_strict_raises(tmp_path):
    # Folder kosong -> error
    with pytest.raises(ValidationError):
        validate_outputs_dir(tmp_path, strict=True)


def test_validate_dir_missing_required(tmp_path):
    # Folder ada tapi tidak ada file
    report = validate_outputs_dir(tmp_path)
    assert not report.ok
    # Setidaknya ada error tentang file wajib hilang
    assert any("wajib hilang" in e for e in report.errors)


def test_validate_dummy_files_match_required(tmp_path):
    from forestwatch.cli.dummy import main as dummy_main

    out_dir = tmp_path / "dummy"
    dummy_main(["--out", str(out_dir), "--n-polygons", "5", "--seed", "1", "--png-size", "32"])

    # Cek 7 file wajib + opsional ada
    files = {p.name for p in out_dir.iterdir()}
    required = {
        "landcover_2021.png",
        "landcover_2021_bounds.json",
        "landcover_2025.png",
        "landcover_2025_bounds.json",
        "deforestation.geojson",
        "statistics.json",
        "legend.json",
    }
    assert required.issubset(files)
    # metrics + model_card (opsional tapi dummy generate)
    assert "metrics.json" in files
    assert "model_card.md" in files


def test_dummy_statistics_internally_consistent(tmp_path):
    """Cross-check: total ha di statistics.json = sum area di geojson."""
    from forestwatch.cli.dummy import main as dummy_main

    out_dir = tmp_path / "dummy"
    dummy_main(["--out", str(out_dir), "--n-polygons", "30", "--seed", "7", "--png-size", "32"])

    fc = json.loads((out_dir / "deforestation.geojson").read_text())
    stats = json.loads((out_dir / "statistics.json").read_text())

    geojson_total = sum(ft["properties"]["area_ha"] for ft in fc["features"])
    stats_total = stats["total_deforestation_ha"]
    # Round ke 1 desimal sesuai konvensi statistics.json
    assert abs(stats_total - round(geojson_total, 1)) < 0.5
