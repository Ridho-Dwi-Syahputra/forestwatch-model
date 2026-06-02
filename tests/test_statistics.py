"""Test statistics builder."""

from __future__ import annotations

import json

from forestwatch.outputs.statistics import build_statistics_json, summarize_geojson_transitions


def test_summarize_aggregates_per_transition(sample_feature_collection):
    summary = summarize_geojson_transitions(sample_feature_collection)
    pt = summary["per_transition_ha"]
    # Dari fixture: 2 sawit (12.4 + 20.5), 1 pertanian (5.1), 1 lahan terbuka (2.8), 1 tambang (1.0)
    assert pt["hutan_ke_sawit"] == 32.9
    assert pt["hutan_ke_pertanian_lain"] == 5.1
    assert pt["hutan_ke_lahan_terbuka"] == 2.8
    assert pt["hutan_ke_tambang"] == 1.0
    assert summary["n_hotspots"] == 5
    assert summary["total_deforestation_ha"] == 41.8


def test_summarize_per_province(sample_feature_collection):
    summary = summarize_geojson_transitions(sample_feature_collection)
    # Convert ke dict untuk akses mudah
    pp = {row["province"]: row["deforestation_ha"] for row in summary["per_province"]}
    assert pp["Papua Selatan"] == 17.5  # 12.4 + 5.1
    assert pp["Papua Tengah"] == 2.8
    assert pp["Papua Barat"] == 1.0
    assert pp["Papua"] == 20.5


def test_build_statistics_json_full_schema(tmp_path, sample_feature_collection, sample_metrics):
    out = tmp_path / "statistics.json"
    stats = build_statistics_json(
        period_from=2021,
        period_to=2024,
        deforestation_geojson=sample_feature_collection,
        per_class_area_ha={
            "Perairan": 100.0,
            "Hutan": 1_000_000.0,
            "Deforestasi": 5000.0,
            "Sawit": 2500.0,
            "Pertanian Lain": 3000.0,
            "Tambang": 50.0,
        },
        model_metrics=sample_metrics,
        out_path=out,
    )
    assert out.exists()
    loaded = json.loads(out.read_text())

    # Field wajib PRD §B.1.2
    for k in (
        "period_from",
        "period_to",
        "total_deforestation_ha",
        "n_hotspots",
        "per_transition_ha",
        "per_province",
        "per_class_area_ha",
        "model_metrics",
    ):
        assert k in loaded
    assert loaded["period_from"] == 2021
    assert loaded["period_to"] == 2024
    assert loaded["n_hotspots"] == 5
    assert loaded["model_metrics"]["mean_iou"] == 0.71
    assert len(loaded["model_metrics"]["per_class"]) == 7


def test_build_statistics_json_default_metrics(sample_feature_collection):
    stats = build_statistics_json(
        period_from=2021,
        period_to=2024,
        deforestation_geojson=sample_feature_collection,
    )
    # Default metrics adalah zeros
    assert stats["model_metrics"]["mean_iou"] == 0.0
    assert stats["model_metrics"]["overall_accuracy"] == 0.0
