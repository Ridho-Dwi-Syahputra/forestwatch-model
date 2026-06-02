"""Test legend builder."""

from __future__ import annotations

import json

from forestwatch.outputs.legend import build_legend_json


def test_build_legend_returns_all_entries():
    legend = build_legend_json()
    assert len(legend) == 7
    ids = [e["id"] for e in legend]
    assert ids == [0, 1, 2, 3, 4, 5, 6]
    names = [e["name"] for e in legend]
    assert "Perairan" in names
    assert "Hutan" in names
    assert "Sawit" in names


def test_build_legend_saves_to_file(tmp_path):
    out = tmp_path / "legend.json"
    legend = build_legend_json(out_path=out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded == legend


def test_legend_color_hex_format():
    legend = build_legend_json()
    for entry in legend:
        assert entry["color"].startswith("#")
        assert len(entry["color"]) == 7
