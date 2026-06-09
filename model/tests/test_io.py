"""Test utils.io."""

from __future__ import annotations

import json

import numpy as np
import pytest

from forestwatch.utils.io import load_json, save_geojson, save_json, save_npz


def test_save_json_creates_parents(tmp_path):
    out = tmp_path / "nested" / "deep" / "file.json"
    saved = save_json({"a": 1, "b": [1, 2]}, out)
    assert saved.exists()
    assert load_json(saved) == {"a": 1, "b": [1, 2]}


def test_save_json_serializes_numpy(tmp_path):
    out = tmp_path / "metrics.json"
    data = {"iou": np.float32(0.71), "arr": np.array([1, 2, 3])}
    save_json(data, out)
    loaded = load_json(out)
    assert loaded["iou"] == pytest.approx(0.71)
    assert loaded["arr"] == [1, 2, 3]


def test_save_geojson_valid(tmp_path):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [100, 0]},
                "properties": {"id": "X"},
            }
        ],
    }
    saved = save_geojson(fc, tmp_path / "out.geojson")
    assert saved.exists()
    loaded = json.loads(saved.read_text())
    assert loaded["type"] == "FeatureCollection"


def test_save_geojson_rejects_invalid():
    with pytest.raises(ValueError, match="FeatureCollection"):
        save_geojson({"type": "Feature"}, "/tmp/x.geojson")
    with pytest.raises(ValueError, match="features"):
        save_geojson({"type": "FeatureCollection"}, "/tmp/x.geojson")


def test_save_npz_compressed(tmp_path):
    out = tmp_path / "patch.npz"
    img = np.random.rand(6, 256, 256).astype("float32")
    lab = np.random.randint(0, 6, (256, 256)).astype("uint8")
    save_npz(out, img=img, lab=lab)
    assert out.exists()
    loaded = np.load(out)
    np.testing.assert_array_equal(loaded["lab"], lab)
    assert loaded["img"].shape == (6, 256, 256)
