"""Test augmentasi offline (TAHAP 2) — label-safe, per-kelas, train-only.

``scripts/`` bukan package; modul dimuat dgn menambah path scripts ke sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import augment_offline  # noqa: E402
from augment_offline import augment_offline as run_aug  # noqa: E402


def _make_patch(tmp_path, name, lab, *, n_bands=6):
    """Tulis satu .npz (img 6-band acak + lab yg diberikan)."""
    h, w = lab.shape
    img = np.random.default_rng(0).random((n_bands, h, w), dtype="float32")
    path = tmp_path / name
    np.savez_compressed(path, img=img, lab=lab.astype("uint8"))
    return path


def _dominant_lab(dominant_cls, size=8):
    """Label hampir seluruhnya ``dominant_cls`` + 1 sudut kelas lain."""
    lab = np.full((size, size), dominant_cls, dtype="uint8")
    lab[0, 0] = (dominant_cls + 1) % 7
    return lab


# ---------------------------------------------------------------------------
# Generasi salinan & penempatan per-slug
# ---------------------------------------------------------------------------


def test_generates_one_copy_per_target_patch(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        _make_patch(src, f"p{i:05d}.npz", _dominant_lab(5))  # Tambang
    for i in range(3, 5):
        _make_patch(src, f"p{i:05d}.npz", _dominant_lab(3))  # Sawit

    out = tmp_path / "aug"
    gen = run_aug(src, out, multiplier=2, seed=1)

    assert gen[5] == 3 and gen[3] == 2
    assert gen[2] == 0 and gen[4] == 0 and gen[6] == 0
    assert len(list((out / "tambang").glob("aug_*.npz"))) == 3
    assert len(list((out / "sawit").glob("aug_*.npz"))) == 2


def test_multiplier_three_makes_two_copies(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_patch(src, "p00001.npz", _dominant_lab(5))
    out = tmp_path / "aug"
    gen = run_aug(src, out, multiplier=3, seed=1)
    assert gen[5] == 2
    assert len(list((out / "tambang").glob("aug_*.npz"))) == 2


# ---------------------------------------------------------------------------
# Label-safe: tak ada kelas baru, metadata benar
# ---------------------------------------------------------------------------


def test_label_safe_no_new_classes_and_metadata(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    lab = _dominant_lab(5)
    orig_vals = set(np.unique(lab).tolist())
    p = _make_patch(src, "p00001.npz", lab)

    out = tmp_path / "aug"
    run_aug(src, out, multiplier=2, seed=7)

    aug_files = list((out / "tambang").glob("aug_*.npz"))
    assert len(aug_files) == 1
    d = np.load(aug_files[0], allow_pickle=True)
    # label augmentasi tak boleh memunculkan kelas yang tak ada di sumber
    assert set(np.unique(d["lab"]).tolist()) <= orig_vals
    assert d["img"].shape == (6, 8, 8)
    assert d["lab"].dtype == np.uint8
    assert str(d["source"]) == "augmented"
    assert str(d["parent"]) == p.name
    assert str(d["slug"]) == "tambang"


# ---------------------------------------------------------------------------
# Gate min_target_frac & skip non-target
# ---------------------------------------------------------------------------


def test_skips_patch_without_minority(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    # Patch 100% Hutan (kelas 1) — bukan target → tidak diaugmentasi.
    _make_patch(src, "p00001.npz", np.ones((8, 8), dtype="uint8"))
    out = tmp_path / "aug"
    gen = run_aug(src, out, multiplier=2, seed=1)
    assert sum(gen.values()) == 0
    assert not any(out.rglob("aug_*.npz"))


def test_min_target_frac_threshold(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    # 64 px Hutan dgn 1 px Tambang → fraksi tambang = 1/64 ≈ 0.0156.
    lab = np.ones((8, 8), dtype="uint8")
    lab[0, 0] = 5
    _make_patch(src, "p00001.npz", lab)

    out_low = tmp_path / "aug_low"
    gen_low = run_aug(src, out_low, multiplier=2, min_target_frac=0.01, seed=1)
    assert gen_low[5] == 1  # 0.0156 >= 0.01 → diaugmentasi

    out_high = tmp_path / "aug_high"
    gen_high = run_aug(src, out_high, multiplier=2, min_target_frac=0.05, seed=1)
    assert sum(gen_high.values()) == 0  # 0.0156 < 0.05 → di-skip


# ---------------------------------------------------------------------------
# Manifest JSON
# ---------------------------------------------------------------------------


def test_manifest_json_written(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_patch(src, "p00001.npz", _dominant_lab(5))
    _make_patch(src, "p00002.npz", _dominant_lab(3))

    out = tmp_path / "aug"
    manifest_path = tmp_path / "manifest.json"
    run_aug(src, out, multiplier=2, seed=1, manifest_path=manifest_path)

    import json

    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert m["total_generated"] == 2
    assert m["by_slug"]["tambang"] == 1
    assert m["by_slug"]["sawit"] == 1
    assert m["params"]["multiplier"] == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_main(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_patch(src, "p00001.npz", _dominant_lab(5))
    out = tmp_path / "aug"
    rc = augment_offline.main(
        ["--src", str(src), "--out", str(out), "--multiplier", "2", "--seed", "1"]
    )
    assert rc == 0
    assert len(list((out / "tambang").glob("aug_*.npz"))) == 1
