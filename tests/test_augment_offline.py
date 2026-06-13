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
# Multiplier per-kelas (dict) & fractional (×1.5, ×2.5, dst.)
# ---------------------------------------------------------------------------


def test_dict_multiplier_per_class(tmp_path):
    """Angka final yang disepakati: 2×1.5, 3×1, 4×1, 5×2, 6×2.5."""
    src = tmp_path / "src"
    src.mkdir()
    _make_patch(src, "p00001.npz", _dominant_lab(5))  # Tambang
    _make_patch(src, "p00002.npz", _dominant_lab(3))  # Sawit
    _make_patch(src, "p00003.npz", _dominant_lab(6))  # Permukiman

    out = tmp_path / "aug"
    manifest_path = tmp_path / "manifest.json"
    gen = run_aug(
        src, out, seed=1,
        multiplier={2: 1.5, 3: 1, 4: 1, 5: 2, 6: 2.5},
        manifest_path=manifest_path,
    )

    assert gen[5] == 1  # mult=2 -> deterministik 1 salinan
    assert gen[3] == 0  # mult=1 -> 0 salinan
    assert gen[6] in (1, 2)  # mult=2.5 -> 1 atau 2 (~50/50)
    assert len(list((out / "tambang").glob("aug_*.npz"))) == gen[5]
    assert len(list((out / "sawit").glob("aug_*.npz"))) == 0
    assert len(list((out / "permukiman").glob("aug_*.npz"))) == gen[6]

    import json

    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert m["params"]["multiplier_map"] == {
        "2": 1.5, "3": 1.0, "4": 1.0, "5": 2.0, "6": 2.5,
    }
    assert m["params"]["multiplier"] == {"2": 1.5, "3": 1, "4": 1, "5": 2, "6": 2.5}


def test_multiplier_one_generates_zero_copies(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_patch(src, "p00001.npz", _dominant_lab(3))  # Sawit
    _make_patch(src, "p00002.npz", _dominant_lab(4))  # Pertanian Lain

    out = tmp_path / "aug"
    gen = run_aug(src, out, classes=(3, 4), multiplier=1, seed=1)

    assert gen[3] == 0
    assert gen[4] == 0
    assert not list((out / "sawit").glob("aug_*.npz"))
    assert not list((out / "pertanian_lain").glob("aug_*.npz"))
    # Dir tetap dibuat oleh clean=True, hanya kosong.
    assert (out / "sawit").is_dir()
    assert (out / "pertanian_lain").is_dir()


def test_dict_multiplier_one_for_specific_class(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_patch(src, "p00001.npz", _dominant_lab(3))  # Sawit
    _make_patch(src, "p00002.npz", _dominant_lab(5))  # Tambang

    out = tmp_path / "aug"
    gen = run_aug(src, out, multiplier={3: 1, 5: 2}, seed=1)

    assert gen[3] == 0
    assert gen[5] == 1
    assert not list((out / "sawit").glob("aug_*.npz"))
    assert len(list((out / "tambang").glob("aug_*.npz"))) == 1


def test_fractional_multiplier_statistical(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    n_patches = 40
    for i in range(n_patches):
        _make_patch(src, f"p{i:05d}.npz", _dominant_lab(2))  # Lahan Terbuka

    out = tmp_path / "aug"
    gen = run_aug(src, out, classes=(2,), multiplier=1.5, seed=123)

    n_total = len(list((out / "lahan_terbuka").glob("aug_*.npz")))
    assert n_total == gen[2]
    # mult=1.5 -> extra=0.5 -> n_copies in {0,1}/patch, E[total]=20, std~3.16.
    assert 7 <= n_total <= 33
    assert n_total not in (0, n_patches)


def test_fractional_multiplier_two_point_five(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    n_patches = 40
    for i in range(n_patches):
        _make_patch(src, f"p{i:05d}.npz", _dominant_lab(6))  # Permukiman

    out = tmp_path / "aug"
    gen = run_aug(src, out, classes=(6,), multiplier=2.5, seed=123)

    n_total = len(list((out / "permukiman").glob("aug_*.npz")))
    assert n_total == gen[6]
    # mult=2.5 -> extra=1.5 -> n_base=1 (selalu) + {0,1} ekstra, E[total]=60.
    assert n_total >= n_patches  # n_base=1 -> minimal 1/patch
    assert n_patches + 7 <= n_total <= n_patches + 33


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


# ---------------------------------------------------------------------------
# clean: idempoten lintas perubahan multiplier
# ---------------------------------------------------------------------------


def test_clean_removes_stale_files_from_previous_multiplier(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_patch(src, "p00001.npz", _dominant_lab(3))  # Sawit

    out = tmp_path / "aug"

    # Run 1: multiplier=2 -> 1 salinan di sawit/.
    gen1 = run_aug(src, out, classes=(3,), multiplier=2, seed=1)
    assert gen1[3] == 1
    assert len(list((out / "sawit").glob("aug_*.npz"))) == 1

    # Run 2: multiplier final=1 (Sawit tak diaugmentasi) pada out_dir SAMA.
    # clean=True (default) harus membuang sisa file dari run 1.
    gen2 = run_aug(src, out, classes=(3,), multiplier=1, seed=1)
    assert gen2[3] == 0
    assert list((out / "sawit").glob("aug_*.npz")) == []


def test_clean_false_preserves_stale_files(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_patch(src, "p00001.npz", _dominant_lab(3))

    out = tmp_path / "aug"
    run_aug(src, out, classes=(3,), multiplier=2, seed=1)
    assert len(list((out / "sawit").glob("aug_*.npz"))) == 1

    gen2 = run_aug(src, out, classes=(3,), multiplier=1, seed=1, clean=False)
    assert gen2[3] == 0
    # clean=False -> file lama dari run 1 tetap ada.
    assert len(list((out / "sawit").glob("aug_*.npz"))) == 1


# ---------------------------------------------------------------------------
# Regresi: nama output unik lintas subfolder sumber (cegah overwrite)
# ---------------------------------------------------------------------------


def test_out_name_unique_when_parent_name_and_stem_collide(tmp_path):
    """Struktur sumber bertingkat (``<region>/<slug>/p#####.npz``) bisa membuat
    ``(parent.name, stem)`` sama utk 2 file BERBEDA (region berbeda, slug & stem
    sama) -> out_name harus tetap unik via ``file_idx``, bukan hanya parent.name
    (lihat docs/model/RENCANA_EKSEKUSI_LANJUTAN.md: 4024 -> 3938 setelah fix
    parent.name, sisa 86 masih collision).
    """
    src = tmp_path / "src"
    (src / "region_a" / "tambang").mkdir(parents=True)
    (src / "region_b" / "tambang").mkdir(parents=True)
    _make_patch(src / "region_a" / "tambang", "p00001.npz", _dominant_lab(5))
    _make_patch(src / "region_b" / "tambang", "p00001.npz", _dominant_lab(5))

    out = tmp_path / "aug"
    gen = run_aug(src, out, classes=(5,), multiplier=2, seed=1)

    assert gen[5] == 2
    found = list((out / "tambang").glob("aug_*.npz"))
    assert len(found) == gen[5]
    assert {f.name for f in found} == {
        "aug_000000_tambang_p00001_00.npz",
        "aug_000001_tambang_p00001_00.npz",
    }


# ---------------------------------------------------------------------------
# Regresi: output augmentasi harus terdeteksi oleh list_patches()
# ---------------------------------------------------------------------------


def test_augmented_output_discoverable_via_list_patches(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        _make_patch(src, f"p{i:05d}.npz", _dominant_lab(5))  # Tambang, mult=2 -> 1 each
    for i in range(3, 5):
        _make_patch(src, f"p{i:05d}.npz", _dominant_lab(3))  # Sawit, mult=2 -> 1 each

    out = tmp_path / "aug"
    gen = run_aug(src, out, multiplier=2, seed=1)

    from forestwatch.data.patches import list_patches

    found = list_patches(out)

    assert len(found) == sum(gen.values())
    assert all(f.name.startswith("aug_") for f in found)
    assert all(f.suffix == ".npz" for f in found)
