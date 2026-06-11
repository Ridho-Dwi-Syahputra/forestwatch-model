"""Test utilitas EDA grid sample per kelas dominan (GATE 1 & GATE 2)."""

from __future__ import annotations

import numpy as np
import pytest

from forestwatch.eda_utils import (
    index_patches_by_dominant_class,
    patch_dominant_class,
)


def _make_patch(tmp_path, name, dominant_cls, *, size=8, n_bands=6):
    """Tulis satu .npz patch (img 6-band + lab) dgn kelas dominan tertentu."""
    img = np.random.default_rng(0).random((n_bands, size, size), dtype="float32")
    lab = np.full((size, size), dominant_cls, dtype="uint8")
    # selipkan sedikit piksel kelas lain agar bukan 100% dominan
    lab[0, 0] = (dominant_cls + 1) % 7
    path = tmp_path / name
    np.savez_compressed(path, img=img, lab=lab)
    return path


# ---------------------------------------------------------------------------
# patch_dominant_class
# ---------------------------------------------------------------------------


def test_patch_dominant_class_basic():
    lab = np.array([[5, 5, 1], [5, 0, 0]], dtype=np.uint8)  # kelas 5 terbanyak (3)
    assert patch_dominant_class(lab) == 5


def test_patch_dominant_class_ignore():
    # Perairan(0) terbanyak, tapi diabaikan → dominan berikutnya = Hutan(1)
    lab = np.array([[0, 0, 0], [0, 1, 1]], dtype=np.uint8)
    assert patch_dominant_class(lab) == 0
    assert patch_dominant_class(lab, ignore_classes=(0,)) == 1


def test_patch_dominant_class_ignore_all_fallback():
    # Semua piksel masuk ignore → fallback ke dominan tanpa pengecualian.
    lab = np.zeros((4, 4), dtype=np.uint8)
    assert patch_dominant_class(lab, ignore_classes=(0,)) == 0


# ---------------------------------------------------------------------------
# index_patches_by_dominant_class
# ---------------------------------------------------------------------------


def test_index_patches_by_dominant_class(tmp_path):
    _make_patch(tmp_path, "p00001.npz", 5)
    _make_patch(tmp_path, "p00002.npz", 5)
    _make_patch(tmp_path, "p00003.npz", 3)
    by_class = index_patches_by_dominant_class(tmp_path)
    assert set(by_class.keys()) == {5, 3}
    assert len(by_class[5]) == 2
    assert len(by_class[3]) == 1


def test_index_patches_accepts_path_list(tmp_path):
    p1 = _make_patch(tmp_path, "p00001.npz", 6)
    p2 = _make_patch(tmp_path, "p00002.npz", 6)
    by_class = index_patches_by_dominant_class([p1, p2])
    assert list(by_class.keys()) == [6]
    assert len(by_class[6]) == 2


# ---------------------------------------------------------------------------
# plot_class_dominant_grid (butuh matplotlib)
# ---------------------------------------------------------------------------


def test_plot_class_dominant_grid_saves_png(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from forestwatch.eda_utils import plot_class_dominant_grid

    # 4 patch dominan Tambang(5), 4 dominan Sawit(3)
    for i in range(4):
        _make_patch(tmp_path, f"p{i:05d}.npz", 5)
    for i in range(4, 8):
        _make_patch(tmp_path, f"p{i:05d}.npz", 3)

    out_dir = tmp_path / "grids"
    figs = plot_class_dominant_grid(
        tmp_path, n_per_class=4, out_dir=out_dir, dpi=50
    )
    cls_ids = [c for c, _ in figs]
    assert 5 in cls_ids and 3 in cls_ids
    # 2 PNG tersimpan (satu per kelas yang punya patch)
    pngs = list(out_dir.glob("grid_kelas_*.png"))
    assert len(pngs) == 2

    matplotlib.pyplot.close("all")


def test_plot_class_dominant_grid_empty_returns_empty(tmp_path):
    pytest.importorskip("matplotlib")
    from forestwatch.eda_utils import plot_class_dominant_grid

    assert plot_class_dominant_grid([]) == []
