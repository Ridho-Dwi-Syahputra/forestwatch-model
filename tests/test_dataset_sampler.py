"""Test bobot WeightedRandomSampler (compute_patch_sampler_weights) — torch-free."""

from __future__ import annotations

import numpy as np

from forestwatch.data.dataset import compute_patch_sampler_weights


def _save_patch(path, lab):
    np.savez_compressed(
        path, img=np.zeros((6, 4, 4), dtype="float32"), lab=lab.astype("uint8")
    )


def test_sampler_weight_higher_for_rare_class(tmp_path):
    # class_weights default proyek (7 kelas): Hutan(1) rendah, Sawit(3)/Tambang(5) tinggi.
    cw = [0.8, 0.4, 2.0, 2.0, 1.0, 2.5, 2.5]
    d = tmp_path / "tile_000"
    d.mkdir()
    _save_patch(d / "p00000.npz", np.full((4, 4), 1))  # semua Hutan
    _save_patch(d / "p00001.npz", np.full((4, 4), 3))  # semua Sawit
    files = [d / "p00000.npz", d / "p00001.npz"]

    w = compute_patch_sampler_weights(files, cw)
    assert len(w) == 2
    # Patch Sawit (bobot kelas 2.0) lebih sering disampling daripada Hutan (0.4).
    assert w[1] > w[0]
    assert abs(w[0] - 0.4) < 1e-6
    assert abs(w[1] - 2.0) < 1e-6


def test_sampler_weight_mixed_patch(tmp_path):
    cw = [0.8, 0.4, 2.0, 2.0, 1.0, 2.5, 2.5]
    d = tmp_path / "t"
    d.mkdir()
    lab = np.full((4, 4), 1)  # 16 piksel; setengah jadi Tambang(5)
    lab[:2, :] = 5
    _save_patch(d / "p0.npz", lab)
    w = compute_patch_sampler_weights([d / "p0.npz"], cw)
    # 0.5*Hutan(0.4) + 0.5*Tambang(2.5) = 1.45
    assert abs(w[0] - 1.45) < 1e-6


def test_sampler_weight_cache_roundtrip(tmp_path):
    cw = [1.0] * 7
    d = tmp_path / "t"
    d.mkdir()
    _save_patch(d / "p0.npz", np.full((4, 4), 1))
    cache = tmp_path / "sampler_weights.json"

    w1 = compute_patch_sampler_weights([d / "p0.npz"], cw, cache_path=cache)
    assert cache.exists()

    # Hapus file patch → call kedua HARUS pakai cache (tidak baca file lagi).
    (d / "p0.npz").unlink()
    w2 = compute_patch_sampler_weights([d / "p0.npz"], cw, cache_path=cache)
    assert w1 == w2
