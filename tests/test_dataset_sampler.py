"""Test bobot WeightedRandomSampler (compute_patch_sampler_weights) — torch-free."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

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


def test_sampler_weight_default_max_workers_is_64():
    # I/O-bound (baca ribuan .npz dari gdrive) -> default tinggi, selaras EDA cell.
    sig = inspect.signature(compute_patch_sampler_weights)
    assert sig.parameters["max_workers"].default == 64


def test_sampler_weight_incremental_save_no_tmp_left_behind(tmp_path):
    cw = [1.0] * 7
    d = tmp_path / "t"
    d.mkdir()
    files = []
    for i in range(4):
        p = d / f"p{i}.npz"
        _save_patch(p, np.full((4, 4), i % 7))
        files.append(p)
    cache = tmp_path / "shared.json"

    w = compute_patch_sampler_weights(files, cw, cache_path=cache, max_workers=2, save_every=1)
    assert len(w) == 4
    assert cache.exists()
    assert not (tmp_path / "shared.json.tmp").exists()
    assert len(json.loads(cache.read_text())) == 4


def test_sampler_weight_shared_cache_skips_recompute_for_other_model(tmp_path):
    """Notebook model ke-2/3 pakai cache_path yang SAMA -> patch yg sudah dihitung
    model pertama tidak dibaca ulang (tetap dapat bobot dari cache)."""
    cw = [1.0] * 7
    d = tmp_path / "t"
    d.mkdir()
    p0, p1 = d / "p0.npz", d / "p1.npz"
    _save_patch(p0, np.full((4, 4), 1))
    _save_patch(p1, np.full((4, 4), 2))
    cache = tmp_path / "patch_sampler_weights_shared.json"

    w_model1 = compute_patch_sampler_weights([p0, p1], cw, cache_path=cache)

    p0.unlink()  # simulasikan: hanya p1 "baru" buat model kedua
    w_model2 = compute_patch_sampler_weights([p0, p1], cw, cache_path=cache)
    assert w_model1 == w_model2


def test_sampler_weight_cache_portable_across_absolute_roots(tmp_path):
    """Cache dihitung di "mesin A" (root absolute X) harus tetap kepakai di
    "mesin B" (root absolute Y berbeda, mis. lab Windows vs Mac) selama struktur
    relatif ``<sumber>/<file>.npz`` sama -> key cache berbasis path relatif,
    bukan absolute path."""
    cw = [1.0] * 7
    cache = tmp_path / "patch_sampler_weights_shared.json"

    root_a = tmp_path / "dataset_local_lab" / "train" / "papua"
    root_a.mkdir(parents=True)
    _save_patch(root_a / "p0.npz", np.full((4, 4), 3))
    w_a = compute_patch_sampler_weights([root_a / "p0.npz"], cw, cache_path=cache)

    # "Mac": root absolute berbeda total, tapi folder induk/nama file sama (papua/p0.npz).
    root_b = tmp_path / "Users" / "mac" / "dataset_local" / "train" / "papua"
    root_b.mkdir(parents=True)
    _save_patch(root_b / "p0.npz", np.full((4, 4), 1))  # isi beda -> buktikan TIDAK dibaca ulang

    w_b = compute_patch_sampler_weights([root_b / "p0.npz"], cw, cache_path=cache)
    assert w_b == w_a


def test_sampler_weight_custom_keys_match_extracted_layout(tmp_path):
    """``keys=`` dipakai saat ``files`` masih path SUMBER (Drive, sebelum ekstrak)
    tapi key cache harus cocok dengan path HASIL EKSTRAK (``train/<sumber>/...``)
    yang dipakai PC/Jetson -> cache yang dihitung dari sumber (mis. di Colab) bisa
    di-hit ulang oleh sisi PC/Jetson yang baca dari path hasil ekstrak."""
    cw = [0.8, 0.4, 2.0, 2.0, 1.0, 2.5, 2.5]
    cache = tmp_path / "patch_sampler_weights_shared.json"

    # "Colab": path sumber Drive, struktur beda total dari hasil ekstrak.
    src_root = tmp_path / "drive" / "ForestWatch_Patches" / "tile_003"
    src_root.mkdir(parents=True)
    _save_patch(src_root / "p00001.npz", np.full((4, 4), 1))  # Hutan -> bobot 0.4

    src_file = src_root / "p00001.npz"
    arcname = "papua/tile_003/p00001.npz"  # arcname create_dataset_archives
    key = "/".join((Path("train") / arcname).parts[-3:])
    w_colab = compute_patch_sampler_weights([src_file], cw, cache_path=cache, keys=[key])
    assert abs(w_colab[0] - 0.4) < 1e-6

    # "PC": path hasil ekstrak (train/papua/tile_003/p00001.npz), key default parts[-3:]
    # harus SAMA -> cache hit, tidak baca ulang (isi file beda -> buktikan dari cache).
    extracted = tmp_path / "dataset_local" / "train" / "papua" / "tile_003"
    extracted.mkdir(parents=True)
    _save_patch(extracted / "p00001.npz", np.full((4, 4), 5))  # isi beda (Tambang)

    w_pc = compute_patch_sampler_weights([extracted / "p00001.npz"], cw, cache_path=cache)
    assert w_pc == w_colab  # dari cache, bukan dibaca ulang (kalau dibaca ulang -> 2.5)


def test_sampler_weight_keys_length_mismatch_raises(tmp_path):
    cw = [1.0] * 7
    d = tmp_path / "t"
    d.mkdir()
    _save_patch(d / "p0.npz", np.full((4, 4), 1))

    import pytest

    with pytest.raises(ValueError):
        compute_patch_sampler_weights([d / "p0.npz"], cw, keys=["a/b/c", "d/e/f"])


def test_sampler_weight_no_collision_between_papua_and_transfer_tile(tmp_path):
    """``ForestWatch_Patches/tile_003/p00001.npz`` (papua) dan
    ``ForestWatch_Patches_Transfer/tambang/tile_003/p00001.npz`` (transfer) punya
    2 komponen path terakhir IDENTIK (``tile_003/p00001.npz``) tapi adalah patch
    BERBEDA -> key cache harus pakai >=3 komponen agar tidak kolusi (dapat bobot
    yang sama padahal label berbeda)."""
    cw = [0.8, 0.4, 2.0, 2.0, 1.0, 2.5, 2.5]

    papua = tmp_path / "train" / "papua" / "tile_003"
    papua.mkdir(parents=True)
    _save_patch(papua / "p00001.npz", np.full((4, 4), 1))  # Hutan -> bobot 0.4

    transfer = tmp_path / "train" / "transfer" / "tambang" / "tile_003"
    transfer.mkdir(parents=True)
    _save_patch(transfer / "p00001.npz", np.full((4, 4), 5))  # Tambang -> bobot 2.5

    files = [papua / "p00001.npz", transfer / "p00001.npz"]
    w = compute_patch_sampler_weights(files, cw)
    assert abs(w[0] - 0.4) < 1e-6
    assert abs(w[1] - 2.5) < 1e-6
