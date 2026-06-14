"""Test bundling/ekstraksi dataset .tar (create_dataset_archives /
extract_dataset_archives) — strategi anti-bottleneck I/O FUSE/rclone."""

from __future__ import annotations

import shutil
import tarfile

import numpy as np
import pytest

from forestwatch.data.dataset import create_dataset_archives, extract_dataset_archives


def _save_patch(path, value=1):
    np.savez_compressed(
        path,
        img=np.zeros((6, 4, 4), dtype="float32"),
        lab=np.full((4, 4), value, dtype="uint8"),
    )


def _make_files(tmp_path, n, subdir):
    d = tmp_path / subdir
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        p = d / f"p{i:05d}.npz"
        _save_patch(p, value=i % 7)
        files.append(p)
    return files


def _items(files, prefix):
    return [(f"{prefix}/{f.name}", f) for f in files]


def _make_all_splits(tmp_path, n_train=2, n_val=1, n_test=1):
    train_files = _make_files(tmp_path, n_train, "train_src")
    val_files = _make_files(tmp_path, n_val, "val_src")
    test_files = _make_files(tmp_path, n_test, "test_src")
    return {
        "train": _items(train_files, "papua"),
        "val": _items(val_files, "papua"),
        "test": _items(test_files, "papua"),
    }, train_files, val_files, test_files


def test_create_dataset_archives_basic_roundtrip(tmp_path):
    splits, train_files, val_files, test_files = _make_all_splits(tmp_path, n_train=3, n_val=2, n_test=2)
    out_dir = tmp_path / "Bahan_Training_Model"

    result = create_dataset_archives(splits, out_dir, n_train_parts=1)

    assert result["train"] == [out_dir / "train" / "train_part01.tar"]
    assert result["val"] == [out_dir / "val" / "val_part01.tar"]
    assert result["test"] == [out_dir / "test" / "test_part01.tar"]
    for tar_paths in result.values():
        for tar_path in tar_paths:
            assert tar_path.exists()

    with tarfile.open(result["train"][0]) as tf:
        names = sorted(tf.getnames())
    assert names == sorted(f"papua/{f.name}" for f in train_files)


def test_create_dataset_archives_splits_train_into_n_parts(tmp_path):
    train_files = _make_files(tmp_path, 7, "train_src")
    out_dir = tmp_path / "Bahan_Training_Model"

    result = create_dataset_archives({"train": _items(train_files, "papua")}, out_dir, n_train_parts=7)

    assert len(result["train"]) == 7
    all_names: list[str] = []
    for tar_path in result["train"]:
        assert tar_path.exists()
        with tarfile.open(tar_path) as tf:
            members = tf.getnames()
        assert len(members) == 1  # 7 file / 7 part = 1 per part
        all_names.extend(members)
    assert sorted(all_names) == sorted(f"papua/{f.name}" for f in train_files)


def test_create_dataset_archives_fewer_items_than_parts(tmp_path):
    train_files = _make_files(tmp_path, 3, "train_src")
    out_dir = tmp_path / "Bahan_Training_Model"

    result = create_dataset_archives({"train": _items(train_files, "papua")}, out_dir, n_train_parts=7)

    # Tidak ada tar kosong -> jumlah part <= jumlah file.
    assert len(result["train"]) == 3
    all_names: list[str] = []
    for tar_path in result["train"]:
        with tarfile.open(tar_path) as tf:
            members = tf.getnames()
        assert len(members) == 1
        all_names.extend(members)
    assert sorted(all_names) == sorted(f"papua/{f.name}" for f in train_files)


def test_create_dataset_archives_only_train_split_gets_n_parts(tmp_path):
    splits, _, val_files, test_files = _make_all_splits(tmp_path, n_train=4, n_val=4, n_test=4)
    out_dir = tmp_path / "Bahan_Training_Model"

    result = create_dataset_archives(splits, out_dir, n_train_parts=4)

    assert len(result["train"]) == 4
    assert len(result["val"]) == 1
    assert len(result["test"]) == 1


def test_create_dataset_archives_idempotent_skip_existing(tmp_path):
    train_files = _make_files(tmp_path, 2, "train_src")
    out_dir = tmp_path / "Bahan_Training_Model"
    splits = {"train": _items(train_files, "papua")}

    create_dataset_archives(splits, out_dir, n_train_parts=1)
    tar_path = out_dir / "train" / "train_part01.tar"
    mtime_before = tar_path.stat().st_mtime_ns

    # Hapus file sumber -> kalau fungsi menulis ulang, akan FileNotFoundError.
    for f in train_files:
        f.unlink()
    create_dataset_archives(splits, out_dir, n_train_parts=1)

    assert tar_path.stat().st_mtime_ns == mtime_before


def test_create_dataset_archives_ignores_orphan_part_file(tmp_path):
    train_files = _make_files(tmp_path, 2, "train_src")
    out_dir = tmp_path / "Bahan_Training_Model"
    split_dir = out_dir / "train"
    split_dir.mkdir(parents=True)
    # Sisa file sementara dari proses lain (crash / sesi paralel) -- harus
    # diabaikan (tidak dihapus, tidak menghalangi pembuatan tar final).
    orphan = split_dir / "train_part01.tar.99999.part"
    orphan.write_bytes(b"leftover-from-another-process")

    result = create_dataset_archives({"train": _items(train_files, "papua")}, out_dir, n_train_parts=1)

    assert orphan.exists()
    tar_path = result["train"][0]
    assert tar_path.exists()
    with tarfile.open(tar_path) as tf:
        assert len(tf.getnames()) == 2


def test_extract_dataset_archives_roundtrip(tmp_path):
    splits, train_files, val_files, test_files = _make_all_splits(tmp_path, n_train=4, n_val=1, n_test=1)
    bahan_dir = tmp_path / "Bahan_Training_Model"
    create_dataset_archives(splits, bahan_dir, n_train_parts=2)

    local_dir = tmp_path / "dataset_local"
    local_dirs = extract_dataset_archives(bahan_dir, local_dir)

    assert local_dirs["train"] == local_dir / "train"
    assert local_dirs["val"] == local_dir / "val"
    assert local_dirs["test"] == local_dir / "test"

    extracted_train = sorted(p.name for p in (local_dir / "train" / "papua").glob("*.npz"))
    assert extracted_train == sorted(f.name for f in train_files)
    assert (local_dir / "train" / ".extracted_ok").exists()
    assert (local_dir / "val" / ".extracted_ok").exists()
    assert (local_dir / "test" / ".extracted_ok").exists()


def test_extract_dataset_archives_idempotent_marker(tmp_path):
    splits, train_files, _, _ = _make_all_splits(tmp_path, n_train=1, n_val=1, n_test=1)
    bahan_dir = tmp_path / "Bahan_Training_Model"
    create_dataset_archives(splits, bahan_dir, n_train_parts=1)

    local_dir = tmp_path / "dataset_local"
    extract_dataset_archives(bahan_dir, local_dir)

    extracted = local_dir / "train" / "papua" / train_files[0].name
    marker = local_dir / "train" / ".extracted_ok"
    assert marker.exists()

    # Tandai file hasil ekstrak -> jika diekstrak ulang, perubahan ini hilang.
    extracted.write_bytes(b"changed-by-test")
    extract_dataset_archives(bahan_dir, local_dir)

    assert extracted.read_bytes() == b"changed-by-test"


def test_extract_dataset_archives_overwrite_forces_redo(tmp_path):
    splits, train_files, _, _ = _make_all_splits(tmp_path, n_train=1, n_val=1, n_test=1)
    bahan_dir = tmp_path / "Bahan_Training_Model"
    create_dataset_archives(splits, bahan_dir, n_train_parts=1)

    local_dir = tmp_path / "dataset_local"
    extract_dataset_archives(bahan_dir, local_dir)

    extracted = local_dir / "train" / "papua" / train_files[0].name
    extracted.write_bytes(b"changed-by-test")
    extract_dataset_archives(bahan_dir, local_dir, overwrite=True)

    assert extracted.read_bytes() != b"changed-by-test"


def test_extract_dataset_archives_missing_tar_raises(tmp_path):
    bahan_dir = tmp_path / "Bahan_Training_Model"
    (bahan_dir / "train").mkdir(parents=True)
    (bahan_dir / "val").mkdir(parents=True)
    (bahan_dir / "test").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        extract_dataset_archives(bahan_dir, tmp_path / "dataset_local")


def test_extract_dataset_archives_insufficient_disk_raises(tmp_path, monkeypatch):
    splits, _, _, _ = _make_all_splits(tmp_path, n_train=2, n_val=1, n_test=1)
    bahan_dir = tmp_path / "Bahan_Training_Model"
    create_dataset_archives(splits, bahan_dir, n_train_parts=1)

    fake_usage = shutil.disk_usage(tmp_path)._replace(free=1)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: fake_usage)

    local_dir = tmp_path / "dataset_local"
    with pytest.raises(OSError):
        extract_dataset_archives(bahan_dir, local_dir)

    # Gagal sebelum ekstraksi -> tidak ada folder split yang ditulis sebagian.
    assert not (local_dir / "train").exists()


def test_extract_dataset_archives_no_disk_check_when_already_extracted(tmp_path, monkeypatch):
    splits, _, _, _ = _make_all_splits(tmp_path, n_train=1, n_val=1, n_test=1)
    bahan_dir = tmp_path / "Bahan_Training_Model"
    create_dataset_archives(splits, bahan_dir, n_train_parts=1)

    local_dir = tmp_path / "dataset_local"
    extract_dataset_archives(bahan_dir, local_dir)

    fake_usage = shutil.disk_usage(tmp_path)._replace(free=1)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: fake_usage)

    # Semua split sudah punya marker -> tidak perlu cek disk, tidak error.
    local_dirs = extract_dataset_archives(bahan_dir, local_dir)
    assert local_dirs["train"] == local_dir / "train"
