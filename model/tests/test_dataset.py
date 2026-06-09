"""Test PapuaDataset & split_files (lightweight, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from forestwatch.data.dataset import split_files


def test_split_files_exact_ratios():
    # 100 file dummy path
    files = [f"p{i:05d}.npz" for i in range(100)]
    train, val, test = split_files(files, train_ratio=0.8, val_ratio=0.1, seed=42)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10


def test_split_files_deterministic_with_seed():
    files = [f"p{i:05d}.npz" for i in range(50)]
    a = split_files(files, seed=42)
    b = split_files(files, seed=42)
    assert [str(x) for x in a[0]] == [str(x) for x in b[0]]


def test_split_files_no_overlap():
    files = [f"p{i:05d}.npz" for i in range(100)]
    train, val, test = split_files(files, seed=42)
    train_set = {str(p) for p in train}
    val_set = {str(p) for p in val}
    test_set = {str(p) for p in test}
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set | val_set | test_set) == 100


# Cek import PapuaDataset hanya bila torch tersedia
torch = pytest.importorskip("torch")


def test_papua_dataset_returns_tensors(tmp_path):
    from forestwatch.data.dataset import PapuaDataset
    from forestwatch.utils.io import save_npz

    # Buat 3 patch dummy
    for i in range(3):
        save_npz(
            tmp_path / f"p{i:05d}.npz",
            img=np.random.rand(6, 32, 32).astype("float32"),
            lab=np.random.randint(0, 6, (32, 32)).astype("uint8"),
        )

    files = sorted(tmp_path.glob("p*.npz"))
    ds = PapuaDataset(files, train=False)
    assert len(ds) == 3

    img, lab = ds[0]
    assert img.shape == (6, 32, 32)
    assert lab.shape == (32, 32)
    assert img.dtype == torch.float32
    assert lab.dtype == torch.long


def test_papua_dataset_with_augmentation(tmp_path):
    from forestwatch.data.dataset import PapuaDataset
    from forestwatch.utils.io import save_npz

    save_npz(
        tmp_path / "p00000.npz",
        img=np.random.rand(6, 32, 32).astype("float32"),
        lab=np.random.randint(0, 6, (32, 32)).astype("uint8"),
    )
    files = sorted(tmp_path.glob("p*.npz"))
    ds = PapuaDataset(files, train=True, augment_p={"horizontal_flip_p": 1.0})
    img, lab = ds[0]
    # Shape sama walaupun di-augment
    assert img.shape == (6, 32, 32)
    assert lab.shape == (32, 32)
