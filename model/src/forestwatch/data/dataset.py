"""PyTorch Dataset + DataLoader builder untuk patch ForestWatch.

Sumber: PRD §A.5 Cell 5 ``PapuaDataset`` — di-refactor dengan augmentasi opsional
dan split deterministik.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class PapuaDataset:
    """Dataset patch Sentinel-2 + label 7 kelas.

    Setiap item return ``(image_tensor (C,H,W) float32, label_tensor (H,W) long)``.
    Augmentasi via albumentations diterapkan jika ``train=True``.
    """

    def __init__(
        self,
        files: Sequence[str | os.PathLike[str]],
        *,
        train: bool = False,
        augment_p: dict[str, float] | None = None,
    ) -> None:
        try:
            import numpy as np  # noqa: PLC0415
            import torch  # noqa: PLC0415, F401
        except ImportError as e:
            raise ImportError(
                "Butuh numpy + torch. Install: pip install -e \".[ml]\""
            ) from e

        self._np = np
        self.files: list[Path] = [Path(f) for f in files]
        self.train = train
        self.aug = self._build_augmentation(augment_p) if train else None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        import torch  # noqa: PLC0415

        data = self._np.load(self.files[i])
        img = data["img"]  # (C, H, W) float32
        lab = data["lab"]  # (H, W) uint8

        if self.aug is not None:
            # albumentations butuh HWC
            img_hwc = self._np.transpose(img, (1, 2, 0))
            transformed = self.aug(image=img_hwc, mask=lab)
            img_hwc = transformed["image"]
            lab = transformed["mask"]
            img = self._np.transpose(img_hwc, (2, 0, 1))

        return (
            torch.from_numpy(self._np.ascontiguousarray(img)).float(),
            torch.from_numpy(self._np.ascontiguousarray(lab)).long(),
        )

    @staticmethod
    def _build_augmentation(p: dict[str, float] | None):
        try:
            import albumentations as A  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "Augmentasi butuh albumentations. Install: pip install -e \".[ml]\""
            ) from e

        p = p or {}
        transforms = [
            A.HorizontalFlip(p=p.get("horizontal_flip_p", 0.5)),
            A.VerticalFlip(p=p.get("vertical_flip_p", 0.5)),
            A.RandomRotate90(p=p.get("rotate_90_p", 0.5)),
        ]
        # Augmentasi radiometrik RINGAN (limit kecil) agar tanda tangan spektral
        # (SWIR pembeda sawit) tidak terdistorsi. Hanya brightness/contrast,
        # tanpa gamma/hue. Default p=0.3.
        bc_p = p.get("brightness_contrast_p", 0.3)
        if bc_p and bc_p > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=p.get("brightness_limit", 0.1),
                    contrast_limit=p.get("contrast_limit", 0.1),
                    p=bc_p,
                )
            )
        return A.Compose(transforms)


def split_files(
    files: Sequence[str | os.PathLike[str]],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split deterministik ke train/val/test."""
    import numpy as np  # noqa: PLC0415

    files_p = [Path(f) for f in files]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(files_p))
    n = len(files_p)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return (
        [files_p[i] for i in train_idx],
        [files_p[i] for i in val_idx],
        [files_p[i] for i in test_idx],
    )


def compute_patch_sampler_weights(
    files: Sequence[str | os.PathLike[str]],
    class_weights: Sequence[float],
    *,
    n_classes: int | None = None,
    cache_path: str | os.PathLike[str] | None = None,
    max_workers: int = 16,
) -> list[float]:
    """Bobot sampling per-patch untuk ``WeightedRandomSampler`` (oversample kelas langka).

    Bobot tiap patch = Σ_c ``class_weights[c]`` · (fraksi piksel kelas c). Patch yang
    banyak memuat kelas langka (bobot kelas tinggi, mis. Sawit/Tambang/Permukiman)
    jadi lebih sering ter-sampling — melengkapi class-weights di loss untuk imbalance
    berat. Dihitung **paralel** (ThreadPoolExecutor) & **di-cache JSON** per-patch
    (keyed by path) agar tahan restart sesi Colab.

    Returns:
        List bobot float sejajar urutan ``files``.
    """
    import json  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from forestwatch.constants import N_CLASSES  # noqa: PLC0415

    n_classes = n_classes or N_CLASSES
    files_p = [Path(f) for f in files]
    keys = [f.as_posix() for f in files_p]
    cw = np.asarray(list(class_weights), dtype=np.float64)

    cache: dict[str, float] = {}
    if cache_path is not None and Path(cache_path).exists():
        try:
            cache = json.load(open(cache_path))
        except (OSError, ValueError):
            cache = {}

    missing = [f for f, k in zip(files_p, keys, strict=False) if k not in cache]
    if missing:
        def _w(p: Path) -> float:
            lab = np.load(p)["lab"]
            cnt = np.bincount(np.asarray(lab).ravel(), minlength=n_classes)[:n_classes]
            tot = float(cnt.sum())
            return 1.0 if tot <= 0 else float((cw * (cnt.astype(np.float64) / tot)).sum())

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for f, w in zip(missing, exe.map(_w, missing), strict=False):
                cache[f.as_posix()] = w
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            json.dump(cache, open(cache_path, "w"))

    weights = [float(cache.get(k, 1.0)) for k in keys]
    if not any(weights):
        weights = [1.0] * len(weights)
    return weights


def _loaders_from_split(
    train_f: Sequence[str | os.PathLike[str]],
    val_f: Sequence[str | os.PathLike[str]],
    test_f: Sequence[str | os.PathLike[str]],
    *,
    batch_size: int,
    num_workers: int,
    augment_p: dict[str, float] | None,
    class_weights: Sequence[float] | None,
    sampler_cache: str | os.PathLike[str] | None,
    persistent_workers: bool = False,
) -> tuple["DataLoader", "DataLoader", "DataLoader"]:
    from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: PLC0415

    # ``persistent_workers`` hanya valid bila ada worker (>0); PyTorch error bila
    # True saat num_workers=0. Jaga worker tetap hidup antar-epoch (kurangi
    # overhead respawn di mesin lab multi-core).
    _pw = bool(persistent_workers) and num_workers > 0

    train_ds = PapuaDataset(train_f, train=True, augment_p=augment_p)
    if class_weights is not None:
        # Oversample patch kelas langka via WeightedRandomSampler (anti-imbalance).
        w = compute_patch_sampler_weights(train_f, class_weights, cache_path=sampler_cache)
        sampler = WeightedRandomSampler(
            weights=w, num_samples=len(train_f), replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=_pw,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=_pw,
        )
    val_loader = DataLoader(
        PapuaDataset(val_f, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=_pw,
    )
    test_loader = DataLoader(
        PapuaDataset(test_f, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=_pw,
    )
    return train_loader, val_loader, test_loader


def build_dataloaders(
    patch_dir: str | os.PathLike[str],
    *,
    batch_size: int = 8,
    num_workers: int = 2,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    augment_p: dict[str, float] | None = None,
    class_weights: Sequence[float] | None = None,
    sampler_cache: str | os.PathLike[str] | None = None,
    persistent_workers: bool = False,
) -> tuple["DataLoader", "DataLoader", "DataLoader"]:
    """Bangun ``train_loader, val_loader, test_loader`` dari satu folder patch.

    Args:
        patch_dir: Folder berisi ``p*.npz``.
        batch_size: Batch size (lihat ``configs/default.yaml``).
        num_workers: Worker DataLoader (2 default di Colab).
        train_ratio, val_ratio: Rasio split (test = 1 - train - val).
        seed: Random seed untuk split deterministik.
        augment_p: Probabilitas augmentasi (lihat ``PapuaDataset``).
        class_weights: Bila diberikan, train_loader memakai ``WeightedRandomSampler``
            (oversample patch kelas langka) menggantikan ``shuffle``.
        sampler_cache: Path cache JSON bobot sampler (per-patch). Restart-safe.
        persistent_workers: Jaga worker DataLoader tetap hidup antar-epoch
            (otomatis ``False`` bila ``num_workers=0``). Berguna di mesin
            multi-core (lab) untuk kurangi overhead respawn worker.

    Returns:
        ``(train_loader, val_loader, test_loader)``.
    """
    try:
        from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: PLC0415, F401
    except ImportError as e:
        raise ImportError("Butuh torch. Install: pip install -e \".[ml]\"") from e

    from forestwatch.data.patches import list_patches  # noqa: PLC0415

    files = list_patches(patch_dir)
    if not files:
        raise FileNotFoundError(
            f"Tidak ada patch di {patch_dir}. Jalankan ``cut_patches`` dahulu."
        )

    train_f, val_f, test_f = split_files(
        files, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    return _loaders_from_split(
        train_f, val_f, test_f,
        batch_size=batch_size, num_workers=num_workers, augment_p=augment_p,
        class_weights=class_weights, sampler_cache=sampler_cache,
        persistent_workers=persistent_workers,
    )


def build_dataloaders_from_files(
    train_files: Sequence[str | os.PathLike[str]],
    val_files: Sequence[str | os.PathLike[str]],
    test_files: Sequence[str | os.PathLike[str]],
    *,
    batch_size: int = 8,
    num_workers: int = 2,
    augment_p: dict[str, float] | None = None,
    class_weights: Sequence[float] | None = None,
    sampler_cache: str | os.PathLike[str] | None = None,
    persistent_workers: bool = False,
) -> tuple["DataLoader", "DataLoader", "DataLoader"]:
    """Bangun ``train_loader, val_loader, test_loader`` dari daftar file eksplisit.

    Dipakai saat split train/val/test sudah ditentukan di luar (mis. Bagian 14:
    train = Papua-train + transfer + augmentasi offline; val/test = Papua-only
    holdout) sehingga tidak di-split ulang dari satu folder seperti
    :func:`build_dataloaders`.

    Args:
        train_files, val_files, test_files: Daftar path ``p*.npz``.
        batch_size: Batch size (lihat ``configs/default.yaml``).
        num_workers: Worker DataLoader (2 default di Colab).
        augment_p: Probabilitas augmentasi (lihat ``PapuaDataset``).
        class_weights: Bila diberikan, train_loader memakai ``WeightedRandomSampler``
            (oversample patch kelas langka) menggantikan ``shuffle``.
        sampler_cache: Path cache JSON bobot sampler (per-patch). Restart-safe.
        persistent_workers: Jaga worker DataLoader tetap hidup antar-epoch
            (otomatis ``False`` bila ``num_workers=0``). Berguna di mesin
            multi-core (lab) untuk kurangi overhead respawn worker.

    Returns:
        ``(train_loader, val_loader, test_loader)``.
    """
    try:
        from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: PLC0415, F401
    except ImportError as e:
        raise ImportError("Butuh torch. Install: pip install -e \".[ml]\"") from e

    if not train_files:
        raise FileNotFoundError("train_files kosong. Jalankan Bagian 14.1/14.2 dahulu.")

    return _loaders_from_split(
        train_files, val_files, test_files,
        batch_size=batch_size, num_workers=num_workers, augment_p=augment_p,
        class_weights=class_weights, sampler_cache=sampler_cache,
        persistent_workers=persistent_workers,
    )
