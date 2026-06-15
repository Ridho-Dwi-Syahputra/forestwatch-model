"""PyTorch Dataset + DataLoader builder untuk patch ForestWatch.

Sumber: PRD §A.5 Cell 5 ``PapuaDataset`` — di-refactor dengan augmentasi opsional
dan split deterministik.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
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
    max_workers: int = 64,
    save_every: int = 5000,
) -> list[float]:
    """Bobot sampling per-patch untuk ``WeightedRandomSampler`` (oversample kelas langka).

    Bobot tiap patch = Σ_c ``class_weights[c]`` · (fraksi piksel kelas c). Patch yang
    banyak memuat kelas langka (bobot kelas tinggi, mis. Sawit/Tambang/Permukiman)
    jadi lebih sering ter-sampling — melengkapi class-weights di loss untuk imbalance
    berat. Dihitung **paralel** (ThreadPoolExecutor, I/O-bound -> default 64 worker)
    dengan progress bar (``tqdm``) & **di-cache JSON** per-patch (keyed by
    ``"<folder_induk>/<nama_file>"``, ditulis atomik tiap ``save_every`` file
    selesai) agar tahan restart sesi Colab/lab. Key relatif (bukan absolute path)
    supaya cache **portable lintas mesin** (mis. dihitung di komputer lab lalu
    cache JSON-nya dipindah ke Mac dengan ``LOCAL_DATA_DIR`` berbeda — key tetap
    cocok selama struktur ``<split>/<sumber>/<file>.npz`` sama). ``cache_path``
    boleh dipakai **bersama lintas model** (mis. satu file di
    ``Bahan_Training_Model/``) — hasil identik selama ``files`` & ``class_weights``
    sama, sehingga notebook model ke-2/3 (atau mesin lain) cukup memuat cache yang
    sudah dihitung model pertama, tanpa baca ulang semua patch.

    Returns:
        List bobot float sejajar urutan ``files``.
    """
    import json  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    from tqdm.auto import tqdm  # noqa: PLC0415

    from forestwatch.constants import N_CLASSES  # noqa: PLC0415

    n_classes = n_classes or N_CLASSES
    files_p = [Path(f) for f in files]
    key_by_path = {f: "/".join(f.parts[-2:]) for f in files_p}
    keys = [key_by_path[f] for f in files_p]
    cw = np.asarray(list(class_weights), dtype=np.float64)

    cache_path_p = Path(cache_path) if cache_path is not None else None
    cache: dict[str, float] = {}
    if cache_path_p is not None and cache_path_p.exists():
        try:
            with open(cache_path_p) as fh:
                cache = json.load(fh)
        except (OSError, ValueError):
            cache = {}

    missing = [f for f, k in zip(files_p, keys) if k not in cache]
    if missing:
        def _w(p: Path) -> float:
            lab = np.load(p)["lab"]
            cnt = np.bincount(np.asarray(lab).ravel(), minlength=n_classes)[:n_classes]
            tot = float(cnt.sum())
            return 1.0 if tot <= 0 else float((cw * (cnt.astype(np.float64) / tot)).sum())

        def _save_cache() -> None:
            if cache_path_p is None:
                return
            cache_path_p.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path_p.with_name(cache_path_p.name + ".tmp")
            with open(tmp, "w") as fh:
                json.dump(cache, fh)
            os.replace(tmp, cache_path_p)

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(_w, f): f for f in missing}
            bar = tqdm(as_completed(futures), total=len(missing), desc="Sampler weights")
            for i, fut in enumerate(bar, 1):
                cache[key_by_path[futures[fut]]] = fut.result()
                if cache_path_p is not None and (i % save_every == 0 or i == len(missing)):
                    _save_cache()

    weights = [float(cache.get(k, 1.0)) for k in keys]
    if not any(weights):
        weights = [1.0] * len(weights)
    return weights


def _chunk_items(
    items: list[tuple[str, Path]], n_parts: int
) -> list[list[tuple[str, Path]]]:
    """Bagi ``items`` jadi maksimal ``n_parts`` potongan berurutan (ukuran serata
    mungkin). Potongan kosong dibuang -> hasil bisa < ``n_parts`` bila
    ``len(items) < n_parts``."""
    if n_parts <= 1 or not items:
        return [items]
    base, rem = divmod(len(items), n_parts)
    chunks: list[list[tuple[str, Path]]] = []
    start = 0
    for i in range(n_parts):
        size = base + (1 if i < rem else 0)
        if size == 0:
            continue
        chunks.append(items[start : start + size])
        start += size
    return chunks


def create_dataset_archives(
    splits: Mapping[str, Sequence[tuple[str, str | os.PathLike[str]]]],
    out_dir: str | os.PathLike[str],
    *,
    n_train_parts: int = 1,
) -> dict[str, list[Path]]:
    """Bundel patch per-split (untuk ``"train"`` harus sudah augmentasi) jadi ``.tar``.

    Mengatasi bottleneck I/O FUSE/rclone (ratusan-ribu file ``.npz`` kecil di
    gdrive) dengan membungkusnya jadi sedikit arsip besar -> sekali ditulis ke
    gdrive, lalu tiap sesi training cukup copy+extract cepat ke disk lokal
    (lihat :func:`extract_dataset_archives`).

    Layout hasil::

        out_dir/
          train/train_part01.tar .. train_part{n_train_parts:02d}.tar
          val/val_part01.tar
          test/test_part01.tar

    Args:
        splits: Mapping nama split (``"train"``/``"val"``/``"test"``) -> daftar
            ``(arcname, source_path)``. ``arcname`` = path relatif di dalam tar
            (mis. ``"papua/tile_000/p00001.npz"``). Untuk split ``"train"``,
            ``source_path`` harus berasal dari ``final_train_files`` (hasil
            **setelah** augmentasi offline), bukan patch mentah.
        out_dir: Folder gdrive tujuan (mis. ``DRIVE_ROOT / 'Bahan_Training_Model'``).
        n_train_parts: Jumlah pecahan ``.tar`` untuk split ``"train"`` (split lain
            selalu 1 file). Memudahkan distribusi/training paralel di beberapa mesin.

    Returns:
        Mapping nama split -> daftar path ``.tar`` (sudah ada atau baru dibuat).

    Idempoten & restart-safe: ``.tar`` final yang sudah ada dilewati (tidak
    ditulis ulang). Penulisan lewat file sementara ber-nama unik per-proses
    (``*.tar.<pid>.part``) + ``os.replace`` atomik -- aman dipanggil bersamaan
    dari beberapa mesin (tak ada tabrakan nama tmp -> tak ada ``.tar`` korup),
    dan sesi yang terhenti otomatis mengulang part tersebut dari awal pada
    proses berikutnya (sisa ``*.<pid>.part`` jadi file sampah tak berbahaya,
    tidak ikut terbaca :func:`extract_dataset_archives`).
    """
    import tarfile  # noqa: PLC0415

    from tqdm.auto import tqdm  # noqa: PLC0415

    out_dir = Path(out_dir)
    result: dict[str, list[Path]] = {}
    for split_name, items in splits.items():
        items_p = [(arcname, Path(src)) for arcname, src in items]
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        n_parts = n_train_parts if split_name == "train" else 1
        chunks = _chunk_items(items_p, max(1, n_parts))

        tar_paths: list[Path] = []
        for i, chunk in enumerate(chunks, start=1):
            tar_path = split_dir / f"{split_name}_part{i:02d}.tar"
            tar_paths.append(tar_path)
            if tar_path.exists():
                continue
            tmp_path = tar_path.with_name(f"{tar_path.name}.{os.getpid()}.part")
            desc = f"Bundling {split_name}_part{i:02d}"
            with tarfile.open(tmp_path, "w") as tf:
                for arcname, src in tqdm(chunk, desc=desc, unit="file"):
                    tf.add(str(src), arcname=arcname)
            os.replace(tmp_path, tar_path)
        result[split_name] = tar_paths
    return result


def _safe_tar_members(tf):
    """Filter member tar sebelum ``extractall``: buang symlink/hardlink & path
    absolut/``..`` (anti path-traversal). Tar di sini selalu hasil
    :func:`create_dataset_archives` sendiri, tapi pengecekan ini murah dan
    portabel ke semua versi Python (parameter ``filter=`` baru ada di 3.12+)."""
    safe = []
    for member in tf.getmembers():
        if member.issym() or member.islnk():
            continue
        if member.name.startswith(("/", "\\")) or ".." in Path(member.name).parts:
            continue
        safe.append(member)
    return safe


def extract_dataset_archives(
    bahan_dir: str | os.PathLike[str],
    local_dir: str | os.PathLike[str],
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    overwrite: bool = False,
    min_free_ratio: float = 1.1,
) -> dict[str, Path]:
    """Ekstrak arsip ``.tar`` (lihat :func:`create_dataset_archives`) ke disk lokal.

    Args:
        bahan_dir: Folder gdrive berisi ``train/``, ``val/``, ``test/`` (masing-masing
            berisi ``<split>_part*.tar``).
        local_dir: Folder disk lokal tujuan (mis. ``Path.home() / 'dataset_local'``).
        splits: Nama split yang diekstrak.
        overwrite: Paksa ekstrak ulang meski marker ``.extracted_ok`` sudah ada.
        min_free_ratio: Margin aman dibanding ukuran data tak terkompresi sebelum
            mulai ekstrak (default 1.1x).

    Returns:
        Mapping nama split -> folder lokal hasil ekstraksi (``local_dir/<split>``).

    Raises:
        FileNotFoundError: Tidak ada ``.tar`` untuk salah satu split yang diminta
            (jalankan :func:`create_dataset_archives` dahulu).
        OSError: Disk lokal tidak cukup untuk hasil ekstraksi -- dicek lebih dulu,
            sebelum menulis apa pun ke ``local_dir``.

    Idempoten: split yang sudah punya marker ``local_dir/<split>/.extracted_ok``
    dilewati, kecuali ``overwrite=True``. Restart-safe untuk sesi Colab/lab yang
    sering terputus.
    """
    import shutil  # noqa: PLC0415
    import tarfile  # noqa: PLC0415

    from tqdm.auto import tqdm  # noqa: PLC0415

    bahan_dir = Path(bahan_dir)
    local_dir = Path(local_dir)

    tar_paths_by_split: dict[str, list[Path]] = {}
    for split_name in splits:
        tar_paths = sorted((bahan_dir / split_name).glob(f"{split_name}_part*.tar"))
        if not tar_paths:
            raise FileNotFoundError(
                f"Tidak ada arsip .tar di {bahan_dir / split_name}. "
                "Jalankan create_dataset_archives dahulu."
            )
        tar_paths_by_split[split_name] = tar_paths

    pending = {
        split_name: tar_paths
        for split_name, tar_paths in tar_paths_by_split.items()
        if overwrite or not (local_dir / split_name / ".extracted_ok").exists()
    }

    if pending:
        needed = 0
        for tar_paths in pending.values():
            for tar_path in tar_paths:
                with tarfile.open(tar_path, "r") as tf:
                    needed += sum(m.size for m in tf.getmembers())

        local_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(local_dir).free
        if needed * min_free_ratio > free:
            raise OSError(
                f"Disk lokal tidak cukup untuk ekstrak dataset: butuh ~"
                f"{needed / 1e9:.2f} GB (margin x{min_free_ratio}), tersedia "
                f"{free / 1e9:.2f} GB di {local_dir}."
            )

    result: dict[str, Path] = {}
    for split_name, tar_paths in tar_paths_by_split.items():
        split_dst = local_dir / split_name
        marker = split_dst / ".extracted_ok"
        if marker.exists() and not overwrite:
            result[split_name] = split_dst
            continue
        split_dst.mkdir(parents=True, exist_ok=True)
        for tar_path in tar_paths:
            with tarfile.open(tar_path, "r") as tf:
                members = _safe_tar_members(tf)
                for member in tqdm(members, desc=f"Extract {tar_path.name}", unit="file"):
                    tf.extract(member, split_dst)
        marker.write_text("ok", encoding="utf-8")
        result[split_name] = split_dst
    return result


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
