"""Patch extraction — potong ubin GeoTIFF jadi patch 256×256 untuk training.

Sumber: PRD §A.5 Cell 5 ``cut_patches()`` — diparameterisasi & sebagai fungsi pure.
"""

from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm

from forestwatch.constants import N_CLASSES
from forestwatch.utils.io import save_npz
from forestwatch.utils.logging import get_logger

_logger = get_logger("forestwatch.data.patches")


def cut_patches(
    tile_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    patch_size: int = 256,
    stride: int | None = None,
    max_nan_ratio: float = 0.3,
    n_channels_image: int = 6,
    tile_glob: str = "*.tif",
    start_index: int = 0,
) -> int:
    """Potong ubin GeoTIFF (6 band citra + 1 band label) menjadi patch ``patch_size×patch_size``.

    Setiap patch disimpan sebagai ``.npz`` berisi:
      - ``img``: float32, shape ``(n_channels_image, H, W)``
      - ``lab``: uint8,   shape ``(H, W)`` (label 0..5)
      - ``tile``: basename ubin asal
      - ``row``, ``col``: posisi origin patch dalam ubin

    Args:
        tile_dir: Folder berisi GeoTIFF ubin (output GEE).
        out_dir: Folder tujuan ``.npz``. Auto-create.
        patch_size: Ukuran patch (default 256).
        stride: Stride sliding window. Default ``patch_size`` (non-overlapping).
        max_nan_ratio: Buang patch dengan rasio NaN > nilai ini.
        n_channels_image: Jumlah band citra di awal stack (sisanya = label).
        tile_glob: Pattern glob untuk ubin (default ``*.tif``).
        start_index: Index awal penamaan ``p{idx:05d}.npz``.

    Returns:
        Jumlah patch yang berhasil disimpan.

    Raises:
        ImportError: Jika ``rasterio`` belum ter-install.
        FileNotFoundError: Jika ``tile_dir`` kosong.
    """
    try:
        import numpy as np  # noqa: PLC0415
        import rasterio  # noqa: PLC0415
        from rasterio.windows import Window  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Butuh numpy + rasterio. Install: pip install numpy rasterio\n"
            "Atau pakai extras: pip install -e \".[gis]\""
        ) from e

    tile_dir_p = Path(tile_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    tile_files = sorted(tile_dir_p.glob(tile_glob))
    if not tile_files:
        raise FileNotFoundError(
            f"Tidak ada file '{tile_glob}' di {tile_dir_p}. "
            "Pastikan ubin sudah diekspor dari GEE."
        )

    if stride is None:
        stride = patch_size

    idx = start_index
    n_dropped_nan = 0
    n_dropped_shape = 0

    for tif_path in tqdm(tile_files, desc="Cutting patches"):
        with rasterio.open(tif_path) as src:
            W, H = src.width, src.height
            for r in range(0, H - patch_size + 1, stride):
                for c in range(0, W - patch_size + 1, stride):
                    win = Window(c, r, patch_size, patch_size)
                    arr = src.read(window=win)  # (n_channels_image+1, H, W)
                    if arr.shape != (n_channels_image + 1, patch_size, patch_size):
                        n_dropped_shape += 1
                        continue
                    img = arr[:n_channels_image]
                    lab = arr[n_channels_image].astype("uint8")
                    nan_ratio = float(np.isnan(img).mean())
                    if nan_ratio > max_nan_ratio:
                        n_dropped_nan += 1
                        continue
                    img = np.nan_to_num(img).astype("float32")
                    save_npz(
                        out_dir_p / f"p{idx:05d}.npz",
                        img=img,
                        lab=lab,
                        tile=tif_path.name,
                        row=r,
                        col=c,
                    )
                    idx += 1
    n_saved = idx - start_index
    _logger.info(
        "Patch extraction selesai: %d disimpan, %d dibuang (NaN), %d dibuang (shape).",
        n_saved,
        n_dropped_nan,
        n_dropped_shape,
    )
    return n_saved


def list_patches(patch_dir: str | os.PathLike[str]) -> list[Path]:
    """Return list path patch ``.npz`` (sorted)."""
    return sorted(Path(patch_dir).rglob("p*.npz"))


def compute_class_distribution(
    patch_dir: str | os.PathLike[str],
    *,
    n_classes: int = N_CLASSES,
    sample_limit: int | None = None,
) -> dict[int, int]:
    """Hitung distribusi kelas (pixel count) dari semua patch label.

    Berguna untuk men-tune ``class_weights`` di trainer.

    Args:
        patch_dir: Folder berisi ``p*.npz``.
        n_classes: Jumlah kelas (default 6).
        sample_limit: Maksimum patch yang dibaca (None = semua).

    Returns:
        Dict ``{class_id: pixel_count}``.
    """
    import numpy as np  # noqa: PLC0415

    files = list_patches(patch_dir)
    if sample_limit:
        files = files[:sample_limit]
    counts: dict[int, int] = {c: 0 for c in range(n_classes)}
    for f in tqdm(files, desc="Counting classes"):
        lab = np.load(f)["lab"]
        u, c = np.unique(lab, return_counts=True)
        for cls, cnt in zip(u.tolist(), c.tolist(), strict=False):
            if 0 <= int(cls) < n_classes:
                counts[int(cls)] += int(cnt)
    return counts
