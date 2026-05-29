"""Inferensi sliding-window tile-by-tile.

Sumber: PRD §A.5 Cell 9 ``infer_tile()``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from forestwatch.utils.logging import get_logger

if TYPE_CHECKING:
    import torch.nn as nn

_logger = get_logger("forestwatch.inference")


def infer_tile(
    tif_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    model: "nn.Module",
    *,
    device: str | None = None,
    patch_size: int = 256,
    stride: int | None = None,
    n_channels_image: int = 6,
) -> Path:
    """Inferensi 1 ubin GeoTIFF → simpan mask kelas uint8 sebagai GeoTIFF.

    Args:
        tif_path: Path ubin input (T1 atau T2; harus berisi ``n_channels_image`` band citra).
        out_path: Path output mask GeoTIFF (1 band uint8).
        model: ``nn.Module`` segmentasi (sudah load weights, di-eval).
        device: ``"cuda"`` atau ``"cpu"``. Default auto-detect.
        patch_size: Ukuran sliding window.
        stride: Stride. Default = ``patch_size`` (non-overlapping).
        n_channels_image: Jumlah band citra.

    Returns:
        Path output absolut.
    """
    try:
        import numpy as np  # noqa: PLC0415
        import rasterio  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Butuh numpy + rasterio + torch. Install: pip install -e \".[ml,gis]\""
        ) from e

    if stride is None:
        stride = patch_size
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device_t).eval()

    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        meta = src.meta.copy()
        meta.update(count=1, dtype="uint8", compress="lzw")
        band_indices = list(range(1, n_channels_image + 1))  # rasterio 1-indexed
        full = src.read(band_indices).astype("float32")  # (C, H, W)
        full = np.nan_to_num(full)

    mask_full = np.zeros((H, W), dtype="uint8")
    with torch.no_grad():
        for r in range(0, H - patch_size + 1, stride):
            for c in range(0, W - patch_size + 1, stride):
                patch = full[:, r : r + patch_size, c : c + patch_size]
                x = torch.from_numpy(patch).unsqueeze(0).to(device_t)
                pred = model(x).argmax(1)[0].cpu().numpy().astype("uint8")
                mask_full[r : r + patch_size, c : c + patch_size] = pred

    with rasterio.open(out, "w", **meta) as dst:
        dst.write(mask_full, 1)

    return out


def infer_tiles_folder(
    tile_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    model: "nn.Module",
    *,
    prefix: str = "mask_",
    tile_glob: str = "*.tif",
    device: str | None = None,
    patch_size: int = 256,
    stride: int | None = None,
    n_channels_image: int = 6,
) -> list[Path]:
    """Inferensi semua ubin di folder.

    Output file: ``{out_dir}/{prefix}{nama_ubin}.tif``.
    """
    tile_dir_p = Path(tile_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    tile_files = sorted(tile_dir_p.glob(tile_glob))
    if not tile_files:
        raise FileNotFoundError(f"Tidak ada '{tile_glob}' di {tile_dir_p}.")

    outputs = []
    for tif in tqdm(tile_files, desc=f"Infer {prefix}"):
        out_path = out_dir_p / f"{prefix}{tif.name}"
        infer_tile(
            tif,
            out_path,
            model,
            device=device,
            patch_size=patch_size,
            stride=stride,
            n_channels_image=n_channels_image,
        )
        outputs.append(out_path)
    _logger.info("Inferensi selesai: %d mask tersimpan di %s", len(outputs), out_dir_p)
    return outputs
