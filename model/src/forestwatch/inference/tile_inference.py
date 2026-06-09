"""Inferensi sliding-window tile-by-tile dengan overlap-blending + opsi TTA.

Sumber: PRD §A.5 Cell 9, ditingkatkan dengan:
- **Overlap-tile + blending** (stride < patch_size): akumulasi probabilitas ke
  buffer float, hilangkan artefak seam antar patch.
- **Test-Time Augmentation (TTA)**: rata-rata softmax atas {asli, hflip, vflip,
  rot180} → prediksi lebih robust (IRUNet, Scientific Reports 2025).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from forestwatch.constants import N_CLASSES
from forestwatch.utils.logging import get_logger

if TYPE_CHECKING:
    import torch.nn as nn

_logger = get_logger("forestwatch.inference")


def _predict_proba_tta(model, x, *, tta: bool):
    """Prediksi probabilitas (softmax) untuk 1 batch patch, opsional TTA.

    TTA: rata-rata softmax atas 4 transformasi dihedral yang reversible
    (asli, hflip, vflip, rot180). Semua dibalik ke orientasi asli sebelum rata.

    Args:
        model: nn.Module (eval).
        x: tensor (B, C, H, W).
        tta: aktifkan TTA.

    Returns:
        tensor probabilitas (B, n_classes, H, W).
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    def _softmax(out):
        return F.softmax(out, dim=1)

    if not tta:
        return _softmax(model(x))

    probs = _softmax(model(x))
    # hflip (dim=3)
    probs = probs + torch.flip(_softmax(model(torch.flip(x, dims=[3]))), dims=[3])
    # vflip (dim=2)
    probs = probs + torch.flip(_softmax(model(torch.flip(x, dims=[2]))), dims=[2])
    # rot180 (dims 2,3)
    probs = probs + torch.flip(_softmax(model(torch.flip(x, dims=[2, 3]))), dims=[2, 3])
    return probs / 4.0


def infer_tile(
    tif_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    model: "nn.Module",
    *,
    device: str | None = None,
    patch_size: int = 256,
    stride: int | None = None,
    n_channels_image: int = 6,
    n_classes: int = N_CLASSES,
    tta: bool = False,
) -> Path:
    """Inferensi 1 ubin GeoTIFF → simpan mask kelas uint8 sebagai GeoTIFF.

    Memakai akumulator probabilitas float (overlap-blending) sehingga aman
    untuk ``stride < patch_size``. Patch tepi yang tidak pas grid tetap
    tercakup (ditambahkan window terakhir yang menempel ke tepi).

    Args:
        tif_path: Path ubin input (berisi ``n_channels_image`` band citra).
        out_path: Path output mask GeoTIFF (1 band uint8).
        model: nn.Module segmentasi (sudah load weights).
        device: ``"cuda"``/``"cpu"``. Default auto.
        patch_size: Ukuran sliding window.
        stride: Stride. Default = ``patch_size`` (non-overlap). Set < patch_size
            untuk overlap-blending (mis. patch_size//2).
        n_channels_image: Jumlah band citra.
        n_classes: Jumlah kelas (untuk buffer probabilitas).
        tta: Aktifkan Test-Time Augmentation.

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

    # Akumulator probabilitas + count (untuk rata-rata di area overlap)
    prob_acc = np.zeros((n_classes, H, W), dtype="float32")
    count = np.zeros((H, W), dtype="float32")

    # Origin grid; pastikan tepi kanan/bawah selalu tercakup
    def _starts(extent: int) -> list[int]:
        if extent <= patch_size:
            return [0]
        xs = list(range(0, extent - patch_size + 1, stride))
        if xs[-1] != extent - patch_size:
            xs.append(extent - patch_size)
        return xs

    rows = _starts(H)
    cols = _starts(W)

    with torch.no_grad():
        for r in rows:
            for c in cols:
                patch = full[:, r : r + patch_size, c : c + patch_size]
                # Patch bisa lebih kecil dari patch_size kalau tile < patch_size
                ph, pw = patch.shape[1], patch.shape[2]
                x = torch.from_numpy(patch).unsqueeze(0).to(device_t)
                probs = _predict_proba_tta(model, x, tta=tta)[0].cpu().numpy()  # (n_classes, ph, pw)
                prob_acc[:, r : r + ph, c : c + pw] += probs[:, :ph, :pw]
                count[r : r + ph, c : c + pw] += 1.0

    count = np.maximum(count, 1e-6)
    mask_full = (prob_acc / count).argmax(axis=0).astype("uint8")

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
    n_classes: int = N_CLASSES,
    tta: bool = False,
) -> list[Path]:
    """Inferensi semua ubin di folder.

    Output file: ``{out_dir}/{prefix}{nama_ubin}.tif``.
    Forward ``stride``, ``tta`` ke ``infer_tile``.
    """
    tile_dir_p = Path(tile_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    tile_files = sorted(tile_dir_p.glob(tile_glob))
    if not tile_files:
        raise FileNotFoundError(f"Tidak ada '{tile_glob}' di {tile_dir_p}.")

    outputs = []
    desc = f"Infer {prefix}" + (" +TTA" if tta else "")
    for tif in tqdm(tile_files, desc=desc):
        out_path = out_dir_p / f"{prefix}{tif.name}"
        infer_tile(
            tif,
            out_path,
            model,
            device=device,
            patch_size=patch_size,
            stride=stride,
            n_channels_image=n_channels_image,
            n_classes=n_classes,
            tta=tta,
        )
        outputs.append(out_path)
    _logger.info(
        "Inferensi selesai: %d mask tersimpan di %s (stride=%s, tta=%s)",
        len(outputs), out_dir_p, stride, tta,
    )
    return outputs
