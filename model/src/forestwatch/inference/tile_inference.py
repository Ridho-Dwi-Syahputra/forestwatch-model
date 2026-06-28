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

from tqdm.auto import tqdm

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
    batch_size: int = 16,
) -> Path:
    """Inferensi 1 ubin GeoTIFF → simpan mask kelas uint8 sebagai GeoTIFF.

    Memakai akumulator probabilitas float (overlap-blending) sehingga aman
    untuk ``stride < patch_size``. Patch tepi yang tidak pas grid tetap
    tercakup (ditambahkan window terakhir yang menempel ke tepi).

    Semua sliding-window di-batch (``batch_size`` window/forward-pass) --
    jauh lebih cepat di GPU drpd 1 window/forward-pass (default sebelumnya).
    Multiprocessing antar-tile TIDAK dipakai krn cuma ada 1 GPU; proses paralel
    akan rebutan GPU yang sama (kontensi), bukan mempercepat.

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
        batch_size: Jumlah window digabung per forward-pass GPU.

    Returns:
        Path output absolut.
    """
    try:
        import numpy as np  # noqa: PLC0415
        import rasterio  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from rasterio.windows import Window  # noqa: PLC0415
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

    # Origin grid; pastikan tepi kanan/bawah selalu tercakup
    def _starts(extent: int) -> list[int]:
        if extent <= patch_size:
            return [0]
        xs = list(range(0, extent - patch_size + 1, stride))
        if xs[-1] != extent - patch_size:
            xs.append(extent - patch_size)
        return xs

    band_indices = list(range(1, n_channels_image + 1))  # rasterio 1-indexed

    # Tanpa overlap (stride >= patch_size), hampir semua piksel cuma disentuh SATU window --
    # tak perlu akumulator probabilitas float32 (n_classes x H x W, bisa >>1 GB utk tile
    # besar). Tulis langsung ke mask uint8 (H x W, ~32x lebih kecil drpd prob_acc+count
    # float32 7-kelas) -- penghematan RAM riil, bukan cuma workaround.
    # Catatan: kalau H/W bukan kelipatan patch_size, window penyesuai-tepi (_starts) bisa
    # sedikit overlap di strip kanan/bawah (maks patch_size-1 px) -- di situ window terakhir
    # "menang" (overwrite), BUKAN di-blend. Tetap prediksi valid, cuma beda dr blending halus
    # yg dipakai saat overlap aktif (stride < patch_size).
    no_overlap = stride >= patch_size

    # Tile sumber bisa >GB-an (6 band float32) -- JANGAN load seluruh tile ke RAM sekaligus
    # (risiko OOM di Colab free, ~12 GB RAM). Tiap window dibaca langsung dari file via
    # rasterio Window saat dibutuhkan -- hasil pikselnya identik dgn slice array penuh,
    # cuma jejak memori jauh lebih kecil (cuma 1 batch window di RAM, bukan 1 tile penuh).
    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        meta = src.meta.copy()
        # nodata=None (override): mask kelas tak punya semantik "no-data" (semua piksel
        # 0..n_classes-1 valid) -- nodata milik citra SUMBER (sering -inf utk float32, mis.
        # komposit GEE getDownloadURL) tak valid utk dtype uint8, rasterio akan raise
        # ValueError saat dibuka utk ditulis kalau nodata lama ikut terbawa.
        meta.update(count=1, dtype="uint8", compress="lzw", nodata=None)

        if no_overlap:
            mask_full = np.zeros((H, W), dtype="uint8")
        else:
            prob_acc = np.zeros((n_classes, H, W), dtype="float32")
            count = np.zeros((H, W), dtype="float32")

        windows = [(r, c) for r in _starts(H) for c in _starts(W)]

        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                chunk = windows[i : i + batch_size]
                patches = []
                shapes = []
                for r, c in chunk:
                    win = Window(c, r, min(patch_size, W - c), min(patch_size, H - r))
                    patch = src.read(band_indices, window=win).astype("float32")
                    patch = np.nan_to_num(patch)
                    # Patch bisa lebih kecil dari patch_size kalau tile < patch_size
                    ph, pw = patch.shape[1], patch.shape[2]
                    shapes.append((ph, pw))
                    if (ph, pw) != (patch_size, patch_size):
                        padded = np.zeros((patch.shape[0], patch_size, patch_size), dtype="float32")
                        padded[:, :ph, :pw] = patch
                        patch = padded
                    patches.append(patch)

                x = torch.from_numpy(np.stack(patches, axis=0)).to(device_t)
                probs_batch = _predict_proba_tta(model, x, tta=tta).cpu().numpy()  # (B, n_classes, ps, ps)

                if no_overlap:
                    preds_batch = probs_batch.argmax(axis=1).astype("uint8")  # (B, ps, ps)
                    for (r, c), (ph, pw), pred in zip(chunk, shapes, preds_batch):
                        mask_full[r : r + ph, c : c + pw] = pred[:ph, :pw]
                else:
                    for (r, c), (ph, pw), probs in zip(chunk, shapes, probs_batch):
                        prob_acc[:, r : r + ph, c : c + pw] += probs[:, :ph, :pw]
                        count[r : r + ph, c : c + pw] += 1.0

    if not no_overlap:
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
    batch_size: int = 16,
) -> list[Path]:
    """Inferensi semua ubin di folder.

    Output file: ``{out_dir}/{prefix}{nama_ubin}.tif``.
    Forward ``stride``, ``tta``, ``batch_size`` ke ``infer_tile``.
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
            batch_size=batch_size,
        )
        outputs.append(out_path)
    _logger.info(
        "Inferensi selesai: %d mask tersimpan di %s (stride=%s, tta=%s)",
        len(outputs), out_dir_p, stride, tta,
    )
    return outputs
