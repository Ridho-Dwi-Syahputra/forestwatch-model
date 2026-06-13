"""Utilitas EDA visual — grid sample per kelas dominan (GATE 1 & GATE 2).

Modul ini menyediakan ``plot_class_dominant_grid`` yang dipakai di DUA gerbang
verifikasi notebook (lihat ``docs/model/RENCANA_EKSEKUSI_LANJUTAN.md``):

* **GATE 1** (Bagian 13, pasca-fusion transfer): pastikan patch hasil
  transfer-learning labelnya masuk akal sebelum di-merge ke pool training.
* **GATE 2** (Bagian 14, train set FINAL pasca-augmentasi): konfirmasi terakhir
  "data siap di-training".

Spesifikasi gerbang (permintaan USER): **7 gambar × 64 sample = 448 sample**,
satu gambar per kelas dominan (grid 8×8 *pasangan*), tiap sample ditampilkan
sebagai **pasangan (RGB asli | Label)** berdampingan — gaya EDA 2 (Bagian 0) —
agar reviewer bisa bandingkan citra asli vs label secara langsung.

Semua dependency berat (numpy/matplotlib) di-*lazy import* di dalam fungsi agar
modul tetap ringan untuk diimpor (mis. di lingkungan tanpa matplotlib).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from forestwatch.constants import CLASS_NAMES, N_CLASSES, PALETTE_RGB
from forestwatch.utils.logging import get_logger

_logger = get_logger("forestwatch.eda_utils")

# Indeks band true-color di stack 6-band (B2,B3,B4,B8,B11,B12) → R=B4,G=B3,B=B2.
_TRUECOLOR_IDX: tuple[int, int, int] = (2, 1, 0)


def patch_dominant_class(
    lab: "object",
    *,
    n_classes: int = N_CLASSES,
    ignore_classes: tuple[int, ...] = (),
) -> int:
    """Kembalikan kelas dengan jumlah piksel terbanyak pada satu patch label.

    Args:
        lab: ndarray ``(H, W)`` label uint8 (0..n_classes-1).
        n_classes: Jumlah kelas valid.
        ignore_classes: Kelas yang diabaikan saat menentukan dominan (mis.
            ``(0, 1)`` untuk fokus ke kelas minoritas). Bila SEMUA piksel patch
            masuk ``ignore_classes``, fallback ke dominan tanpa pengecualian.

    Returns:
        ``int`` id kelas dominan.
    """
    import numpy as np  # noqa: PLC0415

    lab = np.asarray(lab)
    counts = np.bincount(lab.ravel(), minlength=n_classes)[:n_classes].astype(np.int64)
    if ignore_classes:
        masked = counts.copy()
        for c in ignore_classes:
            if 0 <= c < n_classes:
                masked[c] = -1
        if masked.max() > 0:  # ada kelas non-ignore yg benar2 punya piksel
            return int(masked.argmax())
    return int(counts.argmax())


def index_patches_by_dominant_class(
    patches: "str | os.PathLike[str] | list",
    *,
    n_classes: int = N_CLASSES,
    ignore_classes: tuple[int, ...] = (),
    sample_limit: int | None = None,
) -> dict[int, list[Path]]:
    """Kelompokkan patch ``.npz`` berdasarkan kelas dominannya.

    Args:
        patches: Folder berisi ``p*.npz`` ATAU list path ``.npz``.
        n_classes: Jumlah kelas.
        ignore_classes: Diteruskan ke ``patch_dominant_class``.
        sample_limit: Batasi jumlah patch yang dibaca (None = semua).

    Returns:
        Dict ``{class_id: [path, ...]}`` (hanya kelas yang punya ≥1 patch).
    """
    import numpy as np  # noqa: PLC0415

    files = _resolve_patch_paths(patches)
    if sample_limit:
        files = files[:sample_limit]

    by_class: dict[int, list[Path]] = {c: [] for c in range(n_classes)}
    for f in files:
        try:
            lab = np.load(f)["lab"]
        except (OSError, KeyError, ValueError) as exc:  # patch korup/format beda
            _logger.warning("Lewati patch %s (gagal baca label: %s)", f, exc)
            continue
        dom = patch_dominant_class(lab, n_classes=n_classes, ignore_classes=ignore_classes)
        by_class[dom].append(Path(f))
    return {c: paths for c, paths in by_class.items() if paths}


def plot_class_dominant_grid(
    patches: "str | os.PathLike[str] | list | dict",
    *,
    classes: int = N_CLASSES,
    n_per_class: int = 64,
    ignore_classes: tuple[int, ...] = (),
    out_dir: "str | os.PathLike[str] | None" = None,
    dpi: int = 150,
    seed: int = 42,
    class_names: tuple[str, ...] | None = None,
) -> list[tuple[int, "object"]]:
    """Buat 1 grid sample per kelas dominan, tiap sample = pasangan (RGB | Label).

    Untuk tiap kelas yang punya ≥1 patch dominan, dibuat **satu** figure grid
    (mis. 8×8 pasangan untuk ``n_per_class=64``, gaya EDA 2 Bagian 0). Target
    gerbang: 7 kelas × 64 = 448 sample. Kelas dengan kurang dari ``n_per_class``
    patch dominan tetap digambar (sel sisa dikosongkan) + dicatat lewat log
    (lihat catatan GATE 1 di RENCANA: pool transfer mungkin minim Perairan/Hutan
    dominan).

    Args:
        patches: Folder ``.npz``, list path, ATAU dict hasil
            ``index_patches_by_dominant_class``.
        classes: Jumlah kelas (default 7).
        n_per_class: Sample (pasangan RGB|Label) per kelas/grid (default 64 → 8×8).
        ignore_classes: Diteruskan ke pengelompokan dominan (abaikan bila
            ``patches`` sudah berupa dict).
        out_dir: Bila di-set, simpan tiap grid PNG (``grid_kelas_{id}_{nama}.png``)
            pada ``dpi`` ini. Folder auto-create.
        dpi: Resolusi simpan (default 150 — grid pasangan 2x lebih lebar dari
            grid overlay lama, selaras EDA2_Visual_128_Sample).
        seed: Seed pemilihan acak sample dari pool dominan.
        class_names: Override nama kelas (default ``CLASS_NAMES``).

    Returns:
        List ``(class_id, matplotlib.figure.Figure)`` urut kelas. Kosong bila
        tak ada patch sama sekali.

    Raises:
        ImportError: Bila ``matplotlib`` belum terpasang (extras ``ml``).
    """
    import numpy as np  # noqa: PLC0415

    try:
        import matplotlib.patches as mpatches  # noqa: PLC0415
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "plot_class_dominant_grid butuh matplotlib. Install: pip install -e \".[ml]\""
        ) from exc

    class_names = class_names or CLASS_NAMES
    rng = np.random.default_rng(seed)

    if isinstance(patches, dict):
        by_class = {int(k): list(v) for k, v in patches.items() if v}
    else:
        by_class = index_patches_by_dominant_class(
            patches, n_classes=classes, ignore_classes=ignore_classes
        )

    if not by_class:
        _logger.warning("Tidak ada patch dominan untuk digrid (pool kosong).")
        return []

    # Grid SAMPLE (pasangan) ncols x nrows; tiap sample = 2 kolom subplot (RGB, Label).
    ncols = math.ceil(math.sqrt(n_per_class))
    nrows = math.ceil(n_per_class / ncols)

    legend_handles = [
        mpatches.Patch(color=np.array(PALETTE_RGB[c]) / 255.0, label=class_names[c])
        for c in range(classes)
    ]

    out_dir_p: Path | None = None
    if out_dir is not None:
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)

    figures: list[tuple[int, object]] = []
    for cls in range(classes):
        paths = by_class.get(cls, [])
        if not paths:
            _logger.info("Kelas %d (%s): 0 patch dominan — grid dilewati.", cls, class_names[cls])
            continue

        n_avail = len(paths)
        if n_avail < n_per_class:
            _logger.warning(
                "Kelas %d (%s): hanya %d/%d patch dominan tersedia (grid tak penuh).",
                cls, class_names[cls], n_avail, n_per_class,
            )
            chosen = list(paths)
        else:
            idx = rng.choice(n_avail, size=n_per_class, replace=False)
            chosen = [paths[i] for i in idx]

        fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 3.0, nrows * 1.7))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")

        for sample_i, p in enumerate(chosen):
            row, pair = divmod(sample_i, ncols)
            data = np.load(p)
            rgb = _truecolor_from_img(data["img"])
            lab_rgb = _label_rgb(data["lab"])
            ax_rgb, ax_lab = axes[row][pair * 2], axes[row][pair * 2 + 1]
            ax_rgb.imshow(rgb)
            ax_lab.imshow(lab_rgb)
            if row == 0:
                ax_rgb.set_title("RGB", fontsize=7)
                ax_lab.set_title("Label", fontsize=7)

        title = f"Kelas {cls} — {class_names[cls]}  ({min(n_avail, n_per_class)}/{n_per_class} sample"
        title += f", pool {n_avail})" if n_avail >= n_per_class else f", pool {n_avail} — TAK PENUH)"
        fig.suptitle(title, fontsize=12, y=1.10)
        fig.legend(handles=legend_handles, loc="upper center", ncol=classes,
                    fontsize=8, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.90))

        if out_dir_p is not None:
            safe = class_names[cls].lower().replace(" ", "_").replace("/", "_")
            fpath = out_dir_p / f"grid_kelas_{cls}_{safe}.png"
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor="white")
            _logger.info("Grid disimpan: %s", fpath)

        figures.append((cls, fig))

    _logger.info(
        "Selesai: %d grid dibuat (target %d kelas × %d = %d sample).",
        len(figures), classes, n_per_class, classes * n_per_class,
    )
    return figures


# ============================================================================
# Helper internal
# ============================================================================


def _resolve_patch_paths(patches: "str | os.PathLike[str] | list") -> list[Path]:
    """Normalisasi argumen patch → list path ``.npz``.

    Folder di-glob langsung (``p*.npz``, rekursif) — setara
    ``data.patches.list_patches`` tanpa menarik dependency berat (tqdm).
    """
    if isinstance(patches, (str, os.PathLike)):
        return sorted(Path(patches).rglob("p*.npz"))
    return [Path(p) for p in patches]


def _truecolor_from_img(img: "object", *, gain: float = 3.0) -> "object":
    """Citra (C,H,W) reflektansi [0,1] → RGB (H,W,3) true-color ter-stretch.

    Memakai band R=B4, G=B3, B=B2 (indeks 2,1,0), dengan percentile-stretch
    2–98% per-patch + gain ringan agar kontras enak dilihat.
    """
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(img, dtype="float32")
    rgb = np.stack([arr[i] for i in _TRUECOLOR_IDX], axis=-1)  # (H,W,3)
    rgb = np.nan_to_num(rgb, nan=0.0)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    if hi <= lo:
        hi = lo + 1e-6
    rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    return np.clip(rgb * (gain / 3.0) if gain != 3.0 else rgb, 0.0, 1.0)


def _label_rgb(lab: "object") -> "object":
    """Petakan label kelas (H,W) → citra warna (H,W,3) via ``PALETTE_RGB`` (uint8)."""
    import numpy as np  # noqa: PLC0415

    lab = np.asarray(lab)
    out = np.zeros((*lab.shape, 3), dtype="uint8")
    for cls, (r, g, b) in PALETTE_RGB.items():
        out[lab == cls] = (r, g, b)
    return out
