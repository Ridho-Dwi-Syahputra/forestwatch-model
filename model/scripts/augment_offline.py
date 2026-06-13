"""Augmentasi OFFLINE — perbanyak patch kelas minoritas (TRAIN-only) ke Drive.

TAHAP 2 (lihat ``docs/model/RENCANA_EKSEKUSI_LANJUTAN.md``). Dipanggil dari notebook
**Bagian 14** SETELAH split sumber-aware (test & val = Papua holdout; patch transfer
hanya masuk TRAIN). Output dipakai recompute ``class_distribution`` + class weights.

Prinsip (anti-overfit, sesuai RENCANA §"Yang TIDAK dilakukan"):
- Hanya patch **TRAIN minoritas** yang diaugmentasi, multiplier **KECIL (×2–3)** di
  atas basis raw besar dari transfer learning — BUKAN ledakan duplikasi ×3000.
- Transform **LABEL-SAFE**:
  * geometrik (flip H/V + rotate 90°) diterapkan ke **img DAN lab bersama**;
  * brightness/contrast **ringan** HANYA ke ``img`` (lab tak berubah).
- Output per-kelas: ``<out_dir>/<slug>/aug_*.npz`` berisi
  ``img, lab, source="augmented", parent, slug`` (+ ``region`` bila ada di sumber).

Geometri memakai **numpy murni** (tanpa albumentations) → deterministik, bebas
dependency berat, dan aman utk mask segmentation (flip/rot90 tak interpolasi).

Penggunaan CLI (Colab/lokal):

    python scripts/augment_offline.py \
        --src /content/drive/MyDrive/"Satria Data 3.0"/ForestWatch_Patches_Transfer \
        --out /content/drive/MyDrive/"Satria Data 3.0"/Augmented_Patches \
        --classes 2 3 4 5 6 --multiplier 2 --min-target-frac 0.01
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from tqdm.auto import tqdm

from forestwatch.constants import CLASS_SLUGS, N_CLASSES
from forestwatch.utils.io import save_json, save_npz
from forestwatch.utils.logging import get_logger

_logger = get_logger("forestwatch.augment_offline")

# Pool transform geometrik LABEL-SAFE: (flip, k_rot90). Cukup 8 varian unik untuk
# multiplier hingga ×9 tanpa pengulangan identik.
_GEOM_POOL: tuple[tuple[str | None, int], ...] = (
    ("h", 0),
    ("v", 0),
    (None, 1),
    (None, 2),
    (None, 3),
    ("h", 1),
    ("h", 2),
    ("v", 1),
)


def _resolve_patch_files(src_dirs: "str | os.PathLike[str] | list") -> list[Path]:
    """Normalisasi argumen sumber → list path ``p*.npz`` (rekursif, sorted)."""
    if isinstance(src_dirs, (str, os.PathLike)):
        src_dirs = [src_dirs]
    files: list[Path] = []
    for d in src_dirs:
        p = Path(d)
        if p.is_file() and p.suffix == ".npz":
            files.append(p)
        else:
            files.extend(sorted(p.rglob("p*.npz")))
    return files


def _apply_geom(arr, spec):
    """Terapkan satu spec geometrik (flip, k_rot90) pada (C,H,W) atau (H,W)."""
    import numpy as np  # noqa: PLC0415

    flip, k = spec
    out = arr
    if flip == "h":
        out = np.flip(out, axis=-1)
    elif flip == "v":
        out = np.flip(out, axis=-2)
    if k:
        out = np.rot90(out, k, axes=(-2, -1))
    return np.ascontiguousarray(out)


def _apply_brightness(img, rng, *, brightness_limit: float, contrast_limit: float):
    """Brightness/contrast ringan HANYA pada img (reflektansi [0,1]). Lab tak disentuh."""
    import numpy as np  # noqa: PLC0415

    c = 1.0 + float(rng.uniform(-contrast_limit, contrast_limit))
    b = float(rng.uniform(-brightness_limit, brightness_limit))
    return np.clip(img.astype("float32") * c + b, 0.0, 1.0).astype("float32")


def _target_class_of(lab, classes: tuple[int, ...], min_target_frac: float) -> int | None:
    """Pilih kelas target dgn fraksi piksel terbesar (≥ ``min_target_frac``), else None."""
    import numpy as np  # noqa: PLC0415

    counts = np.bincount(np.asarray(lab).ravel(), minlength=N_CLASSES)[:N_CLASSES]
    total = float(counts.sum()) or 1.0
    best_cls, best_frac = None, min_target_frac
    for c in classes:
        frac = counts[c] / total
        if frac >= best_frac:
            best_cls, best_frac = c, frac
    return best_cls


def _augment_one(
    f: Path,
    rng: "np.random.Generator",
    file_idx: int,
    *,
    classes: tuple[int, ...],
    min_target_frac: float,
    multiplier_map: dict[int, float],
    out_root: Path,
    brightness_limit: float,
    contrast_limit: float,
    brightness_p: float,
) -> tuple[int | None, int]:
    """Augmentasi 1 patch sumber → N salinan (N dari ``multiplier_map[target_cls]``).

    Return ``(target_cls, n_generated)``. ``target_cls`` ``None`` bila patch tak
    terbaca atau tak punya kelas minoritas yang cukup (``n_generated`` selalu 0
    dalam kasus itu). Dipanggil dari thread worker — ``rng`` harus independen
    per-file (lihat ``SeedSequence.spawn`` di ``augment_offline``) supaya
    thread-safe & deterministik. ``file_idx`` (posisi ``f`` di list sumber)
    diselipkan ke nama file output agar tetap unik lintas subfolder sumber
    meski ``(parent.name, stem)`` kebetulan sama (struktur sumber bertingkat,
    mis. ``<region>/<slug>/p00000.npz``).
    """
    import numpy as np  # noqa: PLC0415

    try:
        data = np.load(f)
        img = np.asarray(data["img"], dtype="float32")
        lab = np.asarray(data["lab"]).astype("uint8")
    except (OSError, KeyError, ValueError) as exc:
        _logger.warning("Lewati patch %s (gagal baca: %s)", f, exc)
        return None, 0

    target_cls = _target_class_of(lab, classes, min_target_frac)
    if target_cls is None:
        return None, 0  # tak ada kelas minoritas yang cukup → tak diaugmentasi

    # multiplier fractional: extra=mult-1 -> n_base salinan pasti + 1 salinan
    # probabilistik dgn peluang extra_prob (mis. ×1.5 -> 0/1 salinan ~50/50).
    mult = multiplier_map.get(target_cls, 1.0)
    extra = max(0.0, mult - 1.0)
    n_base = int(extra)
    extra_prob = extra - n_base
    n_copies = n_base + (1 if rng.random() < extra_prob else 0)
    if n_copies == 0:
        return target_cls, 0

    slug = CLASS_SLUGS[target_cls]
    cls_dir = out_root / slug
    # region (opsional) ikut diturunkan agar telusur sumber tetap ada.
    region = str(data["region"]) if "region" in getattr(data, "files", []) else ""

    # Pilih spec geometrik unik (atau dgn pengulangan bila multiplier > pool).
    replace = n_copies > len(_GEOM_POOL)
    idx = rng.choice(len(_GEOM_POOL), size=n_copies, replace=replace)

    n_generated = 0
    for k_copy, spec_i in enumerate(idx):
        spec = _GEOM_POOL[int(spec_i)]
        aug_img = _apply_geom(img, spec)
        aug_lab = _apply_geom(lab, spec).astype("uint8")
        if rng.random() < brightness_p:
            aug_img = _apply_brightness(
                aug_img, rng,
                brightness_limit=brightness_limit, contrast_limit=contrast_limit,
            )
        out_name = f"aug_{file_idx:06d}_{f.parent.name}_{f.stem}_{k_copy:02d}.npz"
        save_npz(
            cls_dir / out_name,
            img=aug_img,
            lab=aug_lab,
            source="augmented",
            parent=f.name,
            slug=slug,
            region=region,
        )
        n_generated += 1

    return target_cls, n_generated


def augment_offline(
    src_dirs: "str | os.PathLike[str] | list",
    out_dir: "str | os.PathLike[str]",
    *,
    classes: tuple[int, ...] = (2, 3, 4, 5, 6),
    multiplier: "float | dict[int, float]" = 2.0,
    min_target_frac: float = 0.01,
    brightness_limit: float = 0.1,
    contrast_limit: float = 0.1,
    brightness_p: float = 0.5,
    seed: int = 42,
    manifest_path: "str | os.PathLike[str] | None" = None,
    max_workers: int = 32,
    clean: bool = True,
) -> dict[int, int]:
    """Augmentasi offline patch minoritas → ``out_dir/<slug>/aug_*.npz``.

    Args:
        src_dirs: Folder (atau list) berisi patch TRAIN ``p*.npz`` — HANYA train.
        out_dir: Folder output; sub-folder per-kelas (slug) dibuat otomatis.
        classes: Kelas minoritas yang diaugmentasi (default 2..6).
        multiplier: Total lipatan per patch target. Skalar (uniform utk semua
            ``classes``) atau dict ``{class_id: multiplier}`` per-kelas, mis.
            ``{2: 1.5, 3: 1, 4: 1, 5: 2, 6: 2.5}``. Boleh fractional: ×1.5 berarti
            tiap patch target punya peluang 50% menghasilkan 1 salinan (rata-rata
            0.5 salinan/patch); ×1 = tak ada salinan (extra=0).
        min_target_frac: Patch diaugmentasi hanya bila fraksi salah satu kelas
            target ≥ nilai ini (longgar, default 1%).
        brightness_limit, contrast_limit, brightness_p: Param radiometrik ringan
            (selaras ``configs/default.yaml``; brightness HANYA ke img).
        seed: Seed RNG. Tiap file mendapat RNG anak independen lewat
            ``SeedSequence.spawn`` → hasil deterministik sama persis tak peduli
            urutan eksekusi paralel.
        manifest_path: Bila di-set, simpan manifest JSON.
        max_workers: Jumlah thread paralel (load .npz dari Drive + kompresi
            ``savez_compressed`` sama-sama melepas GIL, jadi ThreadPoolExecutor
            efektif; default 32).
        clean: Bila ``True`` (default), hapus & buat ulang ``out_dir/<slug>/``
            untuk setiap ``c in classes`` sebelum generate — membuat output
            idempoten terhadap ``multiplier`` saat ini (tak ada sisa
            ``aug_*.npz`` dari run sebelumnya dgn multiplier berbeda).

    Returns:
        Dict ``{class_id: n_generated}`` (jumlah patch augmentasi per kelas).
    """
    import numpy as np  # noqa: PLC0415

    out_root = Path(out_dir)

    if isinstance(multiplier, dict):
        multiplier_map = {c: float(multiplier.get(c, 1.0)) for c in classes}
    else:
        multiplier_map = {c: float(multiplier) for c in classes}

    if clean:
        for c in classes:
            slug_dir = out_root / CLASS_SLUGS[c]
            shutil.rmtree(slug_dir, ignore_errors=True)
            slug_dir.mkdir(parents=True, exist_ok=True)

    files = _resolve_patch_files(src_dirs)
    generated: dict[int, int] = {c: 0 for c in classes}
    n_src_used = 0

    if all(m <= 1.0 for m in multiplier_map.values()):
        _logger.warning(
            "Semua multiplier <= 1.0 (%s) → kemungkinan tak ada salinan augmentasi dibuat.",
            multiplier_map,
        )

    # RNG anak independen per file (deterministik & thread-safe), via SeedSequence.spawn.
    child_seeds = np.random.SeedSequence(seed).spawn(len(files))
    rngs = [np.random.default_rng(s) for s in child_seeds]

    worker = partial(
        _augment_one,
        classes=tuple(classes),
        min_target_frac=min_target_frac,
        multiplier_map=multiplier_map,
        out_root=out_root,
        brightness_limit=brightness_limit,
        contrast_limit=contrast_limit,
        brightness_p=brightness_p,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        results = exe.map(worker, files, rngs, range(len(files)))
        for target_cls, n_gen in tqdm(
            results, total=len(files), desc="Augmentasi offline", unit="patch"
        ):
            if target_cls is None:
                continue
            n_src_used += 1
            generated[target_cls] += n_gen

    manifest = {
        "by_class": {str(c): int(n) for c, n in generated.items()},
        "by_slug": {CLASS_SLUGS[c]: int(n) for c, n in generated.items()},
        "total_generated": int(sum(generated.values())),
        "n_source_patches_augmented": int(n_src_used),
        "params": {
            "classes": list(classes),
            "multiplier": multiplier if isinstance(multiplier, dict) else float(multiplier),
            "multiplier_map": {str(c): round(v, 4) for c, v in multiplier_map.items()},
            "min_target_frac": float(min_target_frac),
            "brightness_limit": float(brightness_limit),
            "contrast_limit": float(contrast_limit),
            "brightness_p": float(brightness_p),
            "seed": int(seed),
            "max_workers": int(max_workers),
            "clean": bool(clean),
        },
    }
    if manifest_path is not None:
        save_json(manifest, manifest_path)
    _logger.info(
        "Augmentasi offline selesai: %d patch baru dari %d sumber → %s",
        manifest["total_generated"], n_src_used, out_root,
    )
    return generated


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="augment_offline",
        description="Augmentasi offline patch minoritas (train-only) → Augmented_Patches/<slug>/.",
    )
    p.add_argument("--src", nargs="+", required=True, help="Folder patch TRAIN (boleh >1).")
    p.add_argument("--out", required=True, help="Folder output Augmented_Patches.")
    p.add_argument("--classes", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    p.add_argument("--multiplier", type=float, default=2.0)
    p.add_argument("--min-target-frac", type=float, default=0.01)
    p.add_argument("--brightness-limit", type=float, default=0.1)
    p.add_argument("--contrast-limit", type=float, default=0.1)
    p.add_argument("--brightness-p", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-workers", type=int, default=32, help="Jumlah thread paralel (default 32).")
    p.add_argument("--manifest", default=None, help="Path JSON manifest (opsional).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gen = augment_offline(
        args.src,
        args.out,
        classes=tuple(args.classes),
        multiplier=args.multiplier,
        min_target_frac=args.min_target_frac,
        brightness_limit=args.brightness_limit,
        contrast_limit=args.contrast_limit,
        brightness_p=args.brightness_p,
        seed=args.seed,
        manifest_path=args.manifest,
        max_workers=args.max_workers,
    )
    total = sum(gen.values())
    print(f"Selesai: {total} patch augmentasi dibuat. Per-kelas: {gen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
