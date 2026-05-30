"""Metrics & seeding helper untuk training."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed di python, numpy, torch (CPU + CUDA) untuk reproduktibilitas."""
    random.seed(seed)
    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_confusion_matrix(
    pred: "np.ndarray", target: "np.ndarray", *, n_classes: int = 6
) -> "np.ndarray":
    """Confusion matrix ``(n_classes, n_classes)`` dengan rows=true, cols=pred.

    Tidak butuh sklearn — pakai ``np.bincount`` agar cepat di dataset besar.
    """
    import numpy as np  # noqa: PLC0415

    pred_flat = np.asarray(pred).ravel()
    target_flat = np.asarray(target).ravel()
    if pred_flat.shape != target_flat.shape:
        raise ValueError(f"shape mismatch: pred {pred_flat.shape} vs target {target_flat.shape}")
    mask = (target_flat >= 0) & (target_flat < n_classes) & (pred_flat < n_classes)
    cm = np.bincount(
        n_classes * target_flat[mask].astype(np.int64) + pred_flat[mask].astype(np.int64),
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)
    return cm


def compute_iou_per_class(cm: "np.ndarray", *, eps: float = 1e-9) -> "np.ndarray":
    """IoU per kelas dari confusion matrix.

    IoU(c) = diag(c) / (row_sum(c) + col_sum(c) - diag(c))
    """
    import numpy as np  # noqa: PLC0415

    cm = np.asarray(cm)
    diag = np.diag(cm).astype(np.float64)
    row_sum = cm.sum(axis=1).astype(np.float64)
    col_sum = cm.sum(axis=0).astype(np.float64)
    union = row_sum + col_sum - diag
    return diag / (union + eps)


def compute_f1_per_class(cm: "np.ndarray", *, eps: float = 1e-9) -> "np.ndarray":
    """F1 per kelas dari confusion matrix. F1 = 2*IoU / (1+IoU)."""
    import numpy as np  # noqa: PLC0415

    iou = compute_iou_per_class(cm, eps=eps)
    return 2 * iou / (1 + iou + eps)


def overall_accuracy(cm: "np.ndarray", *, eps: float = 1e-9) -> float:
    """OA = diag.sum() / total.sum()."""
    import numpy as np  # noqa: PLC0415

    total = float(cm.sum())
    if total < eps:
        return 0.0
    return float(np.diag(cm).sum()) / total


def cohen_kappa(cm: "np.ndarray", *, eps: float = 1e-9) -> float:
    """Cohen's Kappa dari confusion matrix.

    κ = (p_o - p_e) / (1 - p_e), dengan p_o = observed agreement (= OA),
    p_e = expected agreement by chance. Metrik standar accuracy assessment
    di remote sensing (mengoreksi kesepakatan akibat kebetulan).

    Interpretasi umum: >0.80 sangat baik, 0.60-0.80 baik, 0.40-0.60 sedang.
    """
    import numpy as np  # noqa: PLC0415

    cm = np.asarray(cm, dtype=np.float64)
    n = cm.sum()
    if n < eps:
        return 0.0
    p_o = np.diag(cm).sum() / n
    p_e = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n)
    denom = 1.0 - p_e
    if abs(denom) < eps:
        return 0.0
    return float((p_o - p_e) / denom)


def median_frequency_weights(
    distribution: "dict[int, int]",
    *,
    n_classes: int = 6,
    clip_min: float = 0.3,
    clip_max: float = 5.0,
    eps: float = 1e-9,
) -> list[float]:
    """Bobot kelas median-frequency balancing (Eigen & Fergus, 2015).

    weight(c) = median(freq) / freq(c), dengan freq(c) = count(c) / total.
    Kelas langka dapat bobot besar, kelas dominan (Hutan) dapat bobot kecil.
    Hasil di-clip ke [clip_min, clip_max] agar training stabil.

    Args:
        distribution: Dict ``{class_id: pixel_count}`` dari
            ``data.patches.compute_class_distribution``.
        n_classes: Jumlah kelas.
        clip_min, clip_max: Batas bobot.

    Returns:
        List bobot panjang ``n_classes`` (urut class_id 0..n-1).
    """
    import numpy as np  # noqa: PLC0415

    counts = np.array([float(distribution.get(c, 0)) for c in range(n_classes)])
    total = counts.sum()
    if total < eps:
        return [1.0] * n_classes
    freqs = counts / total
    # Median hanya dari kelas yang hadir (freq > 0)
    present = freqs[freqs > 0]
    med = float(np.median(present)) if present.size else 1.0
    weights = np.where(freqs > 0, med / (freqs + eps), clip_max)
    weights = np.clip(weights, clip_min, clip_max)
    return [round(float(w), 3) for w in weights]


def metric_summary(
    cm: "np.ndarray",
    *,
    class_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Bundle CM → dict {overall_accuracy, mean_iou, per_class: [...]}."""
    iou = compute_iou_per_class(cm)
    f1 = compute_f1_per_class(cm)
    oa = overall_accuracy(cm)
    kappa = cohen_kappa(cm)
    if class_names is None:
        class_names = tuple(f"class_{i}" for i in range(len(iou)))
    per_class = [
        {"class": class_names[i], "iou": float(iou[i]), "f1": float(f1[i])}
        for i in range(len(iou))
    ]
    return {
        "overall_accuracy": float(oa),
        "mean_iou": float(iou.mean()),
        "kappa": float(kappa),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
