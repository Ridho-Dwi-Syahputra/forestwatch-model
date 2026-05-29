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


def metric_summary(
    cm: "np.ndarray",
    *,
    class_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Bundle CM → dict {overall_accuracy, mean_iou, per_class: [...]}."""
    iou = compute_iou_per_class(cm)
    f1 = compute_f1_per_class(cm)
    oa = overall_accuracy(cm)
    if class_names is None:
        class_names = tuple(f"class_{i}" for i in range(len(iou)))
    per_class = [
        {"class": class_names[i], "iou": float(iou[i]), "f1": float(f1[i])}
        for i in range(len(iou))
    ]
    return {
        "overall_accuracy": float(oa),
        "mean_iou": float(iou.mean()),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
