"""Combined loss = α·CrossEntropy(weighted) + β·DiceLoss.

Sumber: PRD §A.5 Cell 6 ``loss_fn = lambda p, t: 0.6*ce(p,t) + 0.4*dice(p,t)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def combined_loss(
    pred,
    target,
    *,
    ce_loss,
    dice_loss,
    ce_weight: float = 0.6,
    dice_weight: float = 0.4,
):
    """Hitung weighted sum dari CE dan Dice loss."""
    return ce_weight * ce_loss(pred, target) + dice_weight * dice_loss(pred, target)


def make_loss_fn(
    *,
    class_weights: Sequence[float] | None = None,
    ce_weight: float = 0.6,
    dice_weight: float = 0.4,
    device: "str | torch.device | None" = None,
):
    """Factory: return callable ``loss_fn(pred, target) -> scalar``.

    Args:
        class_weights: Bobot per kelas untuk CE (panjang harus = num classes).
        ce_weight, dice_weight: Bobot komponen loss.
        device: Device untuk class_weights tensor.

    Returns:
        Callable.
    """
    try:
        import segmentation_models_pytorch as smp  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Butuh torch + segmentation-models-pytorch. Install: pip install -e \".[ml]\""
        ) from e

    if class_weights is not None:
        cw = torch.tensor(list(class_weights), dtype=torch.float32)
        if device is not None:
            cw = cw.to(device)
        ce = nn.CrossEntropyLoss(weight=cw)
    else:
        ce = nn.CrossEntropyLoss()

    dice = smp.losses.DiceLoss(mode="multiclass")

    def _loss_fn(pred, target):
        return combined_loss(
            pred,
            target,
            ce_loss=ce,
            dice_loss=dice,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
        )

    return _loss_fn
