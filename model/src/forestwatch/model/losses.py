"""Loss functions — configurable combo untuk segmentasi multiclass.

Mendukung kombinasi berbobot dari beberapa komponen loss yang tersedia di
``segmentation_models_pytorch.losses``:

- ``ce``      : CrossEntropy (boleh weighted per kelas)
- ``dice``    : DiceLoss (multiclass)
- ``focal``   : FocalLoss — fokus pada contoh sulit
- ``tversky`` : TverskyLoss — α/β asimetris, β>α menalti false-negative
- ``lovasz``  : LovaszLoss — optimasi IoU langsung

Default proyek (berbasis literatur class-imbalance, lihat PRD §A & review
Jelas dkk. 2024): ``focal_tversky`` = 0.5·Focal + 0.5·Tversky(α=0.3, β=0.7).
β>α menaikkan recall kelas minoritas (Sawit, Tambang).

Sumber: Abraham & Khan (Focal Tversky, 2019); Loss Functions Survey
(arXiv 2312.05391, 2023).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# Preset kombinasi: nama → list (komponen, bobot)
_PRESETS: dict[str, list[tuple[str, float]]] = {
    "ce_dice": [("ce", 0.6), ("dice", 0.4)],
    "focal_tversky": [("focal", 0.5), ("tversky", 0.5)],
    "ce_dice_lovasz": [("ce", 0.4), ("dice", 0.3), ("lovasz", 0.3)],
    "tversky": [("tversky", 1.0)],
    "focal": [("focal", 1.0)],
    "lovasz": [("lovasz", 1.0)],
    "ce": [("ce", 1.0)],
    "dice": [("dice", 1.0)],
}


def combined_loss(
    pred,
    target,
    *,
    ce_loss,
    dice_loss,
    ce_weight: float = 0.6,
    dice_weight: float = 0.4,
):
    """Backward-compat: weighted sum CE + Dice (dipakai test lama)."""
    return ce_weight * ce_loss(pred, target) + dice_weight * dice_loss(pred, target)


def _build_component(
    name: str,
    *,
    class_weights=None,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    focal_gamma: float = 2.0,
):
    """Instansiasi satu komponen loss berdasarkan nama."""
    import segmentation_models_pytorch as smp  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "dice":
        return smp.losses.DiceLoss(mode="multiclass")
    if name == "focal":
        return smp.losses.FocalLoss(mode="multiclass", gamma=focal_gamma)
    if name == "tversky":
        # smp TverskyLoss multiclass; alpha untuk FP, beta untuk FN
        return smp.losses.TverskyLoss(mode="multiclass", alpha=tversky_alpha, beta=tversky_beta)
    if name == "lovasz":
        return smp.losses.LovaszLoss(mode="multiclass")
    raise ValueError(f"Komponen loss '{name}' tidak dikenal.")


def make_loss_fn(
    *,
    loss_type: str = "focal_tversky",
    components: Sequence[tuple[str, float]] | None = None,
    class_weights: Sequence[float] | None = None,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    focal_gamma: float = 2.0,
    device: "str | torch.device | None" = None,
    # Backward-compat lama (dipakai test & notebook lama):
    ce_weight: float | None = None,
    dice_weight: float | None = None,
):
    """Factory: return callable ``loss_fn(pred, target) -> scalar``.

    Args:
        loss_type: Nama preset (lihat ``_PRESETS``). Default ``focal_tversky``.
        components: Override eksplisit ``[(nama, bobot), ...]``. Bila diberikan,
            mengabaikan ``loss_type``.
        class_weights: Bobot per kelas untuk komponen CE.
        tversky_alpha, tversky_beta: Parameter Tversky (β>α → recall naik).
        focal_gamma: Parameter Focal.
        device: Device untuk tensor class_weights.
        ce_weight, dice_weight: **Backward-compat** — jika salah satu di-set,
            paksa pakai kombinasi CE+Dice dengan bobot tsb (perilaku lama).

    Returns:
        Callable loss function.
    """
    try:
        import segmentation_models_pytorch as smp  # noqa: PLC0415, F401
        import torch  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Butuh torch + segmentation-models-pytorch. Install: pip install -e \".[ml]\""
        ) from e

    # Backward-compat: kalau caller lama kirim ce_weight/dice_weight → CE+Dice
    if ce_weight is not None or dice_weight is not None:
        cw = 0.6 if ce_weight is None else ce_weight
        dw = 0.4 if dice_weight is None else dice_weight
        components = [("ce", cw), ("dice", dw)]

    if components is None:
        if loss_type not in _PRESETS:
            raise ValueError(
                f"loss_type '{loss_type}' tidak dikenal. Pilihan: {sorted(_PRESETS)}"
            )
        components = _PRESETS[loss_type]

    # Siapkan class_weights tensor
    cw_tensor = None
    if class_weights is not None:
        cw_tensor = torch.tensor(list(class_weights), dtype=torch.float32)
        if device is not None:
            cw_tensor = cw_tensor.to(device)

    # Bangun tiap komponen sekali (reuse antar batch)
    built: list[tuple[object, float]] = []
    for name, weight in components:
        comp = _build_component(
            name,
            class_weights=cw_tensor,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            focal_gamma=focal_gamma,
        )
        built.append((comp, float(weight)))

    def _loss_fn(pred, target):
        total = None
        for comp, weight in built:
            term = weight * comp(pred, target)
            total = term if total is None else total + term
        return total

    return _loss_fn
