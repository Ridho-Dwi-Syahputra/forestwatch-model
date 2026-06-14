"""Arsitektur ResNet50-U-Net (segmentation-models-pytorch).

Sumber: PRD §A.5 Cell 6.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from forestwatch.constants import N_CLASSES

if TYPE_CHECKING:
    import torch.nn as nn


def build_unet(
    *,
    architecture: str = "unet",
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 6,
    classes: int = N_CLASSES,
    activation: str | None = None,
) -> "nn.Module":
    """Build U-Net (atau variant) dari segmentation-models-pytorch.

    Args:
        architecture: ``"unet"`` (default), ``"unet_scse"`` (Attention U-Net),
            ``"unetpp"`` (U-Net++), atau ``"deeplabv3plus"`` (DeepLabV3+).
        encoder_name: Backbone encoder (default ``resnet50``).
        encoder_weights: ``"imagenet"`` untuk transfer learning, ``None`` untuk scratch.
        in_channels: Jumlah band input (6 untuk Sentinel-2 sesuai PRD).
        classes: Jumlah kelas output (6 untuk PRD §A.3).
        activation: Aktivasi di output. ``None`` = logits (softmax di loss).

    Returns:
        ``torch.nn.Module``.

    Raises:
        ImportError: Jika ``segmentation-models-pytorch`` belum ter-install.
        ValueError: Untuk architecture yang tidak dikenal.
    """
    try:
        import segmentation_models_pytorch as smp  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Butuh segmentation-models-pytorch. Install: pip install -e \".[ml]\""
        ) from e

    common_kwargs = {
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": in_channels,
        "classes": classes,
        "activation": activation,
    }

    if architecture == "unet":
        return smp.Unet(**common_kwargs)
    if architecture == "unet_scse":
        # Attention U-Net (PRD §A.5 Cell 6 catatan: opsi upgrade)
        return smp.Unet(decoder_attention_type="scse", **common_kwargs)
    if architecture == "unetpp" or architecture == "unetplusplus":
        return smp.UnetPlusPlus(**common_kwargs)
    if architecture in ("deeplabv3plus", "deeplabv3+", "deeplabv3p"):
        # DeepLabV3+ (SLR Tabel 6: OA ~96%, IoU ~0.91) — atrous spatial pyramid
        # pooling, bagus untuk pola deforestasi tak beraturan.
        return smp.DeepLabV3Plus(**common_kwargs)
    raise ValueError(
        f"Architecture '{architecture}' tidak dikenal. "
        "Pilih: unet | unet_scse | unetpp | deeplabv3plus"
    )


def count_parameters(model: "nn.Module") -> int:
    """Hitung jumlah parameter trainable."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_to_onnx(
    model: "nn.Module",
    out_path: str | os.PathLike[str],
    *,
    in_channels: int = 6,
    patch_size: int = 256,
    opset_version: int = 13,
    dynamic_batch: bool = True,
) -> Path:
    """Ekspor model ke ONNX untuk reproduktibilitas (PRD §A.5 Cell 8).

    Args:
        model: ``nn.Module`` (sudah dalam mode ``.eval()``).
        out_path: Path output ``.onnx``.
        in_channels, patch_size: Untuk dummy input.
        opset_version: ONNX opset (default 13 untuk kompatibilitas luas).
        dynamic_batch: Bila ``True``, batch dimension dinamis.

    Returns:
        Path absolut file ONNX.
    """
    import torch  # noqa: PLC0415

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    dummy = torch.randn(1, in_channels, patch_size, patch_size, device=device)

    dynamic_axes = (
        {"image": {0: "batch"}, "mask": {0: "batch"}} if dynamic_batch else None
    )

    model.eval()
    torch.onnx.export(
        model,
        dummy,
        out.as_posix(),
        input_names=["image"],
        output_names=["mask"],
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
    )
    return out
