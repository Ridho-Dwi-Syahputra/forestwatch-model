"""Test routing build_unet (unet | unet_scse | unetpp | deeplabv3plus).

Semua test di-skip otomatis bila torch/segmentation-models-pytorch tidak ada
(via pytest.importorskip), sehingga aman di lingkungan minimal.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
smp = pytest.importorskip("segmentation_models_pytorch")

from forestwatch.model.architecture import build_unet  # noqa: E402


def _build(arch: str):
    # encoder_weights=None -> tidak download bobot ImageNet saat test.
    return build_unet(architecture=arch, encoder_weights=None, in_channels=6, classes=7)


def test_unet_default():
    assert isinstance(_build("unet"), smp.Unet)


def test_unet_scse_is_unet_with_attention():
    m = _build("unet_scse")
    assert isinstance(m, smp.Unet)


def test_unetpp_routes_to_unetplusplus():
    assert isinstance(_build("unetpp"), smp.UnetPlusPlus)


def test_deeplabv3plus_routes_to_deeplabv3plus():
    assert isinstance(_build("deeplabv3plus"), smp.DeepLabV3Plus)


@pytest.mark.parametrize("alias", ["deeplabv3+", "deeplabv3p"])
def test_deeplabv3plus_aliases(alias):
    assert isinstance(_build(alias), smp.DeepLabV3Plus)


def test_unknown_architecture_raises():
    with pytest.raises(ValueError, match="tidak dikenal"):
        _build("totally_unknown_arch")
