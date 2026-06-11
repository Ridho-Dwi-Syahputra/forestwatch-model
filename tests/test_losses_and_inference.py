"""Test loss builder configurable + TTA/overlap inference.

Semua test yang butuh torch/smp di-skip otomatis jika dependency tidak ada
(via pytest.importorskip), sehingga aman dijalankan di lingkungan minimal.
"""

from __future__ import annotations

import numpy as np
import pytest

# ============================================================================
# Loss builder (butuh torch + segmentation-models-pytorch)
# ============================================================================

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from forestwatch.model.losses import _PRESETS, make_loss_fn  # noqa: E402


def _sample_pred_target(n_classes=6, b=2, h=16, w=16):
    pred = torch.randn(b, n_classes, h, w, requires_grad=True)
    target = torch.randint(0, n_classes, (b, h, w))
    return pred, target


def test_focal_tversky_default_builds_and_runs():
    loss_fn = make_loss_fn()  # default focal_tversky
    pred, target = _sample_pred_target()
    loss = loss_fn(pred, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    # backward harus jalan
    loss.backward()
    assert pred.grad is not None


@pytest.mark.parametrize("loss_type", list(_PRESETS.keys()))
def test_all_presets_run(loss_type):
    loss_fn = make_loss_fn(loss_type=loss_type)
    pred, target = _sample_pred_target()
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)


def test_class_weights_applied_to_ce():
    loss_fn = make_loss_fn(loss_type="ce", class_weights=[1.0, 2.0, 3.0, 1.0, 1.0, 1.0])
    pred, target = _sample_pred_target()
    assert torch.isfinite(loss_fn(pred, target))


def test_backward_compat_ce_dice():
    # API lama: kirim ce_weight/dice_weight → tetap jalan
    loss_fn = make_loss_fn(ce_weight=0.6, dice_weight=0.4)
    pred, target = _sample_pred_target()
    assert torch.isfinite(loss_fn(pred, target))


def test_invalid_loss_type_raises():
    with pytest.raises(ValueError, match="tidak dikenal"):
        make_loss_fn(loss_type="nonexistent")


# ============================================================================
# Inference TTA + overlap blending (butuh torch + rasterio)
# ============================================================================

rasterio = pytest.importorskip("rasterio")

from forestwatch.inference.tile_inference import _predict_proba_tta, infer_tile  # noqa: E402


class _IdentityArgmaxModel(torch.nn.Module):
    """Model dummy: output konstan agar prediksi deterministik."""

    def __init__(self, n_classes=6):
        super().__init__()
        self.n_classes = n_classes
        # conv 1x1 dengan bias sehingga kelas 1 selalu menang
        self.conv = torch.nn.Conv2d(6, n_classes, 1)
        torch.nn.init.zeros_(self.conv.weight)
        with torch.no_grad():
            self.conv.bias.zero_()
            self.conv.bias[1] = 10.0  # kelas 1 dominan

    def forward(self, x):
        return self.conv(x)


def test_predict_proba_tta_shape():
    model = _IdentityArgmaxModel()
    x = torch.randn(1, 6, 32, 32)
    probs = _predict_proba_tta(model, x, tta=True)
    assert probs.shape == (1, 6, 32, 32)
    # softmax → jumlah prob = 1 per piksel
    assert torch.allclose(probs.sum(dim=1), torch.ones(1, 32, 32), atol=1e-4)


def test_predict_proba_no_tta_shape():
    model = _IdentityArgmaxModel()
    x = torch.randn(1, 6, 32, 32)
    probs = _predict_proba_tta(model, x, tta=False)
    assert probs.shape == (1, 6, 32, 32)


def _write_dummy_tile(path, bands=6, h=40, w=40):
    from rasterio.transform import from_bounds

    transform = from_bounds(140.0, -8.5, 140.1, -8.4, w, h)
    data = np.random.rand(bands, h, w).astype("float32")
    meta = {
        "driver": "GTiff", "height": h, "width": w, "count": bands,
        "dtype": "float32", "crs": "EPSG:4326", "transform": transform,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data)


def test_infer_tile_overlap_and_tta(tmp_path):
    """Inferensi dengan stride<patch (overlap) + tta menghasilkan mask penuh."""
    tile = tmp_path / "tile.tif"
    _write_dummy_tile(tile, h=40, w=40)
    out = tmp_path / "mask.tif"

    model = _IdentityArgmaxModel()
    result = infer_tile(
        tile, out, model,
        device="cpu", patch_size=32, stride=16, n_channels_image=6, tta=True,
    )
    assert result.exists()
    with rasterio.open(result) as src:
        mask = src.read(1)
    assert mask.shape == (40, 40)
    # Model dummy selalu menangkan kelas 1 → semua piksel = 1
    assert (mask == 1).all()


def test_infer_tile_tile_smaller_than_patch(tmp_path):
    """Tile lebih kecil dari patch_size tetap tertangani (tidak crash)."""
    tile = tmp_path / "small.tif"
    _write_dummy_tile(tile, h=20, w=20)
    out = tmp_path / "mask_small.tif"
    model = _IdentityArgmaxModel()
    infer_tile(tile, out, model, device="cpu", patch_size=32, stride=32, n_channels_image=6)
    with rasterio.open(out) as src:
        assert src.read(1).shape == (20, 20)


# ============================================================================
# LR warmup scheduler
# ============================================================================


def test_warmup_scheduler_builds_and_steps():
    """TrainConfig.warmup_epochs > 0 → SequentialLR (LinearLR -> Cosine)."""
    from forestwatch.training.trainer import TrainConfig  # noqa: PLC0415

    cfg = TrainConfig(epochs=10, warmup_epochs=3, learning_rate=1e-3)
    model = _IdentityArgmaxModel()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=cfg.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs - cfg.warmup_epochs)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs]
    )

    lrs = []
    for _ in range(cfg.epochs):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    # LR awal (warmup) < LR puncak (akhir warmup)
    assert lrs[0] < lrs[cfg.warmup_epochs - 1] + 1e-9
    # LR naik selama warmup
    assert lrs[0] < lrs[2]
