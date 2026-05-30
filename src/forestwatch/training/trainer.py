"""Training loop dengan AMP + early stopping + checkpoint.

Sumber: PRD §A.5 Cell 7 — di-refactor sebagai fungsi yang testable + log lengkap.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from forestwatch.constants import N_CLASSES
from forestwatch.training.metrics import set_seed
from forestwatch.utils.logging import get_logger

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

_logger = get_logger("forestwatch.training")


@dataclass
class TrainConfig:
    """Hyperparameters untuk satu run training."""

    epochs: int = 50
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True
    seed: int = 42
    n_classes: int = N_CLASSES
    ckpt_path: str = "outputs/best_model.pt"
    warmup_epochs: int = 3  # linear warmup sebelum cosine (0 = nonaktif)
    log_every: int = 1  # log every N batches in train loop (untuk debug)
    history: list[dict[str, float]] = field(default_factory=list)


def train(
    model,
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    *,
    loss_fn,
    cfg: TrainConfig | None = None,
    optimizer: "torch.optim.Optimizer | None" = None,
    scheduler: "torch.optim.lr_scheduler._LRScheduler | None" = None,
    device: "str | torch.device | None" = None,
) -> dict[str, object]:
    """Training loop dengan AMP + early stopping.

    Args:
        model: ``nn.Module``.
        train_loader, val_loader: PyTorch DataLoaders.
        loss_fn: Callable ``(pred, target) -> scalar`` (lihat ``model.losses``).
        cfg: ``TrainConfig``. Default: parameter dari PRD.
        optimizer: Bila ``None``, AdamW(lr, weight_decay) dipakai.
        scheduler: Bila ``None``, CosineAnnealingLR(T_max=epochs) dipakai.
        device: Default: ``cuda`` if available, else ``cpu``.

    Returns:
        Dict ``{"best_val_iou", "best_epoch", "history": [...], "ckpt_path"}``.
    """
    try:
        import torch  # noqa: PLC0415
        from torch.amp import GradScaler, autocast  # noqa: PLC0415
        from torchmetrics.classification import MulticlassJaccardIndex  # noqa: PLC0415
        from tqdm import tqdm  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("Butuh torch + torchmetrics + tqdm. Install: pip install -e \".[ml]\"") from e

    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device_t)
    use_amp = bool(cfg.amp and device_t.type == "cuda")

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    if scheduler is None:
        warmup = int(getattr(cfg, "warmup_epochs", 0) or 0)
        if warmup > 0 and warmup < cfg.epochs:
            # Linear warmup (lr naik dari 10% → 100%) lalu cosine annealing.
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs - warmup
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup]
            )
            _logger.info("Scheduler: LinearLR warmup %d ep -> CosineAnnealingLR.", warmup)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    iou_metric = MulticlassJaccardIndex(num_classes=cfg.n_classes, average="macro").to(device_t)

    scaler = GradScaler(device_t.type) if use_amp else None
    best_iou = 0.0
    best_epoch = -1
    wait = 0

    ckpt_path = Path(cfg.ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    _logger.info("Mulai training: device=%s, AMP=%s, epochs=%d", device_t, use_amp, cfg.epochs)

    for ep in range(1, cfg.epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss_sum = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {ep:02d} [train]", leave=False)
        t0 = time.time()
        for x, y in train_iter:
            x, y = x.to(device_t, non_blocking=True), y.to(device_t, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with autocast(device_type=device_t.type):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
            train_loss_sum += float(loss.item())
        scheduler.step()
        train_loss = train_loss_sum / max(len(train_loader), 1)

        # ---------------- VAL ----------------
        model.eval()
        iou_metric.reset()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {ep:02d} [val  ]", leave=False):
                x, y = x.to(device_t, non_blocking=True), y.to(device_t, non_blocking=True)
                pred = model(x)
                val_loss_sum += float(loss_fn(pred, y).item())
                iou_metric.update(pred.argmax(1), y)
        val_loss = val_loss_sum / max(len(val_loader), 1)
        val_miou = float(iou_metric.compute().item())

        epoch_time = time.time() - t0
        _logger.info(
            "Ep %02d/%d | train_loss %.4f | val_loss %.4f | val_mIoU %.4f | %.1fs",
            ep, cfg.epochs, train_loss, val_loss, val_miou, epoch_time,
        )
        cfg.history.append(
            {
                "epoch": ep,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_miou": val_miou,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_sec": epoch_time,
            }
        )

        # ---------------- EARLY STOPPING ----------------
        if val_miou > best_iou:
            best_iou = val_miou
            best_epoch = ep
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
            _logger.info("  -> checkpoint disimpan (val_mIoU=%.4f) di %s", best_iou, ckpt_path)
        else:
            wait += 1
            if wait >= cfg.patience:
                _logger.info("Early stopping di epoch %d (patience=%d).", ep, cfg.patience)
                break

    return {
        "best_val_iou": best_iou,
        "best_epoch": best_epoch,
        "history": cfg.history,
        "ckpt_path": str(ckpt_path),
    }


def evaluate(
    model,
    loader: "DataLoader",
    *,
    n_classes: int = N_CLASSES,
    device: "str | torch.device | None" = None,
) -> "tuple[object, object]":
    """Kumpulkan prediksi & label dari ``loader`` (untuk confusion matrix).

    Returns:
        ``(pred_array, target_array)`` numpy flat 1D.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device_t).eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device_t, non_blocking=True)
            p = model(x).argmax(1).cpu().numpy().astype(np.uint8)
            preds.append(p.ravel())
            targets.append(y.numpy().astype(np.uint8).ravel())
    return np.concatenate(preds), np.concatenate(targets)
