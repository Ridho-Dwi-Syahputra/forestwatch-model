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

    epochs: int = 80
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True
    seed: int = 42
    n_classes: int = N_CLASSES
    ckpt_path: str = "outputs/best_model.pt"
    warmup_epochs: int = 3  # linear warmup sebelum cosine (0 = nonaktif)
    log_every: int = 1  # log every N batches in train loop (untuk debug)
    # --- Transfer-learning 2-tahap ---
    freeze_encoder_epochs: int = 3  # encoder ImageNet beku N epoch awal (0 = nonaktif)
    # --- Stabilitas ---
    grad_clip: float = 1.0  # gradient clipping (max-norm); 0 = nonaktif
    # --- Resume (anti sesi Colab mati) ---
    resume: bool = True
    resume_path: str = ""  # default: <ckpt_path>_resume.pt
    log_per_class_iou: bool = True
    history: list[dict[str, float]] = field(default_factory=list)


def set_encoder_trainable(model, trainable: bool) -> bool:
    """Bekukan/buka encoder (transfer-learning 2-tahap).

    Saat beku: ``requires_grad=False`` + ``encoder.eval()`` (BatchNorm stats tetap).
    smp models punya atribut ``.encoder``. Return True bila berhasil di-set.
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return False
    for p in encoder.parameters():
        p.requires_grad = trainable
    if not trainable:
        encoder.eval()  # bekukan running stats BatchNorm selama fase frozen
    return True


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
        from torch.amp import autocast  # noqa: PLC0415  (autocast ada sejak torch 1.10)
        from torchmetrics.classification import MulticlassJaccardIndex  # noqa: PLC0415
        from tqdm.auto import tqdm  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("Butuh torch + torchmetrics + tqdm. Install: pip install -e \".[ml]\"") from e

    # GradScaler generik (device positional) baru ada di torch >= 2.4. Di torch < 2.4
    # (mis. Jetson Orin / JetPack 5.x -> torch 2.1) pakai torch.cuda.amp.GradScaler.
    try:
        from torch.amp import GradScaler  # noqa: PLC0415  (torch >= 2.4)
        _GRADSCALER_NEW = True
    except ImportError:
        from torch.cuda.amp import GradScaler  # noqa: PLC0415  (torch < 2.4)
        _GRADSCALER_NEW = False

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

    iou_macro = MulticlassJaccardIndex(num_classes=cfg.n_classes, average="macro").to(device_t)
    iou_per_class = (
        MulticlassJaccardIndex(num_classes=cfg.n_classes, average=None).to(device_t)
        if cfg.log_per_class_iou else None
    )

    scaler = (GradScaler(device_t.type) if _GRADSCALER_NEW else GradScaler()) if use_amp else None
    best_iou = 0.0
    best_epoch = -1
    wait = 0
    start_epoch = 1

    ckpt_path = Path(cfg.ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path = (
        Path(cfg.resume_path) if cfg.resume_path
        else ckpt_path.with_name(ckpt_path.stem + "_resume.pt")
    )

    # ---------------- RESUME (anti sesi Colab mati) ----------------
    if cfg.resume and resume_path.exists():
        try:
            state = torch.load(resume_path, map_location=device_t)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            if scaler is not None and state.get("scaler") is not None:
                scaler.load_state_dict(state["scaler"])
            start_epoch = int(state.get("epoch", 0)) + 1
            best_iou = float(state.get("best_iou", 0.0))
            best_epoch = int(state.get("best_epoch", -1))
            wait = int(state.get("wait", 0))
            cfg.history = list(state.get("history", cfg.history))
            _logger.info(
                "Resume dari %s -> lanjut epoch %d (best mIoU=%.4f).",
                resume_path, start_epoch, best_iou,
            )
        except Exception as e:  # noqa: BLE001
            _logger.warning("Gagal resume (%s) - mulai dari awal.", e)

    # ---------------- FREEZE ENCODER (fase awal transfer-learning) ----------------
    freeze_n = int(getattr(cfg, "freeze_encoder_epochs", 0) or 0)
    encoder_frozen = False
    if freeze_n > 0 and start_epoch <= freeze_n:
        encoder_frozen = set_encoder_trainable(model, False)
        if encoder_frozen:
            _logger.info("Encoder DIBEKUKAN epoch 1..%d (latih decoder dulu).", freeze_n)

    _logger.info(
        "Mulai training: device=%s, AMP=%s, epochs=%d (mulai ep %d)",
        device_t, use_amp, cfg.epochs, start_epoch,
    )

    for ep in range(start_epoch, cfg.epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        # Unfreeze encoder tepat setelah fase frozen selesai.
        if encoder_frozen and ep > freeze_n:
            set_encoder_trainable(model, True)
            encoder_frozen = False
            _logger.info("Encoder DIBUKA (unfreeze) mulai epoch %d - fine-tune penuh.", ep)
        # Selama fase frozen, model.train() tadi meng-aktifkan BN encoder → eval lagi.
        if encoder_frozen:
            model.encoder.eval()

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
                if cfg.grad_clip and cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = loss_fn(model(x), y)
                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            train_loss_sum += float(loss.item())
        scheduler.step()
        train_loss = train_loss_sum / max(len(train_loader), 1)

        # ---------------- VAL ----------------
        model.eval()
        iou_macro.reset()
        if iou_per_class is not None:
            iou_per_class.reset()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {ep:02d} [val  ]", leave=False):
                x, y = x.to(device_t, non_blocking=True), y.to(device_t, non_blocking=True)
                pred = model(x)
                val_loss_sum += float(loss_fn(pred, y).item())
                pred_lbl = pred.argmax(1)
                iou_macro.update(pred_lbl, y)
                if iou_per_class is not None:
                    iou_per_class.update(pred_lbl, y)
        val_loss = val_loss_sum / max(len(val_loader), 1)
        val_miou = float(iou_macro.compute().item())
        per_class_iou = (
            [round(float(v), 4) for v in iou_per_class.compute().tolist()]
            if iou_per_class is not None else None
        )

        epoch_time = time.time() - t0
        _logger.info(
            "Ep %02d/%d | train_loss %.4f | val_loss %.4f | val_mIoU %.4f | %.1fs",
            ep, cfg.epochs, train_loss, val_loss, val_miou, epoch_time,
        )
        if per_class_iou is not None:
            _logger.info("  IoU per-kelas: %s", per_class_iou)
        rec = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        if per_class_iou is not None:
            rec["val_iou_per_class"] = per_class_iou
        cfg.history.append(rec)

        # ---------------- CHECKPOINT TERBAIK + EARLY STOPPING ----------------
        if val_miou > best_iou:
            best_iou = val_miou
            best_epoch = ep
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
            _logger.info("  -> best checkpoint disimpan (val_mIoU=%.4f) di %s", best_iou, ckpt_path)
        else:
            wait += 1

        # Simpan state PENUH untuk resume tiap epoch (anti sesi Colab mati).
        torch.save(
            {
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_iou": best_iou,
                "best_epoch": best_epoch,
                "wait": wait,
                "history": cfg.history,
            },
            resume_path,
        )

        if wait >= cfg.patience:
            _logger.info("Early stopping di epoch %d (patience=%d).", ep, cfg.patience)
            break

    return {
        "best_val_iou": best_iou,
        "best_epoch": best_epoch,
        "history": cfg.history,
        "ckpt_path": str(ckpt_path),
        "resume_path": str(resume_path),
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
