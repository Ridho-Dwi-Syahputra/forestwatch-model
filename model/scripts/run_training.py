"""CLI: training ResNet50-U-Net dari folder patches.

Penggunaan:

    python scripts/run_training.py \
        --patch-dir data/patches \
        --ckpt-path outputs/best_model.pt \
        --epochs 50 --batch-size 8
"""

from __future__ import annotations

import argparse
import sys

from forestwatch.config import load_config
from forestwatch.constants import CLASS_WEIGHTS_DEFAULT, N_CLASSES
from forestwatch.data.dataset import build_dataloaders
from forestwatch.model.architecture import build_unet, count_parameters
from forestwatch.model.losses import make_loss_fn
from forestwatch.training.trainer import TrainConfig, train
from forestwatch.utils.logging import get_logger

_logger = get_logger("forestwatch.scripts.train")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_training")
    p.add_argument("--patch-dir", required=True, help="Folder berisi p*.npz")
    p.add_argument("--ckpt-path", default="outputs/best_model.pt")
    p.add_argument("--config", default=None, help="YAML config override (default: configs/default.yaml)")
    p.add_argument("--architecture", default=None, help="unet | unet_scse (override config).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import torch  # noqa: PLC0415, F401
    except ImportError:
        print("ERROR: torch belum terinstal. pip install -e \".[ml]\"", file=sys.stderr)
        return 2

    cfg = load_config(args.config)

    architecture = args.architecture or cfg["model"]["architecture"]
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    num_workers = args.num_workers or cfg["training"]["num_workers"]
    seed = args.seed or cfg["project"]["seed"]

    train_loader, val_loader, _ = build_dataloaders(
        args.patch_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=cfg["training"]["split"]["train_ratio"],
        val_ratio=cfg["training"]["split"]["val_ratio"],
        seed=seed,
        augment_p=cfg["training"]["augmentation"],
    )

    model = build_unet(
        architecture=architecture,
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
        activation=cfg["model"].get("activation"),
    )
    _logger.info("Model %s — %d parameters", architecture, count_parameters(model))

    loss_fn = make_loss_fn(
        class_weights=CLASS_WEIGHTS_DEFAULT,
        ce_weight=cfg["training"]["loss"]["ce_weight"],
        dice_weight=cfg["training"]["loss"]["dice_weight"],
    )

    tcfg = TrainConfig(
        epochs=epochs,
        patience=cfg["training"]["patience"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        amp=cfg["training"]["amp"],
        seed=seed,
        n_classes=N_CLASSES,
        ckpt_path=args.ckpt_path,
    )
    summary = train(model, train_loader, val_loader, loss_fn=loss_fn, cfg=tcfg)
    _logger.info("Training selesai. Summary: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
