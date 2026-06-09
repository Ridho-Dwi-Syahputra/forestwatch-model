"""CLI: jalankan inferensi model pada folder ubin T1 & T2, lalu deteksi perubahan.

Penggunaan:

    python scripts/run_inference.py \
        --ckpt outputs/best_model.pt \
        --tiles-t1 data/tiles_t1 --tiles-t2 data/tiles_t2 \
        --mask-dir data/masks
"""

from __future__ import annotations

import argparse
import sys

from forestwatch.config import load_config
from forestwatch.inference.tile_inference import infer_tiles_folder
from forestwatch.model.architecture import build_unet
from forestwatch.utils.logging import get_logger

_logger = get_logger("forestwatch.scripts.infer")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_inference")
    p.add_argument("--ckpt", required=True, help="Path best_model.pt")
    p.add_argument("--tiles-t1", required=True, help="Folder GeoTIFF T1 (citra 2021)")
    p.add_argument("--tiles-t2", required=True, help="Folder GeoTIFF T2 (citra 2024)")
    p.add_argument("--mask-dir", required=True, help="Folder output mask")
    p.add_argument("--config", default=None)
    p.add_argument("--device", default=None, help="cuda | cpu (default auto)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        print("ERROR: torch belum terinstal. pip install -e \".[ml]\"", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    model = build_unet(
        architecture=cfg["model"]["architecture"],
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=None,  # weights akan di-load dari ckpt
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
        activation=cfg["model"].get("activation"),
    )
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    _logger.info("Model loaded dari %s", args.ckpt)

    infer_tiles_folder(
        args.tiles_t1,
        args.mask_dir,
        model,
        prefix="mask_t1_",
        device=args.device,
        patch_size=cfg["inference"]["patch_size"],
        stride=cfg["inference"]["stride"],
    )
    infer_tiles_folder(
        args.tiles_t2,
        args.mask_dir,
        model,
        prefix="mask_t2_",
        device=args.device,
        patch_size=cfg["inference"]["patch_size"],
        stride=cfg["inference"]["stride"],
    )
    _logger.info("Inferensi T1 dan T2 selesai. Mask di: %s", args.mask_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
