"""CLI: orkestrasi 7 file kontrak dari folder mask + metrik JSON.

Penggunaan:

    python scripts/generate_outputs.py \
        --mask-dir data/masks \
        --out outputs/final \
        --metrics outputs/metrics.json \
        --onnx outputs/model.onnx
"""

from __future__ import annotations

import argparse
import sys

from forestwatch.outputs.orchestrator import generate_all_outputs
from forestwatch.utils.io import load_json
from forestwatch.utils.logging import get_logger

_logger = get_logger("forestwatch.scripts.outputs")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="generate_outputs")
    p.add_argument("--mask-dir", required=True, help="Folder mask hasil inferensi (T1 & T2).")
    p.add_argument("--out", required=True, help="Folder output 7 file kontrak.")
    p.add_argument("--period-from", type=int, default=2021)
    p.add_argument("--period-to", type=int, default=2024)
    p.add_argument("--metrics", default=None, help="Path metrics.json (opsional).")
    p.add_argument("--onnx", default=None, help="Path model.onnx (opsional, akan di-copy).")
    p.add_argument("--min-area-ha", type=float, default=0.5)
    p.add_argument("--mask-t1-prefix", default="mask_t1_")
    p.add_argument("--mask-t2-prefix", default="mask_t2_")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = load_json(args.metrics) if args.metrics else None
    paths = generate_all_outputs(
        mask_dir=args.mask_dir,
        out_dir=args.out,
        period_from=args.period_from,
        period_to=args.period_to,
        metrics=metrics,
        onnx_src=args.onnx,
        min_area_ha=args.min_area_ha,
        mask_t1_prefix=args.mask_t1_prefix,
        mask_t2_prefix=args.mask_t2_prefix,
    )
    _logger.info("Selesai. Files: %s", list(paths.keys()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
