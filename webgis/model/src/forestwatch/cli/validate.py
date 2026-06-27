"""CLI: validasi folder output terhadap skema PRD §B.1.

Penggunaan:

    fw-validate --dir outputs/dummy
    python -m forestwatch.cli.validate --dir outputs/dummy
"""

from __future__ import annotations

import argparse
import sys

from forestwatch.validation.schema import validate_outputs_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fw-validate",
        description="Validasi 7 file kontrak ForestWatch (PRD §B.1).",
    )
    p.add_argument(
        "--dir",
        "-d",
        required=True,
        help="Path folder berisi 7 file output (mis. outputs/dummy/).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero jika ada error (default: hanya cetak laporan).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_outputs_dir(args.dir, strict=False)
    print(report.render())
    if args.strict and not report.ok:
        return 1
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
