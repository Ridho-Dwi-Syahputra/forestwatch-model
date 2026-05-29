"""CLI: trigger ekspor GEE 36 ubin Sentinel-2 + label fusion.

CATATAN: butuh akses Google Earth Engine. Lebih praktis dijalankan dari Colab
(``notebooks/01_data_pipeline.ipynb``) karena ``ee.Authenticate()`` interactive.

Penggunaan (Colab atau lokal dengan GEE sudah ter-auth):

    python scripts/run_export.py --year-t2 2024 --year-t1 2021 \
        --folder-t1 ForestWatch_Tiles_T1 --folder-t2 ForestWatch_Tiles_T2 \
        --nx 6 --ny 6
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_export", description="Ekspor 36 ubin GEE ke Google Drive.")
    p.add_argument("--project", default=None, help="GEE project ID (default dari env GEE_PROJECT).")
    p.add_argument("--year-t1", type=int, default=2021)
    p.add_argument("--year-t2", type=int, default=2024)
    p.add_argument("--folder-t1", default="ForestWatch_Tiles_T1")
    p.add_argument("--folder-t2", default="ForestWatch_Tiles_T2")
    p.add_argument("--nx", type=int, default=6)
    p.add_argument("--ny", type=int, default=6)
    p.add_argument("--scale", type=int, default=10)
    p.add_argument("--dry-run", action="store_true", help="Hanya print plan, jangan start.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import ee  # noqa: PLC0415, F401
    except ImportError:
        print("ERROR: earthengine-api belum terinstal. pip install -e \".[gee]\"", file=sys.stderr)
        return 2

    from forestwatch.constants import PAPUA_BBOX  # noqa: PLC0415
    from forestwatch.gee.auth import init_ee  # noqa: PLC0415
    from forestwatch.gee.composite import s2_composite  # noqa: PLC0415
    from forestwatch.gee.export import export_tiles_grid  # noqa: PLC0415
    from forestwatch.gee.label_fusion import build_label  # noqa: PLC0415
    from forestwatch.gee.tiles import make_tiles  # noqa: PLC0415

    init_ee(project=args.project)
    import ee  # noqa: PLC0415

    region = ee.Geometry.Rectangle(list(PAPUA_BBOX))

    img_t2 = s2_composite(args.year_t2, region)
    img_t1 = s2_composite(args.year_t1, region)
    label = build_label(region, args.year_t2)
    stack_t2 = img_t2.addBands(label)

    tiles = make_tiles(region, nx=args.nx, ny=args.ny)
    print(f"Tiles: {len(tiles)} (T1 + T2 = {2*len(tiles)} task).")

    if args.dry_run:
        print("Dry run — tidak start task.")
        return 0

    export_tiles_grid(
        stack_t2,
        tiles,
        name_prefix=f"papua_t2_tile",
        folder=args.folder_t2,
        scale=args.scale,
    )
    export_tiles_grid(
        img_t1,
        tiles,
        name_prefix=f"papua_t1_tile",
        folder=args.folder_t1,
        scale=args.scale,
    )
    print("Semua task dimulai. Pantau di https://code.earthengine.google.com/tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
