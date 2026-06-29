"""Google Earth Engine operations.

Semua import ``ee`` dilakukan *lazy* di dalam fungsi agar modul ini tetap bisa
di-import di lingkungan tanpa ``earthengine-api`` (mis. saat menjalankan tests).
"""

from forestwatch.gee.auth import init_ee
from forestwatch.gee.composite import (
    compute_dnbr,
    fill_composite_gaps,
    mask_scl,
    s2_composite,
    s2_composite_range,
)
from forestwatch.gee.export import export_stack, export_tiles_grid
from forestwatch.gee.label_fusion import build_label
from forestwatch.gee.tiles import make_tiles

__all__ = [
    "init_ee",
    "compute_dnbr",
    "mask_scl",
    "s2_composite",
    "s2_composite_range",
    "fill_composite_gaps",
    "export_stack",
    "export_tiles_grid",
    "build_label",
    "make_tiles",
]
