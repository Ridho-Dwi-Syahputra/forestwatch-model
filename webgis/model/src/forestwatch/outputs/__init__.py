"""Generate 7 file kontrak ke Orang 2 (WebGIS) — PRD §B.1."""

from forestwatch.outputs.landcover_png import (
    mask_to_rgb,
    mosaic_masks_to_png,
    save_mask_array_as_png,
)
from forestwatch.outputs.legend import build_legend_json
from forestwatch.outputs.model_card import render_model_card
from forestwatch.outputs.orchestrator import generate_all_outputs
from forestwatch.outputs.statistics import (
    build_statistics_json,
    summarize_geojson_transitions,
)

__all__ = [
    "mask_to_rgb",
    "mosaic_masks_to_png",
    "save_mask_array_as_png",
    "build_legend_json",
    "render_model_card",
    "generate_all_outputs",
    "build_statistics_json",
    "summarize_geojson_transitions",
]
