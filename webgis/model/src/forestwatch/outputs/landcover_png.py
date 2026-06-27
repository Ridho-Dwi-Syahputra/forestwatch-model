"""Render mask kelas → PNG berwarna untuk Leaflet ``L.imageOverlay``.

Sumber: PRD §A.5 Cell 10.
Output: PNG + sidecar bounds JSON (format Leaflet ``[[s,w],[n,e]]``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from forestwatch.constants import N_CLASSES, PALETTE_RGB
from forestwatch.utils.geo import bounds_to_leaflet, bounds_to_leaflet_from_transform
from forestwatch.utils.io import save_json
from forestwatch.utils.logging import get_logger

_logger = get_logger("forestwatch.outputs.png")


def mask_to_rgb(
    mask: "object",
    palette: dict[int, tuple[int, int, int]] | None = None,
    *,
    n_classes: int = N_CLASSES,
    background: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> "object":
    """Convert numpy ``(H, W)`` mask uint8 → RGBA ``(H, W, 4)`` uint8.

    Piksel di luar range kelas (mis. -1 atau no-data) → ``background``.
    """
    import numpy as np  # noqa: PLC0415

    palette = palette or PALETTE_RGB
    rgb = np.zeros((*mask.shape, 4), dtype="uint8")
    rgb[..., :] = background
    for cls in range(n_classes):
        m = mask == cls
        if not m.any():
            continue
        rgb[m, 0] = palette[cls][0]
        rgb[m, 1] = palette[cls][1]
        rgb[m, 2] = palette[cls][2]
        rgb[m, 3] = 255
    return rgb


def save_mask_array_as_png(
    mask: "object",
    out_png: str | os.PathLike[str],
    *,
    bbox: tuple[float, float, float, float] | None = None,
    out_bounds_json: str | os.PathLike[str] | None = None,
    palette: dict[int, tuple[int, int, int]] | None = None,
) -> tuple[Path, Path | None]:
    """Simpan mask numpy → PNG (optional dengan bounds sidecar).

    Args:
        mask: numpy ``(H, W)`` uint8.
        out_png: Path PNG output.
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` untuk bounds JSON. Bila
            ``None`` dan ``out_bounds_json`` di-set, raise ``ValueError``.
        out_bounds_json: Path JSON bounds. Bila ``None``, tidak menyimpan bounds.
        palette: Override palette default.

    Returns:
        ``(png_path, bounds_path_or_None)``.
    """
    from PIL import Image  # noqa: PLC0415

    rgb = mask_to_rgb(mask, palette=palette)
    out_png_p = Path(out_png)
    out_png_p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGBA").save(out_png_p, optimize=True)
    _logger.info("PNG landcover disimpan: %s (shape=%s)", out_png_p, rgb.shape)

    bounds_path: Path | None = None
    if out_bounds_json is not None:
        if bbox is None:
            raise ValueError("Butuh `bbox` untuk menyimpan bounds sidecar.")
        bounds = bounds_to_leaflet(*bbox)
        bounds_path = save_json(
            {"bounds": bounds, "crs": "EPSG:4326"},
            out_bounds_json,
        )
    return out_png_p, bounds_path


def mosaic_masks_to_png(
    mask_files: list[str | os.PathLike[str]],
    out_png: str | os.PathLike[str],
    out_bounds_json: str | os.PathLike[str],
    *,
    palette: dict[int, tuple[int, int, int]] | None = None,
) -> tuple[Path, Path]:
    """Merge banyak ubin mask GeoTIFF → PNG mosaic + bounds JSON.

    Sumber: PRD §A.5 Cell 10 ``rasterio.merge.merge``.

    Returns:
        ``(png_path, bounds_path)``.
    """
    try:
        import rasterio  # noqa: PLC0415
        from rasterio.merge import merge  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("Butuh rasterio. Install: pip install -e \".[gis]\"") from e

    if not mask_files:
        raise ValueError("mask_files kosong.")
    srcs = [rasterio.open(f) for f in mask_files]
    try:
        mosaic, transform = merge(srcs)
        mask = mosaic[0]  # (H, W) uint8
        H, W = mask.shape
    finally:
        for s in srcs:
            s.close()

    png_path, _ = save_mask_array_as_png(mask, out_png, palette=palette)
    bounds = bounds_to_leaflet_from_transform(H, W, transform)
    bounds_path = save_json({"bounds": bounds, "crs": "EPSG:4326"}, out_bounds_json)
    return png_path, bounds_path


def compute_per_class_area_ha(
    mask: "object",
    *,
    pixel_area_ha: float = 0.01,  # 10m × 10m → 0.01 ha
    n_classes: int = N_CLASSES,
    class_names: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Hitung luas per kelas dari mask numpy. Output dict ``{name: ha}``."""
    import numpy as np  # noqa: PLC0415

    from forestwatch.constants import CLASS_NAMES  # noqa: PLC0415

    class_names = class_names or CLASS_NAMES
    if len(class_names) != n_classes:
        raise ValueError(f"class_names harus panjang {n_classes}.")
    u, c = np.unique(mask, return_counts=True)
    by_id: dict[int, int] = dict(zip(u.tolist(), c.tolist()))
    out: dict[str, float] = {}
    for cls in range(n_classes):
        n_pix = int(by_id.get(cls, 0))
        out[class_names[cls]] = round(n_pix * pixel_area_ha, 1)
    return out


def compute_per_class_area_ha_from_geotiff(
    tif_path: str | os.PathLike[str],
    *,
    pixel_area_ha: float = 0.01,
    n_classes: int = N_CLASSES,
    class_names: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Helper: buka GeoTIFF mask 1 band → ``compute_per_class_area_ha``."""
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("Butuh rasterio.") from e

    with rasterio.open(tif_path) as src:
        mask = src.read(1)
    return compute_per_class_area_ha(
        mask, pixel_area_ha=pixel_area_ha, n_classes=n_classes, class_names=class_names
    )


def _serialize_palette(palette: dict[int, tuple[int, int, int]]) -> dict[str, Any]:
    """Helper internal — tidak dipakai langsung, ada untuk introspection."""
    return {str(k): list(v) for k, v in palette.items()}
