"""Grid tiling untuk ekspor — bagi region menjadi nx × ny ubin."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ee


def make_tiles(region: "ee.Geometry", nx: int = 6, ny: int = 6) -> list["ee.Geometry"]:
    """Bagi ``region`` menjadi grid ``nx × ny`` rectangle.

    Sumber: PRD §A.5 Cell 4. Total ubin = ``nx * ny`` (default 36 untuk Papua).

    Args:
        region: Geometry yang akan dibagi (akan diambil bounds-nya).
        nx: Jumlah kolom grid.
        ny: Jumlah baris grid.

    Returns:
        List ``ee.Geometry.Rectangle`` urut kolom-baris.
    """
    import ee  # noqa: PLC0415

    bounds = region.bounds().coordinates().get(0).getInfo()
    xs = [p[0] for p in bounds]
    ys = [p[1] for p in bounds]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    tiles: list[ee.Geometry] = []
    for i in range(nx):
        for j in range(ny):
            x0 = xmin + i * dx
            x1 = xmin + (i + 1) * dx
            y0 = ymin + j * dy
            y1 = ymin + (j + 1) * dy
            tiles.append(ee.Geometry.Rectangle([x0, y0, x1, y1]))
    return tiles


def make_tiles_from_bbox(
    bbox: tuple[float, float, float, float], nx: int = 6, ny: int = 6
) -> list[tuple[float, float, float, float]]:
    """Versi pure-Python: bagi BBox jadi grid (tanpa GEE).

    Untuk testing & visualisasi awal. Output: list of ``(minLon, minLat, maxLon, maxLat)``.
    """
    xmin, ymin, xmax, ymax = bbox
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    tiles: list[tuple[float, float, float, float]] = []
    for i in range(nx):
        for j in range(ny):
            tiles.append(
                (xmin + i * dx, ymin + j * dy, xmin + (i + 1) * dx, ymin + (j + 1) * dy)
            )
    return tiles
