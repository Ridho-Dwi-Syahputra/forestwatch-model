"""Ekspor batch ke Google Drive."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from forestwatch.utils.logging import get_logger

if TYPE_CHECKING:
    import ee

_logger = get_logger("forestwatch.gee.export")


def export_stack(
    image: "ee.Image",
    *,
    description: str,
    folder: str,
    region: "ee.Geometry",
    scale: int = 10,
    max_pixels: float = 1e13,
    file_format: str = "GeoTIFF",
    start: bool = True,
) -> "ee.batch.Task":
    """Wrapper ``ee.batch.Export.image.toDrive`` dengan default proyek.

    Args:
        image: Image yang akan diekspor.
        description: Nama task (juga prefix file name).
        folder: Subfolder di Google Drive root.
        region: Geometry clip.
        scale: Resolusi (meter/pixel). 10 untuk Sentinel-2.
        max_pixels: Batas max pixel (default 1e13 - lebih dari cukup untuk Papua tile).
        file_format: ``"GeoTIFF"`` (default).
        start: Bila ``True``, langsung ``task.start()``.

    Returns:
        ``ee.batch.Task`` (sudah dimulai bila ``start=True``).
    """
    import ee  # noqa: PLC0415

    task = ee.batch.Export.image.toDrive(
        image=image.clip(region),
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        maxPixels=max_pixels,
        fileFormat=file_format,
    )
    if start:
        task.start()
        _logger.info("Task '%s' dimulai (folder=%s).", description, folder)
    return task


def export_tiles_grid(
    image: "ee.Image",
    tiles: list["ee.Geometry"],
    *,
    name_prefix: str,
    folder: str,
    scale: int = 10,
    max_pixels: float = 1e13,
) -> list["ee.batch.Task"]:
    """Ekspor ``image`` ke setiap tile sebagai task terpisah.

    Args:
        image: Image global (akan di-clip per tile).
        tiles: List geometry tile (dari ``make_tiles``).
        name_prefix: Prefix nama task → ``{prefix}_{idx:02d}``.
        folder: Subfolder Drive.
        scale, max_pixels: Forward ke ``export_stack``.

    Returns:
        List task.
    """
    tasks = []
    for idx, tile in enumerate(tiles):
        description = f"{name_prefix}_{idx:02d}"
        task = export_stack(
            image,
            description=description,
            folder=folder,
            region=tile,
            scale=scale,
            max_pixels=max_pixels,
        )
        tasks.append(task)
    _logger.info("%d task ekspor dimulai untuk prefix '%s'.", len(tasks), name_prefix)
    return tasks


def monitor_tasks(
    tasks: list["ee.batch.Task"],
    *,
    poll_seconds: int = 60,
    timeout_seconds: int = 6 * 3600,
) -> dict[str, int]:
    """Poll status semua task sampai selesai/gagal (atau timeout).

    Returns:
        Dict ``{state: count}`` di akhir.
    """
    elapsed = 0
    while elapsed < timeout_seconds:
        states = [t.status().get("state", "UNKNOWN") for t in tasks]
        counts: dict[str, int] = {}
        for s in states:
            counts[s] = counts.get(s, 0) + 1
        running = counts.get("RUNNING", 0) + counts.get("READY", 0)
        _logger.info("Status tasks: %s", counts)
        if running == 0:
            return counts
        time.sleep(poll_seconds)
        elapsed += poll_seconds
    _logger.warning("Timeout %ds tercapai. Status akhir: %s", timeout_seconds, counts)
    return counts
