"""Otentikasi & inisialisasi Google Earth Engine."""

from __future__ import annotations

import os

from forestwatch.utils.logging import get_logger

_logger = get_logger("forestwatch.gee.auth")


def init_ee(project: str | None = None, *, force_auth: bool = False) -> None:
    """Inisialisasi GEE. Akan trigger ``ee.Authenticate()`` jika belum login.

    Args:
        project: Project ID GEE (mis. ``forestwatch-papua-unand``). Default
            mengambil dari env ``GEE_PROJECT`` atau ``forestwatch-papua-unand``.
        force_auth: Bila ``True``, paksa flow ``ee.Authenticate()`` ulang.

    Raises:
        ImportError: Jika ``earthengine-api`` belum ter-install.
        RuntimeError: Bila inisialisasi GEE gagal setelah autentikasi.
    """
    try:
        import ee  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "earthengine-api belum terinstal. Install dengan:\n"
            "    pip install earthengine-api\n"
            "Atau pakai extras: pip install -e \".[gee]\""
        ) from e

    project_id = project or os.environ.get("GEE_PROJECT", "forestwatch-papua-unand")

    if force_auth:
        _logger.info("Force re-authenticate ke GEE...")
        ee.Authenticate()

    try:
        ee.Initialize(project=project_id)
        _logger.info("GEE siap (project=%s).", project_id)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Init GEE gagal: %s. Mencoba ee.Authenticate()...", exc)
        ee.Authenticate()
        ee.Initialize(project=project_id)
        _logger.info("GEE siap setelah re-auth (project=%s).", project_id)
