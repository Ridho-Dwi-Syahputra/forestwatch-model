"""Auth Google Earth Engine via Service Account -- aman untuk server (tanpa browser login).

Beda dengan ``forestwatch.gee.auth.init_ee`` (dipakai notebook, pakai ``ee.Authenticate()``
interaktif) -- versi ini WAJIB dipakai di backend karena server tidak punya browser/manusia
untuk login saat request masuk.
"""

from __future__ import annotations

from app.core.config import GEE_PROJECT, GEE_SERVICE_ACCOUNT_EMAIL, GEE_SERVICE_ACCOUNT_KEY_PATH

_initialized = False


def init_ee_service_account() -> bool:
    """Inisialisasi GEE sekali (idempotent). Return ``True`` bila siap dipakai.

    Dipanggil saat startup app (lihat ``app/main.py``), bukan per-request --
    auth GEE relatif mahal, tidak perlu diulang tiap kali endpoint dipanggil.
    """
    global _initialized
    if _initialized:
        return True

    if not GEE_SERVICE_ACCOUNT_EMAIL or not GEE_SERVICE_ACCOUNT_KEY_PATH:
        return False

    try:
        import ee
    except ImportError:
        return False

    credentials = ee.ServiceAccountCredentials(GEE_SERVICE_ACCOUNT_EMAIL, GEE_SERVICE_ACCOUNT_KEY_PATH)
    ee.Initialize(credentials, project=GEE_PROJECT)
    _initialized = True
    return True


def is_ready() -> bool:
    return _initialized
