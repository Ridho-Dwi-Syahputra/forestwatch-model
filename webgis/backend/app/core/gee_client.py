"""Auth Google Earth Engine via Service Account -- aman untuk server (tanpa browser login).

Beda dengan ``forestwatch.gee.auth.init_ee`` (dipakai notebook, pakai ``ee.Authenticate()``
interaktif) -- versi ini WAJIB dipakai di backend karena server tidak punya browser/manusia
untuk login saat request masuk.
"""

from __future__ import annotations

from app.core.config import GEE_PROJECT, GEE_SERVICE_ACCOUNT_EMAIL, GEE_SERVICE_ACCOUNT_KEY_PATH

_initialized = False
_init_error: str | None = None


def init_ee_service_account() -> bool:
    """Inisialisasi GEE sekali (idempotent). Return ``True`` bila siap dipakai.

    Dipanggil saat startup app (lihat ``app/main.py``), bukan per-request --
    auth GEE relatif mahal, tidak perlu diulang tiap kali endpoint dipanggil.

    Gagal-aman: kalau auth/permission GEE bermasalah (mis. role IAM kurang), endpoint
    lain (pre-computed) tetap harus jalan -- TIDAK boleh exception di sini menjatuhkan
    seluruh app saat startup (FastAPI startup event yang raise akan gagal start app).
    """
    global _initialized, _init_error
    if _initialized:
        return True

    if not GEE_SERVICE_ACCOUNT_EMAIL or not GEE_SERVICE_ACCOUNT_KEY_PATH:
        _init_error = "GEE_SERVICE_ACCOUNT_EMAIL / GEE_SERVICE_ACCOUNT_KEY_PATH belum diisi di .env"
        return False

    try:
        import ee
    except ImportError as exc:
        _init_error = f"earthengine-api belum terinstall: {exc}"
        return False

    try:
        credentials = ee.ServiceAccountCredentials(
            GEE_SERVICE_ACCOUNT_EMAIL, GEE_SERVICE_ACCOUNT_KEY_PATH
        )
        ee.Initialize(credentials, project=GEE_PROJECT)
    except Exception as exc:  # noqa: BLE001 -- auth/permission GEE, jangan jatuhkan app
        _init_error = f"Gagal inisialisasi Earth Engine: {exc}"
        return False

    _initialized = True
    _init_error = None
    return True


def is_ready() -> bool:
    return _initialized


def get_init_error() -> str | None:
    return _init_error
