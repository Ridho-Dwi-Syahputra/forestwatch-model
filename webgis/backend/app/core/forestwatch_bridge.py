"""Jembatan import ke package ``forestwatch`` (source di ``model/src/``).

Backend memakai logika composite/inferensi/change-detection dari package ``forestwatch``
(daripada duplikasi). Kita tambahkan ``model/src`` ke ``sys.path`` lalu re-export fungsi
yang dipakai endpoint ``/api/analyze``.

Mendukung DUA layout supaya backend jalan baik di monorepo maupun di repo WebGIS standalone:

1. **Monorepo** (``<root>/webgis/backend`` + ``<root>/model/src``) — pakai package asli
   (selalu paling baru). Path: ``_BACKEND_ROOT.parent.parent / "model" / "src"``.
2. **Repo WebGIS standalone** (``<repo>/backend`` + ``<repo>/model/src`` hasil vendoring)
   — fallback ke salinan vendored. Path: ``_BACKEND_ROOT.parent / "model" / "src"``.

Kandidat dicoba berurutan; yang pertama ADA dipakai. Urutan ini sengaja menaruh
original (monorepo) lebih dulu supaya di monorepo tak memakai salinan yang bisa basi.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent

_MODEL_SRC_CANDIDATES = [
    _BACKEND_ROOT.parent.parent / "model" / "src",  # monorepo: <root>/model/src (original)
    _BACKEND_ROOT.parent / "model" / "src",         # standalone: <repo>/model/src (vendored)
]

for _candidate in _MODEL_SRC_CANDIDATES:
    if _candidate.exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

try:
    from forestwatch.constants import CLASS_NAMES, TRANSITION_MAP  # noqa: E402
    from forestwatch.gee.composite import s2_composite  # noqa: E402
    from forestwatch.inference.change_detection import (  # noqa: E402
        detect_transitions_from_arrays,
    )
    from forestwatch.inference.tile_inference import infer_tile  # noqa: E402
    from forestwatch.model.architecture import build_unet  # noqa: E402
    from forestwatch.outputs.statistics import summarize_geojson_transitions  # noqa: E402

    FORESTWATCH_AVAILABLE = True
    FORESTWATCH_IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - hanya saat env belum lengkap
    FORESTWATCH_AVAILABLE = False
    FORESTWATCH_IMPORT_ERROR = str(exc)

    CLASS_NAMES = ()
    TRANSITION_MAP = {}

    def s2_composite(*_args, **_kwargs):  # type: ignore[misc]
        raise RuntimeError(f"Paket forestwatch tidak tersedia: {FORESTWATCH_IMPORT_ERROR}")

    def detect_transitions_from_arrays(*_args, **_kwargs):  # type: ignore[misc]
        raise RuntimeError(f"Paket forestwatch tidak tersedia: {FORESTWATCH_IMPORT_ERROR}")

    def infer_tile(*_args, **_kwargs):  # type: ignore[misc]
        raise RuntimeError(f"Paket forestwatch tidak tersedia: {FORESTWATCH_IMPORT_ERROR}")

    def build_unet(*_args, **_kwargs):  # type: ignore[misc]
        raise RuntimeError(f"Paket forestwatch tidak tersedia: {FORESTWATCH_IMPORT_ERROR}")

    def summarize_geojson_transitions(*_args, **_kwargs):  # type: ignore[misc]
        raise RuntimeError(f"Paket forestwatch tidak tersedia: {FORESTWATCH_IMPORT_ERROR}")
