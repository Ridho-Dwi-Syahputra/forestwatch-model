"""Jembatan import ke package ``forestwatch`` (di ``model/src/``).

Backend ini hidup di monorepo yang sama dengan paket model (``model/src/forestwatch``).
Daripada duplikasi logika composite/inferensi/change-detection, kita tambahkan
``model/src`` ke ``sys.path`` lalu re-export fungsi yang dipakai endpoint ``/api/analyze``.

Asumsi deployment: image Docker menyalin seluruh monorepo (lihat ``Dockerfile``),
bukan hanya folder ``webgis/backend``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent.parent
_MODEL_SRC = _REPO_ROOT / "model" / "src"

if _MODEL_SRC.exists() and str(_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(_MODEL_SRC))

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
