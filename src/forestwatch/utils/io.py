"""IO helper — JSON, GeoJSON, NPZ, dengan create-parent-dir otomatis."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def save_json(
    data: Any,
    path: str | os.PathLike[str],
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """Simpan ``data`` sebagai JSON. Auto-create parent dir.

    Returns: Path absolut file yang disimpan.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=_json_default)
    return p


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load JSON file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_geojson(
    feature_collection: dict[str, Any],
    path: str | os.PathLike[str],
) -> Path:
    """Simpan FeatureCollection sebagai GeoJSON.

    Memvalidasi minimal: type == 'FeatureCollection' dan ada list ``features``.
    """
    if feature_collection.get("type") != "FeatureCollection":
        raise ValueError("Input bukan FeatureCollection (field 'type' tidak cocok).")
    if not isinstance(feature_collection.get("features"), list):
        raise ValueError("FeatureCollection harus punya list 'features'.")
    return save_json(feature_collection, path, indent=2)


def save_npz(
    path: str | os.PathLike[str],
    *,
    compressed: bool = True,
    **arrays: Any,
) -> Path:
    """Wrapper ``numpy.savez(_compressed)`` dengan auto mkdir.

    Lazy import numpy agar modul ini tidak fail jika numpy absent.
    """
    import numpy as np  # noqa: PLC0415

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(p, **arrays)
    else:
        np.savez(p, **arrays)
    return p


def _json_default(obj: Any) -> Any:
    """Fallback serializer: numpy scalar/array, set, dll."""
    try:
        import numpy as np  # noqa: PLC0415

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
