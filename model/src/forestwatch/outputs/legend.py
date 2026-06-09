"""Build ``legend.json`` — PRD §B.1.3."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from forestwatch.constants import CLASS_COLORS, CLASS_IDS, CLASS_NAMES
from forestwatch.utils.io import save_json


def build_legend_json(
    out_path: str | os.PathLike[str] | None = None,
    *,
    class_ids: tuple[int, ...] = CLASS_IDS,
    class_names: tuple[str, ...] = CLASS_NAMES,
    class_colors: dict[int, str] = CLASS_COLORS,
) -> list[dict[str, Any]]:
    """Bangun legend list-of-dict, optional save ke ``out_path``.

    Format (PRD §B.1.3):

        [
          { "id": 0, "name": "Perairan",       "color": "#2A6FDB" },
          ...
        ]
    """
    if len(class_ids) != len(class_names):
        raise ValueError("class_ids dan class_names harus sama panjang.")

    legend = [
        {
            "id": int(class_ids[i]),
            "name": class_names[i],
            "color": class_colors[class_ids[i]],
        }
        for i in range(len(class_ids))
    ]
    if out_path is not None:
        save_json(legend, out_path)
    return legend
