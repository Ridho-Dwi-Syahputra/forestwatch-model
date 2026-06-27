"""Validator schema untuk 7 file kontrak PRD §B.1.

Tidak butuh ``jsonschema`` library — validasi dilakukan dengan pengecekan
field manual (cepat, no extra dependency).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forestwatch.constants import (
    CLASS_COLORS,
    CLASS_IDS,
    CLASS_NAMES,
    OUTPUT_FILES,
    TRANSITION_MAP,
)
from forestwatch.utils.io import load_json


class ValidationError(ValueError):
    """Raised saat validasi gagal (digunakan dalam strict mode)."""


@dataclass
class ValidationReport:
    """Hasil validasi satu/lebih file."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files_checked: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def merge(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.files_checked.extend(other.files_checked)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def render(self) -> str:
        lines = []
        lines.append(f"Files checked: {len(self.files_checked)}")
        for f in self.files_checked:
            lines.append(f"  - {f}")
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  X {e}")
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        if self.ok and not self.warnings:
            lines.append("\nAll 7 files valid against PRD §B.1 schema.")
        elif self.ok:
            lines.append("\nValidation passed with warnings.")
        else:
            lines.append("\nValidation FAILED.")
        return "\n".join(lines)


# ============================================================================
# File-level validators
# ============================================================================


def validate_legend_json(data: list[dict[str, Any]]) -> ValidationReport:
    """Validasi struktur ``legend.json`` (PRD §B.1.3)."""
    report = ValidationReport()
    if not isinstance(data, list):
        report.add_error("legend.json harus berupa list, dapat: " + type(data).__name__)
        return report
    if len(data) != len(CLASS_IDS):
        report.add_error(
            f"legend.json harus berisi {len(CLASS_IDS)} entry (kelas), dapat {len(data)}."
        )
    seen_ids: set[int] = set()
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            report.add_error(f"legend.json[{i}] harus dict.")
            continue
        for required in ("id", "name", "color"):
            if required not in entry:
                report.add_error(f"legend.json[{i}] missing field '{required}'.")
        if "id" in entry:
            if entry["id"] in seen_ids:
                report.add_error(f"legend.json[{i}] id duplikat: {entry['id']}.")
            seen_ids.add(entry["id"])
            if entry["id"] not in CLASS_IDS:
                report.add_warning(f"legend.json[{i}] id={entry['id']} di luar canonical {CLASS_IDS}.")
        if "color" in entry and not _looks_like_hex(entry["color"]):
            report.add_warning(f"legend.json[{i}] color '{entry['color']}' bukan hex valid.")
    return report


def validate_statistics_json(data: dict[str, Any]) -> ValidationReport:
    """Validasi ``statistics.json`` (PRD §B.1.2)."""
    report = ValidationReport()
    if not isinstance(data, dict):
        report.add_error("statistics.json harus dict.")
        return report

    required_top = (
        "period_from",
        "period_to",
        "total_deforestation_ha",
        "n_hotspots",
        "per_transition_ha",
        "per_province",
        "per_class_area_ha",
        "model_metrics",
    )
    for k in required_top:
        if k not in data:
            report.add_error(f"statistics.json missing top-level field '{k}'.")

    if "per_transition_ha" in data:
        pt = data["per_transition_ha"]
        if not isinstance(pt, dict):
            report.add_error("per_transition_ha harus dict.")
        else:
            for tname in TRANSITION_MAP.values():
                if tname not in pt:
                    report.add_warning(f"per_transition_ha missing key '{tname}'.")

    if "per_province" in data:
        pp = data["per_province"]
        if not isinstance(pp, list):
            report.add_error("per_province harus list.")
        else:
            for i, row in enumerate(pp):
                if not isinstance(row, dict) or "province" not in row or "deforestation_ha" not in row:
                    report.add_error(
                        f"per_province[{i}] harus dict dengan 'province' & 'deforestation_ha'."
                    )

    if "per_class_area_ha" in data:
        pca = data["per_class_area_ha"]
        if not isinstance(pca, dict):
            report.add_error("per_class_area_ha harus dict.")

    if "model_metrics" in data:
        mm = data["model_metrics"]
        if not isinstance(mm, dict):
            report.add_error("model_metrics harus dict.")
        else:
            for k in ("overall_accuracy", "mean_iou", "per_class"):
                if k not in mm:
                    report.add_error(f"model_metrics missing '{k}'.")
            if isinstance(mm.get("per_class"), list):
                for i, row in enumerate(mm["per_class"]):
                    if not isinstance(row, dict):
                        report.add_error(f"model_metrics.per_class[{i}] harus dict.")
                        continue
                    for k in ("class", "iou", "f1"):
                        if k not in row:
                            report.add_warning(
                                f"model_metrics.per_class[{i}] missing '{k}'."
                            )

    return report


def validate_deforestation_geojson(data: dict[str, Any]) -> ValidationReport:
    """Validasi ``deforestation.geojson`` (PRD §B.1.1)."""
    report = ValidationReport()
    if not isinstance(data, dict):
        report.add_error("deforestation.geojson harus dict.")
        return report
    if data.get("type") != "FeatureCollection":
        report.add_error("type harus 'FeatureCollection'.")
    if not isinstance(data.get("features"), list):
        report.add_error("features harus list.")
        return report

    valid_transitions = set(TRANSITION_MAP.values())
    seen_ids: set[str] = set()

    for i, feat in enumerate(data["features"]):
        if not isinstance(feat, dict):
            report.add_error(f"features[{i}] harus dict.")
            continue
        if feat.get("type") != "Feature":
            report.add_error(f"features[{i}].type harus 'Feature'.")
        geom = feat.get("geometry")
        if not isinstance(geom, dict) or geom.get("type") not in ("Polygon", "MultiPolygon"):
            report.add_error(
                f"features[{i}].geometry harus Polygon/MultiPolygon dict."
            )
        props = feat.get("properties")
        if not isinstance(props, dict):
            report.add_error(f"features[{i}].properties harus dict.")
            continue
        for required in ("id", "transition_type", "area_ha", "period_from", "period_to"):
            if required not in props:
                report.add_error(f"features[{i}].properties missing '{required}'.")
        if props.get("id") in seen_ids:
            report.add_error(f"features[{i}].id duplikat: '{props.get('id')}'.")
        seen_ids.add(props.get("id"))
        if (
            "transition_type" in props
            and props["transition_type"] not in valid_transitions
        ):
            report.add_error(
                f"features[{i}].properties.transition_type '{props['transition_type']}' "
                f"bukan salah satu dari {sorted(valid_transitions)}."
            )
        if "area_ha" in props and (
            not isinstance(props["area_ha"], (int, float)) or props["area_ha"] < 0
        ):
            report.add_warning(
                f"features[{i}].properties.area_ha invalid: {props['area_ha']}."
            )
    return report


def validate_bounds_json(data: dict[str, Any]) -> ValidationReport:
    """Validasi struktur sidecar bounds JSON."""
    report = ValidationReport()
    if not isinstance(data, dict):
        report.add_error("bounds JSON harus dict.")
        return report
    bounds = data.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 2:
        report.add_error("bounds harus list dengan 2 elemen [[s,w],[n,e]].")
        return report
    for i, point in enumerate(bounds):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            report.add_error(f"bounds[{i}] harus list 2 elemen.")
    return report


def validate_png_exists(path: str | os.PathLike[str]) -> ValidationReport:
    """Sekedar cek file PNG ada & ukuran > 0."""
    report = ValidationReport()
    p = Path(path)
    if not p.exists():
        report.add_error(f"PNG tidak ditemukan: {p}")
        return report
    if p.stat().st_size == 0:
        report.add_error(f"PNG kosong: {p}")
    return report


# ============================================================================
# Directory-level validator
# ============================================================================


_REQUIRED_FILES: tuple[str, ...] = (
    OUTPUT_FILES["landcover_t1_png"],
    OUTPUT_FILES["landcover_t1_bounds"],
    OUTPUT_FILES["landcover_t2_png"],
    OUTPUT_FILES["landcover_t2_bounds"],
    OUTPUT_FILES["deforestation_geojson"],
    OUTPUT_FILES["statistics_json"],
    OUTPUT_FILES["legend_json"],
)

_OPTIONAL_FILES: tuple[str, ...] = (
    OUTPUT_FILES["metrics_json"],
    OUTPUT_FILES["model_onnx"],
    OUTPUT_FILES["model_card"],
)


def validate_outputs_dir(
    directory: str | os.PathLike[str],
    *,
    strict: bool = False,
    required: Iterable[str] = _REQUIRED_FILES,
    optional: Iterable[str] = _OPTIONAL_FILES,
) -> ValidationReport:
    """Cek folder berisi 7 file kontrak (+ optional ONNX, model_card, metrics).

    Args:
        directory: Folder untuk diperiksa.
        strict: Bila ``True``, raise ``ValidationError`` jika ada error.
        required: List file wajib.
        optional: List file opsional (tidak menghasilkan error jika absent).

    Returns:
        ``ValidationReport``.
    """
    report = ValidationReport()
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        report.add_error(f"Direktori tidak ada: {d}")
        if strict:
            raise ValidationError(report.render())
        return report

    # Cek presence
    for fname in required:
        p = d / fname
        if not p.exists():
            report.add_error(f"File wajib hilang: {fname}")
        else:
            report.files_checked.append(fname)
    for fname in optional:
        p = d / fname
        if p.exists():
            report.files_checked.append(fname)
        else:
            report.add_warning(f"File opsional hilang: {fname}")

    # Validasi konten
    legend_path = d / OUTPUT_FILES["legend_json"]
    if legend_path.exists():
        try:
            report.merge(validate_legend_json(load_json(legend_path)))
        except Exception as exc:  # noqa: BLE001
            report.add_error(f"Gagal load legend.json: {exc}")

    stats_path = d / OUTPUT_FILES["statistics_json"]
    if stats_path.exists():
        try:
            report.merge(validate_statistics_json(load_json(stats_path)))
        except Exception as exc:  # noqa: BLE001
            report.add_error(f"Gagal load statistics.json: {exc}")

    geo_path = d / OUTPUT_FILES["deforestation_geojson"]
    if geo_path.exists():
        try:
            report.merge(validate_deforestation_geojson(load_json(geo_path)))
        except Exception as exc:  # noqa: BLE001
            report.add_error(f"Gagal load deforestation.geojson: {exc}")

    for bounds_key in ("landcover_t1_bounds", "landcover_t2_bounds"):
        bp = d / OUTPUT_FILES[bounds_key]
        if bp.exists():
            try:
                report.merge(validate_bounds_json(load_json(bp)))
            except Exception as exc:  # noqa: BLE001
                report.add_error(f"Gagal load {bp.name}: {exc}")

    for png_key in ("landcover_t1_png", "landcover_t2_png"):
        pp = d / OUTPUT_FILES[png_key]
        if pp.exists():
            report.merge(validate_png_exists(pp))

    if strict and not report.ok:
        raise ValidationError(report.render())
    return report


def _looks_like_hex(s: Any) -> bool:
    if not isinstance(s, str) or not s.startswith("#"):
        return False
    h = s[1:]
    if len(h) not in (3, 6, 8):
        return False
    try:
        int(h, 16)
        return True
    except ValueError:
        return False
