"""Sync file statis (DATA_DIR, hasil notebook/Output_Fix) -> database.

Tiap tabel: PARSE file sepenuhnya dulu, baru DELETE+INSERT (urutan ini penting -- kalau parse
gagal di tengah, tabel lama TIDAK boleh ke-wipe duluan, harus tetap di state lama yg valid).
Tiap fungsi `_sync_*` dibungkus try/except sendiri di `sync_all()` -- 1 file hilang/rusak tak
boleh menggagalkan sync tabel lain.

Dipanggil otomatis tiap startup backend (lihat app/main.py) DAN bisa dijalankan manual:
    python -m app.db.sync_static
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import DATA_DIR, FILES
from app.db.orm import DeforestationFeature, Landcover, LegendItem, ProvinceStat, Statistics

logger = logging.getLogger(__name__)

# year -> key prefix dipakai di FILES (landcover_t1_png dst) -- sama dgn pola di landcover.py
_LANDCOVER_YEAR_KEYS = {2021: "t1", 2025: "t2"}


def _sync_legend(db: Session, data_dir: Path) -> int:
    path = data_dir / FILES["legend"]
    items = json.loads(path.read_text(encoding="utf-8"))  # parse dulu

    db.query(LegendItem).delete()
    db.bulk_save_objects(
        [LegendItem(id=item["id"], name=item["name"], color=item["color"]) for item in items]
    )
    db.commit()
    return len(items)


def _sync_landcover(db: Session, data_dir: Path) -> int:
    rows = []
    for year, key in _LANDCOVER_YEAR_KEYS.items():
        png_path = data_dir / FILES[f"landcover_{key}_png"]
        bounds_path = data_dir / FILES[f"landcover_{key}_bounds"]
        if not png_path.exists() or not bounds_path.exists():
            logger.warning("Landcover %s: file tak lengkap (%s / %s), di-skip.", year, png_path, bounds_path)
            continue
        png_bytes = png_path.read_bytes()
        bounds_data = json.loads(bounds_path.read_text(encoding="utf-8"))
        (south, west), (north, east) = bounds_data["bounds"]
        rows.append(
            Landcover(
                year=year,
                png_bytes=png_bytes,
                bounds_south=south, bounds_west=west,
                bounds_north=north, bounds_east=east,
                crs=bounds_data.get("crs", "EPSG:4326"),
            )
        )

    if not rows:
        return 0

    years = [r.year for r in rows]
    db.query(Landcover).filter(Landcover.year.in_(years)).delete(synchronize_session=False)
    db.bulk_save_objects(rows)
    db.commit()
    return len(rows)


def _sync_deforestation(db: Session, data_dir: Path) -> int:
    path = data_dir / FILES["deforestation"]
    fc = json.loads(path.read_text(encoding="utf-8"))  # parse dulu
    features = fc.get("features", [])

    rows = []
    for f in features:
        props = f.get("properties", {})
        rows.append(
            DeforestationFeature(
                id=props["id"],
                transition_type=props["transition_type"],
                province=props.get("province"),
                area_ha=float(props.get("area_ha", 0)),
                period_from=props["period_from"],
                period_to=props["period_to"],
                kawasan_status=props.get("kawasan_status"),
                geometry_json=json.dumps(f.get("geometry")),
            )
        )

    db.query(DeforestationFeature).delete()
    db.bulk_save_objects(rows)
    db.commit()
    return len(rows)


def _sync_statistics(db: Session, data_dir: Path) -> int:
    path = data_dir / FILES["statistics"]
    stats = json.loads(path.read_text(encoding="utf-8"))  # parse dulu

    per_province = stats.get("per_province", [])

    db.query(Statistics).delete()
    db.add(
        Statistics(
            id=1,
            period_from=stats["period_from"],
            period_to=stats["period_to"],
            total_deforestation_ha=stats["total_deforestation_ha"],
            n_hotspots=stats["n_hotspots"],
            per_transition_ha_json=json.dumps(stats.get("per_transition_ha", {})),
            per_class_area_ha_json=json.dumps(stats.get("per_class_area_ha", {})),
            model_metrics_json=json.dumps(stats.get("model_metrics", {})),
        )
    )

    db.query(ProvinceStat).delete()
    db.bulk_save_objects(
        [
            ProvinceStat(province=p["province"], deforestation_ha=p["deforestation_ha"])
            for p in per_province
        ]
    )
    db.commit()
    return 1 + len(per_province)


def sync_all(db: Session, data_dir: Path | None = None) -> dict[str, int]:
    """Full-refresh sync semua tabel statis dari DATA_DIR. Idempotent, non-fatal per tabel.

    Return {nama_tabel: jumlah_baris} -- dipakai utk logging di startup.
    """
    data_dir = data_dir or DATA_DIR
    counts: dict[str, int] = {}

    for name, fn in (
        ("legend", _sync_legend),
        ("landcover", _sync_landcover),
        ("deforestation", _sync_deforestation),
        ("statistics", _sync_statistics),
    ):
        try:
            counts[name] = fn(db, data_dir)
        except Exception as exc:  # noqa: BLE001 -- 1 tabel gagal tak boleh gagalkan yg lain
            db.rollback()
            logger.warning("Sync tabel '%s' gagal (%s) -- tabel ini tetap di state lama.", name, exc)
            counts[name] = -1

    return counts


if __name__ == "__main__":
    from app.db.session import SessionLocal, init_db

    logging.basicConfig(level=logging.INFO)
    init_db()
    session = SessionLocal()
    try:
        result = sync_all(session)
        print("Sync selesai:", result)
    finally:
        session.close()
