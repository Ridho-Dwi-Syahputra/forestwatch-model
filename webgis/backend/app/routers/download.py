"""Endpoint download file data -- dibangun dari DB (reuse helper serialisasi router lain,
jangan re-query+rebuild independen)."""

from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.routers.deforestation import feature_to_dict, query_deforestation_features
from app.routers.legend import serialize_legend
from app.routers.statistics import load_stats

router = APIRouter()

VALID_DOWNLOAD_FILES = {"geojson", "legend", "metrics"}


@router.get("/download/{file_type}")
def download_file(file_type: str, db: Session = Depends(get_db)):
    if file_type not in VALID_DOWNLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"file_type tidak valid. Pilihan: {sorted(VALID_DOWNLOAD_FILES)}",
        )

    if file_type == "legend":
        payload = serialize_legend(db)
    elif file_type == "metrics":
        # Konsolidasi: metrics.json lama cuma duplikat statistics.model_metrics -- derive dari
        # situ (bare dict, BUKAN dibungkus {"model_metrics": {...}}, sesuai bentuk file asli).
        payload = load_stats(db)["model_metrics"]
    else:  # "geojson"
        rows = query_deforestation_features(db)
        payload = {
            "type": "FeatureCollection",
            "features": [feature_to_dict(r) for r in rows],
            "total": len(rows),
        }

    return Response(
        content=json.dumps(payload).encode("utf-8"),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={file_type}.json"},
    )


@router.get("/download/deforestation/csv")
def download_deforestation_csv(db: Session = Depends(get_db)):
    """Konversi data deforestasi -> CSV untuk unduhan."""
    rows = query_deforestation_features(db)
    if not rows:
        raise HTTPException(status_code=404, detail="Tidak ada data deforestasi")

    features = [feature_to_dict(r) for r in rows]

    all_keys = set()
    for f in features:
        all_keys.update(f.get("properties", {}).keys())
    fieldnames = sorted(all_keys)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for f in features:
        writer.writerow(f.get("properties", {}))

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=deforestation.csv"},
    )
