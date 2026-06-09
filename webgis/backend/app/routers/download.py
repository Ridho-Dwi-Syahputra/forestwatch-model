"""Endpoint download file data."""

import json
import csv
import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from app.core.config import DATA_DIR, FILES

router = APIRouter()

DOWNLOADABLE = {
    "geojson": FILES["deforestation"],
    "legend":  FILES["legend"],
    "metrics": FILES["metrics"],
}


@router.get("/download/{file_type}")
def download_file(file_type: str):
    if file_type not in DOWNLOADABLE:
        raise HTTPException(
            status_code=400,
            detail=f"file_type tidak valid. Pilihan: {list(DOWNLOADABLE.keys())}"
        )

    path = DATA_DIR / DOWNLOADABLE[file_type]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{DOWNLOADABLE[file_type]} tidak ditemukan")

    return FileResponse(
        path=str(path),
        filename=path.name,
        media_type="application/json",
    )


@router.get("/download/deforestation/csv")
def download_deforestation_csv():
    """Konversi deforestation.geojson → CSV untuk unduhan."""
    path = DATA_DIR / FILES["deforestation"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="deforestation.geojson tidak ditemukan")

    fc = json.loads(path.read_text(encoding="utf-8"))
    features = fc.get("features", [])

    if not features:
        raise HTTPException(status_code=404, detail="Tidak ada data deforestasi")

    # Kumpulkan semua property keys
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
