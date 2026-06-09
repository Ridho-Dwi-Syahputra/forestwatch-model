import json
from fastapi import APIRouter, HTTPException
from app.core.config import DATA_DIR, FILES

router = APIRouter()


@router.get("/legend")
def get_legend():
    path = DATA_DIR / FILES["legend"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="legend.json tidak ditemukan")
    return json.loads(path.read_text(encoding="utf-8"))
