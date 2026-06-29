from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.orm import LegendItem
from app.db.session import get_db

router = APIRouter()


def serialize_legend(db: Session) -> list[dict]:
    """Reused oleh /api/download/legend -- jangan duplikasi query+dict-comprehension."""
    rows = db.query(LegendItem).order_by(LegendItem.id).all()
    return [{"id": r.id, "name": r.name, "color": r.color} for r in rows]


@router.get("/legend")
def get_legend(db: Session = Depends(get_db)):
    items = serialize_legend(db)
    if not items:
        raise HTTPException(status_code=404, detail="Data legend belum tersedia di database")
    return items
