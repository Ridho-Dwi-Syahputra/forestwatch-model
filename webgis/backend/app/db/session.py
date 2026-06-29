"""Engine/session SQLAlchemy (MySQL) + dependency FastAPI + pembuatan database/tabel.

Tanpa Alembic (simplifikasi sengaja -- proyek kecil/solo, skema jarang berubah).
"""

from __future__ import annotations

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import DATABASE_SERVER_URL, DATABASE_URL, DB_NAME
from app.db.orm import Base

logger = logging.getLogger(__name__)

# pool_pre_ping=True: hindari error "MySQL server has gone away" akibat koneksi idle timeout
# (umum kalau backend dibiarkan jalan lama tanpa request, koneksi pool jadi basi).
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Buat database (kalau belum ada) lalu buat semua tabel. Idempotent -- aman dipanggil
    tiap startup."""
    server_engine = create_engine(DATABASE_SERVER_URL, future=True)
    try:
        with server_engine.connect() as conn:
            conn.execute(
                text(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4")
            )
            conn.commit()
    finally:
        server_engine.dispose()

    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency -- yield 1 Session, selalu ditutup setelah request selesai."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
