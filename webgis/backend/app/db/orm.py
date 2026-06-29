"""Skema tabel SQLAlchemy 2.0 -- dashboard data (sync dari DATA_DIR) + log/hasil
``POST /api/analyze``.

Kolom BLOB (PNG) pakai ``LargeBinary().with_variant(LONGBLOB, "mysql")`` dan kolom JSON-teks
(geojson/statistik) pakai ``Text().with_variant(LONGTEXT, "mysql")``: di MySQL, tipe polos
``BLOB``/``TEXT`` keduanya cap 64KB (TERKONFIRMASI NYATA -- deforestation_geojson hasil
analyze sungguhan ke-truncate diam-diam di 65535 byte sebelum fix ini, dites langsung lewat
MySQL asli bukan tebakan). Di backend lain (mis. SQLite saat test) keduanya jatuh ke tipe
generik yang kapasitasnya praktis tak terbatas -- jadi aman dipakai lintas dialect tanpa kode
terpisah.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.dialects.mysql import LONGBLOB, LONGTEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_PngBlob = LargeBinary().with_variant(LONGBLOB, "mysql")
_JsonText = Text().with_variant(LONGTEXT, "mysql")


class LegendItem(Base):
    __tablename__ = "legend_items"

    # autoincrement=False: id berasal dari legend.json (mulai dari 0), BUKAN identity DB --
    # tanpa ini, MySQL AUTO_INCREMENT memperlakukan literal 0 sbg "generate otomatis" (bukan
    # nilai 0 sungguhan), merusak insert id=0.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    color: Mapped[str] = mapped_column(String(16), nullable=False)


class Landcover(Base):
    """Satu baris per tahun (2021, 2025 saat ini -- VALID_YEARS bisa bertambah)."""

    __tablename__ = "landcover"

    year: Mapped[int] = mapped_column(Integer, primary_key=True)
    png_bytes: Mapped[bytes] = mapped_column(_PngBlob, nullable=False)
    bounds_south: Mapped[float] = mapped_column(Float, nullable=False)
    bounds_west: Mapped[float] = mapped_column(Float, nullable=False)
    bounds_north: Mapped[float] = mapped_column(Float, nullable=False)
    bounds_east: Mapped[float] = mapped_column(Float, nullable=False)
    crs: Mapped[str] = mapped_column(String(32), nullable=False, default="EPSG:4326")


class DeforestationFeature(Base):
    """1 baris = 1 polygon perubahan. transition_type/province/area_ha indexed -- jadi WHERE SQL
    ganti filter list-comprehension yang dipakai versi file-based lama."""

    __tablename__ = "deforestation_features"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # mis. "DF-00000"
    transition_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    province: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    area_ha: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    period_from: Mapped[int] = mapped_column(Integer, nullable=False)
    period_to: Mapped[int] = mapped_column(Integer, nullable=False)
    kawasan_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    geometry_json: Mapped[str] = mapped_column(_JsonText, nullable=False)


class Statistics(Base):
    """Singleton (id selalu 1) -- full-refresh: hapus baris lama, insert baru tiap sync.

    per_transition_ha / per_class_area_ha / model_metrics tak pernah di-query parsial oleh
    endpoint manapun (selalu dikembalikan utuh) -- disimpan sbg kolom JSON teks, bukan
    dinormalisasi ke tabel sendiri.
    """

    __tablename__ = "statistics"

    # id selalu 1 (singleton) -- bukan identity DB, autoincrement=False sama alasannya dgn LegendItem.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    period_from: Mapped[int] = mapped_column(Integer, nullable=False)
    period_to: Mapped[int] = mapped_column(Integer, nullable=False)
    total_deforestation_ha: Mapped[float] = mapped_column(Float, nullable=False)
    n_hotspots: Mapped[int] = mapped_column(Integer, nullable=False)
    per_transition_ha_json: Mapped[str] = mapped_column(_JsonText, nullable=False)
    per_class_area_ha_json: Mapped[str] = mapped_column(_JsonText, nullable=False)
    model_metrics_json: Mapped[str] = mapped_column(_JsonText, nullable=False)


class ProvinceStat(Base):
    """Dinormalisasi (beda dgn Statistics) krn sudah tabular & dipakai langsung oleh
    GET /api/statistics/per-province."""

    __tablename__ = "province_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    province: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    deforestation_ha: Mapped[float] = mapped_column(Float, nullable=False)


class AnalyzeRequestLog(Base):
    """1 baris per panggilan POST /api/analyze, sukses ATAU gagal."""

    __tablename__ = "analyze_request_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    requested_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)
    aoi_lon_min: Mapped[float] = mapped_column(Float, nullable=False)
    aoi_lat_min: Mapped[float] = mapped_column(Float, nullable=False)
    aoi_lon_max: Mapped[float] = mapped_column(Float, nullable=False)
    aoi_lat_max: Mapped[float] = mapped_column(Float, nullable=False)
    year_t1: Mapped[int] = mapped_column(Integer, nullable=False)
    year_t2: Mapped[int] = mapped_column(Integer, nullable=False)
    min_area_ha: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # "success" | "error"
    http_status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    result: Mapped["AnalyzeResult | None"] = relationship(
        back_populates="request_log", uselist=False
    )


class AnalyzeResult(Base):
    """1:1 dgn AnalyzeRequestLog -- cuma dibuat utk request yang sukses."""

    __tablename__ = "analyze_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_log_id: Mapped[int] = mapped_column(
        ForeignKey("analyze_request_log.id"), nullable=False, unique=True
    )
    deforestation_geojson: Mapped[str] = mapped_column(_JsonText, nullable=False)
    statistics_json: Mapped[str] = mapped_column(_JsonText, nullable=False)
    bounds_south: Mapped[float] = mapped_column(Float, nullable=False)
    bounds_west: Mapped[float] = mapped_column(Float, nullable=False)
    bounds_north: Mapped[float] = mapped_column(Float, nullable=False)
    bounds_east: Mapped[float] = mapped_column(Float, nullable=False)
    landcover_t1_png: Mapped[bytes | None] = mapped_column(_PngBlob, nullable=True)
    landcover_t2_png: Mapped[bytes | None] = mapped_column(_PngBlob, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    request_log: Mapped[AnalyzeRequestLog] = relationship(back_populates="result")
