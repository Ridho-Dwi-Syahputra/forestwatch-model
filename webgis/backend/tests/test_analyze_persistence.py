"""Unit test app/db/analyze_persistence.py -- TANPA FastAPI/GEE/torch, cuma ORM + schema."""

import base64
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.analyze_persistence import log_analyze_request, save_analyze_result
from app.db.orm import AnalyzeRequestLog, AnalyzeResult, Base
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'analyze_test.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    yield session
    session.close()


def _sample_request() -> AnalyzeRequest:
    return AnalyzeRequest(aoi=[138.0, -5.0, 138.2, -4.8], year_t1=2021, year_t2=2025, min_area_ha=0.5)


def _sample_response() -> AnalyzeResponse:
    png_data_url = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nFAKE").decode("ascii")
    return AnalyzeResponse(
        deforestation={"type": "FeatureCollection", "features": [], "total": 0},
        statistics={"total_deforestation_ha": 0.0},
        bounds=[[-4.8, 138.0], [-5.0, 138.2]],
        year_t1=2021, year_t2=2025,
        landcover_t1_png=png_data_url,
        landcover_t2_png=png_data_url,
    )


def test_save_analyze_result_sukses(db_session):
    req = _sample_request()
    resp = _sample_response()

    save_analyze_result(db_session, req, resp, duration_ms=1234)

    log_row = db_session.query(AnalyzeRequestLog).one()
    assert log_row.status == "success"
    assert log_row.http_status_code == 200
    assert log_row.duration_ms == 1234
    assert log_row.year_t1 == 2021 and log_row.year_t2 == 2025

    result_row = db_session.query(AnalyzeResult).one()
    assert result_row.request_log_id == log_row.id
    assert json.loads(result_row.deforestation_geojson) == resp.deforestation
    assert json.loads(result_row.statistics_json) == resp.statistics
    assert result_row.landcover_t1_png == b"\x89PNG\r\n\x1a\nFAKE"


def test_log_analyze_request_error_path_tanpa_result(db_session):
    req = _sample_request()

    log_analyze_request(
        db_session, req, status="error", http_status_code=503,
        error_message="GEE belum siap", duration_ms=42,
    )

    log_row = db_session.query(AnalyzeRequestLog).one()
    assert log_row.status == "error"
    assert log_row.http_status_code == 503
    assert log_row.error_message == "GEE belum siap"
    assert db_session.query(AnalyzeResult).count() == 0


def test_png_data_url_didecode_jadi_bytes_mentah(db_session):
    req = _sample_request()
    resp = _sample_response()

    save_analyze_result(db_session, req, resp, duration_ms=1)

    result_row = db_session.query(AnalyzeResult).one()
    assert isinstance(result_row.landcover_t1_png, bytes)
    assert result_row.landcover_t1_png.startswith(b"\x89PNG")


def test_kegagalan_db_tertelan_tidak_raise(db_session):
    """Jaminan inti: kegagalan tulis log/hasil TIDAK BOLEH merusak response live."""
    req = _sample_request()
    resp = _sample_response()
    db_session.close()  # session ditutup duluan -> operasi DB di bawah pasti gagal

    log_analyze_request(db_session, req, status="error", http_status_code=500)
    save_analyze_result(db_session, req, resp, duration_ms=1)
    # tidak ada exception yang lolos sampai sini -> test ini lulus
