"""Test endpoint backend FastAPI menggunakan TestClient.

DB test pakai SQLite terpisah (tmp file) -- bukan MySQL milik app/db/session.py. Data
dimasukkan langsung sbg baris ORM (bukan via file+sync_static), lalu di-override ke
app.dependency_overrides[get_db]. TestClient dipakai TANPA `with`-context-manager, jadi
event startup FastAPI (yang konek ke MySQL) tidak pernah jalan -- pytest aman tanpa
MySQL server menyala. Lihat catatan di plan migrasi DB.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    # Paksa fitur "Analisis Custom" tampak unconfigured di test ini, TERLEPAS dari isi .env
    # nyata di mesin dev (load_dotenv() di config.py akan memuatnya kalau ada) -- supaya
    # test_analyze_unconfigured_returns_503 deterministik, bukan kebetulan lolos krn .env kosong.
    os.environ["CORS_ORIGINS"] = "http://localhost:5173"
    os.environ["GEE_SERVICE_ACCOUNT_EMAIL"] = ""
    os.environ["GEE_SERVICE_ACCOUNT_KEY_PATH"] = ""
    os.environ["MODEL_CHECKPOINT_PATH"] = str(tmp_path_factory.mktemp("data") / "nonexistent_model.pt")

    import importlib

    import app.core.config as cfg
    importlib.reload(cfg)

    from app.db.orm import (
        Base, DeforestationFeature, Landcover, LegendItem, ProvinceStat, Statistics,
    )
    from app.db.session import get_db
    from app.main import app as fastapi_app

    db_path = tmp_path_factory.mktemp("db") / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    session = TestSessionLocal()
    session.add_all(
        LegendItem(id=i, name=f"Kelas {i}", color="#ABABAB") for i in range(6)
    )
    session.add(
        Statistics(
            id=1, period_from=2021, period_to=2025,
            total_deforestation_ha=1000.0, n_hotspots=10,
            per_transition_ha_json=json.dumps({
                "hutan_ke_lahan_terbuka": 200.0,
                "hutan_ke_sawit": 400.0,
                "hutan_ke_pertanian_lain": 300.0,
                "hutan_ke_terbakar": 100.0,
            }),
            per_class_area_ha_json=json.dumps({"Hutan": 5000.0}),
            model_metrics_json=json.dumps(
                {"overall_accuracy": 0.86, "mean_iou": 0.71, "per_class": []}
            ),
        )
    )
    session.add(ProvinceStat(province="Papua Selatan", deforestation_ha=600.0))
    session.add(
        DeforestationFeature(
            id="DF-00001", transition_type="hutan_ke_sawit", province="Papua Selatan",
            area_ha=12.4, period_from=2021, period_to=2025, kawasan_status=None,
            geometry_json=json.dumps(
                {"type": "Polygon", "coordinates": [[[130, -5], [131, -5], [131, -4], [130, -5]]]}
            ),
        )
    )

    from io import BytesIO

    import numpy as np
    from PIL import Image

    buf = BytesIO()
    Image.fromarray(np.zeros((10, 10, 4), dtype="uint8"), mode="RGBA").save(buf, format="PNG")
    png_bytes = buf.getvalue()
    for year in (2021, 2025):
        session.add(
            Landcover(
                year=year, png_bytes=png_bytes,
                bounds_south=-9.5, bounds_west=130.0, bounds_north=0.5, bounds_east=141.2,
                crs="EPSG:4326",
            )
        )
    session.commit()
    session.close()

    def _override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    return TestClient(fastapi_app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_legend(client):
    r = client.get("/api/legend")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 6


def test_statistics(client):
    r = client.get("/api/statistics")
    assert r.status_code == 200
    data = r.json()
    assert data["period_from"] == 2021
    assert data["n_hotspots"] == 10


def test_statistics_per_province(client):
    r = client.get("/api/statistics/per-province")
    assert r.status_code == 200
    assert "data" in r.json()


def test_statistics_per_transition(client):
    r = client.get("/api/statistics/per-transition")
    assert r.status_code == 200
    assert "data" in r.json()


def test_statistics_summary(client):
    r = client.get("/api/statistics/summary")
    assert r.status_code == 200
    data = r.json()
    assert "total_deforestation_ha" in data
    assert "n_hotspots" in data


def test_deforestation_no_filter(client):
    r = client.get("/api/deforestation")
    assert r.status_code == 200
    data = r.json()
    assert data["type"] == "FeatureCollection"
    assert data["total"] == 1


def test_deforestation_filter_transition(client):
    r = client.get("/api/deforestation?transition_type=hutan_ke_sawit")
    assert r.status_code == 200
    assert r.json()["total"] == 1

    r2 = client.get("/api/deforestation?transition_type=hutan_ke_terbakar")
    assert r2.json()["total"] == 0


def test_deforestation_invalid_transition(client):
    r = client.get("/api/deforestation?transition_type=invalid")
    assert r.status_code == 400


def test_deforestation_filter_province(client):
    r = client.get("/api/deforestation?province=Papua Selatan")
    assert r.status_code == 200
    assert r.json()["total"] == 1


def test_deforestation_filter_min_area(client):
    r = client.get("/api/deforestation?min_area_ha=100")
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_download_geojson(client):
    r = client.get("/api/download/geojson")
    assert r.status_code == 200
    assert "FeatureCollection" in r.text


def test_download_legend(client):
    r = client.get("/api/download/legend")
    assert r.status_code == 200
    assert len(r.json()) == 6


def test_download_metrics(client):
    r = client.get("/api/download/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["mean_iou"] == 0.71
    assert "model_metrics" not in data  # bare dict, bukan dibungkus


def test_download_deforestation_csv(client):
    r = client.get("/api/download/deforestation/csv")
    assert r.status_code == 200
    assert "transition_type" in r.text
    assert "DF-00001" in r.text


def test_download_invalid(client):
    r = client.get("/api/download/invalid_type")
    assert r.status_code == 400


def test_landcover(client):
    r = client.get("/api/landcover/2021")
    assert r.status_code == 200
    data = r.json()
    assert data["year"] == 2021
    assert data["image_url"].endswith("/api/landcover/2021/image")
    assert data["bounds"] == [[-9.5, 130.0], [0.5, 141.2]]


def test_landcover_invalid_year(client):
    r = client.get("/api/landcover/1999")
    assert r.status_code == 400


def test_landcover_image(client):
    r = client.get("/api/landcover/2021/image")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"  # magic bytes PNG


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()
    assert "POST /api/analyze" in r.json()["endpoints"]


def test_analyze_unconfigured_returns_503(client):
    """Tanpa GEE_SERVICE_ACCOUNT_EMAIL & checkpoint model di env test, endpoint analyze harus
    gagal jelas (503) -- bukan crash 500 -- supaya sisa API tetap bisa didemokan tanpa fitur ini."""
    r = client.post(
        "/api/analyze",
        json={"aoi": [138.0, -5.0, 138.2, -4.8], "year_t1": 2021, "year_t2": 2025},
    )
    assert r.status_code == 503


def test_analyze_invalid_aoi_returns_422(client):
    r = client.post(
        "/api/analyze",
        json={"aoi": [138.2, -4.8, 138.0, -5.0], "year_t1": 2021, "year_t2": 2025},
    )
    assert r.status_code == 422
