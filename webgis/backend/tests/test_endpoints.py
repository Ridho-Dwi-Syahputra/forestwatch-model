"""Test endpoint backend FastAPI menggunakan TestClient."""

import json
import pytest
from fastapi.testclient import TestClient
from pathlib import Path


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    """TestClient dengan data dummy di tmp folder."""
    import os

    # Buat dummy data
    data_dir = tmp_path_factory.mktemp("data")
    static_dir = tmp_path_factory.mktemp("static")

    # legend.json
    legend = [
        {"id": i, "name": f"Kelas {i}", "color": f"#{'AB' * 3}"}
        for i in range(6)
    ]
    (data_dir / "legend.json").write_text(json.dumps(legend))

    # statistics.json
    stats = {
        "period_from": 2021, "period_to": 2025,
        "total_deforestation_ha": 1000.0, "n_hotspots": 10,
        "per_transition_ha": {
            "hutan_ke_lahan_terbuka": 200.0,
            "hutan_ke_sawit": 400.0,
            "hutan_ke_pertanian_lain": 300.0,
            "hutan_ke_terbakar": 100.0,
        },
        "per_province": [{"province": "Papua Selatan", "deforestation_ha": 600.0}],
        "per_class_area_ha": {"Hutan": 5000.0},
        "model_metrics": {"overall_accuracy": 0.86, "mean_iou": 0.71, "per_class": []},
    }
    (data_dir / "statistics.json").write_text(json.dumps(stats))

    # deforestation.geojson
    fc = {
        "type": "FeatureCollection", "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[130, -5], [131, -5], [131, -4], [130, -5]]]},
                "properties": {
                    "id": "DF-00001", "transition_type": "hutan_ke_sawit",
                    "area_ha": 12.4, "period_from": 2021, "period_to": 2025,
                    "province": "Papua Selatan",
                },
            }
        ]
    }
    (data_dir / "deforestation.geojson").write_text(json.dumps(fc))

    # PNG + bounds dummy
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.zeros((10, 10, 4), dtype="uint8"), mode="RGBA")
    img.save(data_dir / "landcover_2025.png")
    img.save(data_dir / "landcover_2021.png")
    bounds = {"bounds": [[-9.5, 130.0], [0.5, 141.2]], "crs": "EPSG:4326"}
    (data_dir / "landcover_2025_bounds.json").write_text(json.dumps(bounds))
    (data_dir / "landcover_2021_bounds.json").write_text(json.dumps(bounds))

    # metrics.json
    (data_dir / "metrics.json").write_text(json.dumps({"mean_iou": 0.71}))

    # Patch config
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["CORS_ORIGINS"] = "http://localhost:5173"

    # Import setelah env di-set
    import importlib
    import app.core.config as cfg
    importlib.reload(cfg)

    from app.main import app as fastapi_app
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


def test_download_invalid(client):
    r = client.get("/api/download/invalid_type")
    assert r.status_code == 400


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
