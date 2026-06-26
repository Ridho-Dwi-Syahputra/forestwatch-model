# PENTING: build context untuk Dockerfile ini adalah ROOT MONOREPO (bukan webgis/backend/),
# karena app/core/forestwatch_bridge.py butuh akses ke model/src/forestwatch (POST /api/analyze).
# Struktur folder di image SENGAJA disamakan dgn repo (webgis/backend/ + model/src/) supaya
# perhitungan path relatif di forestwatch_bridge.py (BACKEND_ROOT.parent.parent / "model/src")
# tetap valid tanpa logika khusus untuk Docker.
#
# Build dari root repo:
#   docker build -f webgis/backend/Dockerfile -t forestwatch-backend .
#
# Kalau fitur /api/analyze TIDAK dipakai (GEE_SERVICE_ACCOUNT_EMAIL kosong di env), folder
# model/src boleh tidak ada -- backend tetap jalan normal untuk endpoint pre-computed lainnya.

FROM python:3.11-slim

WORKDIR /app

COPY webgis/backend/requirements.txt webgis/backend/requirements.txt
RUN pip install --no-cache-dir -r webgis/backend/requirements.txt

COPY webgis/backend/ webgis/backend/
COPY model/src/forestwatch model/src/forestwatch

WORKDIR /app/webgis/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
