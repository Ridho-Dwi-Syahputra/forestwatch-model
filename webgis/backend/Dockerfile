# PENTING: build context Dockerfile ini = ROOT REPO WEBGIS (folder yang berisi
# backend/ + model/), BUKAN folder backend/. Sebab app/core/forestwatch_bridge.py butuh
# akses ke ../model/src/forestwatch untuk fitur POST /api/analyze. Struktur folder di image
# SENGAJA disamakan dgn repo (backend/ + model/src/) supaya path relatif di bridge
# (BACKEND_ROOT.parent / "model/src") tetap valid tanpa logika khusus Docker.
#
# Build (dari root repo webgis):
#   docker build -f backend/Dockerfile -t forestwatch-backend .
# Di monorepo, context-nya folder webgis/:
#   docker build -f webgis/backend/Dockerfile -t forestwatch-backend webgis
#
# Kalau fitur /api/analyze TIDAK dipakai (GEE_SERVICE_ACCOUNT_EMAIL kosong di env), folder
# model/ boleh tidak ada -- backend tetap jalan normal untuk endpoint pre-computed lainnya.

FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY model/ model/

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
