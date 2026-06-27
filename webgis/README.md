# ForestWatch Papua — WebGIS

Dashboard + WebGIS interaktif untuk visualisasi tutupan lahan & deteksi deforestasi Papua
(skema **7 kelas**), hasil model segmentasi Sentinel-2. Repo ini **self-contained**: frontend,
backend, dan package model (`forestwatch`, di-vendor) ada di sini semua.

## Struktur

```
webgis/
├── frontend/        React + Vite + Leaflet + Chart.js (7 halaman)
│   ├── src/         App.jsx (dashboard, peta, statistik, akurasi, metodologi, unduh, analisis custom)
│   ├── dummy-data/  data contoh (dipakai bila backend tak diset) -- ikut di-commit
│   └── Dockerfile
├── backend/         FastAPI (8 endpoint kontrak + POST /api/analyze)
│   ├── app/
│   ├── data/        7 file kontrak (.gitignored kecuali .gitkeep -- isi saat deploy)
│   └── Dockerfile
├── model/           package `forestwatch` (VENDORED dari repo model -- lihat model/README.md)
│   ├── src/forestwatch/
│   └── configs/
└── docker-compose.yml
```

## Menjalankan (mode dev)

### Frontend (cukup ini untuk demo UI)
```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```
Tanpa konfigurasi apa pun, frontend memakai `dummy-data/`. Untuk menyambung ke backend, isi
`frontend/.env` → `VITE_API_URL=http://localhost:8000` lalu jalankan ulang.

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # sesuaikan
uvicorn app.main:app --reload --port 8000             # http://localhost:8000/docs
```

## Menjalankan (Docker)
```bash
docker compose up --build
# backend  -> http://localhost:8000
# frontend -> http://localhost:4173
```

## Endpoint backend
| Endpoint | Fungsi |
|---|---|
| `GET /api/health` | health check |
| `GET /api/legend` | kode kelas + warna |
| `GET /api/landcover/{2021\|2025}` | PNG tutupan lahan + bounds |
| `GET /api/deforestation` | GeoJSON polygon perubahan |
| `GET /api/statistics` | KPI, per-provinsi, per-transisi, per-kelas |
| `GET /api/download/{geojson\|metrics\|legend}` | unduh artefak |
| `POST /api/analyze` | **analisis wilayah custom on-demand** (GEE + model) |

`POST /api/analyze` butuh: (a) kredensial GEE Service Account (`GEE_SERVICE_ACCOUNT_*` di
`backend/.env`), (b) checkpoint model (`MODEL_CHECKPOINT_PATH`), (c) package `forestwatch`
(sudah di `model/`). Kalau `GEE_SERVICE_ACCOUNT_EMAIL` kosong, endpoint ini balas **503**
dengan pesan jelas; endpoint lain tetap normal.

## Package `forestwatch` (vendored)
Backend memakai logika composite/inferensi/change-detection dari package `forestwatch` di
`model/`. `backend/app/core/forestwatch_bridge.py` menambahkan `model/src` ke `sys.path` saat
runtime (tak wajib `pip install`). Ini **salinan** — sumber kebenaran ada di repo model.
Detail & cara re-vendor: lihat [`model/README.md`](model/README.md).

## Catatan data
- `frontend/dummy-data/` — data contoh, **ikut di-commit** (supaya frontend jalan stand-alone).
- `backend/data/` — diisi 7 file kontrak hasil model saat deploy (gitignored).
- Checkpoint `.pt` / `.onnx` & raster `.tif` **tidak** di-commit (terlalu besar; lihat `.gitignore`).
