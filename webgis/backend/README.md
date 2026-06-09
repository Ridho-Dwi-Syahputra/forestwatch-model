# ForestWatch Papua — Backend API

REST API FastAPI untuk WebGIS dashboard. Menserve 7 file kontrak dari Orang 1 (Model) ke frontend React.

## Quickstart

```powershell
cd "D:\Local Disk D\Tugas\SATRIA DATA\webgis\backend"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Buka browser: **http://localhost:8000/docs** (Swagger UI)

## Endpoints

| Method | URL | Keterangan |
| --- | --- | --- |
| GET | `/api/health` | Status server |
| GET | `/api/legend` | 6 kelas + warna |
| GET | `/api/landcover/{year}` | URL PNG + bounds (year: 2021 atau 2025) |
| GET | `/api/deforestation` | GeoJSON + filter query params |
| GET | `/api/statistics` | Statistik lengkap |
| GET | `/api/statistics/per-province` | Breakdown per provinsi |
| GET | `/api/statistics/per-transition` | Breakdown per transisi |
| GET | `/api/statistics/summary` | Ringkasan untuk card UI |
| GET | `/api/download/geojson` | Download deforestation.geojson |
| GET | `/api/download/deforestation/csv` | Download sebagai CSV |

### Query Params `/api/deforestation`

| Param | Tipe | Contoh |
| --- | --- | --- |
| `transition_type` | string | `hutan_ke_sawit` |
| `province` | string | `Papua Selatan` |
| `min_area_ha` | float | `5.0` |

## Struktur Folder

```
backend/
├── app/
│   ├── main.py          # FastAPI app + CORS + routes
│   ├── core/config.py   # Path data, CORS origins, nama file
│   ├── routers/         # 1 file per endpoint group
│   └── schemas/         # Pydantic models
├── data/                # 7 file kontrak (dari Orang 1)
├── static/              # PNG landcover (auto-copy dari data/)
├── tests/               # pytest endpoint tests
├── Dockerfile
└── requirements.txt
```

## Mengganti Data Dummy dengan Data Asli

Saat Orang 1 selesai (Minggu 3), cukup replace file di folder `data/`:

```powershell
# Copy dari Google Drive ForestWatch_Outputs
Copy-Item "PATH_DRIVE\ForestWatch_Outputs\*" "data\" -Force
```

Tidak perlu ubah kode apapun — schema sudah sesuai PRD §B.1.

## Environment Variables

Salin `.env.example` → `.env`:

```
DATA_DIR=./data
CORS_ORIGINS=http://localhost:5173,https://forestwatch-papua.vercel.app
```

## Deploy (Production)

Backend di-deploy ke **Railway** atau **Render** (free tier):

1. Push repo ke GitHub
2. Connect Railway ke repo
3. Set env var `CORS_ORIGINS` ke URL frontend production
4. Deploy — Railway auto-detect Dockerfile
