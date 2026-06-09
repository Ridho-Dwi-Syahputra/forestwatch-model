# MASTER PLAN — Role WEBGIS (Orang 2)

**Proyek:** ForestWatch Papua — Sistem Deteksi Dini Deforestasi Berbasis Deep Learning
**Kompetisi:** Statistics Essay Competition (SEC) SATRIA DATA 2026 · Subtema 1
**Institusi:** Universitas Andalas
**Deadline akhir:** 30 Juni 2026, 16.00 WIB
**Dokumen ini:** Panduan eksekusi harian untuk Developer Lead (Full-Stack)

> Dokumen ini adalah turunan dari [PRD v2.0](../PRD_ForestWatch_Papua_v2.md) dan [Buku Panduan v2.0](../Buku_Panduan_ForestWatch_Papua_v2.docx). PRD menyebut Leaflet vanilla; dokumen ini meng-upgrade ke **arsitektur full-stack FastAPI + React** sesuai keputusan tim. Skema data (PRD §B.1) tetap dipakai persis.

---

## 1. Ringkasan Peran & Tujuan

Sebagai **Developer Lead (Orang 2)**, tugas utama adalah membangun **WebGIS dashboard interaktif** untuk seluruh Papua yang menampilkan hasil deteksi deforestasi dari model Orang 1. WebGIS harus dapat **diakses publik via URL**, semua fitur berfungsi, dan **tahan crash saat presentasi**.

### Target Keberhasilan

- WebGIS deployed dengan URL live (frontend + backend)
- Semua **14 fitur wajib** dari PRD §B.2.3 berfungsi
- Studi kasus Merauke punya tombol/zoom preset yang jalan
- 4 filter jenis transisi berfungsi
- Slider waktu 2021 ↔ 2024 instan
- Video demo 2 menit tersedia
- Tested di Chrome dan Safari
- **Tahan crash** saat demo (pre-computed products, tidak ada inferensi real-time)

---

## 2. Arsitektur Full-Stack

```
┌─────────────────────────────────────┐
│   User Browser                      │
│   (Chrome / Safari)                 │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   FRONTEND                          │
│   React 18 + Vite + react-leaflet   │
│   Deployed: Vercel / Netlify        │
│   Folder: docs/webgis/frontend/     │
└─────────────────┬───────────────────┘
                  │  fetch JSON via REST
                  ▼
┌─────────────────────────────────────┐
│   BACKEND                           │
│   FastAPI + uvicorn                 │
│   Deployed: Railway / Render        │
│   Folder: docs/webgis/backend/      │
└─────────────────┬───────────────────┘
                  │  baca file lokal
                  ▼
┌─────────────────────────────────────┐
│   DATA (dari Orang 1)               │
│   7 file di backend/data/           │
│   - landcover_*.png + bounds.json   │
│   - deforestation.geojson           │
│   - statistics.json                 │
│   - legend.json + metrics.json      │
│   - model.onnx (referensi saja)     │
└─────────────────────────────────────┘
```

**Catatan:** Backend **TIDAK** menjalankan inferensi ONNX. Hanya membaca file pre-computed yang dihasilkan Orang 1. Ini sesuai semangat PRD §0.2 — minimalkan risiko crash saat demo.

---

## 3. Stack Teknologi

### Backend (FastAPI)

| Library | Versi | Fungsi |
| --- | --- | --- |
| Python | 3.10+ | Runtime |
| `fastapi` | latest | Framework REST |
| `uvicorn[standard]` | latest | ASGI server |
| `pydantic` | v2 | Schema validation |
| `geopandas` | latest | Operasi GeoJSON (filter, simpan) |
| `rasterio` | latest | Operasi raster (opsional, untuk on-demand crop) |
| `shapely` | latest | Geometry ops (simplify, intersect) |
| `python-multipart` | latest | File download endpoint |

### Frontend (React + Vite)

| Library | Versi | Fungsi |
| --- | --- | --- |
| Node.js | 18+ | Runtime |
| `vite` | 5+ | Build tool |
| `react` + `react-dom` | 18+ | UI framework |
| `react-leaflet` | 4+ | Leaflet bindings |
| `leaflet` | 1.9+ | Peta dasar |
| `leaflet.markercluster` | 1.5+ | Clustering marker |
| `chart.js` + `react-chartjs-2` | 4+ | Pie chart & bar chart |
| `tailwindcss` | 3+ | Styling |
| `axios` | latest | HTTP client |

---

## 4. Struktur Folder

### Backend — `docs/webgis/backend/`

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + CORS middleware
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py           # DATA_DIR, CORS_ORIGINS
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── landcover.py        # GET /api/landcover/{year}
│   │   ├── deforestation.py    # GET /api/deforestation (filter)
│   │   ├── statistics.py       # GET /api/statistics + breakdowns
│   │   ├── legend.py           # GET /api/legend
│   │   ├── download.py         # GET /api/download/{file}
│   │   └── health.py           # GET /api/health
│   └── schemas/
│       ├── __init__.py
│       ├── feature.py          # GeoJSON feature pydantic
│       └── statistics.py       # Statistics pydantic
├── data/                       # Copy/symlink dari ForestWatch_Outputs/
│   ├── landcover_2024.png
│   ├── landcover_2024_bounds.json
│   ├── landcover_2021.png
│   ├── landcover_2021_bounds.json
│   ├── deforestation.geojson
│   ├── statistics.json
│   └── legend.json
├── static/                     # Serve PNG landcover via /static
├── tests/
│   └── test_endpoints.py
├── .env.example                # CORS_ORIGINS=http://localhost:5173
├── requirements.txt
├── Dockerfile                  # Untuk Railway/HuggingFace
└── README.md
```

### Frontend — `docs/webgis/frontend/`

```
frontend/
├── src/
│   ├── main.jsx                # Entry point
│   ├── App.jsx                 # Root component (layout 2-kolom)
│   ├── components/
│   │   ├── MapView.jsx         # MapContainer + layer compose
│   │   ├── layers/
│   │   │   ├── LandcoverOverlay.jsx
│   │   │   └── DeforestationLayer.jsx
│   │   ├── controls/
│   │   │   ├── TimeSlider.jsx        # 2021 ↔ 2024
│   │   │   ├── TransitionFilter.jsx  # 4 checkbox
│   │   │   ├── OpacitySlider.jsx
│   │   │   └── MeraukeButton.jsx
│   │   ├── panel/
│   │   │   ├── StatsPanel.jsx
│   │   │   ├── TransitionPieChart.jsx
│   │   │   ├── ProvinceBarChart.jsx
│   │   │   └── MetricsTable.jsx
│   │   ├── Legend.jsx
│   │   ├── PopupContent.jsx
│   │   ├── AboutModal.jsx
│   │   └── DownloadButton.jsx
│   ├── api/
│   │   └── client.js           # axios instance + endpoints
│   ├── hooks/
│   │   ├── useDeforestation.js
│   │   └── useStatistics.js
│   ├── styles/
│   │   └── index.css           # Tailwind directives
│   └── constants/
│       ├── colors.js           # Palette 6 kelas + 4 transisi
│       └── papua.js            # Center, zoom, bounds
├── public/
│   ├── favicon.ico
│   └── logo.png
├── index.html
├── vite.config.js              # Set VITE_API_URL env
├── tailwind.config.js
├── postcss.config.js
├── package.json
├── .env.example                # VITE_API_URL=http://localhost:8000
└── README.md
```

---

## 5. Pra-Pekerjaan (Setup Day 0)

Selesai dalam **1 hari**:

- [ ] Node.js 18+ terinstal (`node --version`)
- [ ] Python 3.10+ terinstal (`python --version`)
- [ ] VS Code dengan extension: Python, ESLint, Prettier, Tailwind IntelliSense
- [ ] Repo GitHub `forestwatch-papua-webgis` dibuat
- [ ] Subfolder `backend/` dan `frontend/` sudah ada (sudah dibuat sebelumnya)
- [ ] Buat Python venv di `backend/`: `python -m venv .venv`
- [ ] Scaffold Vite + React di `frontend/`: `npm create vite@latest . -- --template react`
- [ ] Install Tailwind di frontend: `npm i -D tailwindcss postcss autoprefixer && npx tailwindcss init -p`
- [ ] Sepakati **kontrak data** dengan Orang 1 (PRD §B.1 + lihat §6 di dokumen ini)
- [ ] **Bikin dummy file** di `backend/data/` agar dev bisa lanjut tanpa menunggu Orang 1:
  - `deforestation.geojson` (10–20 polygon karangan di koordinat Papua)
  - `statistics.json` (angka karangan dengan schema sesuai PRD §B.1.2)
  - `legend.json` (copy persis dari PRD §B.1.3)
  - `landcover_2024.png` (placeholder PNG hijau polos + bounds)
  - `landcover_2021.png` (placeholder PNG)
- [ ] Test backend hello-world berjalan: `uvicorn app.main:app --reload`
- [ ] Test frontend dev server: `npm run dev`

---

## 6. Kontrak API — REST Endpoints

| Method | Endpoint | Query Params | Response |
| --- | --- | --- | --- |
| GET | `/api/health` | — | `{ "status": "ok" }` |
| GET | `/api/legend` | — | Array 6 objek `{ id, name, color }` |
| GET | `/api/landcover/{year}` | `year ∈ {2021, 2024}` | `{ image_url, bounds: [[s,w],[n,e]] }` |
| GET | `/api/deforestation` | `transition_type?`, `province?`, `min_area_ha?` | GeoJSON FeatureCollection |
| GET | `/api/statistics` | — | Object lengkap (PRD §B.1.2) |
| GET | `/api/statistics/per-province` | — | Array of `{ province, deforestation_ha }` |
| GET | `/api/statistics/per-transition` | — | Object `{ transition_type: ha }` |
| GET | `/api/download/{file}` | `file ∈ {geojson, csv, metrics}` | Binary download |

### Detail: `GET /api/landcover/2024`

Response:
```json
{
  "image_url": "/static/landcover_2024.png",
  "bounds": [[-9.5, 130.0], [0.5, 141.2]]
}
```

### Detail: `GET /api/deforestation?transition_type=hutan_ke_sawit`

Response: FeatureCollection sesuai PRD §B.1.1 (filtered).

### Detail: `GET /api/statistics`

Response: Object sesuai PRD §B.1.2 (full).

### CORS

Set `CORSMiddleware` di `app/main.py` dengan whitelist:
- `http://localhost:5173` (Vite dev)
- URL production frontend (mis. `https://forestwatch-papua.vercel.app`)

---

## 7. Timeline Eksekusi 4 Minggu

### Minggu 1 — Scaffold + Mock Data + Kontrak

**Tujuan minggu:** Backend hello-world + frontend dengan peta Papua kosong. Kontrak data sudah disepakati dengan Orang 1, dummy file sudah ada.

**Backend tasks:**
1. Init FastAPI app di `app/main.py` dengan endpoint `/api/health`.
2. Tambahkan CORSMiddleware (allow `http://localhost:5173`).
3. Buat semua router file kosong (return mock JSON statis).
4. Setup `core/config.py` dengan `DATA_DIR = "data/"`.
5. Buat `requirements.txt`. Test `uvicorn app.main:app --reload`.

**Frontend tasks:**
1. Scaffold Vite + React di `frontend/`.
2. Install dependencies (lihat §3).
3. Setup Tailwind: edit `tailwind.config.js` content path + `index.css` directives.
4. Buat `MapView.jsx` dengan `MapContainer` dari react-leaflet, basemap OSM, center `[-4.5, 138]`, zoom 6.
5. Buat layout 2-kolom di `App.jsx` (peta kiri 70%, panel kanan 30% dengan placeholder).
6. Setup `api/client.js` dengan `axios` baseURL dari `VITE_API_URL`.
7. Test fetch `/api/health` dan tampilkan status di pojok bawah.

**Koordinasi:**
- Hari 1: rapat dengan Orang 1, finalisasi skema 7 file. Kirim mockup schema ke Orang 1 untuk konfirmasi.
- Hari 3: terima dummy file dari Orang 1 (atau bikin sendiri kalau Orang 1 belum siap), letakkan di `backend/data/`.

**DoD:**
- Backend respond `/api/health` di `http://localhost:8000`
- Frontend render peta Papua di `http://localhost:5173`
- Frontend berhasil fetch `/api/health` (CORS lolos)
- Dummy 7 file ada di `backend/data/`

**Checklist Minggu 1:**
- [ ] Setup Day 0 selesai
- [ ] FastAPI hello-world berjalan
- [ ] CORS middleware ter-set
- [ ] Semua router kosong dibuat (return mock)
- [ ] Vite + React + Tailwind ter-scaffold
- [ ] react-leaflet basemap Papua render
- [ ] Layout 2-kolom ada
- [ ] axios client connected to backend
- [ ] Skema 7 file disepakati dengan Orang 1
- [ ] Dummy 7 file ada di `backend/data/`
- [ ] Commit awal ke GitHub

---

### Minggu 2 — Fitur Penuh dengan Dummy Data

**Tujuan minggu:** WebGIS **100% fungsional** memakai dummy data. UI tidak akan berubah lagi saat data asli datang di Minggu 3 — yang berubah hanya isi file.

**Backend tasks:**
1. Implementasi `routers/landcover.py`: baca bounds.json + return image URL ke `/static/landcover_{year}.png`. Mount `/static` di main.py.
2. Implementasi `routers/deforestation.py` dengan `geopandas.read_file()` + filter query params (`transition_type`, `province`, `min_area_ha`).
3. Implementasi `routers/statistics.py` + sub-endpoint per-province dan per-transition (baca `statistics.json`).
4. Implementasi `routers/legend.py` (baca `legend.json`).
5. Implementasi `routers/download.py` dengan `FileResponse`.
6. Tulis pydantic schemas di `app/schemas/` untuk response validation.
7. Test semua endpoint dengan curl atau FastAPI `/docs`.

**Frontend tasks:**
1. `LandcoverOverlay.jsx`: gunakan `ImageOverlay` dari react-leaflet, fetch bounds dari `/api/landcover/{year}`.
2. `DeforestationLayer.jsx`: fetch `/api/deforestation`, render dengan `GeoJSON` component, styling per `transition_type` (4 warna).
3. `TimeSlider.jsx`: toggle antara 2021 dan 2024, swap `LandcoverOverlay`.
4. `TransitionFilter.jsx`: 4 checkbox, state global (Context atau Zustand), trigger refetch `/api/deforestation?transition_type=...`.
5. `OpacitySlider.jsx`: slider 0–1 untuk landcover overlay (default 0.7).
6. `Legend.jsx`: tampilkan 6 kotak warna dari `/api/legend`, position bottom-right.
7. `StatsPanel.jsx`: card total deforestasi, n_hotspots dari `/api/statistics`.
8. `TransitionPieChart.jsx`: Chart.js pie dari `/api/statistics/per-transition`. Klik segmen → set filter.
9. `ProvinceBarChart.jsx`: Chart.js bar dari `/api/statistics/per-province`. Klik bar → zoom map.
10. `PopupContent.jsx`: popup format dari PRD §B.2.5 (ID, transisi, luas, provinsi, status).

**DoD:**
- Semua 8 endpoint respond dengan dummy data
- UI menampilkan semua 14 fitur PRD §B.2.3 (kecuali fitur yang butuh data asli seperti `kawasan_status`)
- Filter transisi berfungsi (peta update saat checkbox di-toggle)
- Slider waktu berfungsi (overlay 2021 ↔ 2024 swap)
- Marker punya popup
- Pie & bar chart render

**Checklist Minggu 2:**
- [ ] `/api/landcover/{year}` berfungsi + serve PNG via /static
- [ ] `/api/deforestation` dengan 3 query filter
- [ ] `/api/statistics` + 2 sub-endpoint
- [ ] `/api/legend` dan `/api/download/{file}`
- [ ] Pydantic schemas di-define
- [ ] Test endpoint via Swagger `/docs`
- [ ] LandcoverOverlay render PNG ke peta
- [ ] DeforestationLayer dengan styling per-transisi
- [ ] TimeSlider berfungsi
- [ ] TransitionFilter (4 checkbox) berfungsi
- [ ] OpacitySlider berfungsi
- [ ] Legend bottom-right
- [ ] StatsPanel render card total
- [ ] TransitionPieChart render + klik segmen → filter
- [ ] ProvinceBarChart render + klik bar → zoom
- [ ] Popup pada marker
- [ ] Marker cluster (jika dummy > 200 titik)
- [ ] Commit ke GitHub

---

### Minggu 3 — Integrasi Data Asli + Studi Kasus Merauke + Polish

**Tujuan minggu:** Ganti dummy dengan **data asli** dari Orang 1. Tambahkan fitur Merauke, modal Tentang, dan tombol unduh data.

**Backend tasks:**
1. Hari 1: terima `ForestWatch_Outputs/` dari Orang 1. Copy 7 file ke `backend/data/` (atau symlink).
2. Verifikasi schema match dengan pydantic — fix mismatch di backend bila perlu.
3. Bila GeoJSON > 5 MB → tambahkan `shapely.simplify(tolerance=0.0001)` di backend.
4. Bila marker > 1000 → pertimbangkan pagination atau spatial index.
5. Re-test semua endpoint dengan data asli.

**Frontend tasks:**
1. `MeraukeButton.jsx`: tombol "Pusat ke Merauke" → `map.flyTo([-8.5, 140.4], 9)`. Highlight semua poligon Papua Selatan dengan stroke tebal.
2. Tambah filter `province` di state global (sebenarnya server-side filter).
3. `AboutModal.jsx`: modal dengan penjelasan metodologi singkat (ambil dari PRD §0.1 + §A.4 ringkasan).
4. `MetricsTable.jsx`: tabel di panel kanan dengan mIoU, OA, IoU per kelas dari `/api/statistics`.
5. `DownloadButton.jsx`: dropdown menu — unduh `deforestation.geojson`, `statistics.csv` (backend convert dari JSON).
6. Sub-fitur: studi kasus Merauke = modal/sidebar yang otomatis muncul saat tombol diklik, dengan ringkasan luas hutan-ke-pertanian-lain di Papua Selatan.
7. UI polish: loading state, error handling, empty state untuk filter tanpa hasil.

**DoD:**
- Data asli dari Orang 1 tampil di WebGIS
- Tombol Merauke berfungsi
- Modal Tentang berisi konten
- MetricsTable menampilkan angka asli dari model
- Tombol unduh data berfungsi (download file)
- 14 fitur PRD §B.2.3 semua tercapai
- Angka di UI konsisten dengan `statistics.json`

**Checklist Minggu 3:**
- [ ] Data asli dari Orang 1 ada di `backend/data/`
- [ ] Schema validation pass
- [ ] Backend handle GeoJSON besar (simplify bila perlu)
- [ ] MeraukeButton berfungsi (flyTo + highlight)
- [ ] Filter province berfungsi
- [ ] AboutModal dengan konten metodologi
- [ ] MetricsTable render mIoU, OA, IoU
- [ ] DownloadButton berfungsi (geojson + csv)
- [ ] Loading state ada pada fetch
- [ ] Error state ada pada gagal fetch
- [ ] Empty state untuk filter tanpa hasil
- [ ] Cross-check angka WebGIS = `statistics.json`
- [ ] Sync dengan Orang 3 untuk angka esai
- [ ] Tested di Chrome & Safari (manual)
- [ ] Commit ke GitHub

---

### Minggu 4 — Deploy + Video Demo + Final Polish

**Tujuan minggu:** WebGIS **deployed dengan URL publik**, video demo 2 menit selesai, siap presentasi.

**Backend deployment:**
1. Buat `Dockerfile` di `backend/` untuk Railway / HuggingFace Spaces.
2. Push backend repo terpisah / monorepo ke GitHub.
3. Deploy ke Railway: connect repo → set env vars (`CORS_ORIGINS`) → deploy.
4. Verifikasi backend reachable via URL Railway.
5. Update CORS dengan URL frontend production.

**Frontend deployment:**
1. Build production: `npm run build` → output di `dist/`.
2. Test build lokal: `npm run preview`.
3. Set `VITE_API_URL` ke URL backend Railway.
4. Deploy ke Vercel: connect repo → set env var `VITE_API_URL` → deploy.
5. Verifikasi frontend reachable, fetch ke backend berhasil (CORS lolos).

**Video demo:**
1. Skrip video 2 menit:
   - 0:00–0:15 — Intro: judul, tim, URL
   - 0:15–0:45 — Peta Papua + overlay landcover + slider waktu
   - 0:45–1:15 — Filter transisi + popup marker + panel statistik
   - 1:15–1:45 — Studi kasus Merauke (tombol)
   - 1:45–2:00 — MetricsTable + URL & outro
2. Rekam dengan OBS Studio atau Loom. Tambah subtitle bila perlu.
3. Upload ke YouTube unlisted atau Drive, share URL ke Orang 3.

**Final polish:**
1. README.md di repo: deskripsi, URL live, screenshot, cara run local.
2. Test responsiveness: minimal desktop (1280×720) dan laptop (1366×768). Mobile boleh diabaikan.
3. Performance check: lighthouse score, lazy load chart bila lambat.

**DoD:**
- URL backend live (Railway)
- URL frontend live (Vercel) dan dapat akses backend
- Video demo 2 menit
- README lengkap
- Tested di Chrome & Safari di production

**Checklist Minggu 4:**
- [ ] Dockerfile backend dibuat
- [ ] Backend deployed ke Railway/Render
- [ ] CORS production di-update
- [ ] Frontend build production sukses
- [ ] Frontend deployed ke Vercel/Netlify
- [ ] End-to-end test di production URL
- [ ] Video demo 2 menit selesai
- [ ] Video di-upload, URL tersedia
- [ ] README.md frontend
- [ ] README.md backend
- [ ] URL live dikirim ke Orang 3 untuk dimasukkan ke esai
- [ ] Lighthouse score check
- [ ] Backup repo final di GitHub
- [ ] Stand by untuk demo Q&A

---

## 8. Strategi Koordinasi dengan Orang 1

| Waktu | Aksi |
| --- | --- |
| Hari 1 Minggu 1 | Rapat dengan Orang 1, finalisasi skema 7 file (PRD §B.1) |
| Hari 3 Minggu 1 | Terima dummy 7 file dari Orang 1, atau bikin sendiri kalau Orang 1 belum siap |
| Hari 1 Minggu 2 | Cek ulang dummy data — schema masih cocok |
| Hari 7 Minggu 2 | Check-in: apakah Orang 1 sudah punya 1 ubin asli untuk integration awal |
| Hari 1 Minggu 3 | Orang 1 hands-off semua 7 file. Ganti dummy. |
| Hari 5 Minggu 3 | Cross-check angka di WebGIS = `statistics.json` = bahan esai Orang 3 |
| Minggu 4 | Stand by kalau Orang 1 update file (mis. metrik berubah setelah re-train) |

**Aturan emas:** WebGIS tidak menunggu model. Jika data asli telat, demo tetap jalan dengan dummy (asal schema sama).

---

## 9. Deployment Plan

### Pilihan Platform (Gratis)

| Komponen | Platform | Alasan |
| --- | --- | --- |
| Backend FastAPI | **Railway** (free $5 credit/bulan) atau **Render** free tier | Support Python, auto-deploy dari GitHub |
| Backend (alternatif) | **HuggingFace Spaces** (Docker SDK) | Gratis, persistent storage |
| Frontend React | **Vercel** | Optimized untuk Vite/React, CDN global, deploy 1-klik |
| Frontend (alternatif) | **Netlify** atau **GitHub Pages** (build dengan base path) | Sama-sama gratis |

### Environment Variables

**Backend (`.env`):**
```
DATA_DIR=./data
CORS_ORIGINS=https://forestwatch-papua.vercel.app,http://localhost:5173
```

**Frontend (`.env.production`):**
```
VITE_API_URL=https://forestwatch-papua-api.up.railway.app
```

### Dockerfile Backend (template)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 10. Risiko & Mitigasi

| Risiko | Mitigasi |
| --- | --- |
| Backend Railway/Render mati saat demo | Frontend cache hasil API di localStorage saat init. Punya fallback dummy data di repo frontend (build-time). |
| Data asli dari Orang 1 telat | Develop full UI dengan dummy di Minggu 2 — tidak menunggu. |
| GeoJSON terlalu besar (>5MB → lambat fetch) | `shapely.simplify(tolerance=0.0001)` di backend, atau paginate via query param. |
| Marker > 1000 lag di browser | Pakai `leaflet.markercluster` (sudah di-plan); turunkan threshold render |
| CORS error di production | Set `CORSMiddleware` whitelist domain frontend; verifikasi via DevTools |
| PNG landcover Papua terlalu besar (>10 MB) | Convert ke WebP, atau pakai tile pyramid (`gdal2tiles.py`) |
| Free tier Railway/Vercel sleep saat idle | Ping endpoint `/health` via cron eksternal (uptimerobot.com gratis) sebelum demo |
| Schema dari Orang 1 berubah mendadak | Punya pydantic validation di backend yang lempar error jelas; isolasi perubahan ke 1 file |
| Vite build error karena leaflet CSS | Import `leaflet/dist/leaflet.css` di `main.jsx`; fix icon path bug Leaflet standar |
| Browser Safari tidak render layer | Test sejak Minggu 2 di Safari, jangan tunggu Minggu 4 |

---

## 11. Referensi Cepat ke PRD

| Tugas | Lihat |
| --- | --- |
| Kontrak 7 file (skema lengkap) | PRD §B.1 |
| Skema GeoJSON | PRD §B.1.1 |
| Skema statistics.json | PRD §B.1.2 |
| Skema legend.json + warna transisi | PRD §B.1.3 |
| Stack rekomendasi (Leaflet) | PRD §B.2.1 |
| Struktur file (versi vanilla) | PRD §B.2.2 |
| **Daftar 14 fitur wajib** | PRD §B.2.3 |
| Mockup layout 2-kolom | PRD §B.2.4 |
| Code skeleton Leaflet | PRD §B.2.5 (referensi adaptasi ke React) |
| Milestone vanilla (untuk komparasi) | PRD §B.2.6 |

---

## 12. Checklist Akhir (Sebelum Submit — dari PRD Lampiran 2)

**Dari PRD:**
- [ ] WebGIS dapat diakses via URL publik
- [ ] Semua 14 fitur wajib (PRD §B.2.3) berfungsi
- [ ] Filter 4 jenis transisi berfungsi
- [ ] Studi kasus Merauke punya tombol/zoom preset
- [ ] Video demo 2 menit tersedia
- [ ] Tested di Chrome dan Safari

**Tambahan untuk full-stack:**
- [ ] Backend health endpoint OK di production
- [ ] CORS test pass (frontend fetch backend tanpa error)
- [ ] Tidak ada hard-coded localhost URL di production build
- [ ] Backend `requirements.txt` lengkap dan up-to-date
- [ ] Frontend `package.json` lengkap dan up-to-date
- [ ] README.md frontend & backend lengkap
- [ ] Lighthouse score frontend ≥ 80 (performance & accessibility)
- [ ] Data asli (bukan dummy) tampil di production
- [ ] Angka di WebGIS = angka di esai Orang 3 = angka di `statistics.json`
- [ ] URL live dikirim ke Orang 3 (sertakan di esai)
- [ ] Backup full repo di GitHub
- [ ] Demo dry-run dengan tim sebelum submit

---

## 13. Daftar 14 Fitur Wajib (Quick Reference dari PRD §B.2.3)

| # | Fitur | Status |
| --- | --- | --- |
| 1 | Peta dasar Papua (OSM/Esri Satellite) | [ ] |
| 2 | Overlay tutupan lahan (PNG + bounds) dengan opacity slider | [ ] |
| 3 | Slider waktu T1 ↔ T2 (2021 ↔ 2024) | [ ] |
| 4 | Marker deforestasi (GeoJSON dengan styling per transisi) | [ ] |
| 5 | Popup detail (ID, transisi, luas, provinsi, status) | [ ] |
| 6 | Filter 4 jenis transisi (checkbox) | [ ] |
| 7 | Panel statistik (total ha, hotspot, breakdown) | [ ] |
| 8 | Grafik per transisi (pie chart, klik → filter) | [ ] |
| 9 | Grafik per provinsi (bar chart, klik → zoom) | [ ] |
| 10 | Tabel metrik model (mIoU, OA, IoU per kelas) | [ ] |
| 11 | Legenda (6 kotak warna + nama) | [ ] |
| 12 | Tombol "Pusat ke Merauke" | [ ] |
| 13 | Studi kasus Merauke (modal/highlight Papua Selatan) | [ ] |
| 14 | Tombol unduh data (GeoJSON & CSV) + Modal Tentang | [ ] |

---

*Master Plan WebGIS · ForestWatch Papua · SEC SATRIA DATA 2026 · v1.0*
*Untuk perubahan substansial pada skema data atau arsitektur, koordinasi dengan Orang 1 dan update [PRD v2.0](../PRD_ForestWatch_Papua_v2.md) + dokumen ini secara bersamaan.*
