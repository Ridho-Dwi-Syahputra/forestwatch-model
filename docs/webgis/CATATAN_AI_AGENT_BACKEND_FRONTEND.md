# Catatan untuk AI Agent — Backend & Frontend WebGIS ForestWatch Papua

> **Untuk siapa:** AI agent (atau developer) yang membangun **backend + frontend** WebGIS.
> **Tujuan:** menyambungkan model deteksi deforestasi (sudah jadi, hasil training tim model)
> ke aplikasi web yang bisa diminta user: *"tampilkan deforestasi Papua antara tahun A dan tahun B"*.
> **Stack yang sudah diputuskan:** Backend **FastAPI (Python)**, Frontend **React + Leaflet**.
> **Bahasa proyek:** Indonesia.

---

## 0. TL;DR — Alur Lengkap End-to-End

```
[User di Frontend]
   pilih: Area (AOI) + Tahun T1 (mis. 2021) + Tahun T2 (mis. 2026)
        │
        ▼
[Frontend → Backend]  POST /api/analyze  { aoi, year_t1, year_t2 }
        │
        ▼
[Backend] langkah-langkah:
   1. Minta citra Sentinel-2 ke GEE untuk (AOI, T1)  → komposit 6-band  ──┐
   2. Minta citra Sentinel-2 ke GEE untuk (AOI, T2)  → komposit 6-band  ──┤  (preprocessing IDENTIK training)
   3. Jalankan model.onnx pada citra T1 → peta 7-kelas T1 (raster)        │
   4. Jalankan model.onnx pada citra T2 → peta 7-kelas T2 (raster)        │
   5. Bandingkan T1 vs T2 per-piksel (change detection)                   │
      → poligon area "Hutan → {Tambang/Sawit/Lahan Terbuka/Pertanian}"    │
   6. Hitung statistik (total ha, per-transisi, per-provinsi)             │
        │
        ▼
[Backend → Frontend]  JSON:
   - deforestation.geojson  (poligon perubahan + koordinat)
   - statistics.json        (ringkasan angka)
   - landcover_t1 / t2      (peta tutupan lahan tiap tahun, opsional)
        │
        ▼
[Frontend] render di peta Leaflet:
   - layer poligon deforestasi (warna per jenis transisi)
   - panel statistik (total hektar, grafik per transisi/provinsi)
```

**Inti yang harus dipahami:**
- **Model itu "pembaca peta", bukan "pendeteksi perubahan".** Ia hanya melihat 1 citra dan
  menjawab *"tiap titik ini tutupan lahannya apa (dari 7 kelas)"*. Tidak tahu apa-apa soal waktu.
- **Deteksi perubahan = membandingkan dua peta hasil model (T1 vs T2).** Ini logika murni
  di backend, kodenya **sudah ada** (`change_detection.py`), tinggal dipanggil.
- **Model dijalankan dengan citra tahun berapa pun** — tahun cuma parameter tanggal saat
  meminta citra ke GEE. Tidak perlu training ulang untuk tahun baru.

---

## 1. ⚠️ ATURAN PALING KRITIS: Preprocessing Citra HARUS Identik dengan Training

Model dilatih pada citra yang diolah dengan cara spesifik. **Jika backend memberi citra dengan
preprocessing berbeda, prediksi model akan kacau** (walaupun tidak error — hasilnya salah diam-diam).

Backend WAJIB memproduksi citra dengan spesifikasi PERSIS berikut (sumber:
`model/src/forestwatch/gee/composite.py`, fungsi `s2_composite`):

| Parameter | Nilai WAJIB | Keterangan |
|---|---|---|
| Koleksi GEE | `COPERNICUS/S2_SR_HARMONIZED` | Sentinel-2 Surface Reflectance Harmonized |
| Band (urutan!) | `["B2","B3","B4","B8","B11","B12"]` | **6 band, urutan TEPAT seperti ini** |
| Rentang tanggal | `{tahun}-01-01` s/d `{tahun}-12-31` | komposit 1 tahun penuh |
| Filter awan | `CLOUDY_PIXEL_PERCENTAGE < 40` | buang scene berawan tebal |
| Cloud masking | mask SCL kelas `[3, 8, 9, 10, 11]` | bayangan, awan medium/tinggi, cirrus, salju |
| Reduksi temporal | `.median()` | komposit median antar-scene |
| Normalisasi | `.divide(10000)` | reflektansi ke skala [0, 1] |
| Resolusi (scale) | `10` meter/piksel | |
| CRS | `EPSG:4326` (WGS84) | |

**Cara termudah & paling aman:** backend **memakai ulang kode Python yang sama**
(`forestwatch.gee.composite.s2_composite`) — jangan tulis ulang preprocessing dari nol di
JavaScript/lainnya. Paket `forestwatch` bisa di-`pip install -e` di environment backend.

```python
# Contoh di backend (FastAPI) — pakai kode model yang sudah ada:
import ee
from forestwatch.gee.composite import s2_composite

ee.Initialize(...)  # auth service account (lihat §3)
aoi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
img_t1 = s2_composite(2021, aoi)   # ee.Image 6-band, reflektansi [0,1]
img_t2 = s2_composite(2026, aoi)
```

---

## 2. Spesifikasi Model (`model.onnx`)

Tim model menghasilkan `model.onnx` (lihat `model/src/forestwatch/model/architecture.py:export_to_onnx`).

| Aspek | Nilai |
|---|---|
| Format | ONNX (opset 13), jalankan via `onnxruntime` |
| Arsitektur | U-Net (encoder ResNet50), single-frame semantic segmentation |
| **Input** | nama `"image"`, shape `(batch, 6, 256, 256)`, dtype `float32`, reflektansi [0,1] |
| **Output** | nama `"mask"`, shape `(batch, 7, 256, 256)` = **logits per kelas** |
| Batch | dinamis (dimensi 0 boleh berapa saja) |
| Kelas (argmax channel) | 0=Perairan, 1=Hutan, 2=Lahan Terbuka, 3=Sawit, 4=Pertanian Lain, 5=Tambang, 6=Permukiman |

**Catatan penting input model:**
- Model menerima **patch 256×256**, bukan citra utuh. Citra AOI besar harus **dipotong jadi
  patch 256×256** (sliding window), diprediksi per-patch, lalu **dijahit kembali** jadi peta utuh.
- Logika sliding-window + overlap-blending + penjahitan **sudah ada** di
  `model/src/forestwatch/inference/tile_inference.py` (`infer_tile`, `infer_tiles_folder`).
  Backend sangat disarankan **memakai ulang** modul ini (versi PyTorch) ATAU mereplikasi logikanya
  bila pakai onnxruntime murni. Output akhir: 1 raster mask uint8 (nilai 0–6) per AOI per tahun.

```python
# Opsi A (paling simpel): pakai modul inference yang sudah ada (butuh torch + checkpoint .pt)
from forestwatch.inference.tile_inference import infer_tile
infer_tile("aoi_t1.tif", "mask_t1.tif", model)   # model = nn.Module hasil build_unet + load_state_dict

# Opsi B (onnxruntime murni, tanpa torch): replikasi sliding-window di tile_inference.py,
#         tiap patch 256x256 → ort_session.run; argmax pada axis kelas → mask uint8.
```

---

## 3. Backend — Detail Implementasi (FastAPI)

### 3.1 Koneksi ke Google Earth Engine (GEE)

- GEE Python API butuh autentikasi. Untuk **server (backend, tanpa interaksi user)**, gunakan
  **Service Account**, BUKAN `ee.Authenticate()` interaktif.
  - Buat service account di Google Cloud, beri akses ke Earth Engine, unduh key JSON.
  - Project GEE yang dipakai tim model: **`forestwatch-papua-unand`**.
  ```python
  import ee
  credentials = ee.ServiceAccountCredentials(EMAIL, "key.json")
  ee.Initialize(credentials, project="forestwatch-papua-unand")
  ```
- **Cara menarik data piksel dari GEE ke backend (tanpa download file manual):**
  - Untuk AOI kecil/medium: `ee.Image.getDownloadURL()` atau `geemap.ee_to_numpy()` /
    `ee.data.computePixels()` → langsung jadi numpy array di memori. **Tidak perlu** simpan
    GeoTIFF besar lebih dulu (beda dengan pipeline training yang export ke Drive karena
    skalanya SE-PAPUA — untuk WebGIS skalanya per-AOI, jauh lebih kecil).
  - Untuk AOI besar: tetap mungkin perlu strategi tiling + `Export` async; mulai dari AOI
    kecil dulu untuk MVP.

### 3.2 Pipeline yang dipanggil backend (kode sudah tersedia di paket `forestwatch`)

| Langkah | Modul/Fungsi yang dipakai ulang | Output |
|---|---|---|
| 1. Komposit S2 per tahun | `gee.composite.s2_composite(year, aoi)` | `ee.Image` 6-band |
| 2. (jika perlu) Tarik ke raster | `geemap.ee_to_numpy` / `ee.data.computePixels` | numpy `(H,W,6)` |
| 3. Inferensi model T1 & T2 | `inference.tile_inference.infer_tile` | mask uint8 `(H,W)` per tahun |
| 4. **Deteksi perubahan** | `inference.change_detection.detect_transitions_from_arrays(mask_t1, mask_t2, transform, period_from, period_to)` | list feature GeoJSON |
| 5. Statistik | `outputs.statistics.summarize_geojson_transitions(fc)` | dict ringkasan |
| 6. Legend (statis) | `outputs.legend.build_legend_json()` | list legend |

**Detail change detection** (`change_detection.py`):
- Hanya mendeteksi transisi **dari Hutan (kelas 1)** ke salah satu dari:
  `2=Lahan Terbuka, 3=Sawit, 4=Pertanian Lain, 5=Tambang, 6=Permukiman`.
- Aturan: piksel yang `T1 == Hutan` DAN `T2 == kelas_target` → ditandai berubah.
- Poligonisasi: `rasterio.features.shapes` → `shapely` → GeoJSON Polygon (EPSG:4326).
- Filter noise: poligon < `min_area_ha` (default 0.5 ha) dibuang. **Naikkan** nilai ini
  (mis. 1–5 ha) bila terlalu banyak poligon kecil/noise.

### 3.3 Endpoint API yang disarankan

```
POST /api/analyze
  body: { "aoi": [lon_min, lat_min, lon_max, lat_max],
          "year_t1": 2021, "year_t2": 2026,
          "min_area_ha": 1.0 }   # opsional
  resp: { "deforestation": <GeoJSON FeatureCollection>,
          "statistics": <statistics.json>,
          "bounds_t1": {...}, "bounds_t2": {...} }   # bounds untuk overlay PNG (opsional)

GET  /api/legend
  resp: <legend.json>   # statis, bisa di-cache di frontend

GET  /api/landcover?aoi=...&year=2021   # opsional: peta tutupan lahan 1 tahun (PNG/raster)
```

> **Catatan performa / "real-time":** memproses GEE + inferensi butuh **detik s/d menit**
> (tergantung luas AOI & ada/tidaknya GPU), JADI **bukan real-time instan**. Rekomendasi:
> - Untuk AOI besar, jalankan sebagai **job async** (mis. Celery/RQ atau BackgroundTasks),
>   frontend polling status.
> - **Cache hasil** per `(aoi, year_t1, year_t2)` agar permintaan sama tidak dihitung ulang.
> - Untuk demo/lomba, sediakan **beberapa AOI preset** (mis. Merauke, area tambang) yang
>   hasilnya sudah dipra-hitung → tampil instan.

---

## 4. Kontrak Output (Skema JSON yang Dikirim ke Frontend)

Skema ini **sama** dengan "7 file kontrak" yang dihasilkan pipeline model
(`model/src/forestwatch/outputs/`), jadi backend & frontend bisa pakai bentuk yang sama.

### 4.1 `deforestation.geojson` — poligon area deforestasi (output UTAMA)
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Polygon", "coordinates": [[[lon, lat], ...]] },
      "properties": {
        "id": "DF-00001",
        "transition_type": "hutan_ke_tambang",
        "area_ha": 12.4,
        "period_from": 2021,
        "period_to": 2026,
        "province": "Papua Tengah"
      }
    }
  ]
}
```
`transition_type` salah satu dari: `hutan_ke_lahan_terbuka`, `hutan_ke_sawit`,
`hutan_ke_pertanian_lain`, `hutan_ke_tambang`, `hutan_ke_permukiman`.

### 4.2 `statistics.json` — ringkasan untuk panel/grafik
```json
{
  "total_deforestation_ha": 1234.5,
  "n_hotspots": 87,
  "per_transition_ha": {
    "hutan_ke_lahan_terbuka": 200.1,
    "hutan_ke_sawit": 540.3,
    "hutan_ke_pertanian_lain": 88.0,
    "hutan_ke_tambang": 350.6,
    "hutan_ke_permukiman": 55.5
  },
  "per_province": [
    { "province": "Papua", "deforestation_ha": 300.2 },
    { "province": "Papua Tengah", "deforestation_ha": 500.1 }
  ]
}
```

### 4.3 `legend.json` — pemetaan warna 7 kelas (statis)
```json
[
  { "id": 0, "name": "Perairan",       "color": "#2A6FDB" },
  { "id": 1, "name": "Hutan",          "color": "#0B3D0B" },
  { "id": 2, "name": "Lahan Terbuka",  "color": "#E03B24" },
  { "id": 3, "name": "Sawit",          "color": "#F97316" },
  { "id": 4, "name": "Pertanian Lain", "color": "#E9C46A" },
  { "id": 5, "name": "Tambang",        "color": "#8E24AA" },
  { "id": 6, "name": "Permukiman",     "color": "#757575" }
]
```

### 4.4 Warna per jenis transisi (untuk layer deforestasi di peta)
```
hutan_ke_lahan_terbuka : #7F1D1D
hutan_ke_sawit         : #F97316
hutan_ke_pertanian_lain: #EAB308
hutan_ke_tambang       : #8E24AA
hutan_ke_permukiman    : #757575
```

---

## 5. Frontend — Detail Implementasi (React + Leaflet)

### 5.1 Komponen input (yang kamu sebut di WA — benar)
- **Pemilih Area (AOI):** dropdown preset (Papua, Merauke, dll) ATAU gambar kotak/poligon di peta.
- **Pemilih Tahun T1 & T2:** dua input tahun (mis. 2021 dan 2026). Kirim ke `POST /api/analyze`.
- Tombol **"Analisis Deforestasi"** → panggil backend → tampilkan loading (bisa lama, lihat §3.3).

### 5.2 Layer peta
- **Basemap:** OpenStreetMap / satelit.
- **Layer deforestasi:** render `deforestation.geojson`; warnai poligon by `transition_type`
  (§4.4); klik poligon → popup berisi `area_ha`, `transition_type`, `period_from→to`, `province`.
- **(Opsional) Layer tutupan lahan T1/T2:** overlay PNG 7-warna (pakai `legend.json` untuk legenda).
  Sediakan toggle untuk lihat peta 2021 vs 2026.
- **Center peta default:** Papua `lat -4.5, lon 138.0`. Preset Merauke `lat -8.5, lon 140.4`.

### 5.3 Panel statistik
- Kartu ringkasan: `total_deforestation_ha`, `n_hotspots`.
- Grafik batang: `per_transition_ha` (deforestasi per jenis).
- Grafik/tabel: `per_province` (deforestasi per provinsi).
- Daftar hotspot terurut luas (dari `features`), klik → zoom ke poligon di peta.

---

## 6. Daftar Hal yang KURANG / Perlu Ditegaskan dari Penjelasan Awal (WhatsApp)

Penjelasan di WA sudah **benar secara konsep**. Tambahan yang WAJIB diperhatikan tim WebGIS:

1. **[KRITIS] Preprocessing identik (§1).** "Minta data via geemap + cloud masking" saja belum cukup —
   harus 6 band spesifik + urutan tepat + median + divide 10000. Cara aman: pakai ulang
   `s2_composite` dari paket model, jangan tulis ulang.
2. **"Catatan" model = raster per-piksel, bukan angka agregat.** Backend yang menghitung
   "hutan 2021 segini, 2025 segini" dengan membandingkan dua raster (sudah ada di `change_detection.py`).
3. **Model butuh input patch 256×256**, jadi citra AOI harus dipotong-jahit (sudah ada di
   `tile_inference.py`).
4. **Hanya transisi DARI HUTAN yang dihitung sebagai deforestasi** (Hutan→Tambang/Sawit/dst).
   Perubahan lain (mis. Sawit→Permukiman) TIDAK masuk output deforestasi.
5. **Bukan real-time instan** — butuh detik s/d menit. Pakai job async + cache + AOI preset (§3.3).
6. **Auth GEE pakai Service Account** untuk server, bukan login interaktif (§3.1).
7. **Sumber kebenaran kelas/warna/transisi** ada di `model/src/forestwatch/constants.py` —
   jangan hardcode ulang nilai berbeda; impor atau salin dari sana agar konsisten.

---

## 7. Referensi Kode (paket `forestwatch`, di `model/src/`)

| Kebutuhan | File |
|---|---|
| Preprocessing citra (komposit S2) | `forestwatch/gee/composite.py` → `s2_composite()` |
| Konstanta kelas, warna, transisi, band, asset GEE | `forestwatch/constants.py` |
| Inferensi tile (sliding-window + jahit) | `forestwatch/inference/tile_inference.py` |
| Deteksi perubahan → GeoJSON | `forestwatch/inference/change_detection.py` |
| Statistik & legend | `forestwatch/outputs/statistics.py`, `outputs/legend.py` |
| Export ONNX (spesifikasi I/O model) | `forestwatch/model/architecture.py` → `export_to_onnx()` |

> Sebelum mulai, jalankan `pip install -e ".[ml,gis]"` di folder `model/` agar paket
> `forestwatch` (+ rasterio, shapely, onnxruntime/torch) tersedia di environment backend.

---

*Dokumen ini acuan untuk membangun backend + frontend. Bila ada perbedaan antara dokumen ini
dan kode di `model/src/forestwatch/`, **kode yang menang** — perbarui dokumen ini agar sinkron.*
