# Design Brief — WebGIS ForestWatch Papua

> **Untuk:** Pembuatan desain frontend (Claude Design / Figma).
> **Sifat:** WebGIS **sederhana** — peta interaktif + panel statistik. **TIDAK ada** fitur unduh data, ekspor, login, atau editor kompleks.
> **Bahasa UI:** Bahasa Indonesia.
> **Nada:** Netral & ilmiah. Sebut "perubahan/transisi tutupan lahan", **bukan** narasi politik atau proyek tertentu.

---

## 1. Konteks Singkat

ForestWatch Papua memvisualisasikan hasil model deep learning (ResNet50-U-Net) yang mengklasifikasikan tutupan lahan Papua dari citra Sentinel-2 ke **7 kelas**, lalu mendeteksi **perubahan hutan** antara **2021 → 2025** beserta 5 jenis transisinya.

Frontend hanya **menampilkan data jadi** (pre-computed) dari backend. Tidak ada inferensi/komputasi berat di browser → ringan, cepat, tahan saat presentasi.

**Stack target:** React + Vite + react-leaflet (peta) + Chart.js (grafik) + Tailwind CSS (styling).

---

## 2. Sumber Data (dari Backend FastAPI)

Frontend mengambil semua data via REST API. Field-nya sudah pasti:

| Endpoint | Dipakai untuk | Bentuk data |
| --- | --- | --- |
| `GET /api/legend` | Legenda peta | Array 7 objek `{id, name, color}` |
| `GET /api/landcover/2021` & `/2025` | Layer raster peta | `{image_url, bounds: [[s,w],[n,e]], crs}` |
| `GET /api/deforestation` | Marker/poligon perubahan | GeoJSON FeatureCollection |
| `GET /api/statistics` | Panel statistik & grafik | Objek (lihat di bawah) |
| `GET /api/statistics/summary` | Kartu ringkasan | `{period_from, period_to, total_deforestation_ha, n_hotspots, mean_iou, overall_accuracy}` |

**Isi `statistics.json`:**
```json
{
  "period_from": 2021,
  "period_to": 2025,
  "total_deforestation_ha": 198710.8,
  "n_hotspots": 60,
  "per_transition_ha": {
    "hutan_ke_lahan_terbuka": 40369.7,
    "hutan_ke_sawit": 71788.1,
    "hutan_ke_pertanian_lain": 48961.5,
    "hutan_ke_tambang": 37591.5,
    "hutan_ke_permukiman": 8420.3
  },
  "per_province": [ {"province": "Papua Selatan", "deforestation_ha": 108067.8}, ... ],
  "per_class_area_ha": { "Hutan": 117110413.1, "Sawit": 4497119.9, ... },
  "model_metrics": {
    "overall_accuracy": 0.86, "mean_iou": 0.71, "kappa": 0.8276,
    "per_class": [ {"class": "Perairan", "iou": 0.91, "f1": 0.95}, ... ]
  }
}
```

**Properti tiap feature di `deforestation.geojson`:**
`id`, `transition_type`, `area_ha`, `period_from`, `period_to`, `province`, `kawasan_status` (opsional).

---

## 3. Sistem Warna (WAJIB konsisten)

### 7 Kelas Tutupan Lahan
| Kelas | Warna | Hex |
| --- | --- | --- |
| Perairan | biru | `#2A6FDB` |
| Hutan | hijau gelap | `#0B3D0B` |
| Lahan Terbuka | merah | `#E03B24` |
| Sawit | oranye | `#F97316` |
| Pertanian Lain | kuning | `#E9C46A` |
| Tambang | ungu | `#8E24AA` |
| Permukiman | abu-abu | `#757575` |

### 5 Jenis Transisi (marker perubahan)
| Transisi | Label tampilan | Hex |
| --- | --- | --- |
| `hutan_ke_lahan_terbuka` | Hutan → Lahan Terbuka | `#7F1D1D` |
| `hutan_ke_sawit` | Hutan → Sawit | `#F97316` |
| `hutan_ke_pertanian_lain` | Hutan → Pertanian Lain | `#EAB308` |
| `hutan_ke_tambang` | Hutan → Tambang | `#8E24AA` |
| `hutan_ke_permukiman` | Hutan → Permukiman | `#757575` |

**Aksen UI:** hijau hutan (`#0B3D0B`) sebagai warna brand utama, dengan latar terang/putih agar peta menonjol.

---

## 4. Layout Utama (Desktop)

Layout **2 kolom**: peta besar di kiri, panel statistik di kanan.

```
┌───────────────────────────────────────────────────────────────┐
│  HEADER:  🌳 ForestWatch Papua            [Tentang & Metodologi] │
├──────────────────────────────────────────┬────────────────────┤
│                                          │  PANEL STATISTIK    │
│                                          │  ┌────────────────┐ │
│                                          │  │ Total Perubahan│ │
│                                          │  │  198.710 ha    │ │
│            PETA LEAFLET                  │  │  60 titik      │ │
│       (Papua + overlay tutupan lahan)    │  │  2021 → 2025   │ │
│                                          │  └────────────────┘ │
│   [Slider waktu: 2021 ●———— 2025]        │  ┌────────────────┐ │
│                                          │  │ Per Jenis      │ │
│   [Opacity overlay: ───●──── 70%]        │  │ Transisi (pie) │ │
│                                          │  └────────────────┘ │
│   Filter transisi:                       │  ┌────────────────┐ │
│   ☑ Lahan Terbuka ☑ Sawit ☑ Tambang      │  │ Per Provinsi   │ │
│   ☑ Pertanian Lain ☑ Permukiman          │  │ (bar chart)    │ │
│                                          │  └────────────────┘ │
│   [Legenda 7 kelas]                      │  ┌────────────────┐ │
│                                          │  │ Akurasi Model  │ │
│                                          │  │ OA / mIoU /κ   │ │
│                                          │  └────────────────┘ │
└──────────────────────────────────────────┴────────────────────┘
```

---

## 5. Komponen yang Harus Ada

### A. Header (bar atas)
- Logo + judul "ForestWatch Papua" dengan subjudul kecil: "Pemantauan Tutupan Lahan & Perubahan Hutan 2021–2025".
- Tombol kanan: **"Tentang & Metodologi"** (buka modal).
- Tinggi ramping, warna brand hijau gelap atau putih dengan aksen hijau.

### B. Peta Leaflet (komponen utama, kiri ~70% lebar)
- **Basemap:** OpenStreetMap atau Esri Satellite (default OSM).
- **Default view:** center Papua `[-4.5, 138]`, zoom 6.
- **Overlay tutupan lahan:** gambar PNG (dari `/api/landcover/{year}`) dipasang sebagai image overlay sesuai `bounds`. Bisa diatur transparansinya.
- **Layer perubahan:** poligon/marker dari GeoJSON, diwarnai per `transition_type`. Jika titik banyak (>200), gunakan clustering.
- **Popup** saat marker diklik (lihat E).
- Kontrol zoom standar di pojok kiri atas.

### C. Kontrol Peta (overlay di atas/bawah peta)
1. **Slider Waktu** — toggle/slider 2 posisi: **2021 ↔ 2025**. Mengganti overlay tutupan lahan ke tahun terpilih. Beri label tahun yang jelas.
2. **Slider Opacity** — atur transparansi overlay tutupan lahan (0–100%, default 70%).
3. **Filter Transisi** — 5 checkbox (Lahan Terbuka, Sawit, Pertanian Lain, Tambang, Permukiman), masing-masing dengan kotak warnanya. Default semua aktif. Mematikan satu = sembunyikan marker jenis itu.
4. **Legenda** — kotak kecil (pojok kanan-bawah peta) berisi 7 kelas + warna + nama.

### D. Panel Statistik (kanan ~30% lebar, scrollable)
Empat kartu bertumpuk:

1. **Kartu Ringkasan** (paling atas, paling menonjol)
   - Angka besar: **Total perubahan hutan** (mis. `198.710 ha`).
   - Sub-angka: jumlah titik perubahan (`60 titik`), periode (`2021 → 2025`).

2. **Kartu Per Jenis Transisi** — **Pie/Donut chart** dari `per_transition_ha` (5 segmen, warna sesuai tabel transisi). Tooltip tampilkan ha + persen.

3. **Kartu Per Provinsi** — **Bar chart** horizontal dari `per_province` (6 provinsi, urut dari terbesar). Warna seragam hijau.

4. **Kartu Akurasi Model** — tabel ringkas:
   - Overall Accuracy (mis. `86%`), Mean IoU (`0.71`), **Cohen's Kappa** (`0.83`).
   - Tabel kecil IoU per kelas (6 baris) — opsional, bisa di-collapse.

### E. Popup Detail (saat marker diklik)
Kartu kecil berisi:
```
ID: DF-00001
Jenis: Hutan → Sawit
Luas: 12,4 ha
Periode: 2021 → 2025
Provinsi: Papua Selatan
```
Warna header popup mengikuti warna jenis transisinya.

### F. Modal "Tentang & Metodologi"
Teks statis singkat (netral, ilmiah):
- Apa itu ForestWatch Papua (1 paragraf).
- Metode: Sentinel-2 + ResNet50-U-Net, 7 kelas, deteksi perubahan 2021→2025.
- Sumber data (Sentinel-2, ESA WorldCover, Dynamic World, Hansen GFC, BIOPAMA).
- Keterbatasan singkat (tutupan awan, dll).
- Tombol tutup.

---

## 6. Responsif (Mobile/Tablet) — Sederhana Saja

- **Desktop (≥1024px):** layout 2 kolom seperti di atas.
- **Mobile (<768px):** peta full-width di atas, panel statistik menumpuk **di bawah** peta (stack vertikal). Kontrol peta jadi tombol/drawer yang bisa dibuka.
- Tidak perlu desain mobile yang rumit — cukup tumpuk vertikal.

---

## 7. Yang TIDAK Perlu Dibuat (biar tetap sederhana)

- ❌ Tombol unduh data / ekspor GeoJSON / CSV / PDF
- ❌ Login, akun pengguna, dashboard admin
- ❌ Upload citra / inferensi langsung di browser
- ❌ Animasi time-series kompleks (cukup toggle 2021/2025)
- ❌ Pengaturan/preferensi pengguna
- ❌ Multi-bahasa

---

## 8. Prinsip Visual

- **Bersih & fokus ke peta** — peta adalah bintang utama, panel statistik pendukung.
- **Flat & modern** — kartu dengan sudut membulat lembut, bayangan tipis, banyak ruang putih.
- **Hirarki jelas** — angka total perubahan paling menonjol; grafik mendukung; tabel metrik paling kecil.
- **Konsisten warna** — selalu pakai palet kelas/transisi di Bagian 3 di seluruh peta, legenda, grafik, dan popup.
- **Font** — sans-serif (Inter/Roboto), angka besar untuk metrik kunci.

---

## 9. Daftar Layar untuk Didesain

1. **Layar Utama** — peta + panel statistik (desktop).
2. **Layar Utama (mobile)** — versi tumpuk vertikal.
3. **State Popup** — marker diklik, popup detail muncul.
4. **Modal Tentang & Metodologi** — overlay di atas peta.
5. *(Opsional)* **State loading** — skeleton/spinner saat data peta dimuat.

---

*Design Brief WebGIS ForestWatch Papua · SEC SATRIA DATA 2026 · Universitas Andalas*
*Acuan data: `webgis/backend` (FastAPI) + 7 file output model di `ForestWatch_Outputs/`.*
