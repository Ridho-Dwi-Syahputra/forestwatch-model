> ⚠️ **DOKUMEN USANG (v1.0).** Skema di bawah sudah lama (6 kelas, T2=2024, "Lahan Terbakar", BIOPAMA, 4 transisi, "food estate"). Skema final sekarang **7 kelas** (…, Lahan Terbuka, Tambang, Permukiman), **T2=2025**, dataset upgrade (FDP Palm 2025a + Mining union TW+Maus). **Rencana eksekusi terkini → [`RENCANA_EKSEKUSI_LANJUTAN.md`](RENCANA_EKSEKUSI_LANJUTAN.md).** Isi di bawah dibiarkan utuh sebagai arsip historis.

# MASTER PLAN — Role MODEL (Orang 1)

**Proyek:** ForestWatch Papua — Sistem Deteksi Dini Deforestasi Berbasis Deep Learning
**Kompetisi:** Statistics Essay Competition (SEC) SATRIA DATA 2026 · Subtema 1
**Institusi:** Universitas Andalas
**Deadline akhir:** 30 Juni 2026, 16.00 WIB
**Dokumen ini:** Panduan eksekusi harian untuk Machine Learning Lead

> Dokumen ini adalah turunan dari [PRD v2.0](../PRD_ForestWatch_Papua_v2.md) dan [Buku Panduan v2.0](../Buku_Panduan_ForestWatch_Papua_v2.docx). PRD adalah sumber kebenaran teknis (kode, skema data). Dokumen ini memecah pekerjaan jadi langkah harian yang dapat dieksekusi dan ditick.

---

## 1. Ringkasan Peran & Tujuan

Sebagai **Machine Learning Lead (Orang 1)**, tugas utama adalah membangun model segmentasi semantik **ResNet50-U-Net** yang mengklasifikasikan tutupan lahan Papua dari citra **Sentinel-2** ke dalam **7 kelas**, lalu menjalankan **deteksi perubahan 2021 vs 2024** untuk menghasilkan peta deforestasi beserta **4 jenis transisi** (hutan ke lahan terbuka, hutan ke sawit, hutan ke pertanian lain, hutan ke terbakar).

### Target Keberhasilan (dari PRD §A.1)

| Metrik | Minimum | Ideal | Catatan |
| --- | --- | --- | --- |
| Mean IoU (mIoU) | ≥ 0,60 | ≥ 0,75 | Laporkan angka **asli** |
| IoU Deforestasi/Lahan Terbuka | ≥ 0,55 | ≥ 0,70 | Kelas terpenting |
| IoU Sawit | ≥ 0,55 | ≥ 0,70 | Relevansi naratif tinggi |
| Overall Accuracy | ≥ 0,80 | ≥ 0,90 | Mudah tinggi karena hutan dominan |
| 7 File output ke Orang 2 | Lengkap | — | Wajib agar WebGIS jalan |

> **Prinsip emas:** Lebih baik mIoU 0,68 yang **nyata** daripada 0,90 yang **fiktif**. Juri akan menguji angka di sesi tanya jawab.

---

## 2. Stack Teknologi & Tools

| Tool | Versi/Tier | Fungsi |
| --- | --- | --- |
| Google Earth Engine | Project `forestwatch-papua-unand` (Community tier) | Akses 5 dataset, label fusion, ekspor ubin |
| Google Colab | Free/Pro dengan GPU T4 | Training model, inferensi |
| Google Drive | Min 15 GB tersedia | Penyimpanan ubin, patch, model, output |
| VS Code (lokal) | Latest | Editor + Colab integration |
| GitHub | Repo tim | Backup notebook & kode |

### Library Python (install via Cell 1 PRD)

```bash
pip install -q earthengine-api geemap segmentation-models-pytorch
pip install -q rasterio geopandas numpy matplotlib tqdm albumentations torchmetrics
pip install -q shapely fiona pyproj
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Struktur Folder Google Drive

```
MyDrive/
├── ForestWatch_Tiles_T2/        # 36 GeoTIFF (citra 2024 + label)
├── ForestWatch_Tiles_T1/        # 36 GeoTIFF (citra 2021)
├── ForestWatch_Patches/         # patch 256x256 .npz untuk training
│   └── best_model.pt
└── ForestWatch_Outputs/         # 7 file kontrak untuk Orang 2
    ├── landcover_2024.png
    ├── landcover_2024_bounds.json
    ├── landcover_2021.png
    ├── landcover_2021_bounds.json
    ├── deforestation.geojson
    ├── statistics.json
    ├── legend.json
    ├── metrics.json
    └── model.onnx + model_card.md
```

---

## 3. Pra-Pekerjaan (Setup Day 0)

Sebelum mulai Minggu 1, pastikan semua item ini selesai dalam **1 hari**:

- [ ] Login GEE di console.cloud.google.com/earth-engine dengan project `forestwatch-papua-unand`
- [ ] Tier registrasi: **Noncommercial → Public academic institution → Andalas University → Community**
- [ ] Test autentikasi GEE di Colab: jalankan `ee.Authenticate(); ee.Initialize(project='forestwatch-papua-unand')`
- [ ] Mount Google Drive di Colab, buat 4 folder utama (lihat tabel di atas)
- [ ] Verifikasi runtime Colab → Change runtime type → **T4 GPU** aktif (`!nvidia-smi`)
- [ ] Buat GitHub repo `forestwatch-papua-model` dan commit notebook kosong sebagai baseline
- [ ] Test akses 5 dataset dengan satu `print(ee.Image('ESA/WorldCover/v200/2021').getInfo())` dst.
- [ ] Bookmarks: [GEE Code Editor](https://code.earthengine.google.com/), [Tasks](https://code.earthengine.google.com/tasks), [PRD §A.5](../PRD_ForestWatch_Papua_v2.md)

---

## 4. Timeline Eksekusi 4 Minggu

### Minggu 1 — Data Pipeline & Label Fusion (Uji Coba)

**Tujuan minggu:** Membuktikan pipeline label fusion 6 aturan bekerja pada **1 ubin uji**, lalu menyepakati kontrak data dengan Orang 2.

**Urutan eksekusi:**
1. Jalankan Cell 1 (instalasi) dan Cell 2 (auth GEE) di Colab.
2. Adopsi Cell 3 PRD: definisi area Papua (bbox 130°–141,2°E, -9,5°–0,5°N), komposit S2 bebas awan untuk 2024, label fusion 6 aturan.
3. Pilih **1 ubin kecil** (~1° × 1° di sekitar Merauke, koordinat ~140,4°E, -8,5°S) untuk uji coba — lebih cepat daripada 36 ubin.
4. Ekspor stack `(6 band citra + 1 band label)` ke Drive dengan `Export.image.toDrive`.
5. Pantau task di [GEE Tasks Console](https://code.earthengine.google.com/tasks).
6. Setelah selesai, download ubin GeoTIFF ke laptop, buka di QGIS atau plot dengan `rasterio` + `matplotlib`.
7. Verifikasi visual: apakah hutan = hijau, sawit terdeteksi di area perkebunan known, deforestasi muncul di area gundul known.
8. **Sepakati skema 7 file kontrak dengan Orang 2** (lihat §5). Kirim contoh dummy `deforestation.geojson` (5–10 polygon karangan) dan `statistics.json` ke Orang 2 — **paling lambat hari ke-3** agar Orang 2 bisa develop UI paralel.

**Definition of Done (DoD):**
- 1 GeoTIFF ubin uji (7 band) tersimpan di `ForestWatch_Tiles_T2/`
- Visualisasi label fusion menunjukkan ≥ 5 kelas (Lahan Terbakar boleh nol di area uji)
- Dummy file kontrak terkirim ke Orang 2
- Notebook tersimpan di GitHub

**Checklist Minggu 1:**
- [ ] Setup Day 0 selesai
- [ ] Cell 1–3 PRD ter-adopsi di notebook Colab
- [ ] Komposit S2 2024 untuk Papua berhasil tanpa error
- [ ] Label fusion 6 aturan menghasilkan citra label 6 kelas (visual cek di geemap)
- [ ] Ekspor 1 ubin uji (Merauke) ke Drive berhasil
- [ ] Verifikasi visual di QGIS/rasterio: distribusi kelas masuk akal
- [ ] Skema 7 file kontrak (PRD §B.1) disepakati dengan Orang 2
- [ ] Dummy `deforestation.geojson` & `statistics.json` dikirim ke Orang 2

---

### Minggu 2 — Ekspor Penuh + Training Model

**Tujuan minggu:** Punya **36 ubin penuh** Papua + model **ResNet50-U-Net** terlatih dengan **mIoU ≥ 0,60** pada validation set.

**Urutan eksekusi:**
1. Adopsi Cell 4 PRD: `make_tiles(papua, 6, 6)` → 36 ubin. Ekspor `stack_t2` (untuk training) dan `img_t1` (untuk deteksi perubahan).
2. **72 task ekspor** akan berjalan paralel di GEE (~2–4 jam). Pantau, re-jalankan ubin yang gagal.
3. Setelah selesai, jalankan Cell 5: `cut_patches()` memotong ubin jadi patch 256×256 `.npz`. Buang patch dengan >30% NaN.
4. Hitung **jumlah patch real** (`len(files)`). Target: ribuan patch.
5. **Hitung distribusi kelas asli** dari label patches → sesuaikan `class_weights` di Cell 6 (jangan pakai nilai default PRD bila distribusi sangat berbeda).
6. Adopsi Cell 6: definisikan model, loss kombinasi (0,6 × CE berbobot + 0,4 × Dice), optimizer AdamW, scheduler Cosine.
7. Jalankan Cell 7: training 50 epoch dengan AMP, early stopping patience=10. Simpan `best_model.pt` ke Drive.
8. Monitor mIoU per epoch. Bila tidak naik setelah 15 epoch, cek: batch size, learning rate, distribusi label.

**DoD:**
- 72 GeoTIFF ubin tersimpan di Drive
- ≥ 80% patch valid (≤ 20% dibuang karena NaN)
- `best_model.pt` tersimpan dengan mIoU validation **≥ 0,60**
- Log training tersimpan (.txt atau screenshot terakhir)

**Checklist Minggu 2:**
- [ ] 36 task ekspor T2 dimulai
- [ ] 36 task ekspor T1 dimulai
- [ ] Semua 72 ubin sukses (re-run jika ada yang gagal)
- [ ] `cut_patches()` selesai, jumlah patch dicatat
- [ ] Distribusi kelas dihitung dan `class_weights` disesuaikan
- [ ] Model didefinisikan dengan `in_channels=6, classes=6`
- [ ] Training berjalan dengan AMP (tidak OOM)
- [ ] Early stopping checkpoint disimpan ke Drive
- [ ] mIoU validation ≥ 0,60 tercapai
- [ ] Log training disimpan
- [ ] Notebook di-commit ke GitHub
- [ ] Update progress ke tim (rapat mingguan)

**Mitigasi cepat:**
- Bila OOM → turunkan `batch_size` dari 8 ke 4, atau patch size 128.
- Bila mIoU stuck di < 0,50 → periksa kembali distribusi label, normalisasi reflektansi (`/10000`), dan augmentasi aktif.
- Bila task GEE timeout → kecilkan tile, atau ulangi task yang gagal.

---

### Minggu 3 — Inferensi + Deteksi Perubahan + 7 File Output

**Tujuan minggu:** Hasilkan semua **7 file output** sesuai kontrak [PRD §B.1](../PRD_ForestWatch_Papua_v2.md) dan **serah-terima** ke Orang 2 di hari pertama minggu.

**Urutan eksekusi:**
1. Load model terbaik dari Drive.
2. Adopsi Cell 9 PRD: jalankan `infer_tile()` untuk **36 ubin T1** dan **36 ubin T2**. Hasil = 72 mask GeoTIFF di `ForestWatch_Masks/`.
3. Adopsi blok deteksi perubahan di Cell 9: untuk tiap ubin, hitung selisih T1 vs T2 dengan `TRANSITION_MAP`. Polygonisasi area berubah dengan `rasterio.features.shapes`, simpan `deforestation.geojson`.
4. Filter polygon < 0,5 ha (noise).
5. Adopsi Cell 10 PRD: merge mask T2 jadi raster Papua tunggal, render PNG berwarna dengan PALETTE, simpan bounds JSON.
6. Ulangi untuk T1.
7. Generate `legend.json` (PRD §B.1.3) dan `statistics.json` (PRD §B.1.2) — termasuk `per_class_area_ha`, `per_transition_ha`, `n_hotspots`, `total_deforestation_ha`, dan embedded `model_metrics`.
8. Ekspor `model.onnx` (Cell 8) dengan opset 13.
9. Tulis `model_card.md` singkat: arsitektur, training set, metric, keterbatasan.
10. **Validasi schema 7 file** dengan jq atau Python: pastikan struktur sesuai PRD §B.1.1, §B.1.2, §B.1.3.
11. Copy/share folder `ForestWatch_Outputs/` ke Orang 2.

**DoD:**
- 7 file lengkap di `ForestWatch_Outputs/`
- Schema setiap file sesuai PRD §B.1 (validasi manual)
- Total deforestasi (ha) dan jumlah hotspot konsisten antara `deforestation.geojson` dan `statistics.json`
- Orang 2 sudah punya akses ke folder Drive

**Checklist Minggu 3:**
- [ ] Inferensi T1 (36 ubin) selesai
- [ ] Inferensi T2 (36 ubin) selesai
- [ ] Deteksi perubahan 4 transisi berhasil
- [ ] `deforestation.geojson` dihasilkan, > 0 feature
- [ ] `landcover_2024.png` + bounds JSON dihasilkan
- [ ] `landcover_2021.png` + bounds JSON dihasilkan
- [ ] `legend.json` sesuai PRD §B.1.3
- [ ] `statistics.json` sesuai PRD §B.1.2 (semua field lengkap)
- [ ] `metrics.json` dengan confusion matrix
- [ ] `model.onnx` dengan opset 13
- [ ] `model_card.md` ditulis (≥ 200 kata)
- [ ] Validasi cross-file: angka konsisten
- [ ] Folder `ForestWatch_Outputs/` di-share ke Orang 2 (akses Editor)
- [ ] Notif di grup tim bahwa data asli siap

---

### Minggu 4 — Finalisasi & Dokumentasi

**Tujuan minggu:** Polish, dokumentasi reproducible, dan sync angka final ke Orang 3 untuk esai.

**Urutan eksekusi:**
1. **(Opsional bila waktu cukup)** Re-train dengan `decoder_attention_type='scse'` (Attention U-Net) untuk mengejar mIoU 0,75. Bandingkan dengan baseline.
2. Generate confusion matrix sebagai figure PNG (untuk lampiran esai).
3. Generate plot training curve (loss vs epoch, mIoU vs epoch).
4. Hitung **breakdown per provinsi** menggunakan shapefile batas administratif Papua. Update `statistics.json` field `per_province`.
5. Tulis ulang `model_card.md` lengkap (PRD-style):
   - Arsitektur, jumlah parameter
   - Dataset training (sumber, distribusi kelas)
   - Hasil evaluasi (mIoU, OA, IoU per kelas)
   - Limitation (cloud, BIOPAMA 2019, sawit muda)
   - Intended use (lomba, pilot WebGIS — bukan keputusan kebijakan tanpa validasi lapangan)
6. Sinkronisasi angka dengan Orang 3:
   - Total deforestasi (ha)
   - 4 breakdown transisi (ha + %)
   - Top 3 provinsi terdampak
   - mIoU, OA, IoU per kelas
   - Spesifik Merauke: luas hutan-ke-pertanian-lain di Papua Selatan
7. Final commit notebook ke GitHub dengan README.md yang menjelaskan cara menjalankan.

**DoD:**
- Notebook fully reproducible (run from top → bottom tanpa error pada machine baru)
- `model_card.md` final
- Confusion matrix PNG + training curve PNG tersimpan
- Angka final dikirim ke Orang 3 dalam format teks/markdown rapi

**Checklist Minggu 4:**
- [ ] (Opsional) Attention U-Net comparison dilakukan
- [ ] Confusion matrix figure tersimpan di `ForestWatch_Outputs/figures/`
- [ ] Training curve figure tersimpan
- [ ] `per_province` di `statistics.json` di-update dengan angka real
- [ ] `model_card.md` final (≥ 500 kata, sesuai template)
- [ ] Angka final dikirim ke Orang 3 (markdown bullet)
- [ ] Notebook reproducible (testing manual)
- [ ] README.md repo GitHub lengkap dengan instruksi
- [ ] Backup terakhir semua file ke Drive + GitHub
- [ ] Stand by untuk Q&A juri (kuasai semua angka)

---

## 5. Kontrak Output untuk Orang 2 (WAJIB)

Semua file di folder `ForestWatch_Outputs/` Google Drive, sistem koordinat **EPSG:4326**.

| # | Nama file | Format | Isi | Wajib di Minggu |
| --- | --- | --- | --- | --- |
| 1 | `landcover_2024.png` + `landcover_2024_bounds.json` | PNG + JSON | Raster segmentasi T2 berwarna + bounds Leaflet | 3 |
| 2 | `landcover_2021.png` + `landcover_2021_bounds.json` | PNG + JSON | Raster T1 untuk slider waktu | 3 |
| 3 | `deforestation.geojson` | GeoJSON | Poligon transisi + atribut | 3 |
| 4 | `statistics.json` | JSON | Statistik agregat + metrik model | 3 |
| 5 | `legend.json` | JSON | id → {nama, warna} 6 kelas | 3 |
| 6 | `metrics.json` | JSON | Metrik model lengkap | 3 |
| 7 | `model.onnx` + `model_card.md` | ONNX + Markdown | Model & metadata | 4 |

### Skema `deforestation.geojson` (lihat PRD §B.1.1)

Properti tiap feature:
- `id`: string `DF-XXXXX`
- `transition_type`: salah satu dari `hutan_ke_lahan_terbuka`, `hutan_ke_sawit`, `hutan_ke_pertanian_lain`, `hutan_ke_terbakar`
- `area_ha`: float
- `period_from`: 2021
- `period_to`: 2024
- `province`: string (Papua / Papua Selatan / dll)
- `kawasan_status`: string opsional (mis. "APL / Food Estate Merauke")

### Skema `statistics.json` (lihat PRD §B.1.2)

Field wajib:
- `period_from`, `period_to`
- `total_deforestation_ha`
- `n_hotspots`
- `per_transition_ha` (4 key)
- `per_province` (array of 6 object)
- `per_class_area_ha` (6 key sesuai nama kelas)
- `model_metrics` (objek dengan `overall_accuracy`, `mean_iou`, `per_class[]`)

### **PENTING — Deadline Dummy Data**

Kirim **dummy** versi 7 file ini ke Orang 2 **paling lambat akhir Minggu 1** (data karangan dengan schema valid) — agar Orang 2 bisa develop UI tanpa menunggu model selesai. Saat data asli siap di Minggu 3, Orang 2 cukup ganti file.

---

## 6. Risiko Teknis & Mitigasi (dari PRD §A.6)

| Jika terjadi | Lakukan |
| --- | --- |
| OOM Colab pada batch 8, patch 256×256, 6 band | Turunkan `batch_size` ke 4, atau patch size 128. AMP sudah ON. |
| Patch didominasi NaN karena awan | Perpanjang rentang tanggal komposit (misal 2024-01 s/d 2025-06), tambah median window. |
| Ketidakseimbangan kelas membuat hutan over-predicted | Naikkan `class_weights` untuk kelas minoritas; tambah Dice loss share. |
| Sawit muda (0–3 tahun) tidak terdeteksi | Akui jujur di esai; tangkap via deteksi perubahan. |
| Kelas Lahan Terbakar terlalu sedikit (< 100 patch) | Gabung ke Kelas 2 (Deforestasi); turunkan jumlah kelas ke 5. Update `legend.json`. |
| Hansen 30m vs Sentinel 10m noise di tepi | `focal_min(radius=1)` sudah dipasang (erosi 1 piksel). |
| Sesi Colab terputus saat training | Checkpoint per epoch sudah ke Drive; lanjutkan dari `best_model.pt`. |
| GEE export timeout | Pembagian 36 ubin sudah menekan risiko; re-run task yang gagal. |
| Model overfit (val IoU turun setelah naik) | Tambah augmentasi, turunkan learning rate, kurangi epoch. Early stopping sudah aktif. |

---

## 7. Referensi Cepat ke PRD

| Tugas | Lihat |
| --- | --- |
| Daftar lengkap dataset + asset ID | PRD §A.2 |
| Definisi 6 kelas | PRD §A.3 + Buku Panduan §4.3 |
| 6 aturan label fusion | PRD §A.4 |
| Kode lengkap notebook | PRD §A.5 (Cell 1–10) |
| Risiko + mitigasi | PRD §A.6 |
| Skema 7 file output | PRD §B.1 |
| Skema GeoJSON | PRD §B.1.1 |
| Skema statistics.json | PRD §B.1.2 |
| Skema legend.json | PRD §B.1.3 |
| Justifikasi naratif (untuk Orang 3) | Buku Panduan §4 |

---

## 8. Checklist Akhir (Sebelum Submit — dari PRD Lampiran 2)

- [ ] Semua 7 file output di `ForestWatch_Outputs/` lengkap dan tervalidasi
- [ ] Metrik model dilaporkan **jujur** dan konsisten antar file (mIoU di `metrics.json` = `statistics.json.model_metrics.mean_iou`)
- [ ] Studi kasus Merauke punya koordinat spesifik (cek `feature.properties.province == 'Papua Selatan'`)
- [ ] Confusion matrix tersimpan sebagai PNG + JSON
- [ ] Kode notebook (.ipynb) di GitHub repo tim
- [ ] `model.onnx` tervalidasi dengan `onnxruntime` (test inference 1 patch)
- [ ] `model_card.md` final
- [ ] Sinkronisasi angka dengan Orang 3 selesai
- [ ] Tim sudah uji buka WebGIS Orang 2 dengan data asli
- [ ] Siap Q&A juri tentang semua angka

---

## 9. Alur Komunikasi Tim

| Waktu | Aksi |
| --- | --- |
| Setiap akhir minggu | Rapat tim 15–30 menit. Share progress, blocker. |
| Hari ke-3 Minggu 1 | Kirim dummy 7 file ke Orang 2 |
| Hari ke-7 Minggu 2 | Update progress training ke tim (mIoU current) |
| Hari ke-1 Minggu 3 | Hands-off `ForestWatch_Outputs/` ke Orang 2 |
| Hari ke-5 Minggu 3 | Cross-check angka dengan Orang 3 untuk esai |
| Minggu 4 | Stand by untuk re-generate file bila ada bug ditemukan Orang 2 |

---

*Master Plan Model · ForestWatch Papua · SEC SATRIA DATA 2026 · v1.0*
*Untuk perubahan substansial pada metodologi atau skema data, update [PRD v2.0](../PRD_ForestWatch_Papua_v2.md) dan dokumen ini secara bersamaan.*
