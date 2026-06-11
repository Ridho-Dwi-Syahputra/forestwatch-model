# Panduan Lengkap: Menjalankan ForestWatch Papua di Google Colab

## Jawaban Singkat

| Pertanyaan | Jawaban |
| --- | --- |
| Upload file apa ke Colab? | **Tidak ada** — semua dari GitHub |
| Bisa matikan laptop? | **Ya** — tapi biarkan tab browser Colab terbuka |
| Output tersimpan di mana? | **Google Drive** — otomatis, tidak hilang walau runtime mati |
| Perlu setup ulang tiap sesi? | **Ya** — 2 cell setup (clone + mount Drive), ~3 menit |

---

## Persiapan Sekali Saja (Lakukan Sebelum Mulai)

### Langkah 1 — Buat GitHub Repository

1. Buka [github.com/new](https://github.com/new)
2. Nama repo: `forestwatch-papua-model`
3. Visibility: **Public** (agar Colab bisa clone tanpa token)
4. Klik **Create repository**

### Langkah 2 — Push Folder `model/` ke GitHub

Buka terminal di VS Code (folder `model/`):

```bash
git init
git add .
git commit -m "initial: forestwatch papua pipeline"
git branch -M main
git remote add origin https://github.com/USERNAME/forestwatch-papua-model.git
git push -u origin main
```

Ganti `USERNAME` dengan username GitHub kamu.

### Langkah 3 — Simpan Notebook ke Google Drive

1. Buka Google Drive di browser
2. Buat folder `ForestWatch` di My Drive
3. Upload file: `notebooks/forestwatch_papua_full_pipeline.ipynb` ke folder itu

Selesai — setup awal tidak perlu diulang lagi.

---

## Alur Kerja Tiap Sesi (Setiap Kali Buka Colab)

### Step A — Buka Notebook dari Drive

1. Buka [colab.research.google.com](https://colab.research.google.com)
2. File → Open notebook → Google Drive → pilih `forestwatch_papua_full_pipeline.ipynb`

### Step B — Set GPU T4

```
Runtime → Change runtime type → T4 GPU → Save
```

Klik **Connect** (pojok kanan atas). Tunggu sampai RAM/Disk meter muncul.

### Step C — Jalankan Cell Setup (Bagian 0 di notebook)

**Cell 1 — Clone repo + install package:**
```python
!git clone https://github.com/USERNAME/forestwatch-papua-model.git forestwatch
%cd forestwatch
!pip install -q -e ".[gee,gis,ml]"
```
Durasi: ~3–5 menit (install PyTorch, rasterio, dll).

**Cell 2 — Mount Google Drive (WAJIB untuk simpan output):**
```python
from google.colab import drive
drive.mount('/content/drive')
```
Akan minta izin akses — klik Allow.

**Cell 3 — Verifikasi GPU:**
```python
import torch
print("GPU:", torch.cuda.get_device_name(0))
```
Harus tampil: `Tesla T4`

**Cell 4 — Set path (sudah ada di notebook, pastikan path-nya):**
```python
from pathlib import Path
DRIVE_ROOT = Path('/content/drive/MyDrive')
TILES_T1   = DRIVE_ROOT / 'ForestWatch_Tiles_T1'
TILES_T2   = DRIVE_ROOT / 'ForestWatch_Tiles_T2'
PATCH_DIR  = DRIVE_ROOT / 'ForestWatch_Patches'
MASK_DIR   = DRIVE_ROOT / 'ForestWatch_Masks'
OUT_DIR    = DRIVE_ROOT / 'ForestWatch_Outputs'
CKPT_PATH  = PATCH_DIR / 'best_model.pt'
```

Setelah 4 cell ini → lanjutkan dari Bagian yang relevan di notebook.

---

## Bisa Matikan Laptop?

```
Laptop kamu              Colab server (Google)
┌──────────┐             ┌──────────────────────────┐
│ Browser  │ ──putus──   │ Training tetap jalan ✓   │
│ (mati)   │             │ GPU T4                   │
└──────────┘             │ Simpan ke Drive tiap     │
                         │ epoch ✓                  │
                         └──────────────────────────┘
```

**Boleh:**
- Minimize laptop / layar gelap
- Laptop sleep (kalau browser tetap terbuka)

**Hindari:**
- Tutup tab Colab di browser → Colab deteksi idle, disconnect setelah ~30–90 menit
- Shutdown laptop → browser tutup → sama dengan di atas

**Solusi kalau harus matikan laptop:**
1. Pastikan training sudah berjalan minimal beberapa epoch (checkpoint sudah ada di Drive)
2. Saat hidup lagi → buka Colab → reconnect runtime → jalankan ulang cell setup
3. Training akan lanjut dari checkpoint terakhir (sudah otomatis di trainer.py)

---

## Di Mana Output Tersimpan?

Semua output tersimpan di **Google Drive** (`/content/drive/MyDrive/`):

```
MyDrive/
├── ForestWatch_Tiles_T1/       # 36 GeoTIFF citra 2021 (ekspor GEE ~2-4 jam)
├── ForestWatch_Tiles_T2/       # 36 GeoTIFF citra 2024 + label
├── ForestWatch_Patches/        # patch .npz + best_model.pt + training_curve.png
├── ForestWatch_Masks/          # 72 mask hasil inferensi
└── ForestWatch_Outputs/        # 7 file kontrak untuk WebGIS
    ├── landcover_2021.png
    ├── landcover_2021_bounds.json
    ├── landcover_2024.png
    ├── landcover_2024_bounds.json
    ├── deforestation.geojson
    ├── statistics.json
    ├── legend.json
    ├── metrics.json
    ├── model.onnx
    └── model_card.md
```

**Tidak ada yang hilang** walau runtime Colab mati — semua sudah di Drive.

---

## Timeline per Minggu

### Minggu 1
1. Setup GitHub + push repo
2. Buka Colab → jalankan **Bagian 1** (generate dummy untuk Orang 2)
3. Jalankan **Bagian 2** (auth GEE + komposit S2)
4. Jalankan **Bagian 3** (ekspor 36 ubin) → **bisa tinggal**, proses ~2–4 jam di server GEE

### Minggu 2
1. Buka Colab lagi
2. Jalankan **Bagian 4** (cut patches)
3. Jalankan **Bagian 5** (training) → **bisa tinggal**, proses ~1–2 jam di GPU T4

### Minggu 3
1. Jalankan **Bagian 6–8** (evaluasi + inferensi + generate 7 file output)
2. Kirim folder `ForestWatch_Outputs/` ke Orang 2 (WebGIS)
3. Kirim angka ke Orang 3 (esai)

### Minggu 4
- Backup notebook ke GitHub
- Stand by untuk Q&A juri

---

## Tips Penting

1. **Jangan jalankan `Run All`** langsung — jalankan per Bagian sesuai jadwal minggu
2. **Ekspor GEE dan training bisa ditinggal** — cukup pantau via Gmail (Colab kirim notif kalau selesai/error)
3. **Kalau runtime mati saat training** → cek Drive apakah ada `best_model.pt` → kalau ada, lanjut dari cell evaluasi/inferensi
4. **Kalau `pip install` lambat** → normal, ~3–5 menit karena PyTorch besar
5. **Pastikan Drive punya ruang** — estimasi total: ~20–30 GB (ubin GEE + patches + model)

---

*Panduan ini melengkapi [README.md](README.md) dan [docs/model/MASTER_PLAN.md](../docs/model/MASTER_PLAN.md)*
