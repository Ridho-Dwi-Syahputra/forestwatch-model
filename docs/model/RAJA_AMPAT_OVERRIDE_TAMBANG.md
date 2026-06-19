# Raja Ampat & Override Manual Tambang — Catatan untuk Tahap 2

> Pengingat: apa yang diubah, di cell mana, dan apa yang **wajib dicek ulang** sebelum
> lanjut training skala penuh (se-Papua) pakai `Bahan_Training_Fix` di Tahap 2.

---

## 1. Kenapa ada perubahan ini?

Notebook `train_merauke_boven_digoel_attention_unet.ipynb` (Tahap 1) butuh sinyal **Tambang**
asli-lokal Papua. Merauke+Boven Digoel sendiri driver deforestasinya sawit/food estate, hampir
tidak ada tambang nyata — jadi ditambahkan ekspor GEE baru untuk 3 kandidat pulau tambang nikel
di Raja Ampat: **Gag** (PT Gag Nikel, aktif sejak 2001), **Kawe** (PT KSM), **Manuran** (PT ASP).

## 2. Temuan kritis: footprint global TIDAK cover Raja Ampat sama sekali

Label kelas **Tambang (5)** di pipeline ini **bukan** hasil deteksi dari piksel citra — murni
overlay poligon statis dari dua dataset:
- `Tang & Werner (2023)` — Global Mining Footprint
- `Maus dkk. (2022)`

Setelah cek histogram label langsung (bukan tebak dari warna RGB), hasilnya **ketiga pulau =
0 piksel Tambang**, termasuk Gag yang tambangnya sudah berjalan 24 tahun dan jelas terlihat di
citra. Kesimpulan: kedua dataset global itu memang **tidak mencakup Raja Ampat**, kemungkinan
karena cakupannya fokus ke hub nikel utama Indonesia (Sulawesi: Morowali, Konawe), bukan
pulau-pulau kecil terpencil di Papua Barat Daya.

## 3. Solusi yang diterapkan

- **Kawe & Manuran dikeluarkan** dari `RAJA_AMPAT_REGIONS` — tidak memberi nilai unik (0%
  Tambang), tidak layak diekspor.
- **Gag dipertahankan + override manual**: piksel dipaksa jadi kelas 5 jika *(a)* NDVI rendah
  (`< 0.25`, non-vegetasi/tanah terbuka), *(b)* bukan Perairan (hasil DW base), *(c)* berada
  dalam sub-bbox estimasi konsesi `GAG_MINING_SUBBBOX = (129.82, -0.50, 129.96, -0.38)`.
  Hasil terverifikasi: **27.388 piksel Tambang (~273,9 ha)**, masuk akal dibanding dokumentasi
  kumulatif 623 ha (2001–2024) — order of magnitude konsisten, bukan asal jadi.
- Preview overlay magenta (cek visual) **wajib dijalankan & dicek** sebelum submit export —
  ada di notebook, jangan dilewati kalau bbox/threshold diubah lagi nanti.

## 4. Cell yang berubah — `train_merauke_boven_digoel_attention_unet.ipynb`

| Cell | Isi | Catatan |
|---|---|---|
| 5 | Setup `RAJA_AMPAT_REGIONS` | Sekarang **cuma `gag`** |
| 6 | Preview RGB (wajib) | Tidak berubah |
| 7 | **Baru** — `gag_tambang_override()` + preview overlay magenta | Override manual, threshold di sini |
| 8 | Verifikasi histogram label | Sekarang menerapkan override utk `gag` sebelum hitung |
| 9 | Submit export GEE | Sekarang menerapkan override utk `gag` sebelum stack+export |
| 10–11 | Monitor task + cut patches | Tidak berubah |

## 5. Cell baru — `optimize_dataset.ipynb`

Ditambahkan **cell 4–5** (setelah path setup, sebelum scan distribusi): cek apakah **10 region
transfer Tambang** yang sudah dipakai di `Bahan_Training_Fix` (Sangatta/KPC, Tanah Bumbu,
Morowali/IMIP, Batu Hijau, Chuquicamata, Escondida, Bingham Canyon, Kalgoorlie, Hunter Valley,
Hambach) punya masalah footprint-coverage yang sama atau tidak — scan langsung dari patch yang
**sudah ada** di `ForestWatch_Patches_Transfer/tambang/`, baca metadata `tile` per patch untuk
identifikasi region asal, tanpa perlu panggil GEE lagi. Output: tabel piksel Tambang per region +
flag eksplisit kalau ada yang 0 piksel.

**Belum dijalankan** — hipotesis saya (region-region ini tambang raksasa & lama, persis jenis
operasi yang ditarget paper footprint global) kemungkinan besar aman, tapi ini **harus
diverifikasi dengan menjalankan cell-nya**, bukan dipercaya begitu saja — pelajaran langsung
dari kasus Gag/Kawe/Manuran di atas.

## 6. Status checklist sebelum/di awal Tahap 2 (training se-Papua)

1. **Jalankan cell verifikasi baru di `optimize_dataset.ipynb`** (cell "Verifikasi: apakah
   region transfer Tambang...") dan baca hasilnya. **BELUM DIJALANKAN** — ini satu-satunya item
   yang masih perlu aksi MANUAL dari user (perlu Drive + GEE env, tidak bisa saya jalankan).
   - Kalau semua region > 0 piksel → `Bahan_Training_Fix` aman dipakai apa adanya.
   - Kalau ada region yang 0 piksel → perlu override manual serupa Gag untuk region itu
     (sub-bbox + threshold NDVI/bare-soil), sebelum data dipakai training skala penuh.
2. ✅ **SELESAI** — Patch Gag (Raja Ampat) sudah digabung ke pipeline 40GB. Lihat detail di
   §7 di bawah.
3. Jangan lupa: ambang `NDVI_BARE_THRESHOLD=0.25` dan `GAG_MINING_SUBBBOX` adalah **estimasi**,
   bukan survei presisi — kalau nanti ada akses ke data konsesi resmi (shapefile KLHK/ESDM),
   override ini sebaiknya diganti dengan poligon asli.
4. ✅ **SELESAI** — Estimasi "97GB" di cell 5 `train_model_1/2/3_*.ipynb` sudah diperbaiki.
   Ukuran aktual `Bahan_Training_Fix` ~40GB (muat di disk Colab ~73GB) karena `.npz` tersimpan
   **terkompresi** (`save_npz(..., compressed=True)` → [io.py](../../model/src/forestwatch/utils/io.py)),
   bukan ~97GB seperti dihitung dari array mentah. Cell 5 sekarang **selalu**
   `extract_dataset_archives()` apa pun `ENV`-nya (termasuk Colab) — lepas dari bottleneck I/O
   Drive FUSE yang sebelumnya bikin epoch 2-3 jam tak selesai.
5. **Alternatif arsitektur Tahap 2 (kalau GPU Colab limit/habis kuota)**: training `Bahan_Training_Fix`
   (40GB) bisa dipindah ke **Jetson sebagai GPU server**, dieksekusi dari VS Code di PC Lab via
   *remote Jupyter kernel* — bukan via Drive, lewat **SMB/CIFS share** dari HDD PC (data sudah
   diekstrak ke `train/val/test` + `class_weights.json` + `patch_sampler_weights_shared.json`).
   Catatan penting: kernel + GPU itu satu paket (sama seperti Colab — kernel jalan di VM Google,
   bukan di laptop kamu), jadi Jetson **wajib** punya cara baca data PC (mount SMB), tidak bisa
   "pinjam GPU doang" tanpa itu.

   Langkah:
   1. **PC**: pastikan folder data sudah terekstrak (bukan `.tar`), lalu *share* folder itu
      (klik kanan → Properties → Sharing), catat IP PC (`ipconfig`).
   2. **Jetson**: `sudo apt install -y cifs-utils` → `sudo mkdir -p /mnt/forestwatch_dataset` →
      `sudo mount -t cifs //<IP_PC>/<share> /mnt/forestwatch_dataset -o username=...,password=...,vers=3.0`
      → verifikasi `ls /mnt/forestwatch_dataset` muncul `train/ val/ test/ class_weights.json`.
   3. **Jetson**: `jupyter notebook --no-browser --ip=0.0.0.0 --port=8888` → catat URL+token,
      ganti host jadi IP Jetson (`hostname -I`).
   4. **VS Code (PC)**: kernel picker → "Select Another Kernel" → "Existing Jupyter Server" →
      masukkan URL Jetson.
   5. Notebook cell 2: `ENV = "jetson"`, `DATA_ROOT = Path("/mnt/forestwatch_dataset")` (cocokkan
      mount point). Jalankan, harus print `CUDA=True` dengan nama GPU Orin (bukan GPU PC yang lemah).

   Alternatif lain tanpa setup jaringan sama sekali: **Kaggle Notebook** (GPU P100 16GB atau
   2×T4, kuota ~30 jam/minggu, sesi ~9-12 jam) — upload `Bahan_Training_Fix` via Kaggle API
   (bukan drag-drop browser, untuk ukuran 40GB), lalu ganti `DRIVE_ROOT`/`DATA_ROOT` ke
   `/kaggle/input/...`. Cek ulang kuota/limit Kaggle terbaru di kaggle.com sebelum bergantung
   penuh padanya untuk deadline ketat — kebijakan platform bisa berubah.

## 7. Raja Ampat (Gag) digabung ke pipeline 40GB se-Papua — detail implementasi

**`optimize_dataset.ipynb`**:
- Cell 3 (`else` branch, non-jetson): tambah `PATCHES_RAJA_AMPAT = DRIVE_ROOT /
  'ForestWatch_Patches_RajaAmpat'`, `raja_ampat_files = list_patches(...)`, digabung ke
  `train_files` (sejajar dengan `transfer_files`/`aug_files` — train-only, sama seperti
  region transfer lain, val/test tetap holdout murni se-Papua tanpa Raja Ampat).
- Cell "Subsampling": `subsample_to_targets()` dapat parameter baru `exempt` — patch Raja
  Ampat **selalu dipertahankan penuh**, tidak ikut dikelompokkan ke cap Perairan/Hutan.
  Alasan: island bbox Gag ~94% air, sebagian besar dari 144 patch-nya *dominant*-Perairan —
  tanpa exempt ini, risiko nyata patch yang justru mengandung piksel Tambang (minoritas di
  patch yang dominan Hutan/Perairan) ikut terbuang random saat subsampling kelas mayoritas.
- `to_arcname()`: tambah prefix `(PATCHES_RAJA_AMPAT, "rajaampat")`.
- Cell bundling: item Raja Ampat (arcname berprefiks `rajaampat/`) dipisah dari
  `train_items_fix`, dibungkus sebagai **split tersendiri** `train_rajaampat` →
  `Bahan_Training_Fix/train_rajaampat/train_rajaampat_part01.tar` — **BUKAN** dicampur ke
  `train_part01-07.tar`. Tujuan: kalau Raja Ampat berubah (tambah pulau, ganti threshold
  override), tidak perlu re-bundle 7 part train yang besar, cukup re-bundle 1 tar kecil ini.
- Cell sampler weights: `sampler_keys_fix` digabung dari `train_items_fix` (prefix `"train"`)
  + `ra_items_fix` (prefix `"train_rajaampat"`) — supaya key cocok dengan path lokal hasil
  ekstraksi nanti (lihat poin berikut).

**`train_model_1/2/3_*.ipynb`** (cell 5, "Siapkan artefak training"):
- `extract_dataset_archives(BAHAN_DIR, LOCAL_DATA_DIR, splits=(...))` — `splits` sekarang
  dinamis: otomatis tambah `'train_rajaampat'` kalau folder itu ada di `BAHAN_DIR` (deteksi
  via `.exists()`, jadi tetap aman dijalankan terhadap `Bahan_Training_Fix` versi LAMA yang
  belum punya Raja Ampat — tidak akan error).
- `final_train_files` = patch dari `local_dirs['train']` **+** `local_dirs['train_rajaampat']`
  (kalau ada) — digabung sebelum dipakai ke `build_dataloaders_from_files`.

## 8. `OUTPUT_DIR` ditambahkan ke `train_model_1/2/3_*.ipynb`

Menyamakan pola dengan notebook Merauke (`train_merauke_boven_digoel_attention_unet.ipynb`):
- Cell 9 ("Identitas model"): tambah `OUTPUT_DIR = MODEL_DIR / 'output'` (khusus gambar),
  dibuat via `mkdir`.
- Cell plot history: `training_curve.png` sekarang disimpan ke `OUTPUT_DIR`, bukan `MODEL_DIR`
  langsung (`training_history.json` tetap di `MODEL_DIR`, itu bukan gambar).
- Cell evaluasi test: `confusion_matrix.png` sekarang ke `OUTPUT_DIR` (`metrics.json` tetap
  di `MODEL_DIR`).
- `compare_and_select_best_model.ipynb` (cell "Tetapkan pemenang"): disesuaikan, sekarang
  copy `confusion_matrix.png` dari `WIN_DIR / 'output'`, bukan `WIN_DIR` langsung.
