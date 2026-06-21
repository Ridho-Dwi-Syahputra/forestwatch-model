# Ekspor Gelombang 3: Tambang & Permukiman (+20-30 jt px/kelas) + Sawit BARU (Sumatra/Kalimantan, +40 jt px) — ADDITIF

> Catatan eksekusi untuk **Gelombang 3** (lanjutan additif dari Bagian 12.6-12.8 / gel.1-gel.2) di
> `model/notebooks/forestwatch_papua_full_pipeline.ipynb` (Bagian **12.9 / 12.9a / 12.9b / 12.9c**).
> Semua **ADDITIF**: tidak ada data lama yang dihapus atau di-export ulang.

---

## 1. Kenapa ada gelombang ini? (status distribusi)

Distribusi TOTAL (Papua + Transfer) terakhir vs target (Tambang 50jt, lainnya 100jt):

| Kelas | Total | % target | Status |
|---|---|---|---|
| Lahan Terbuka | 112.6 jt | 112.6% | ✅ aman |
| Pertanian Lain | 244.7 jt | 244.7% | ✅ aman |
| Sawit | 69.9 jt | 69.9% | ✅ aman (≥50%), tapi diminta tambah lagi |
| **Tambang** | 21.8 jt | **43.6%** | ❌ masih < 50% |
| **Permukiman** | 40.2 jt | **40.2%** | ❌ masih < 50% |

Target Gelombang 3 (lebih agresif dari sekadar "tutup gap ke 50%"):
- **Tambang**: tambahan **+20-30 jt px** → total jadi ±41.8-51.8 jt (≈84-104% dari target 50 jt).
- **Permukiman**: tambahan **+20-30 jt px** → total jadi ±60.2-70.2 jt (≈60-70% dari target 100 jt).
- **Sawit**: tambahan **minimal +40 jt px**, fokus **Sumatra & Kalimantan** (densitas sangat tinggi
  di sana) — gelombang additif **PERTAMA** untuk kelas Sawit.

---

## 2. Prinsip ADDITIF (kenapa aman, tidak merusak yang lama)

- Ekspor ke folder Drive **yang SAMA** per-kelas (`folder='tambang'` / `'permukiman'` / `'sawit'`),
  **bukan** folder baru. Folder `sawit` sudah dibuat sejak Bagian 12.0 (`TRANSFER_SLUGS`).
- Nama tile diberi **prefix yang urut alfabetis di AKHIR** tile lama:

| Slug | Urutan alfabetis tile lama → baru |
|---|---|
| tambang | `tambang_{nama}_*` (a–u) < gel.1 `tambang_zz_*` < gel.2 `tambang_zzz_*` < **gel.3 `tambang_zzzz_*`** |
| permukiman | `permukiman_{nama}_*` (b/j/m/s) < gel.1 `permukiman_zz_*` < **gel.3 `permukiman_zzz_*`** |
| sawit | `sawit_{kalbar/kalteng/riau_pelalawan/sumut}_*` < **gel.3 (BARU) `sawit_zz_*`** |

- Karena `cut_patches_resilient` (Bagian 12.5) memberi index `tile_000, tile_001, ...` menurut
  **urutan alfabetis** file `.tif`, tile baru selalu dapat index BARU di akhir → tile lama
  **tidak tergeser** dan **di-skip** lewat penanda `_DONE` saat 12.5 dijalankan ulang.
- **Dataset identik** dengan ekspor lama: semua cell memakai `s2_composite(T2)` + `build_label()`
  (T2=2025, dari `cfg['periods']['t2']`) yang memfusikan 6 sumber yang sama (Sentinel-2, ESA
  WorldCover, Dynamic World, Hansen GFC, FDP Palm, Global Mining Footprint).
- Bagian **12.3b** (pemindah tile nyasar) dan **12.5** (cut patches) sudah generik untuk
  `'sawit'` — **tidak perlu diubah** untuk gelombang ini.

---

## 3. Cell baru yang ditambahkan

### Bagian 12.9 (markdown)
Header & ringkasan gelombang 3 + urutan jalankan.

### Bagian 12.9a — Setup: 19 region baru (7 Tambang + 6 Permukiman + 6 Sawit), revisi v2

> **Revisi v2** (pasca run nyata 12.9b #1 di Colab): set 14-region awal (5+5+4) ternyata
> **`[mungkin kurang]` di SEMUA kelas** + `kaltim_paser` gate **`[X]` (0.0% Tambang)`**.
> Hasil gate #1 (estimasi tambahan): Tambang +8.52jt/20jt, Permukiman +16.30jt/20jt,
> Sawit +18.60jt/40jt. Region di bawah sudah **direvisi**: `kaltim_paser` dibuang,
> `jambi_tebo`+`kalsel_kotabaru` (8.4% keduanya) dibuang, dan 9 region baru ditambah
> (3 Tambang + 1 Permukiman + 5 Sawit... lihat detail per-kelas) berdasarkan benchmark
> region terbaik dari gate #1 (`australia_huntervalley` 15.6%, `bogor_depok` 47.8%,
> `riau_rokanhilir` 48.1%). **WAJIB ulangi 12.9a → 12.9b** sebelum 12.9c untuk verifikasi
> verdict `[cukup]` pada set v2 ini.

**Tambang gel.3** (`tambang_zzzz_`, bbox 0.40-0.50°, megamine batubara/bijih besi/oil sands
berdensitas tinggi — `kaltim_paser` dibuang krn 0% pada gate #1):

| Region | BBox (lon0, lat0, lon1, lat1) | Keterangan |
|---|---|---|
| `kaltim_sangatta` | (117.40, 0.25, 117.90, 0.75) | Kaltim Prima Coal, Sangatta (gate #1: 6.4%) |
| `sumsel_tanjungenim` | (103.65, -3.90, 104.10, -3.45) | Bukit Asam, Tanjung Enim (gate #1: 3.5%) |
| `safrica_sishen` | (22.70, -28.10, 23.20, -27.60) | Sishen-Kolomela, bijih besi (gate #1: 5.9%) |
| `australia_huntervalley` | (150.70, -32.75, 151.20, -32.25) | Hunter Valley, batubara (gate #1: 15.6%, terbaik) |
| `canada_athabasca` **(BARU)** | (-111.75, 56.95, -111.30, 57.40) | Athabasca oil sands, Alberta — footprint tambang terluas dunia |
| `australia_bowenbasin` **(BARU)** | (148.00, -22.00, 148.50, -21.50) | Bowen Basin (Goonyella-Peak Downs), kluster batubara terpadat Qld |
| `brazil_carajas` **(BARU)** | (-50.40, -6.20, -50.00, -5.80) | Carajas (Vale S11D), kompleks bijih besi terbesar dunia |

**Permukiman gel.3** (`permukiman_zzz_`, bbox 0.35-0.40°, kota provinsi besar BARU — belum
dipakai di `TRANSFER_REGIONS` maupun gel.1; gap terkecil di gate #1, hanya tambah 1 kota):

| Region | BBox | Keterangan |
|---|---|---|
| `bogor_depok` | (106.65, -6.75, 107.05, -6.35) | Bogor-Depok metro (gate #1: 47.8%, terbaik) |
| `padang` | (100.20, -1.15, 100.60, -0.75) | Padang, ibu kota Sumbar |
| `banjarmasin` | (114.40, -3.50, 114.80, -3.10) | Banjarmasin, ibu kota Kalsel (gate #1: 10.8%) |
| `pontianak` | (109.10, -0.20, 109.50, 0.20) | Pontianak, ibu kota Kalbar |
| `manado` | (124.65, 1.30, 125.05, 1.70) | Manado, ibu kota Sulut |
| `yogyakarta` **(BARU)** | (110.25, -8.00, 110.60, -7.65) | Yogyakarta metro (Kartamantul), kota besar baru |

**Sawit BARU** (`sawit_zz_`, bbox ~0.40°, Sumatra & Kalimantan, distrik BEDA dari 4 region
`TRANSFER_REGIONS['sawit']` (riau_pelalawan, sumut, kalbar, kalteng); `jambi_tebo` &
`kalsel_kotabaru` dibuang krn 8.4% (lemah) — gap terbesar di gate #1, tambah 4 sabuk sawit baru):

| Region | BBox | Keterangan |
|---|---|---|
| `sumsel_musibanyuasin` | (103.80, -2.85, 104.20, -2.45) | Sumsel, Musi Banyuasin (gate #1: 29.4%) |
| `riau_rokanhilir` | (100.90, 1.50, 101.30, 1.90) | Riau, Rokan Hilir (gate #1: 48.1%, terbaik) |
| `riau_kampar` **(BARU)** | (101.00, 0.00, 101.40, 0.40) | Riau, Kampar — sentra sawit industri (Asian Agri dll) |
| `riau_indragirihilir` **(BARU)** | (102.80, -0.70, 103.20, -0.30) | Riau, Indragiri Hilir — sawit gambut terbesar |
| `sumut_labuhanbatu` **(BARU)** | (99.90, 1.85, 100.30, 2.25) | Sumut, Labuhanbatu — sabuk sawit historis (PTPN/Lonsum) |
| `kalbar_ketapang` **(BARU)** | (110.20, -1.80, 110.60, -1.40) | Kalbar, Ketapang — konsesi sawit besar |

Ketiganya digabung ke `GEL3_REGIONS = {slug: (prefix, regions, target_class), ...}`.

### Bagian 12.9b — EDA/GATE pasca-fusion (gaya Bagian 6 & 12.1b)

Untuk **setiap region** di `GEL3_REGIONS`:
- `s2_composite(T2, box)` → `build_label(box, T2, composite=img)`.
- `label.reduceRegion(ee.Reducer.frequencyHistogram(), box, scale=100m)` → persentase piksel
  kelas target di region tersebut.
- Estimasi kontribusi piksel native = luas BBox (px) × persentase kelas target.
- **Gate per region**: `>=1%` piksel kelas target (peringatan saja, tidak `raise` — pola 12.1b).
- **Verdict per kelas**: jumlah estimasi semua region per slug dibandingkan target tambahan
  (Tambang +20jt, Permukiman +20jt, Sawit +40jt) → `[cukup]` / `[mungkin kurang]`.
- Hasil disimpan ke `Distribution_Reports/pixel_estimate_gel3.json`.

Jika ada `[X]` atau `[mungkin kurang]`, **revisi `GEL3_REGIONS` di 12.9a** (ganti/tambah region)
SEBELUM menjalankan 12.9c.

### Bagian 12.9c — Export ADDITIF GABUNGAN (1 cell, 3 label)

Loop tunggal `for slug, (prefix, regions, _) in GEL3_REGIONS.items(): for name, bbox in regions: ...`
— pola ekspor identik 12.6c/12.7c/12.8c (`s2_composite` → `build_label` → `addBands` →
`make_tiles(nx=2, ny=2)` → `export_tiles_grid(name_prefix=f'{prefix}{name}_tile', folder=slug, ...)`).

19 region × 4 tile (2×2) = **76 task export**, ke 3 folder (`tambang`, `permukiman`, `sawit`).

---

## 4. Urutan menjalankan (TUTORIAL)

> Prasyarat: sudah pernah jalan sampai Bagian 12.8 (folder transfer siap, termasuk `sawit`),
> `TRANSFER_REGIONS`, `s2_composite`, `build_label`, `save_json`, `DIST_DIR`, dan `cfg`/`T2`
> sudah ter-load di sesi.

1. **12.9a** → cek 19 region tercetak (7 tambang + 6 permukiman + 6 sawit, set v2).
2. **12.9b** → lihat tabel `% kelas target` per region + **verdict per kelas**
   (`[cukup]`/`[mungkin kurang]`). Jika masih ada `[mungkin kurang]` atau region dengan `[X]`,
   revisi/ganti region di 12.9a (set v3, dst.), lalu ulangi 12.9b.
3. **12.9c** → submit **76 task** ekspor gabungan (Tambang+Permukiman+Sawit) ke folder masing-masing.
4. **Tunggu** SEMUA task berstatus **`COMPLETED`** di
   <https://code.earthengine.google.com/tasks>.
5. **Jalankan ULANG** `Bagian 12.3b` — pindahkan `.tif` nyasar dari `Augmented_Patches/<slug>/`
   ke `ForestWatch_Tiles_Transfer/<slug>/` (sudah meng-handle `tambang`, `permukiman`, `sawit`).
6. **Jalankan ULANG** `Bagian 12.5` — cut patches FASE 2. Tile lama **`SKIP - komplit`**, tile
   baru (`tile_0xx` index baru) dipotong jadi patch.
7. **Jalankan ULANG** `13.1 → 13.2 → 13.2b` — lihat distribusi Transfer & TOTAL terbaru,
   bandingkan dengan target tambahan di atas.

---

## 5. Catatan penting

- **T2 = 2025** (`cfg['periods']['t2']`) — sama seperti semua ekspor sebelumnya, dataset identik.
- **Tidak ada data lama dihapus / di-export ulang.** Hanya menumpuk tile baru.
- `TRANSFER_REGIONS` **tidak diubah** — region Gelombang 3 murni hidup di `GEL3_REGIONS` (12.9a),
  terpisah dari baseline.
- **File `.tif` boleh 100-200MB+** — tidak masalah selama label target ada (dikonfirmasi via 12.9b).
- **Sync wajib:** notebook dijalankan di Colab dari clone repo `forestwatch-model`. Sebelum run,
  pastikan notebook terbaru sudah ter-sync (push notebook + `git pull`/clone ulang di Colab).
- Jika setelah 13.2b Tambang/Permukiman/Sawit masih kurang dari target, tambah region lagi dengan
  pola yang sama (prefix urut di akhir: `tambang_zzzzz_`, `permukiman_zzzz_`, `sawit_zzz_`).
- **Catatan `australia_huntervalley`:** bbox ini beririsan sebagian (~56% area) dengan
  `TRANSFER_REGIONS['tambang']['aus_huntervalley']` (sudah diekspor di Bagian 11). Tidak fatal —
  data yang diekspor tetap valid & menambah piksel Tambang nyata di 13.2b — tapi sebagian dari
  estimasi 15.6% kemungkinan "menghitung ulang" piksel yang sudah ada di dataset. Dipertahankan
  di v2 karena ini region berdensitas terbaik di gate #1 dan gap Tambang masih besar.
