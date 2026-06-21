# Ekspor Tambahan: Tambang (gel.2) + Permukiman — ADDITIF

> Catatan eksekusi untuk menutup gap distribusi piksel **Tambang** & **Permukiman**
> di `model/notebooks/forestwatch_papua_full_pipeline.ipynb` (Bagian **12.7** & **12.8**).
> Semua **ADDITIF**: tidak ada data lama yang dihapus atau di-export ulang.

---

## 1. Kenapa ada bagian ini? (gap distribusi)

Distribusi TOTAL (Papua + Transfer) terakhir vs target (Tambang 50jt, lainnya 100jt):

| Kelas | Total | % target | Status |
|---|---|---|---|
| Lahan Terbuka | 110.7 jt | 110.7% | ✅ aman |
| Pertanian Lain | 234.9 jt | 234.9% | ✅ aman |
| Sawit | 67.6 jt | 67.6% | ✅ aman (≥50%) |
| **Tambang** | 20.4 jt | **40.8%** | ⚠️ kurang ~4.6 jt px ke 50% |
| **Permukiman** | 29.4 jt | **29.4%** | ❌ kurang ~20.6 jt px ke 50% (terlemah) |

Tujuan: tambah data **Tambang** & **Permukiman** sampai mendekati/melewati ambang 50% target,
**tanpa menyentuh** data/region yang sudah ada (Sawit/Lahan Terbuka/Pertanian Lain sudah aman,
tidak disentuh).

---

## 2. Prinsip ADDITIF (kenapa aman, tidak merusak yang lama)

- Ekspor ke folder Drive **yang SAMA** per-kelas (`folder='tambang'` / `folder='permukiman'`),
  **bukan** folder baru.
- Nama tile diberi **prefix yang urut alfabetis di AKHIR** tile lama:
  - Tambang: lama `tambang_{nama}_*` < gel.1 `tambang_zz_*` < **gel.2 `tambang_zzz_*`** (3 huruf z).
  - Permukiman: lama `permukiman_{nama}_*` (b/j/m/s) < **baru `permukiman_zz_*`** (z).
- Karena `cut_patches_resilient` (Bagian 12.5) memberi index `tile_000, tile_001, ...` menurut
  **urutan alfabetis** file `.tif`, tile baru selalu dapat index BARU di akhir → tile lama
  **tidak tergeser** dan **di-skip** lewat penanda `_DONE` saat 12.5 dijalankan ulang.
- **Dataset identik** dengan ekspor lama: semua cell memakai `s2_composite()` + `build_label()`
  yang memfusikan 6 sumber yang sama (Sentinel-2, ESA WorldCover, Dynamic World, Hansen GFC,
  FDP Palm, Global Mining Footprint). Tidak ada sumber/preprocessing yang berbeda.

---

## 3. Cell baru yang ditambahkan

### Bagian 12.7 — TAMBANG tambahan (gelombang 2)
- **12.7a** — definisi `TAMBANG_EXTRA2_REGIONS`: 3 open-pit besar, bbox ~0.2–0.3°:
  - `papua_grasberg` (Grasberg/Tembagapura — **IN-DOMAIN Papua**, emas-tembaga)
  - `peru_antamina` (tembaga/seng)
  - `safrica_witbank` (Mpumalanga, batubara strip-mine)
- **12.7b** — preview 8 sample/region (RGB | label) + cetak **% piksel Tambang/region**.
  Cek **Tambang dominan**; kalau ada region ~0%, hapus dari `TAMBANG_EXTRA2_REGIONS` di 12.7a.
- **12.7c** — submit task GEE export (prefix `tambang_zzz_`, folder `tambang`, tile 2×2).

### Bagian 12.8 — PERMUKIMAN tambahan
- **12.8a** — definisi `PERMUKIMAN_EXTRA_REGIONS` = dua grup:
  - **Grup A (kota besar BARU, built-up dominan, penyumbang utama gap):**
    `palembang`, `semarang`, `denpasar`, `balikpapan`, `pekanbaru`.
    (Makassar & Medan **tidak diulang** — sudah ada di `TRANSFER_REGIONS`.)
  - **Grup B (permukiman KECIL 3T, bbox ketat ~0.12°):**
    `papua_jayapura`, `papua_merauke`, `papua_timika`, `papua_sorong`, `papua_biak` (semua **IN-DOMAIN**),
    `maluku_ambon`, `ntt_kupang`.
- **12.8b** — preview 8 sample/region + cetak **% piksel Permukiman/region**.
- **12.8c** — submit task GEE export (prefix `permukiman_zz_`, folder `permukiman`, tile 2×2).

> **Soal "label target harus dominan":** untuk **kota besar** terpenuhi (built-up padat).
> Untuk **permukiman kecil 3T**, Permukiman **tidak** akan dominan — kota kecil dikelilingi
> hutan/vegetasi. Ini **disengaja** demi realisme wilayah 3T Papua. Mitigasi: bbox dibuat
> **ketat** (bukan raksasa) supaya built-up tetap fraksi yang berarti, dan **12.8b melaporkan
> %/region** agar region yang built-up-nya ~0% bisa dibuang sebelum export.

---

## 4. Urutan menjalankan (TUTORIAL)

> Prasyarat: sudah pernah jalan sampai Bagian 12 (folder transfer siap), `TRANSFER_REGIONS`,
> `s2_composite`, `build_label`, helper 11.0, dan `cfg`/`T2` sudah ter-load di sesi.

1. **12.7a** → cek 3 region tambang tercetak.
2. **12.7b** → lihat preview & **% Tambang/region**. Buang region yang ~0% bila ada.
3. **12.7c** → submit export tambang gel.2.
4. **12.8a** → cek daftar region permukiman (5 besar + 7 kecil).
5. **12.8b** → lihat preview & **% Permukiman/region** (kota besar dominan; kecil cukup > 0%).
6. **12.8c** → submit export permukiman.
7. **Tunggu** SEMUA task (12.7c + 12.8c) berstatus **`COMPLETED`** di
   <https://code.earthengine.google.com/tasks>.
8. **Jalankan ULANG** `Bagian 12.3b` — pindahkan `.tif` nyasar dari `Augmented_Patches/<slug>/`
   ke `ForestWatch_Tiles_Transfer/<slug>/` (sudah meng-handle `tambang` & `permukiman`).
9. **Jalankan ULANG** `Bagian 12.5` — cut patches FASE 2. Tile lama akan **`SKIP - komplit`**,
   tile baru (`tile_0xx` index baru) dipotong jadi patch.
10. **Jalankan ULANG** `13.1 → 13.2 → 13.2b` — lihat distribusi Transfer & TOTAL terbaru.

---

## 5. Catatan penting

- **Tidak ada data lama dihapus / di-export ulang.** Hanya menumpuk tile baru.
- **File `.tif` boleh 100MB+** — tidak masalah selama label target ada (dikonfirmasi via preview).
- **Sync wajib:** notebook dijalankan di Colab dari clone repo `forestwatch-model`. Sebelum run,
  pastikan notebook terbaru sudah ter-sync (push notebook + `git pull`/clone ulang di Colab),
  jika tidak cell 12.7/12.8 tidak akan muncul di sesi Colab.
- Jika setelah 13.2b Tambang/Permukiman masih < 50%, tinggal tambah region lagi dengan pola
  yang sama (prefix urut di akhir: tambang → `tambang_zzzz_`, permukiman → `permukiman_zzz_`).
