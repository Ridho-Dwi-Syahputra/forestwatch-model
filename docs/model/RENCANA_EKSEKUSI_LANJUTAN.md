# 🧭 RENCANA EKSEKUSI LANJUTAN — Model ForestWatch Papua

**Versi:** 8.0 · **Status:** Rencana aktif (menggantikan rencana Bagian 11–16 di PROGRESS_SAAT_INI v5.0)
**Untuk:** Lead Data Scientist (USER) & AI Agent penerus
**Prinsip:** jujur, bertahap, modular — **JANGAN rusak Bagian 1–10 yang sudah mapan.**

> Dokumen ini menyatukan & **mengoreksi** dokumen sebelumnya:
> `PROGRESS_SAAT_INI.md` (v5.0), `FUTURE BEST PLAN.md`, `SARAN_TAMBAHAN_PIXEL_PATCH.md`,
> `RECOVERY_LABEL_ANOMALI.md`. `MASTER_PLAN.md` sudah **USANG** (skema 6 kelas/2024).

---

## 0. Kondisi Terkini (Angka Kebenaran — single source of truth)

**Notebook nyata (64 sel, KONDISI SAAT INI):** Bagian 0–10 (pipeline + EDA + label fusion DW-base) ✅ ·
**10B Data Healing NDVI** ✅ · 11 Training · 12 Evaluasi · 13 Inferensi · 14 Output.
Transfer-learning **belum ada di notebook** (baru rencana).

**Struktur notebook TARGET** (setelah TAHAP 0–2, lihat skema penomoran §2): 0–10 + 10B (tetap) → **11 EDA Pra-Ekspor Transfer (BARU)** → **12 Ekspor+Fusion+Cut Transfer (BARU)** → **13 EDA Pasca-Fusion + Grid 448-sample / GATE 1 (BARU)** → **14 Split+Aug Offline+Reweight+Verifikasi Final / GATE 2 (BARU)** → **15 Training** (dulu 11) → **16 Evaluasi** (dulu 12) → **17 Inferensi** (dulu 13) → **18 Output** (dulu 14).

**Skema final 7 kelas** (sudah benar di `constants.py`):
`0 Perairan · 1 Hutan · 2 Lahan Terbuka · 3 Sawit · 4 Pertanian Lain · 5 Tambang · 6 Permukiman`.
Catatan: **"Deforestasi" = PERISTIWA** (transisi Hutan→X di deteksi perubahan), **bukan kelas**.

**Distribusi piksel — PAKAI angka pasca-healing sebagai acuan terbaru** (dari 193.453 patch Papua):

| Kelas | Pasca-Healing (ACUAN) | Pra-Healing (arsip) | Catatan |
|---|---|---|---|
| Perairan (0) | ~66,4% | 72,08% | turun karena rawa dikembalikan |
| Hutan (1) | ~32,0% | 26,36% | naik (hutan rawa pulih) |
| Pertanian Lain (4) | ~0,66% | 0,66% | |
| Lahan Terbuka (2) | ~0,47% | 0,47% | |
| Sawit (3) | ~0,30% | 0,30% (~39 jt px) | cukup |
| Permukiman (6) | ~0,09% | 0,091% (~11,6 jt px) | kurang |
| **Tambang (5)** | **~0,002–0,003%** | **0,0027% (~343 rb px ≈ 5 patch)** | **⚠️ KRITIS** |

**Yang sudah ada di kode (jangan dibuat ulang):**
- `dataset.py`: **WeightedRandomSampler + median-frequency class weights + augmentasi online (albumentations)**. → Augmentasi online akan **DIMATIKAN** (diganti offline; lihat §1.1). Logika transform-nya bisa **dipakai ulang** untuk skrip augmentasi offline.
- `trainer.py`: Focal-Tversky loss, transfer-learning 2-tahap (freeze→unfreeze), gradient clipping, **resume checkpoint**, IoU per-kelas tiap epoch.
- Label fusion DW-base + overlay (Hansen→Lahan Terbuka, Mining union TW+Maus→Tambang, FDP Palm→Sawit).
- Bagian 10B: healing NDVI (hutan rawa yang salah jadi air → dikembalikan ke Hutan).

---

## 1. KOREKSI PENTING atas dokumen lama (baca sebelum eksekusi)

### 1.1 ✅ Augmentasi OFFLINE (keputusan USER) — yang dikoreksi hanya TARGET-nya
**Keputusan:** pakai **augmentasi OFFLINE** (patch hasil augmentasi disimpan jadi `.npz`, lalu DataLoader tinggal membaca). **Augmentasi ONLINE DIMATIKAN.**
**Alasan (valid):** augmentasi online berjalan di **CPU DataLoader**; di Colab (CPU ~2 core) ini jadi *bottleneck* → GPU menunggu data → training lambat. Offline = transform dihitung sekali di depan, training tinggal baca file → GPU tetap kenyang, iterasi lebih cepat.

**Yang DIKOREKSI dari PROGRESS v5.0 = "1–2 miliar" via DUPLIKASI; target "250 juta" DIPERTAHANKAN tapi sebagai BASIS RAW (TAHAP 1), bukan hasil augmentasi:**
- ❌ **Versi PROGRESS v5.0 lama:** terapkan "250 juta" ke RAW Papua-only (mis. Tambang 343K) lalu "ledakkan ke 1–2 miliar" via augmentasi → multiplier ~3000–6000× → **menggandakan 5 situs tambang yang sama** ribuan kali → overfitting parah.
- ✅ **KEPUTUSAN BARU (USER, skala penuh):** **TAHAP 1 (Transfer Learning) membawa RAW tiap kelas 2–6 ke ~250 juta px** — basis ini berasal dari **data NYATA & BERAGAM** (banyak region di luar Papua), bukan duplikasi. **TAHAP 2 (Augmentasi Offline)** baru diterapkan SETELAHNYA di atas basis 250 juta ini → hasil final "beda lagi" (lebih besar), dengan multiplier kecil karena basisnya sudah besar.

**Tabel target RAW pasca-transfer (TAHAP 1) — kelas 2–6:**

| Kelas | Raw Papua sekarang | Target Raw pasca-Transfer | Tambahan via Transfer |
|---|---:|---:|---:|
| Lahan Terbuka (2) | 60.040.158 | 250.000.000 | +189,96 jt |
| Sawit (3) | 39.037.196 | 250.000.000 | +210,96 jt |
| Pertanian Lain (4) | 84.689.947 | 250.000.000 | +165,31 jt |
| Permukiman (6) | 11.617.443 | 250.000.000 | +238,38 jt |
| **Tambang (5)** | 343.406 | 250.000.000 | +249,66 jt |
| **TOTAL tambahan** | | | **≈1,05 miliar px** (~16.000 patch, ~25 GB raw) |

- TAHAP 2 (augmentasi offline) diterapkan **DI ATAS** basis 250 juta ini, dengan **multiplier awal kecil (×2, placeholder — disesuaikan setelah tahu kapasitas storage/compute riil pasca-TAHAP 1)**. Estimasi final per kelas ≈ **~500 juta px** (bisa naik jadi ×3 ≈ ~750 juta bila storage memungkinkan).
- ✅ Setelah TAHAP 2, **`class_distribution.json` & class weights di-RECOMPUTE** dari distribusi baru → bobot Tambang otomatis turun dari ~4,93 ke level lebih moderat (mencegah *double-counting*: basis raw sudah besar + loss-weight ekstrem sekaligus).
- 📊 **Dampak komposisi total (estimasi @×2):** 5 kelas minoritas gabungan naik dari **~1,54%** → **~16–17%** dari total dataset (≈15 miliar px). Ini **masih jauh dari parity** (14,3%/kelas bila 7 kelas sama rata = ~71,5% gabungan utk 5 kelas), tapi kenaikan signifikan (~11×) — **wajib dipantau di TAHAP 4** untuk false-positive rate (lihat §4).
> Ringkas: **250 juta = lantai RAW (TAHAP 1, dari data nyata beragam luar Papua)**; augmentasi offline (TAHAP 2) jadi lapisan TAMBAHAN di atasnya, bukan satu-satunya sumber volume — sehingga model tidak overfit ke segelintir situs Papua.

### 1.2 ✅ Filosofi imbalance yang BENAR (dari SARAN_TAMBAHAN)
Kelas **tidak perlu** disamakan 14% rata. *Real-world prior* penting: memaksa Tambang setara Hutan → **over-sensitivity → false positive massal** (lapangan bola ditebak tambang). Focal-Tversky (β=0,7) + class weights + sampler sudah memberi prioritas ke kelas langka.

### 1.3 ⚠️ Bug akar `DW_TO_CLASS[3]=0` masih ada di sumber
`constants.py` masih memetakan **Flooded Vegetation (DW 3) → Perairan (0)**. Hutan rawa Papua (Asmat/Mamberamo) sempat jadi "air"; sudah di-heal **pasca-hoc** di patch Papua (NDVI>0,3 → Hutan), TAPI **mapping sumber belum diperbaiki** → akan **terulang** saat memotong patch transfer (Kalimantan/Sumatra/Jawa).
> **Aksi (TAHAP 0):** perbaiki di sumber — opsi (a) ubah `DW_TO_CLASS[3] = 1`, atau (b) bake aturan NDVI ke `build_label`/proses cut sehingga healing otomatis berlaku untuk **semua region**. Jangan andalkan healing manual per-region.

### 1.4 ⚠️ Domain shift (latih pakai data luar Papua)
Sah sebagai *domain generalization*, asalkan:
- **Test-set Papua-only (holdout)** untuk pelaporan IoU — angka yang dipakai di esai HARUS dari Papua, bukan data Kalimantan/Jawa.
- Augmentasi tekstur/radiometrik ringan → model belajar **bentuk/morfologi**, bukan menghafal warna tanah.
- Morfologi: tambang Kalimantan = batubara dataran rendah (beda dari Grasberg = tembaga *open-pit* dataran tinggi). Keragaman bagus, tapi jangan over-andalkan satu lokasi; idealnya 2–3 lokasi.

### 1.5 🗂️ Dokumen lain
- `MASTER_PLAN.md` **USANG** (6 kelas, T2=2024, Lahan Terbakar, BIOPAMA, food estate) → ditandai usang.
- `FUTURE BEST PLAN.md` & `SARAN_TAMBAHAN_PIXEL_PATCH.md` → **prinsipnya tetap rujukan** (real-world prior, multi-region, selective cutting, domain-shift mitigation), tapi **skala numerik (~750 patch/20-30MB) DI-OVERRIDE** oleh keputusan USER ke ~250 jt px raw/kelas (§1.1) — ~20× lebih besar.
- Tambang tetap akan jadi kelas terlemah meski di-enrich — **laporkan jujur** (nilai plus di mata juri).

---

## 2. ROADMAP PER-TAHAP

**Aturan tetap:** inkremental & modular. Isi Bagian 0–10 + 10B tidak diubah (lihat catatan renumbering di bawah untuk Bagian 11–14 lama).

**Skema penomoran notebook (RENUMBERING — keputusan USER, menggantikan skema sub-huruf 10C/10D sebelumnya):**
4 Bagian BARU (hasil TAHAP 1 & TAHAP 2) disisipkan **berurutan setelah 10B**, diberi nomor **11–14**. Bagian lama 11–14 (Training/Evaluasi/Inferensi/Output) **DIGESER** menjadi **15–18** — isi sel dipindah apa adanya, hanya isi yang diedit sesuai TAHAP 3–6.

| Nomor | Isi | Tahap |
|---|---|---|
| 0–10 | Pipeline + EDA + Label Fusion Papua | tidak berubah |
| 10B | Healing NDVI Papua | tidak berubah |
| **11 (BARU)** | **EDA Pra-Ekspor** Wilayah Transfer (per region, gaya Bagian 1–6) | TAHAP 1a |
| **12 (BARU)** | Ekspor GEE + Label Fusion + Cut Patches Transfer | TAHAP 1b |
| **13 (BARU)** | **EDA Pasca-Fusion** Transfer: distribusi piksel + grid 448-sample (**GATE 1**) | TAHAP 1c |
| **14 (BARU)** | Split Jujur + Augmentasi Offline + Re-weight + **Verifikasi Final** (**GATE 2**) | TAHAP 2 |
| **15** (dulu 11) | Training | TAHAP 3 — isi diedit |
| **16** (dulu 12) | Evaluasi | TAHAP 4 — isi diedit |
| **17** (dulu 13) | Inferensi + Deteksi Perubahan | TAHAP 5 — isi nyaris sama |
| **18** (dulu 14) | 7 File Kontrak + Output | TAHAP 6 — isi diedit |

**Aksi struktural (bagian dari TAHAP 0, dilakukan SEKALI di awal):**
1. Geser seluruh sel Bagian 11→15, 12→16, 13→17, 14→18 (judul + kode, isi tidak diubah dulu).
2. Siapkan 4 slot judul placeholder Bagian 11–14 baru (kosong, diisi TAHAP 1 & 2).
3. **Update referensi silang**: catatan di Bagian 0 (±baris 849) yang menyebut *"...dihitung di deteksi perubahan (Bagian 13)..."* → ubah jadi **"(Bagian 17)"** (Inferensi & Deteksi Perubahan, dulu Bagian 13).

### TAHAP 0 — Konsolidasi, Perbaikan Akar & Reorganisasi Notebook
**Tujuan:** fondasi bersih + slot notebook siap, sebelum menambah data.
**Langkah:**
1. Perbaiki akar Flooded-Veg (§1.3) di `src/forestwatch/constants.py` / `gee/label_fusion.py` (pilih opsi a/b). Tambah tes unit kecil.
2. Pastikan logika healing dapat dipakai ulang untuk region transfer (jadikan fungsi, bukan sel ad-hoc).
3. Kunci "angka kebenaran terkini" (tabel §0) di dokumen ini.
4. **Reorganisasi notebook** (lihat tabel skema penomoran di atas): geser Bagian 11→15, 12→16, 13→17, 14→18 (isi sel dipindah apa adanya); siapkan 4 judul placeholder Bagian 11–14 baru (kosong, diisi TAHAP 1 & 2). Update referensi silang "Bagian 13" (±baris 849, Bagian 0) → "Bagian 17".
5. Buat fungsi reusable `plot_class_dominant_grid(patches, classes=7, n_per_class=64)` (mis. di `src/forestwatch/eda_utils.py`) — menghasilkan 1 grid 8×8 (RGB + label overlay, 64 sample) per kelas yang punya ≥64 patch dominan. Dipakai ulang di Bagian 13 (GATE 1) & Bagian 14 (GATE 2).
**File:** `constants.py`/`label_fusion.py` (+ test); `eda_utils.py` (fungsi grid baru); notebook (reorganisasi sel 11-14→15-18 + 4 slot baru); `MASTER_PLAN.md` (banner usang).
**DoD:** patch baru region mana pun otomatis bebas bug rawa; pytest hijau; Bagian 0–10+10B isinya tak berubah; Bagian 15–18 berisi sel lama (Training/Eval/Inferensi/Output) tanpa kehilangan kode/output; referensi "Bagian 13"→"Bagian 17" terupdate; `plot_class_dominant_grid` siap pakai (uji coba dengan data Papua existing).
**Risiko:** mengubah mapping → distribusi sedikit bergeser (wajar, hanya pengaruhi flooded-veg). Reorganisasi notebook → risiko salah pindah sel/output ter-reset; mitigasi: pindahkan satu Bagian per langkah, jalankan ulang & verifikasi sebelum lanjut ke Bagian berikutnya.

### TAHAP 1 — Transfer-Learning Data Enrichment SKALA BESAR (Bagian 11–13 BARU, target raw 250 juta px/kelas, 2–6)
**Tujuan:** bawa RAW tiap kelas 2–6 dari level Papua-saja ke **~250 juta px** (lihat tabel §1.1) via ekspor GEE multi-region di luar Papua + patch cutting — dengan **dua gerbang EDA** (sebelum & sesudah ekspor) agar kualitas data terverifikasi visual sebelum lanjut. Ini **~20× lipat** dari rekomendasi awal SARAN_TAMBAHAN (~750 patch/20–30 MB) — **TAHAP 1 adalah critical path terbesar di seluruh roadmap** dan sebaiknya **dimulai paling pertama / paralel dengan TAHAP 0** karena antrian task ekspor GEE bisa berjalan multi-hari di background.

**1.1 Strategi ROI — MULTI-REGION per kelas (bukan 1 BBox kecil):**
Karena gap sebesar ini (total ~1,05 miliar px / ~16.000 patch), satu hotspot 30–40 km tidak cukup. Gabungkan beberapa region per kelas:
- **Tambang (5)** — gap terbesar (+249,7 jt): Kalimantan Timur (sabuk batubara Sangatta/Kutai Timur/Berau) + Kalimantan Selatan (Tanah Bumbu/Tanah Laut) + Sulawesi Tengah (Morowali, nikel — variasi morfologi, mitigasi warna via augmentasi radiometrik ringan).
- **Permukiman (6)** — gap +238,4 jt: Jabodetabek + Bandung Raya + Surabaya Raya + Medan + Makassar (gabungan kota besar agar densitas tinggi).
- **Sawit (3)** — gap +211,0 jt: Riau + Sumatra Utara + Kalimantan Barat + Kalimantan Tengah (sabuk sawit utama Indonesia, area realistis tersedia).
- **Lahan Terbuka (2)** — gap +190,0 jt: tepi-tepi tambang/perkebunan baru (Kalimantan & Sumatra) + area pasca-kebakaran/pembukaan lahan.
- **Pertanian Lain (4)** — gap +165,3 jt: dataran rendah persawahan Jawa (Pantura) + ladang Sumatra.

**1.2 (Bagian 11 BARU) — EDA Pra-Ekspor per Region (gaya Bagian 1–6, SEBELUM submit ekspor GEE penuh):**
Tujuan: lihat dulu data mentahnya — sama seperti Bagian 1–6 untuk Papua — supaya tidak buang kuota/waktu ekspor GEE untuk AOI yang ternyata berawan/datanya jelek/labelnya aneh.
- Sub-bagian **11.1–11.5** (satu per kelompok kelas/region di §1.1: Tambang, Permukiman, Sawit, Lahan Terbuka, Pertanian Lain).
- Per sub-bagian: definisikan BBox kandidat → ambil 2–3 sample tile representatif (bukan seluruh BBox) → tampilkan **true color, false color, NDVI, cloud probability** (sama seperti visualisasi Bagian 1–6) → jalankan `build_label` (versi sudah-fix TAHAP 0) pada sample tile → tampilkan overlay label untuk cek cepat (terutama bug rawa & kewajaran kelas target).
**DoD 11:** tiap region punya ≥1 preview visual (citra + NDVI + label overlay) yang **direview USER** sebelum lanjut ke ekspor penuh Bagian 12. Region dengan tutupan awan tinggi/data jelek → ganti BBox sebelum lanjut.

**1.3 (Bagian 12 BARU) — Ekspor GEE + Label Fusion + Cut Patches:**
1. Definisikan BBox skala kabupaten/provinsi per region (jauh lebih besar dari sample preview Bagian 11).
2. Ekspor `s2_composite` + `build_label` **yang sama persis** dengan Papua (sudah bebas bug rawa setelah TAHAP 0) → `ForestWatch_Tiles_Transfer/<region>/`.
3. **Patch cutting threshold DILONGGARKAN**: dari `>5%` (FUTURE BEST PLAN) menjadi `>1–2%` atau bahkan ambil semua patch dalam ROI inti — supaya volume cukup untuk capai target. Densitas tinggi tetap diprioritaskan saat sampling/augmentasi di TAHAP 2.
4. Simpan ke `ForestWatch_Patches_Transfer/<region>/`, tandai metadata sumber (`source: transfer`, `region: <nama>`).

**1.4 (Bagian 13 BARU) — EDA Pasca-Fusion + Grid 448-Sample (GATE 1):**
Tujuan: setelah cut patches, pastikan datanya **siap** sebelum di-merge ke pool utama — pola sama dengan EDA pasca-preprocessing Papua (Bagian 10), diterapkan ke patch transfer.
- **Distribusi piksel**: hitung ulang per region & gabungan, bandingkan ke target tabel §1.1 (gap closure %).
- **Visual grid** (`plot_class_dominant_grid`, fungsi dari TAHAP 0 langkah 5): untuk tiap kelas yang punya ≥64 patch dominan (target: kelas 2–6, idealnya juga 0–1 jika ada di pool transfer) → 1 grid **8×8 = 64 sample** (RGB + label overlay) → hingga **7 gambar = 448 sample**.
- Review manual: cek label masuk akal per kelas (mis. grid "Tambang" benar-benar tambang bukan jalan/awan, grid "Permukiman" benar-benar permukiman, dst.), cek ulang bug rawa di region baru.
**GATE 1:** ⛔ **lanjut ke Bagian 14 (TAHAP 2) hanya setelah USER review & approve grid + distribusi ini.**

**File:** notebook **Bagian 11–13 (BARU)**, disisipkan setelah 10B (lihat skema penomoran §2); folder Drive `ForestWatch_Tiles_Transfer/<region>/` + `ForestWatch_Patches_Transfer/`.
**DoD:** raw tambahan per kelas mendekati target tabel §1.1 (**toleransi ±20%** mengingat tekanan waktu — capaian parsial tetap dicatat & dilaporkan jujur); patch terorganisir per region; **GATE 1 (Bagian 13) di-approve** sebelum TAHAP 2.
**Risiko (BESAR):**
- **Kuota & antrian task GEE** untuk ~16.000 patch-equivalent (puluhan—ratusan task export) → mitigasi: mulai SEGERA, jalankan paralel per region, jangan tunggu satu region selesai sebelum memulai region lain.
- **Storage Drive** (~25 GB+ raw .tif sebelum cut) → mitigasi: hapus `.tif` mentah setelah berhasil di-cut ke `.npz` terkompresi.
- **Domain shift** makin dominan karena mayoritas piksel minoritas kini dari luar Papua → mitigasi lanjut di TAHAP 2/3 (test tetap Papua-only).
- **Waktu vs deadline 19 hari** → jika mendekati deadline TAHAP 1 belum 100%, **prioritaskan kelas dengan dampak terbesar (Tambang, Permukiman)** dan terima capaian parsial untuk Lahan Terbuka/Sawit/Pertanian Lain — laporkan apa adanya di TAHAP 6.
- **Grid 448-sample tidak penuh**: kelas tertentu (mis. Perairan/Hutan) mungkin <64 patch dominan di pool transfer-only → tampilkan yang tersedia + catat; gap akan tertutup oleh patch Papua saat merge (Bagian 14/GATE 2).

### TAHAP 2 — Split Jujur → Augmentasi OFFLINE → Merge + Re-weight + Verifikasi Final (Bagian 14 BARU)
**Tujuan:** dataset siap latih (kelas langka cukup tebal via augmentasi offline) + evaluasi adil tanpa kebocoran + **konfirmasi visual terakhir** sebelum training.
**URUTAN KRITIS (split dulu, baru augmentasi, baru verifikasi):**
1. Gabungkan patch transfer (yang sudah lolos GATE 1) ke pool (tandai asal: `papua` vs `transfer`).
2. **SPLIT dulu** train/val/test. **test = 100% Papua (holdout)**; patch transfer hanya boleh masuk **train**.
3. **Augmentasi OFFLINE — HANYA pada patch TRAIN, untuk 5 kelas minoritas** (Lahan Terbuka, Sawit, Pertanian Lain, Tambang, Permukiman — dari Papua + transfer bila ada):
   - Transform: rotasi 90/180/270, flip H/V, brightness/contrast ringan (pakai ulang logika `dataset.py`).
   - **Multiplier KECIL di atas basis ~250 jt raw dari TAHAP 1** (placeholder **×2**, naik ke ×3 bila storage memungkinkan setelah TAHAP 1 selesai) → estimasi final **~500–750 juta px/kelas** (lihat §1.1).
     > ✅ **ANGKA FINAL (terkunci & diimplementasikan)** — `augment_offline.py` (param `multiplier`
     > sbg dict per-kelas, fractional) + notebook Bagian 14.2:
     > `{Lahan Terbuka(2): ×1.5, Sawit(3): ×1, Pertanian Lain(4): ×1, Tambang(5): ×2, Permukiman(6): ×2.5}`
     > (×1.5/×2.5 = peluang ~50% patch target dapat 1 salinan ekstra). Sekaligus **fix bug
     > `list_patches()`** (`patches.py`, glob `"p*.npz"` → `"*.npz"`) — sebelumnya
     > `AUGMENTED_PATCHES` (isi `aug_*.npz`) **tidak pernah terbaca** sehingga augmentasi
     > offline inert (tidak masuk `final_train_files`/`dist_final`/class weights/sampler).
     > Bukti: `model/tests/test_augment_offline.py` (15 tes, termasuk regresi `list_patches`).
   - Simpan ke folder baru `Augmented_Patches/` sebagai `.npz` (mis. `tambang_aug_00001.npz`).
   - ⚠️ **JANGAN augmentasi patch val/test** → kalau augmented copy dari patch test bocor ke train = evaluasi tidak jujur.
4. Hitung ulang `class_distribution.json` + **median-frequency class weights** dari train (termasuk augmented).
5. (Opsional, disarankan) **stage** folder train+augmented ke disk lokal `/content` sebelum training → baca cepat (hindari bottleneck I/O Google Drive).
6. **Verifikasi Final (GATE 2)** — jalankan ulang `plot_class_dominant_grid` (fungsi sama dari TAHAP 0 langkah 5) pada **train set FINAL** (papua + transfer + augmented) → **7 gambar × 64 = 448 sample**, 1 gambar per kelas (0–6), masing-masing 64 patch yang didominasi kelas tsb. Tampilkan juga distribusi piksel akhir berdampingan dengan target §1.1.

**GATE 2:** ⛔ **lanjut ke Bagian 15 (Training) hanya setelah USER review & approve grid 448-sample + distribusi akhir ini** — inilah konfirmasi "data sudah siap di-training".

**File:** notebook **Bagian 14 (BARU)**, disisipkan setelah Bagian 13 (lihat skema penomoran §2); skrip `scripts/augment_offline.py` (baru); `data/dataset.py` (split sumber-aware + baca 3 folder: `ForestWatch_Patches`, `ForestWatch_Patches_Transfer`, `Augmented_Patches`; **augmentasi online OFF**); `class_distribution.json`.
**DoD:** test = 100% Papua tanpa augmentasi; kelas langka train tebal (ribuan patch); weights ter-update; tak ada kebocoran augmented→test; **GATE 2 di-approve USER**.
**Risiko:** kebocoran augmented (mitigasi: augment SETELAH split, hanya train) · I/O Drive lambat (mitigasi: stage ke lokal) · GATE 2 gagal (grid menunjukkan label salah/komposisi timpang) → kembali perbaiki TAHAP 1/2 sebelum training, jangan paksa lanjut.

### TAHAP 3 — Training (Bagian 15, dulu 11 — isi diedit)
**Prasyarat:** GATE 2 (Bagian 14) sudah di-approve USER.
**Tujuan:** model ResNet50-U-Net terlatih, stabil pada kelas langka.
**Langkah (Bagian 15, dulu 11) — baca patch siap-pakai (augmentasi sudah offline):**
- **Augmentasi online OFF** (DataLoader hanya membaca `.npz` Papua + transfer + augmented → GPU tidak menunggu CPU).
- Loss **Focal-Tversky** (α=0,3 β=0,7) + **class weights** (sampler opsional/ringan karena kelas langka sudah tebal pasca-augmentasi offline).
- **Transfer-learning 2-tahap** (freeze encoder ImageNet → unfreeze) + **resume checkpoint** (anti sesi Colab putus).
- Pantau **IoU per-kelas tiap epoch** (deteksi dini bila Tambang/Permukiman kolaps).
**File:** notebook Bagian 15, dulu 11 (pakai `trainer.py` apa adanya); `best_model.pt` di Drive.
**DoD:** training selesai/early-stop; `best_model.pt` tersimpan; kurva loss/mIoU + IoU per-kelas tercatat.
**Risiko:** kelas langka tetap rendah → wajar, laporkan jujur; OOM → turunkan batch/patch.

### TAHAP 4 — Evaluasi Jujur (Bagian 16, dulu 12 — isi diedit)
**Tujuan:** angka yang bisa dipertanggungjawabkan di hadapan juri.
**Langkah:** evaluasi pada **test Papua**; hitung mIoU, OA, Cohen's Kappa, **IoU/F1 per-kelas + precision per-kelas** (lihat §3 — pantauan false-positive akibat naiknya proporsi minoritas); confusion matrix (PNG). Catat Tambang/Permukiman apa adanya. Ekspor `model.onnx`.
**File:** notebook Bagian 16, dulu 12; `metrics.json`, confusion_matrix.png, `model.onnx`.
**DoD:** metrik tersimpan & konsisten; tidak ada angka fiktif; Tambang dilaporkan dengan keterbatasannya.
**Risiko:** mIoU di bawah target → opsi A/B arsitektur (`unet_scse`/`unetpp`) bila waktu cukup.

### TAHAP 5 — Inferensi Papua + Deteksi Perubahan + 7 File Kontrak (Bagian 17, dulu 13 — isi nyaris sama)
**Tujuan:** output untuk WebGIS (Orang 2) & esai (Orang 3).
**Langkah:** inferensi **khusus Papua** T1(2021) & T2(2025) — **data transfer TIDAK masuk peta/output**; deteksi perubahan **5 transisi** (Hutan→Lahan Terbuka/Sawit/Pertanian/Tambang/Permukiman); generate + **validasi schema** 7 file; hands-off ke Orang 2.
**File:** notebook Bagian 17, dulu 13; `ForestWatch_Outputs/` (landcover PNG+bounds, deforestation.geojson, statistics.json, legend.json, metrics.json, model.onnx, model_card.md).
**DoD:** 7 file valid; angka konsisten lintas file; "total deforestasi" = jumlah semua transisi; referensi internal "Bagian 17" (eks "Bagian 13", lihat TAHAP 0 langkah 4) konsisten.
**Risiko:** kebocoran data transfer ke output → pastikan pipeline inferensi hanya baca ubin Papua.

### TAHAP 6 — Finalisasi & Sinkron Esai (Bagian 18, dulu 14 — isi diedit)
**Tujuan:** dokumentasi + serah angka final.
**Langkah:** update `model_card.md` + `Buku_Panduan_v3.docx`; sinkron angka ke Orang 3 (total deforestasi, breakdown 5 transisi, top provinsi, mIoU, IoU per-kelas, studi kasus Merauke/Timika); catatan **keterbatasan jujur** (Tambang langka + domain-shift transfer, healing rawa, footprint mining ~2019).
**File:** notebook Bagian 18, dulu 14.
**DoD:** angka final terkirim; keterbatasan terdokumentasi; notebook reproducible (Bagian 0–18 berurutan tanpa nomor bentrok).

---

## 3. Yang TIDAK Dilakukan (penegasan)
- ✅ Augmentasi **offline** DIPAKAI (keputusan USER, demi kecepatan GPU).
- ✅ Volume besar (~250 jt raw/kelas + augmentasi) **DISETUJUI**, TAPI basisnya = **data nyata beragam dari transfer learning (TAHAP 1)**, bukan duplikasi ~5 patch Papua. Inilah yang membedakan dari "1–2 miliar via duplikasi" PROGRESS lama.
- ❌ Augmentasi (TAHAP 2) sebagai **satu-satunya** sumber volume — augmentasi hanya multiplier KECIL (×2–3) di atas basis raw besar, bukan ×3000–6000 dari raw kecil.
- ❌ Augmentasi pada patch val/test (hanya train).
- ❌ Memakai **konsesi tambang (1,2 juta ha)** sebagai label (mayoritas masih hutan) — tetap pakai **footprint nyata**.
- ❌ Mengubah/merusak Bagian 1–10 + 10B.
- ❌ Memakai data Papua-luar (transfer) di val/test atau di output peta.
- ⚠️ **Wajib dipantau** (bukan "tidak dilakukan", tapi konsekuensi skala baru): proporsi 5 kelas minoritas gabungan naik dari ~1,54% → **~16–17%** dari total dataset training. Pantau **precision per-kelas** di TAHAP 4 — bila false-positive melonjak, turunkan multiplier TAHAP 2 (×2→×1) atau naikkan kembali bobot kelas mayoritas.

## 4. Tabel Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| **Skala ekspor GEE ~20× lipat** (~1,05 miliar px / ~16.000 patch, multi-region) — bottleneck waktu/kuota terbesar di seluruh roadmap | Mulai TAHAP 1 SEGERA (paralel TAHAP 0); jalankan ekspor semua region bersamaan; terima capaian parsial (±20%) bila waktu mepet |
| **Storage Drive membengkak** (~25 GB+ raw sebelum cut, lebih lagi pasca-augmentasi ×2-3) | Hapus `.tif` mentah setelah di-cut ke `.npz`; gunakan kompresi `.npz` |
| Domain shift (Kalimantan/Jawa/Sumatra ≠ Papua), makin dominan krn basis besar dari luar Papua | Test Papua-only; augmentasi tekstur/radiometrik ringan; multi-region per kelas; loss Focal-Tversky |
| **Proporsi minoritas naik ~11× (1,5%→16-17%)** → potensi over-sensitivity/false positive | Pantau **precision per-kelas** (bukan cuma IoU/recall) di TAHAP 4; bila FP tinggi, turunkan multiplier TAHAP 2 ke ×1 |
| Tambang tetap lemah meski di-enrich (Grasberg = morfologi unik) | Laporkan jujur; multi-region (Kaltim+Kalsel+Sulteng) untuk variasi |
| Bug rawa terulang di patch transfer | TAHAP 0 perbaiki di sumber (mapping/NDVI) sebelum cut transfer di SEMUA region |
| Overfitting kelas langka | Basis raw besar & beragam (bukan duplikasi); multiplier aug kecil (×2-3); early stopping; per-class IoU monitor |
| Sesi Colab putus saat training/ekspor | Resume checkpoint (sudah ada); GEE export berjalan server-side (tidak terganggu disconnect) |
| Kebocoran transfer→evaluasi | Split sumber-aware; test 100% Papua; augmentasi hanya pada patch train |
| Kebocoran augmented→test | **Augmentasi SETELAH split, hanya patch train** |
| I/O Google Drive lambat saat baca dataset besar | Stage train+augmented ke disk lokal `/content` sebelum training |
| OOM (dataset jauh lebih besar) | batch 8→4 atau patch 256→128; AMP aktif; WeightedRandomSampler agar tak perlu load semua per epoch |

## 5. Definition of Done Ringkas (checklist global)
- [x] **TAHAP 0 (kode lokal SELESAI 2026-06-11):** bug rawa diperbaiki di sumber (`constants.py` NDVI_FOREST_THRESHOLD + `label_fusion.py` `heal_swamp_forest` di `build_label` & `apply_label_fusion_numpy`, wired di `run_export.py`) ✅ · pytest hijau (78 tes; label_fusion+eda_utils+constants) ✅ · notebook direorganisasi (11-14→15-18, 4 slot 11-14 baru = EDA Pra-Ekspor/Ekspor+Cut/GATE1/GATE2 disiapkan, referensi "Bagian 13"→"Bagian 17" terupdate) ✅ · `plot_class_dominant_grid` siap di `forestwatch.eda_utils` ✅. **Sisa (butuh Colab/GPU/GEE):** re-run Bagian 15-18 di Colab untuk pastikan sel lama tetap jalan pasca-reorg.
- [ ] TAHAP 1 (Bagian 11-13): EDA pra-ekspor per region (Bagian 11) di-review; raw kelas 2-6 mendekati ~250 jt px/kelas (toleransi ±20%) via transfer multi-region (Bagian 12); **GATE 1** — grid 448-sample + distribusi (Bagian 13) di-approve USER
- [ ] TAHAP 2 (Bagian 14): split dulu (test Papua-only) → augmentasi offline ×2-3 di atas basis besar (HANYA train) → weights re-computed → **GATE 2** — grid 448-sample final di-approve USER (baru boleh lanjut training)
- [ ] TAHAP 3 (Bagian 15): `best_model.pt` + kurva + IoU per-kelas
- [ ] TAHAP 4 (Bagian 16): metrik jujur (Papua test) + IoU & precision per-kelas + confusion matrix + ONNX
- [ ] TAHAP 5 (Bagian 17): 7 file kontrak valid (Papua-only) → Orang 2
- [ ] TAHAP 6 (Bagian 18): angka final + keterbatasan → Orang 3 (esai)

---

*RENCANA_EKSEKUSI_LANJUTAN v8.0 · ForestWatch Papua · SEC SATRIA DATA 2026 · Universitas Andalas*
*Menggantikan rencana Bagian 11–16 di PROGRESS_SAAT_INI v5.0. Notebook direnumber: 11-14(lama)→15-18; slot 11-14 baru = Transfer Learning+EDA (GATE 1) → Split/Aug/Reweight+Verifikasi (GATE 2). Rujukan benar: FUTURE BEST PLAN + SARAN_TAMBAHAN.*
