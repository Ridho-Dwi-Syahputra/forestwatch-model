> ⚠️ **DOKUMEN USANG (v5.0).** Rencana **BAB 3 & BAB 4** di bawah (Bagian 11–16: Setup ROI, EDA, Export, Offline Augmentation, Training Hibrida) sudah **DIGANTIKAN** oleh skema **Bagian 11–18** di [`RENCANA_EKSEKUSI_LANJUTAN.md`](RENCANA_EKSEKUSI_LANJUTAN.md) v8.0 (4 bagian baru disisipkan sebagai 11–14 dengan **GATE 1**/**GATE 2**; Training/Evaluasi/Inferensi/Output lama digeser jadi 15–18). Tiga koreksi utama:
> 1. Target augmentasi offline **BUKAN** "1–2 Miliar px/kelas via Inverse Frequency Multiplier" (BAB 3, Bagian 15) — yang benar: basis **~250 juta px/kelas RAW** dari transfer learning (Bagian 12), lalu augmentasi offline **×2–3 saja** → final **~500–750 juta px/kelas**.
> 2. Loss function tetap **Focal-Tversky** (sudah ada & teruji di `trainer.py`), **BUKAN** "Focal Loss + Dice Loss" (BAB 4, TUGAS 4).
> 3. Wajib ada **dua gerbang EDA** (GATE 1 & GATE 2, grid 448-sample = 7×64) sebelum training — **TIDAK ADA** di dokumen ini.
>
> BAB 1–2 (riwayat & status Bagian 1–10, termasuk Data Healing) **tetap akurat** dan berguna sebagai konteks. **Rencana eksekusi terkini → [`RENCANA_EKSEKUSI_LANJUTAN.md`](RENCANA_EKSEKUSI_LANJUTAN.md).** Isi BAB 3–4 di bawah dibiarkan utuh sebagai arsip historis.

# 🌟 DOKUMEN MASTER STATE & HANDOFF KOMPREHENSIF
## PROYEK FORESTWATCH PAPUA (SATRIA DATA 2026)

---

**Status Proyek:** ACTIVE | **Fase Saat Ini:** Pra-Pelatihan (Augmentasi Data Eksternal) | **Tingkat Prioritas:** KRITIS
**Target Pembaca:** AI Agent Pengganti (Next-Gen Assistant) & Lead Data Scientist (USER)
**Versi Dokumen:** 5.0 (Restrukturisasi Pipeline Bagian 11-16 & Offline Augmentation)

---

## 🛑 PERINGATAN KERAS (STRICT DIRECTIVE) 🛑
**PROGRESS BAGIAN 1 HINGGA 10 DI JUPYTER NOTEBOOK TELAH BERJALAN SANGAT LANCAR DAN SEMPURNA!**
Segala bentuk perubahan, penambahan kode, atau eksekusi *Transfer Learning* ke depannya **SAMA SEKALI TIDAK BOLEH MERUSAK ATAU MENGGANGGU** kode yang sudah mapan di Bagian 1 hingga 10. File notebook utama telah tervalidasi. Seluruh penambahan fitur (fusi data eksternal, augmentasi, dll) **WAJIB** bersifat inkremental (melanjutkan dari Bagian 11 ke atas) atau modular tanpa menghancurkan alur kerja Papua yang sudah ada. Jika Anda merusak *pipeline* Bagian 1-10, proyek ini akan gagal!

---

## BAB 1: PENDAHULUAN & KONTEKS PROYEK

Proyek **ForestWatch Papua** adalah inisiatif kompetitif untuk ajang bergengsi SATRIA DATA 2026. Tujuannya adalah mendeteksi perubahan tutupan lahan dan laju deforestasi secara presisi tinggi di Pulau Papua antara tahun 2021 hingga 2025 menggunakan pencitraan satelit **Sentinel-2**. Arsitektur pemodelan yang ditargetkan adalah **ResNet50-UNet**.

Tantangan terbesar yang sedang dihadapi adalah geografi Papua itu sendiri yang didominasi oleh hutan murni dan perairan. Hal ini menciptakan **Class Imbalance Ekstrem**, di mana kelas seperti Tambang dan Permukiman hampir tidak eksis. Dokumen ini merangkum secara komprehensif seluruh sejarah proyek dan struktur *pipeline* final untuk menyelesaikannya.

---

## BAB 2: RIWAYAT PEMROSESAN DATA (BAGIAN 1 - 10)

Seluruh proses di **Bagian 1 hingga 10** telah tereksekusi tanpa cela:

### 2.1 Sourcing Data Satelit (Integrasi 6 Sumber Superposisi)
Proyek ini mengintegrasikan 6 sumber data raksasa dari Google Earth Engine (GEE):
1.  **Sentinel-2 SR Harmonized (`COPERNICUS/S2_SR_HARMONIZED`)**
2.  **Dynamic World V1 (`GOOGLE/DYNAMICWORLD/V1`)**: Peta Dasar Mutlak.
3.  **ESA WorldCover v200 (`ESA/WorldCover/v200/2021`)**
4.  **Hansen GFC v1.13 (`UMD/hansen/global_forest_change_2025_v1_13`)**
5.  **FDP Palm Probability 2025a (`projects/forestdatapartnership/assets/palm/model_2025a`)**
6.  **Global Mining Footprint (Tang & Werner + Maus dkk)**

### 2.2 Tahap Ekspor & Pemotongan Matriks (Tiling & Patching)
*   **STATUS: SELESAI MUTLAK (COMPLETED)**
*   Citra Papua telah diekspor ke Google Drive.
*   Puluhan ribu file `.npz` (256x256 piksel) telah dipotong dan tersimpan di folder `ForestWatch_Patches`.

### 2.3 Resolusi Anomali (Operasi "Data Healing")
*   *Bug Awal:* Fusi interseksi ketat `ESA AND DW` menghancurkan label Hutan menjadi 24% dan Sawit 0%.
*   *Solusi Sempurna:* Penggunaan *Dynamic World Base Map* yang ditimpa oleh *Thematic Overlays* (Hansen, FDP Palm, Mining Footprint).
*   **Hasil Akhir (Distribusi Piksel Sembuh):**
    *   `Perairan (0)`: 66.42%
    *   `Hutan (1)`: 32.02%
    *   `Pertanian Lain (4)`: 0.66%
    *   `Lahan Terbuka (2)`: 0.47%
    *   `Sawit (3)`: 0.30%
    *   `Permukiman (6)`: 0.09%
    *   `Tambang (5)`: 0.002% (Ini yang memicu kebutuhan Transfer Learning).

### 2.4 Metodologi "EDA-First"
Seluruh modul EDA telah disempurnakan:
*   **EDA 1 (Distribusi Kelas Linear Scale Miliar)**.
*   **EDA 2 (128 Patch Visual Validation)**: Bebas *deadlock* berkat penggunaan `ThreadPoolExecutor`.
*   **EDA 3 (Profil Spektral Rata-rata)**.
*   Semua gambar EDA otomatis tersimpan (HD 300 DPI) ke `Drive/Satria Data 3.0/EDA_Visualizations/`.

---

## BAB 3: RESTRUKTURISASI PIPELINE (BAGIAN 11 HINGGA 16)

Karena Bagian 1-10 sudah sangat lancar, kita harus melakukan **peningkatan (upgrade)** metodologi untuk menyelamatkan model dari bias akibat *Class Imbalance*. Rencana lama (di mana Bagian 11 langsung menjadi *Training*) telah **DIHAPUS**. 

Untuk menghindari beban *runtime* yang terlalu berat saat *training* (*Augmentation Bottleneck*), **semua augmentasi akan dilakukan secara offline dan disimpan permanen di Google Drive**.

Berikut adalah struktur *Jupyter Notebook* resmi yang baru (Mulai dari Bagian 11):

### 📌 Bagian 11: Setup Global Spatial Transfer Learning
Menentukan *Bounding Box* (Koordinat ROI) dari luar Papua untuk menargetkan "Sweet Spot" kelas minoritas:
*   Tambang: Kalimantan & Sulawesi.
*   Sawit: Sumatera Utara & Riau.
*   Permukiman & Pertanian: Jawa.

### 📌 Bagian 12: EDA Pra-Fusi & Pasca-Fusi (Transfer Learning)
Melakukan visualisasi distribusi piksel secara *real-time* di GEE (sebelum diekspor) untuk memastikan bahwa koordinat ROI yang dipilih pada Bagian 11 benar-benar mengandung kelas target yang diharapkan.

### 📌 Bagian 13: GEE Export & Tiling (Data Transfer)
Mengeksekusi *Export* dari Earth Engine untuk ROI Jawa, Kalimantan, dan Sumatera ke Google Drive. Dilanjutkan dengan pemotongan *patches* (Tiling) khusus data luar Papua ini.

### 📌 Bagian 14: EDA Pasca-Export & Distribusi Piksel Akhir
Validasi ketat. Melakukan perhitungan jumlah *patches* dan kalkulasi ulang persentase piksel dari data hasil *Transfer Learning*. Memastikan target awal murni sebesar **250 Juta piksel per kelas minoritas** tercapai pada fase ini (sebelum augmentasi).

### 📌 Bagian 15: Offline Data Augmentation & Final EDA
**PENTING:** Melakukan augmentasi data (Rotasi, Flip, Color Jittering, dll) **SEBELUM** siklus *training* dimulai. Semua gambar hasil augmentasi harus disimpan ke Google Drive.
*   **Target Augmentasi:** Meledakkan jumlah piksel dari "modal awal" 250 Juta menjadi jauh lebih besar (melalui metode *Inverse Frequency Multiplier*, bisa menyentuh angka 1 - 2 Miliar piksel per kelas) agar benar-benar kokoh menghadapi kelas mayoritas.
*   Tujuan: Meringankan beban GPU/CPU PyTorch saat iterasi *training*.
*   Diakhiri dengan EDA Final untuk melihat bentuk fisik dari matriks yang telah diaugmentasi.

### 📌 Bagian 16: Pelatihan Model (ResNet50-UNet)
Fase puncak. Menghubungkan PyTorch *DataLoader* langsung ke dataset yang sudah matang di Google Drive, dikombinasikan dengan *Loss Function* Hibrida (*Focal Loss + Dice Loss*).

---

## BAB 4: BUKU PANDUAN UNTUK AI AGENT SELANJUTNYA (SYSTEM HANDOFF)

**🚨 BACA DENGAN SEKSAMA 🚨**
Anda adalah AI Agent penerus. Tugas Anda adalah mengawal eksekusi **Bagian 11 hingga Bagian 16**. Jangan mengubah apapun di Bagian 1-10.

### TUGAS 1: Skrip Ekstraksi Modular
Patuhi struktur Bagian 11-13. Pastikan kode ekspor GEE untuk Transfer Learning menggunakan parameter yang sama dengan Papua (6 Band + 1 Label), namun arahkan destinasi penyimpanannya secara eksklusif ke `Drive/Satria Data 3.0/Global Spatial Transfer Learning/`.

### TUGAS 2: Implementasi Offline Augmentation (Bagian 15)
Buat satu sel atau file `.py` terpisah yang membaca *patches* langka (seperti Tambang), memutarnya (90, 180, 270 derajat), membaliknya (*flip*), lalu menyimpannya kembali sebagai file `.npz` baru (misalnya `patch_tambang_aug_001.npz`). Ini krusial agar model tinggal menarik data tanpa perlu menghitung augmentasi secara dinamis saat iterasi lambat.

### TUGAS 3: Refaktor Arsitektur `dataset.py`
Jangan menghancurkan *class* lama. Modifikasi atau buat kelas baru yang mampu mengenali folder `ForestWatch_Patches` (Papua asli), `Global Spatial Transfer Learning` (Data Asli Luar Papua), dan `Augmented_Patches` (Data Augmentasi Offline).

### TUGAS 4: Loss Function Hibrida
Pada **Bagian 16**, pastikan Anda menerapkan perhitungan **Focal Loss + Dice Loss**. Tanah Kalimantan dan Papua mungkin berbeda warna secara optik (*Domain Shift*). *Loss* ini akan menyelamatkan metrik IoU model saat berhadapan dengan perbedaan tersebut.

---

**KESIMPULAN:**
Progress Bagian 1-10 berjalan mulus bagai baja. Tantangan kompetisi SATRIA DATA saat ini berada di arsitektur data. Ikuti peta jalan Bagian 11 hingga 16 dengan ketelitian absolut. Selamat bekerja!
