> ℹ️ **CATATAN STATUS.** Prinsip umum dokumen ini (real-world prior, multi-region sourcing, mitigasi domain-shift) **masih jadi rujukan** — lihat [`RENCANA_EKSEKUSI_LANJUTAN.md`](RENCANA_EKSEKUSI_LANJUTAN.md) v8.0 §1.5. Namun beberapa detail teknis sudah **DIKOREKSI** (lihat catatan inline ⚠️/⛔ di bawah): lokasi tambang (§A.1), threshold patch cutting (§3.3), dan **alur penggabungan dataset (§3.4) — JANGAN diikuti apa adanya**, ikuti RENCANA TAHAP 1 (Bagian 12) & TAHAP 2 (Bagian 14) untuk menghindari kebocoran data transfer ke test set.

# FUTURE BEST PLAN: Global Spatial Transfer Learning
**Strategi Mengatasi Extreme Class Imbalance pada Model Pemantauan Hutan Papua**

## 1. Latar Belakang Masalah
Dalam proyek ForestWatch Papua, kita menghadapi masalah *extreme class imbalance*. Kelas dominan seperti **Perairan (72.15%)** dan **Hutan (26.39%)** mendominasi jutaan piksel, sedangkan kelas minoritas yang krusial untuk pemantauan deforestasi sangat langka di daratan Papua:
- **Tambang**: ~0.00%
- **Permukiman**: ~0.00%
- **Sawit**: 0.31%

Melatih arsitektur segmentasi (seperti U-Net) dengan data yang sangat timpang ini dapat menyebabkan model mengabaikan kelas minoritas secara total, meskipun *class weights* (pembobotan loss) sudah digunakan.

## 2. Solusi Utama: Global Spatial Transfer Learning (Domain Generalization)
Karena satelit Sentinel-2 mengambil spektrum cahaya yang sama di seluruh belahan bumi, kita tidak dibatasi oleh batas administratif Papua. Kita akan memperkaya (*enrichment*) dataset *training* dengan mengambil "sampel kaya" (*hotspots*) dari wilayah lain di Indonesia maupun dunia yang memiliki karakteristik spektral dan morfologis serupa.

### A. Sourcing Hotspots (Lokasi Pencurian Data)
1. **Tambang (Mining - Kelas 5)**
   - **Lokasi Ideal**: Pegunungan Andes (Peru/Chile) atau Kalimantan.
   - **Alasan**: Tambang emas/tembaga di Grasberg, Papua memiliki morfologi *open pit* berbatu abu-abu pucat di dataran tinggi. Mengambil data tambang tembaga dari Peru/Chile akan memberikan *Domain Match* yang sempurna. Data tambang batubara/emas dari Kalimantan bisa menambah keragaman (*variance*) bentuk galian tambang.
   - *(Catatan: Hindari eksklusivitas tambang Nikel Sulawesi (Morowali) yang berwarna laterit merah tua agar model tidak mengalami over-fitting pada warna tanah tertentu).*

   > ⚠️ **Update RENCANA v8.0 (§1.1):** Morowali (Sulteng, nikel) justru **DIPAKAI** sebagai salah satu dari 3 region Tambang (bersama Kaltim + Kalsel) untuk variasi morfologi — bukan dihindari, tapi dimitigasi via augmentasi radiometrik ringan (selaras prinsip "jangan eksklusif 1 lokasi" di atas). Opsi Pegunungan Andes (Peru/Chile) **TIDAK diadopsi** — fokus region domestik Indonesia.

2. **Permukiman (Built-up - Kelas 6)**
   - **Lokasi Ideal**: Pulau Jawa (Jabodetabek atau Surabaya).
   - **Alasan**: Kepadatan lahan terbangun sangat tinggi, memastikan model belajar membedakan struktur atap, beton, dan jalan aspal terhadap lahan terbuka/tanah kosong.

3. **Sawit (Oil Palm - Kelas 3)**
   - **Lokasi Ideal**: Sumatera (Riau atau Sumatera Utara).
   - **Alasan**: Pusat perkebunan kelapa sawit dunia. Spektrum inframerah (NIR/SWIR) tajuk sawit di Sumatera identik dengan perkebunan sawit di selatan Papua.

## 3. Rencana Implementasi Teknis (Selective Pipeline)
Untuk mencegah ledakan data (menghindari mendownload jutaan piksel hutan/laut dari pulau lain), *pipeline* GEE akan dimodifikasi menjadi sistem **Selective Patch Cutting**:

1. **Definisi Bounding Box (BBox) Spesifik**
   - Membuat poligon kecil berukuran 1x1 derajat yang hanya memusat tepat di atas *hotspot* target.
   
2. **Eksekusi GEE Ekspor Parsial**
   - Menjalankan fungsi `s2_composite` dan `build_label` persis seperti di Papua, namun hanya untuk BBox yang telah ditentukan.
   - Ekspor ubin (`.tif`) ke direktori Google Drive terpisah (misal: `ForestWatch_Tiles_Transfer`).

3. **Selective Patch Extraction (Pemotongan Tersaring)**
   - Saat proses *sliding window* (256x256 piksel), *patch* **TIDAK** langsung disimpan.
   - Diterapkan *Threshold Logic*:
     ```python
     # Contoh: Hanya simpan jika area tambang menutupi minimal 5% dari patch
     target_pixels = (lab == TARGET_CLASS).sum()
     total_pixels = 256 * 256
     if (target_pixels / total_pixels) > 0.05:
         np.savez_compressed(...)
     ```

   > ⚠️ **Update RENCANA v8.0 (TAHAP 1, Bagian 12 §1.3):** threshold `> 0.05` (5%) di atas **DILONGGARKAN** menjadi `> 1–2%` (atau ambil semua patch dalam ROI inti) agar volume cukup mencapai target ~250 juta px/kelas.

4. **Penggabungan Dataset (Dataset Merging)**
   - *Patch* `.npz` berkualitas tinggi yang lolos filter seleksi ini langsung dicampur ke dalam direktori utama `ForestWatch_Patches`.
   - Modul `compute_class_distribution` akan secara otomatis membaca penambahan data ini, menyesuaikan persentase, dan menormalkan kembali *class weights* sebelum proses *training*.

   > ⛔ **KOREKSI KRITIS RENCANA v8.0 (TAHAP 1 §1.3 & TAHAP 2):** JANGAN langsung mencampur ke `ForestWatch_Patches`. Simpan ke folder **terpisah** `ForestWatch_Patches_Transfer/<region>/` dengan metadata `source: transfer`. Penggabungan ke pool training **hanya** terjadi **setelah** split train/val/test (test = 100% Papua holdout) — lihat TAHAP 2 langkah 1–2. Jika "langsung dicampur" sebelum split diikuti, data transfer bisa **bocor** ke test set Papua dan evaluasi jadi tidak jujur.

## 4. Antisipasi Risiko (Domain Shift)
Meskipun teknik ini sangat *powerful*, kita harus waspada terhadap fenomena **Domain Shift** (Pergeseran Domain), di mana fitur target di area *training* memiliki distribusi warna yang sedikit berbeda dari area *deployment* (Papua).
- **Mitigasi**: Penggunaan data augmentasi spasial (*flip*, *rotation*) dan radiometrik (*brightness*, *contrast* terbatas) pada *Dataloader* untuk memaksa *Convolutional Neural Network* fokus mempelajari *bentuk/morfologi* dan *tekstur* (spatial correlation) alih-alih hanya menghafal intensitas warna (spectral reflectance) spesifik dari pulau asal.
