# Laporan Resolusi Anomali Data (Data Healing)
**Koreksi Miskasifikasi Hutan Rawa menjadi Perairan**

## 1. Latar Belakang Penemuan Masalah
Selama fase *Exploratory Data Analysis* (EDA) Tahap 2 (Validasi Visual), ditemukan sebuah anomali fatal pada data *ground truth*: berhektar-hektar area yang secara visual merupakan **Hutan Hijau Lebat** dilabeli oleh sistem sebagai **Perairan/Air (Kelas 0 - Biru)**.

## 2. Akar Penyebab (*Root Cause Analysis*)
Setelah melakukan penelusuran (*debugging*) ke dalam arsitektur ekstraksi data Google Earth Engine (GEE), akar masalah ditemukan pada file konfigurasi `src/forestwatch/constants.py`:

```python
# REMAP DYNAMIC WORLD (9 kelas) → 6 KELAS PROYEK
DW_TO_CLASS: dict[int, int] = {
    ...
    3: 0,  # Flooded Veg → Perairan (Rawa/Wetland)
    ...
}
```
Peta dasar *Dynamic World* (DW) memiliki **Kelas 3 (Flooded Vegetation)** yang secara khusus merepresentasikan ekosistem rawa, mangrove, dan vegetasi yang terendam air. Geografi Papua memiliki ekosistem Hutan Rawa (seperti Asmat dan Mamberamo) terluas di Asia. 

Karena aturan konfigurasi (*mapping*) di atas secara eksplisit mengubah Kelas 3 menjadi Kelas 0 (Perairan), jutaan piksel Hutan Rawa secara paksa dikonversi menjadi Air. Jika dibiarkan, hal ini akan menyebabkan *Catastrophic Forgetting* pada model U-Net, di mana model akan mengira tekstur pohon rimbun adalah bentang laut.

## 3. Metodologi Penyembuhan (Label Healer)
Alih-alih mengunduh ulang 12.6 Miliar piksel dari GEE (yang akan memakan waktu berhari-hari), diterapkan perbaikan data tingkat piksel (*Pixel-level Data Healing*) berbasis **Hukum Fisika Optik**:

*   **Teori:** Air murni memantulkan sangat sedikit cahaya Inframerah Dekat (NIR), sedangkan klorofil pada daun hijau memantulkan NIR dengan sangat kuat.
*   **Indikator:** *Normalized Difference Vegetation Index* (NDVI).
*   **Logika Perbaikan:**
    > Jika sebuah piksel berlabel `0` (Perairan), **TETAPI** memiliki nilai `NDVI > 0.3` (karakteristik vegetasi lebat), maka secara fisik mustahil piksel tersebut adalah genangan air murni. Piksel tersebut adalah kanopi pohon (Hutan Rawa), sehingga labelnya dipaksa kembali menjadi `1` (Hutan).

## 4. Eksekusi Teknis
Pemulihan dilakukan menggunakan arsitektur pemrosesan paralel (36 *Workers* `ProcessPoolExecutor`):
- Membuka ±193.453 file `.npz` secara sekuensial.
- Melakukan operasi raster matematika pada matriks 256x256 piksel.
- Menimpa (*overwrite*) label asli jika anomali ditemukan.

## 5. Hasil dan Implikasi
- Ratusan juta piksel Hutan Rawa berhasil dikembalikan ke habitat aslinya (Kelas 1).
- Distribusi kelas (terutama persentase Hutan vs Perairan) kembali akurat dan mencerminkan prioritas alam nyata (*Real-world prior*).
- Anomali lain seperti "belang-belang" pada area pertambangan (*Tailings Pond*) **dibiarkan**, karena secara fisik genangan limbah tambang memang merupakan benda cair. Anomali kecil ini (*Noisy Labels*) akan ditangani secara otomatis oleh toleransi algoritmik dari **Focal Tversky Loss** selama fase *training*.
