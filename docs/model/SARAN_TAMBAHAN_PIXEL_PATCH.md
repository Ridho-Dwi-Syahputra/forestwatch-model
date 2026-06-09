# Rencana Akuisisi Data Ekstraksi (Domain Generalization)

Dokumen ini merangkum analisis distribusi piksel terkini (Papua) dan strategi penambahan *patch* dari luar wilayah Papua untuk menangani *class imbalance* tanpa mengorbankan rasio alamiah data (Real-World Prior).

## 1. Mengapa Kelas Tidak Perlu Disetarakan (Sama Rata 14%)?

Pendekatan *Deep Learning* modern tidak mensyaratkan setiap kelas memiliki jumlah data yang persis sama, dengan alasan berikut:

1. **Real-World Prior (Kenyataan Alamiah):** Di dunia nyata, lautan (72%) dan hutan (26%) mendominasi bentang alam. Memaksa rasio Tambang (0.00%) disamakan dengan Hutan (26%) akan membuat model mengalami **Over-Sensitivitas**. Akibatnya, model akan sangat agresif mencari Tambang dan memunculkan *False Positive* massal (contoh: lapangan bola ditebak sebagai tambang).
2. **Information Saturation:** Jaringan syaraf tiruan (U-Net) hanya membutuhkan "jumlah variasi minimal" untuk mengenali pola tekstur. Setelah AI melihat 15 Juta piksel tambang dari berbagai sudut dan cuaca, menambahkan 1 Miliar piksel tambang lagi tidak akan membuat model lebih akurat, melainkan hanya memperlama proses *training*.
3. **Penanganan Algoritmik (Focal-Tversky Loss):** Ketimpangan data mentah ditangani di tingkat Loss Function. Dengan `Tversky Beta = 0.7`, kita menginstruksikan algoritma untuk "memberikan penalti kerugian yang sangat besar jika gagal mengenali 1 piksel kelas minoritas (Tambang/Sawit), namun memberikan penalti kecil bila salah mengenali Perairan." Oleh karenanya, data yang sedikit akan mendapatkan perhatian prioritas (Bobot kelas Tambang = 4.93 vs Hutan = 0.05).

---

## 2. Analisis Distribusi Saat Ini (Papua)

Distribusi diambil dari 193.453 *patch* aktual Papua:
- **Kelas 0 (Perairan):** 9.13 Miliar piksel (72.08%)
- **Kelas 1 (Hutan):** 3.34 Miliar piksel (26.36%)
- **Kelas 2 (Lahan Terbuka):** 60 Juta piksel (0.47%)
- **Kelas 3 (Sawit):** 39 Juta piksel (0.30%)
- **Kelas 4 (Pertanian Lain):** 84 Juta piksel (0.66%)
- **Kelas 5 (Tambang):** 343 Ribu piksel (0.0027%) ⚠️ **Kritis**
- **Kelas 6 (Permukiman):** 11.6 Juta piksel (0.091%) ⚠️ **Kurang**

---

## 3. Rekomendasi Penambahan Pixel & Patch (Selective Patch Cutting)

Kita hanya akan mengambil *Bounding Box* (BBOX) berukuran spesifik pada area-area konsentrasi (Hotspot) untuk menghindari penumpukan data *background* yang tidak berguna.

Satu *Patch* berukuran 256x256 piksel mencakup area ± 2.5 km x 2.5 km (65.536 piksel).

### Prioritas 1: Tambang (Kelas 5)
*   **Kondisi Saat Ini:** 343.406 piksel (~5 patch utuh). Sangat rentan *overfitting*.
*   **Target Tambahan:** ~15 Juta hingga 20 Juta piksel.
*   **Ekuivalen Patch:** ~250 hingga 300 *patch* kaya tambang.
*   **Kebutuhan Tile (BBOX):** 1 Tile berukuran ± 40 km x 40 km.
*   **Lokasi Optimal:** 
    *   Tambang Batubara Kalimantan (Kutai Timur/Sangatta).
    *   Pegunungan Andes, Peru/Chile (Morfologi tambang *open pit* dataran tinggi berbatu yang sangat mirip dengan Grasberg Papua).

### Prioritas 2: Permukiman (Kelas 6)
*   **Kondisi Saat Ini:** 11.6 Juta piksel (~177 patch utuh). Kurang padat untuk membedakan struktur beton raksasa.
*   **Target Tambahan:** ~20 Juta piksel.
*   **Ekuivalen Patch:** ~300 *patch* padat penduduk.
*   **Kebutuhan Tile (BBOX):** 1 Tile berukuran ± 40 km x 40 km.
*   **Lokasi Optimal:** Pulau Jawa (Pinggiran Jabodetabek atau Surabaya Raya). Memberikan kepadatan struktur semen dan aspal jalan tol yang ekstrem.

### Prioritas 3: Sawit (Kelas 3)
*   **Kondisi Saat Ini:** 39 Juta piksel (~600 patch utuh). Sudah lumayan kuat.
*   **Target Tambahan:** ~10 Juta piksel.
*   **Ekuivalen Patch:** ~150 *patch* perkebunan sawit.
*   **Kebutuhan Tile (BBOX):** 1 Tile berukuran ± 30 km x 30 km.
*   **Lokasi Optimal:** Riau atau Sumatera Utara. Sawit di Papua sebagian besar merupakan tanaman muda/pembukaan lahan baru. Data dari Sumatera memberikan varians pola kanopi sawit dewasa yang ditanam rapat dan rapi.

### Kesimpulan Beban Komputasi
Penambahan data eksternal ini secara total hanya membutuhkan **sekitar 750 Patch tambahan** (sekitar 0.38% dari total 193.453 patch Papua yang ada saat ini).
Beban penyimpanan (*storage*) ekstra hanya sekitar **20 - 30 Megabyte**, namun lonjakan akurasi deteksi kelas langka (IoU) yang dihasilkan akan sangat eksponensial.
