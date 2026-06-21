# Catatan Tambahan — Batas Skala AOI (Live vs Precompute) & Struktur Halaman WebGIS

> **Status:** Dokumen ini adalah **koreksi & pelengkap** untuk
> `docs/webgis/CATATAN_AI_AGENT_BACKEND_FRONTEND.md`, khususnya bagian §3.3
> ("Catatan performa / real-time"). Tabel performa di dokumen tersebut **terlalu optimis**
> untuk skala kabupaten/provinsi — dokumen ini menggantikannya dengan angka yang lebih realistis.

---

## 1. Kenapa perlu koreksi?

Dokumen utama bilang AOI sebesar **kabupaten/provinsi** bisa diproses "live" dalam
**detik–menit**. Setelah dihitung ulang berdasarkan resolusi 10m & batas ukuran
`computePixels`/`getDownloadURL` GEE (puluhan MB per request), klaim itu **terlalu optimis**
untuk AOI di atas skala kota/kecamatan.

---

## 2. Tabel realistis: skala AOI vs kelayakan "live"

Asumsi: resolusi 10m, 1 patch model = 256×256 = 65.536 piksel, 6 band.

| Skala AOI | Contoh | ≈ Luas | ≈ Jumlah patch 256×256 | Kelayakan "live" |
|---|---|---|---|---|
| Kota kecil / kecamatan | Kota Jayapura (~940 km²) | ratusan km² | ~140 patch | ✅ **Live OK** — puluhan detik s/d ~2 menit |
| Kabupaten | Merauke (~46.800 km²) | puluhan ribu km² | ~7.100 patch (~11 GB data mentah) | ⚠️ **Borderline** — perlu banyak panggilan `computePixels` (limit per-request puluhan MB), realistisnya **5–15+ menit**. Lebih cocok **job async** dengan progress bar, bukan "instan" |
| Provinsi / se-Papua | Papua, Papua Selatan, dst (puluhan–ratusan ribu km²) | sangat besar | puluhan ribu patch | ❌ **Tidak live** — masalahnya sama dengan "se-Papua": **harus precompute** (batch job offline) |

**Kesimpulan:** AOI yang benar-benar "live" (detik–menit, cocok untuk interaksi user) itu skala
**kota/kecamatan**, BUKAN kabupaten/provinsi penuh.

---

## 3. Dua jenis "permintaan citra" — jangan disamakan

| Jenis | Tujuan | Batasan ukuran AOI | Cara |
|---|---|---|---|
| **Layer visual (basemap citra)** | Ditampilkan di peta sebagai latar | **Tidak ada batas** — GEE tile server (`getMapId` → XYZ tile URL) handle tiling sendiri, berapapun luas & zoom | Leaflet load tile URL langsung, tanpa lewat backend/model |
| **Data piksel untuk model** | Input `model.onnx` (inferensi 7-kelas) | **Dibatasi** oleh `computePixels`/`getDownloadURL` (puluhan MB/request) → lihat tabel §2 | Backend tarik numpy array ke memori, potong jadi patch 256×256, jalankan model |

Jadi: **menampilkan citra se-Papua sebagai latar peta = gratis & instan**. Yang mahal adalah
**menjalankan model** di atasnya.

---

## 4. Provinsi di LUAR Papua (mis. Sumatra) — dua masalah berbeda

1. **Infrastruktur (ukuran/kuota GEE):** sama saja seperti Papua — tidak ada perbedaan
   teknis berdasarkan lokasi. Tabel §2 berlaku di mana pun di dunia.
2. **Akurasi model (domain shift) — ini yang lebih krusial:** model dilatih khusus pada
   karakteristik **Papua** (pola hutan, sawit, tambang, permukiman). Sumatra punya
   karakteristik sangat berbeda (sawit industrial jauh lebih masif, lahan gambut, kepadatan
   permukiman berbeda). Menjalankan model Papua di Sumatra **kemungkinan menghasilkan
   klasifikasi yang salah secara sistematis** — bukan bug, tapi keterbatasan generalisasi model.

**Rekomendasi:** scope WebGIS **tetap Papua** (sesuai nama proyek "ForestWatch Papua" &
target SATRIA DATA 2026). Dukungan wilayah lain butuh data training tambahan — di luar scope
saat ini.

---

## 5. Rekomendasi struktur WebGIS: 1 halaman, 2 mode

Tidak perlu 2 halaman terpisah. Lebih praktis: **1 halaman map** dengan **2 mode/tab**
(basemap & state peta tetap konsisten saat user pindah mode):

### Mode A — "Peta Papua" (default, instan)
- Tampilkan hasil **precompute** (sudah dihitung offline sebelumnya): peta tutupan lahan
  + poligon deforestasi + statistik untuk **seluruh Papua**.
- Load cepat karena hanya serve file GeoJSON/JSON statis yang sudah jadi — **tidak**
  memanggil GEE/model saat user buka halaman.

### Mode B — "Analisis Custom" (live)
- User gambar/pilih AOI **kecil** (skala kota/kecamatan, sesuai §2) + pilih tahun T1 & T2.
- Backend proses live (`POST /api/analyze`), tampilkan loading/progress, hasil **detik–menit**.
- Untuk AOI di luar batas kota/kecamatan (kabupaten ke atas), **tolak atau alihkan ke
  job async** dengan peringatan "proses bisa beberapa menit" — jangan diam-diam timeout.

---

## 6. Endpoint API tambahan

```
GET  /api/papua-overview
  resp: { "deforestation": <GeoJSON precompute se-Papua>,
          "statistics": <statistics.json precompute>,
          "generated_at": "2026-06-01T00:00:00Z" }   # kapan batch terakhir dijalankan
  # Statis — backend cukup baca file hasil precompute (disk/object storage), TIDAK panggil GEE/model.

POST /api/analyze   (sama seperti §3.3 dokumen utama, untuk Mode B)
  body: { "aoi": [...], "year_t1": ..., "year_t2": ..., "min_area_ha": ... }
  # Validasi: tolak jika luas AOI > batas kota/kecamatan (lihat §2), beri pesan jelas ke frontend.
```

---

## 7. Cara hasilkan `papua-overview` (job precompute)

Job ini dijalankan **offline/sekali** (bukan per-request user), idenya **sama** dengan
pipeline export training (`make_tiles` + tiling):

1. Pecah Papua jadi banyak tile (sama prinsip `make_tiles` di `forestwatch.gee.tiles`).
2. Tiap tile: `s2_composite()` → `infer_tile()` → mask 7-kelas (untuk T1 dan T2).
3. Jahit semua mask jadi raster utuh per tahun.
4. `detect_transitions_from_arrays()` pada raster gabungan → GeoJSON + statistik.
5. Simpan hasil (`papua_overview.geojson` + `papua_overview_stats.json`) ke storage yang
   diakses `GET /api/papua-overview`.

Job ini berat (jam, bukan menit) — jalankan **sebelum demo**, atau jadwalkan periodik
(mis. tiap kali ada update data/model), bukan dipicu user.

---

*Dokumen ini melengkapi `CATATAN_AI_AGENT_BACKEND_FRONTEND.md`. Bila ada perbedaan, gunakan
angka & rekomendasi di dokumen ini untuk soal skala AOI/performa (§3.3 dokumen utama dianggap
diperbarui oleh dokumen ini).*
