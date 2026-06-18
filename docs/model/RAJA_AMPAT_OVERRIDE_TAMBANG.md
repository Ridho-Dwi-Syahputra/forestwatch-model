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

## 6. Yang WAJIB dilakukan setelah Tahap 1 berhasil, sebelum Tahap 2 (training se-Papua)

1. **Jalankan cell verifikasi baru di `optimize_dataset.ipynb`** (cell 5) dan baca hasilnya.
   - Kalau semua region > 0 piksel → `Bahan_Training_Fix` aman dipakai apa adanya.
   - Kalau ada region yang 0 piksel → perlu override manual serupa Gag untuk region itu
     (sub-bbox + threshold NDVI/bare-soil), sebelum data dipakai training skala penuh.
2. **Pertimbangkan apakah patch Gag (Raja Ampat)** — saat ini hanya hidup di
   `Training_Merauke_Boven_Digoel/` (folder khusus Tahap 1) — perlu digabung juga ke
   `ForestWatch_Patches_Transfer` / `Bahan_Training_Fix` supaya sinyal Tambang lokal Papua ini
   ikut terpakai di training se-Papua, bukan cuma di subset Merauke+BD.
3. Jangan lupa: ambang `NDVI_BARE_THRESHOLD=0.25` dan `GAG_MINING_SUBBBOX` adalah **estimasi**,
   bukan survei presisi — kalau nanti ada akses ke data konsesi resmi (shapefile KLHK/ESDM),
   override ini sebaiknya diganti dengan poligon asli.
