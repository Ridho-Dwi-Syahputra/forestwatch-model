# forestwatch (vendored)

Salinan package Python **`forestwatch`** dari repo model utama
(`forestwatch-papua-model`, folder `model/`), di-vendor ke dalam repo WebGIS ini
supaya backend bisa berdiri sendiri **tanpa** harus meng-clone repo model.

Dipakai oleh backend untuk endpoint **`POST /api/analyze`** (analisis deforestasi
wilayah custom on-demand): komposit Sentinel-2 dari GEE → inferensi model 7-kelas →
deteksi transisi Hutan. Lihat `backend/app/core/forestwatch_bridge.py`.

## Cara dipakai backend
`forestwatch_bridge.py` menambahkan `model/src` ke `sys.path` saat runtime — jadi
**tidak wajib** `pip install`. Dependensi runtime (`torch`, `segmentation-models-pytorch`,
`earthengine-api`, `rasterio`, `numpy`, dst.) di-install lewat `backend/requirements.txt`.

Alternatif (opsional): `pip install -e ./model` lalu `import forestwatch` jalan tanpa
sys.path hack.

## ⚠️ Sumber kebenaran (source of truth)
Ini **salinan**. Kalau ada perbaitan pada package model, lakukan di repo model dulu,
lalu **re-vendor** (copy ulang) `src/forestwatch` + `configs/` ke sini. Jangan edit
salinan ini sebagai satu-satunya tempat perubahan.

Isi yang di-vendor:
- `src/forestwatch/` — seluruh package (semua submodule, agar tak ada import internal yang putus)
- `configs/` — `default.yaml`, `classes.yaml`, `papua_bbox.geojson` (dipakai `forestwatch.config` bila `load_config()` dipanggil)
- `pyproject.toml` — metadata package (untuk opsi `pip install -e ./model`)
