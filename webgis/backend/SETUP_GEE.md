# Setup GEE Service Account untuk fitur "Analisis Wilayah Custom"

Halaman **Analisis Wilayah Custom** (`POST /api/analyze`) menarik komposit Sentinel-2 dari
Google Earth Engine (GEE) **secara langsung saat tombol ditekan**, lalu menjalankan model di
wilayah yang diminta. Karena server tidak punya browser untuk login manual, GEE diakses lewat
**Service Account** (akun robot + file kunci JSON). Dokumen ini panduan membuatnya.

> Endpoint lain (Dashboard, Peta, Statistik, Akurasi, Unduh) **tidak** butuh ini — mereka baca
> file hasil model yang sudah jadi. Yang butuh GEE hanya `POST /api/analyze`.

Project yang dipakai: **`forestwatch-papua-3`** (sudah diset di `.env`).

---

## Langkah-langkah (kamu yang jalankan — butuh login Google-mu)

### 1. Buka Google Cloud Console
https://console.cloud.google.com/ → pojok kiri atas, pilih project **`forestwatch-papua-3`**.
(Pastikan project ini yang sudah terdaftar/aktif untuk Earth Engine.)

### 2. Aktifkan Earth Engine API
- Menu ☰ → **APIs & Services** → **Enabled APIs & services** → **+ ENABLE APIS AND SERVICES**.
- Cari **"Google Earth Engine API"** → klik → **Enable**.

### 3. Buat Service Account
- Menu ☰ → **IAM & Admin** → **Service Accounts** → **+ CREATE SERVICE ACCOUNT**.
- *Service account name*: `forestwatch-backend` (bebas).
- Klik **Create and Continue**.
- *Grant this service account access*: pilih role **Earth Engine Resource Viewer**
  (kalau tidak muncul, pakai **Editor**). Klik **Continue** → **Done**.
- Catat **email**-nya, formatnya:
  `forestwatch-backend@forestwatch-papua-3.iam.gserviceaccount.com`

### 4. Buat & unduh kunci JSON
- Di daftar Service Accounts, klik service account tadi → tab **KEYS**.
- **ADD KEY** → **Create new key** → pilih **JSON** → **Create**.
- File `.json` otomatis terunduh. **Pindahkan & ganti namanya** menjadi:
  ```
  webgis/backend/secrets/gee-service-account.json
  ```
  (Folder `secrets/` sudah ada dan sudah di-`.gitignore` — file ini TIDAK akan ter-commit.)

### 5. Daftarkan Service Account ke Earth Engine
Service account perlu izin memakai Earth Engine:
- Buka https://code.earthengine.google.com (login akun yang punya project), atau
- Daftarkan di https://signup.earthengine.google.com/#!/service_accounts (tempel email SA dari
  langkah 3).
- Pastikan SA terdaftar di project `forestwatch-papua-3`.

### 6. Isi `.env`
Buka `webgis/backend/.env`, isi baris email (key path sudah benar):
```
GEE_SERVICE_ACCOUNT_EMAIL=forestwatch-backend@forestwatch-papua-3.iam.gserviceaccount.com
GEE_SERVICE_ACCOUNT_KEY_PATH=./secrets/gee-service-account.json
```

### 7. Jalankan & verifikasi
```bash
cd webgis/backend
pip install -r requirements.txt          # sekali saja (butuh earthengine-api, torch, rasterio, dll)
uvicorn app.main:app --reload
```
Cek kesiapan:
```bash
curl http://localhost:8000/api/health
```
Harus muncul `"gee_ready": true`, `"model_ready": true`, `"analyze_ready": true`.
Kalau salah satu `false`, lihat `"model_error"` atau pesan startup di terminal.

Lalu jalankan frontend dengan API URL di-set agar memanggil backend (bukan data statis):
```bash
cd webgis/frontend
# .env.local:
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev
```
Buka halaman **Analisis Custom** → pilih AOI **Merauke** → **Jalankan Analisis**. Hasil + dua
peta mini (Sebelum/Sesudah) akan muncul.

---

## Catatan keamanan
- **JANGAN** commit `secrets/` maupun `.env` (keduanya sudah di `.gitignore`).
- **JANGAN** share file kunci JSON ke siapa pun / chat / repo publik. Kalau bocor, hapus key di
  Cloud Console (tab Keys → hapus) dan buat yang baru.

## Batasan teknis fitur
- **AOI maksimum ~12 km/sisi** (`ANALYZE_MAX_SIDE_KM` di `.env`). Lebih besar ditolak (HTTP 400)
  karena batas ukuran unduh `getDownloadURL` GEE (komposit 6 band float32 @10 m).
- Komposit median **1 tahun penuh**; tahun berjalan yang belum lengkap bisa kurang bebas awan.
- Preview "Sebelum/Sesudah" adalah **klasifikasi model**, bukan foto satelit tahun tersebut
  (citra satelit historis per-tahun tidak tersedia dari tile gratis).
