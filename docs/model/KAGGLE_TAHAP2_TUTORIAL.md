# Tutorial Training Tahap 2 di Kaggle (dataset 40GB)

Panduan menjalankan `train_model_1/2/3_*.ipynb` di **Kaggle Notebook** untuk dataset penuh
`Bahan_Training_Fix` (~40GB). Dipakai karena Colab free sesi cuma ~1j20m (tak cukup 60 epoch),
sedangkan Kaggle sesi 9–12 jam → ekstrak sekali, training tembus sekali jalan.

> Notebook sudah mendukung `ENV = "kaggle"` (branch di cell 2 + jalur jetson-like di cell 3/4/5).
> Tutorial ini fokus ke **upload data** + **setup notebook**, yang harus kamu lakukan manual.

---

## 0. Kendala kunci Kaggle yang menentukan strategi

- **Maks ~20GB per dataset privat.** `Bahan_Training_Fix` ~40GB → **harus dipecah jadi 2 dataset**.
- `/kaggle/input/*` = **SSD lokal cepat** (bukan FUSE seperti Drive) → ekstraksi tar cepat (~5 mnt),
  tak ada bottleneck I/O.
- `/kaggle/temp` = scratch besar (tempat ekstrak); `/kaggle/working` ~20GB, **persisten via "Save
  Version"** (tempat checkpoint/output).
- **Internet harus ON** di notebook (Settings → Internet) supaya bisa `git clone` + `pip install`.

---

## 1. Prasyarat

1. Akun Kaggle + **verifikasi nomor HP** (wajib untuk GPU + dataset privat).
2. API token: Kaggle → Account → "Create New API Token" → unduh `kaggle.json`.
3. `kaggle` CLI di mesin tempat upload:
   ```bash
   pip install -q kaggle
   mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```

---

## 2. Struktur 2 dataset

`Bahan_Training_Fix/` di Drive berisi: `train/`, `val/`, `test/`, `train_rajaampat/`,
`class_weights.json`, `patch_sampler_weights_shared.json`, `optimize_cache/`.

Pecah jadi 2 folder upload (`optimize_cache/` TIDAK perlu diupload):

| Dataset | Isi | Perkiraan |
|---|---|---|
| **`fw-papua-train`** | `train/` (7 tar) + `train_rajaampat/` + `class_weights.json` + `patch_sampler_weights_shared.json` | ~19,4 GB |
| **`fw-papua-valtest`** | `val/` + `test/` | ~19,6 GB |

> Kedua json ditaruh di dataset-1; cell 2 notebook menyatukan semua mount via symlink, jadi letak
> json di dataset mana pun tetap kebaca. Kalau `fw-papua-valtest` ditolak karena >20GB, pecah lagi
> jadi `fw-papua-val` & `fw-papua-test` (jadi 3 dataset) — notebook tetap jalan (symlink semua mount).

---

## 3. Upload (rekomendasi: dari Colab, Drive ter-mount → hindari download 40GB ke lokal)

Di sebuah Colab notebook (Drive mounted), siapkan 2 folder staging lalu upload. `cp` di Drive
lambat untuk 40GB (~30–60 mnt sekali jalan) tapi cuma sekali:

```python
from pathlib import Path
import shutil, subprocess, json, os
FIX = Path('/content/drive/MyDrive/Satria Data 3.0/Bahan_Training_Fix')
STG = Path('/content/kaggle_upload'); STG.mkdir(exist_ok=True)

# --- Dataset 1: train + raja ampat + json ---
d1 = STG / 'fw-papua-train'; d1.mkdir(exist_ok=True)
for item in ['train', 'train_rajaampat']:
    if (FIX/item).exists(): shutil.copytree(FIX/item, d1/item, dirs_exist_ok=True)
for j in ['class_weights.json', 'patch_sampler_weights_shared.json']:
    shutil.copy(FIX/j, d1/j)

# --- Dataset 2: val + test ---
d2 = STG / 'fw-papua-valtest'; d2.mkdir(exist_ok=True)
for item in ['val', 'test']:
    shutil.copytree(FIX/item, d2/item, dirs_exist_ok=True)

# metadata + upload (butuh ~/.kaggle/kaggle.json)
!pip install -q kaggle
for d, slug, title in [(d1,'fw-papua-train','FW Papua Train'), (d2,'fw-papua-valtest','FW Papua ValTest')]:
    meta = {"title": title, "id": f"<USERNAME_KAGGLE>/{slug}", "licenses": [{"name": "CC0-1.0"}]}
    (d/'dataset-metadata.json').write_text(json.dumps(meta))
    subprocess.run(f"kaggle datasets create -p {d} --dir-mode tar", shell=True)
```

Ganti `<USERNAME_KAGGLE>` dengan username Kaggle-mu. `--dir-mode tar` membungkus subfolder jadi tar
saat upload (lebih cepat untuk banyak file). **Catat slug** kedua dataset.

> Alternatif: kalau tar sudah ada di HDD PC lab, jalankan `kaggle datasets create -p <folder>`
> langsung dari PC (tanpa Colab).

---

## 4. Setup Kaggle Notebook

1. Di Kaggle: **+ New Notebook**. Settings (panel kanan):
   - **Accelerator → GPU** (T4 ×2 atau P100 — T4 disarankan, codebase pakai AMP/Tensor Core).
   - **Internet → On**.
2. **Add Input** → cari & attach **kedua** dataset (`fw-papua-train`, `fw-papua-valtest`).
   Keduanya mount di `/kaggle/input/<slug>/`.
3. Upload / copy isi `train_model_1_attention_unet.ipynb` (atau 2/3) ke notebook Kaggle.
4. Cell 2: ubah baris pertama jadi:
   ```python
   ENV = "kaggle"
   ```
5. Jalankan cell sesuai urutan (sama seperti panduan Colab):
   - **RUN**: `2 → 3 → 4 → 5 → 7 → 9 → 10 → 11 → 12 → 13 → 14`
   - **boleh SKIP**: cell 6 (EDA distribusi, cuma info).
   - Cell 2 (kaggle) otomatis: clone repo, `pip install`, symlink mount, **ekstrak tar ke
     `/kaggle/temp/dataset_local`** (~5 mnt), salin json. Cell 4/5 lalu baca folder terekstrak itu
     (jalur "jetson", tanpa Drive/folder mentah).

> Catatan: konfigurasi training saat ini = **80 epoch, sampler aktif, lr 1e-3, loss
> `focal_tversky`** (regime yang sama dipakai di Colab — lihat Tahap 1/2 sebelumnya). Eksperimen
> Deferred Re-Weighting (DRW) yang pernah ditambahkan sempat ditarik kembali supaya `git pull` di
> sesi manapun (Colab/Kaggle) tidak diam-diam mengganti hyperparameter di tengah training yang
> sedang berjalan. Kalau nanti mau dicoba lagi, perlu di-set ulang secara sengaja, bukan otomatis.

---

## 5. Yang diamati

- Cell 2: `extract_dataset_archives` selesai tanpa error; muncul `+ N patch Raja Ampat` di cell 4.
- Cell 11: log `Mulai training: ... epochs=80 (mulai ep 1)`, IoU per-kelas tercetak tiap epoch.
- Bandingkan `best val mIoU` akhir vs hasil Tahap 1 (Merauke, 0,3146) dan run Tahap 2 sebelumnya.

---

## 6. Ambil hasil

- Output ada di `/kaggle/working/Model_Comparison/model_1_attention_unet/`
  (`best_model.pt`, `metrics.json`, `output/*.png`, `model.onnx`, `summary.json`).
- Klik **"Save Version"** (Commit) supaya `/kaggle/working` tersimpan permanen — jika tidak,
  hilang saat sesi berakhir. Atau download manual file penting dari panel Output.

---

## 7. Resume multi-sesi di Kaggle (PENTING bila training melebihi 1 sesi)

80 epoch × ~16 mnt/epoch ≈ **>12 jam**, sedangkan 1 sesi Kaggle maks ~9–12 jam → kemungkinan butuh
**lebih dari 1 sesi** (early-stopping biasanya berhenti lebih awal dari plafon 80, tapi tetap
siapkan rencana resume). Resume sudah didukung (`cfg.resume=True`, checkpoint
`best_model_resume.pt`), persis sama dengan mekanisme yang dipakai saat ganti-akun di Colab.

**TAPI gotcha Kaggle:** `/kaggle/working` pada sesi **interaktif** TIDAK otomatis persisten saat
sesi berakhir — beda dari Drive. Supaya checkpoint selamat antar-sesi, pakai mode **batch**:

1. **Sesi 1** — jangan run interaktif sampai mati. Pakai **"Save Version" → "Save & Run All
   (Commit)"**: notebook jalan sebagai batch (sampai ~12 jam), dan **output `/kaggle/working`
   otomatis tersimpan jadi versi** (termasuk bila kena limit waktu — partial output ikut tersimpan).
   Saat commit selesai, `best_model_resume.pt` ada di output notebook versi itu.
2. **Sesi 2** — di notebook yang sama: **Add Input → Notebook Output** (pilih output versi
   sebelumnya), lalu di awal sebelum cell 11, salin checkpoint lama ke `/kaggle/working`:
   ```python
   import shutil, glob, os
   for src in glob.glob('/kaggle/input/**/best_model*.pt', recursive=True):
       dst = src.replace('/kaggle/input/' + src.split('/')[2], str(OUTPUTS_ROOT)).replace(
           os.path.dirname(src), str(CKPT_PATH.parent))
       os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst)
   ```
   (atau manual: copy `best_model.pt` + `best_model_resume.pt` ke
   `OUTPUTS_ROOT/Model_Comparison/<MODEL_KEY>/`). Lalu **Save & Run All** lagi → cell 11 resume
   dari epoch terakhir.
3. Ulang sampai 80 epoch (plafon) / early-stop tercapai.

> Re-ekstraksi tar (~5 mnt) terjadi tiap sesi (`/kaggle/temp` ephemeral) — itu wajar, training
> progress tidak hilang selama checkpoint dibawa lewat output→input di atas.

## 8. Troubleshoot
- **`pip` mau reinstall torch**: Kaggle sudah punya torch CUDA; biasanya `pip install -e .[ml]`
  skip torch (versi sudah memenuhi). Kalau memaksa downgrade, hapus `torch`/`torchvision` dari
  pin atau install manual hanya `segmentation-models-pytorch torchmetrics albumentations`.
- **Disk penuh saat ekstrak**: pastikan total ekstrak (~40GB) muat di `/kaggle/temp`
  (scratch ~besar). Input mount tidak menghitung kuota. Kalau mepet, kurangi `max_workers`.
- **Dataset >20GB ditolak**: pecah `fw-papua-valtest` jadi `val` & `test` terpisah (3 dataset);
  notebook tetap jalan (cell 2 symlink semua mount otomatis).

---

> Catatan kejujuran: metrik utama dan satu-satunya yang dilaporkan adalah `metrics.json`
> (evaluasi di TEST set Papua-holdout, cell 13). Tidak ada cell diagnostik terpisah di
> notebook ini — jangan dicampur dengan eksperimen lain yang mungkin punya metrik berbeda.
