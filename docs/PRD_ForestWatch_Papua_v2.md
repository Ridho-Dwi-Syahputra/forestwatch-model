# PRD v2.0 — ForestWatch Papua

### Product Requirements Document: Training Model, Integrasi WebGIS, dan Strategi Esai

**Proyek:** Sistem Deteksi Dini Deforestasi Berbasis Deep Learning  
**Kompetisi:** Statistics Essay Competition (SEC) SATRIA DATA 2026  
**Subtema 1:** Transformasi Digital dan Tata Kelola Data Nasional  
**Institusi:** Universitas Andalas  
**Deadline:** 30 Juni 2026, 16.00 WIB

**Versi dokumen:** v2.0 (revisi komprehensif pasca-validasi dataset dan tinjauan literatur)
**Status:** Final — siap diimplementasikan oleh tim

---

## 0. Ringkasan Eksekutif

Dokumen ini adalah satu-satunya sumber kebenaran (*single source of truth*) yang mengikat ketiga peran tim. Inti dari proyek tiga-peran ini bukan sekadar "membuat model lalu membuat web", melainkan **kontrak serah-terima data** antar peran: output Orang 1 adalah input Orang 2, dan angka dari keduanya adalah bahan baku Orang 3. PRD ini mendefinisikan kontrak tersebut secara presisi agar tidak ada pekerjaan yang menunggu, salah format, atau diulang.

PRD v2.0 ini menggantikan v1.0 secara keseluruhan. Perubahan utama:

1. **Lima dataset** (bukan dua) — Sentinel-2, ESA WorldCover v200, Dynamic World V1, Hansen GFC 2024 v1.12, BIOPAMA Global Oil Palm v1.
2. **Skema 6 kelas direvisi** — Hutan Primer + Hutan Sekunder digabung menjadi "Hutan"; Sawit dipisahkan dari Pertanian Lain.
3. **Label fusion 6 aturan** — *consensus labeling* antara ESA + Dynamic World, BIOPAMA untuk pemisahan sawit, Hansen + *morphological erosion* untuk deforestasi.
4. **Deteksi perubahan kaya transisi** — 4 jenis transisi deforestasi (ke Lahan Terbuka, ke Sawit, ke Pertanian Lain, ke Lahan Terbakar) yang dilaporkan terpisah.
5. **Kode lengkap** untuk semua tahap — siap ditempel ke Colab oleh Orang 1.

### 0.1. Validasi Metode terhadap Literatur (Ringkasan)

| Temuan literatur | Implikasi untuk proyek |
| --- | --- |
| U-Net adalah arsitektur segmentasi deforestasi paling banyak dipakai (≈45% studi) dan paling akurat (Jelas dkk., 2024, *Frontiers*). | Pilihan **ResNet50-U-Net dipertahankan**. |
| Attention U-Net mencapai F1 0,955–0,977 di Amazon (John & Zhang, 2022). | **Opsi upgrade** jika waktu cukup — `decoder_attention_type='scse'`. |
| Untuk tropis berawan, fusi radar Sentinel-1 adalah solusi standar (Reiche dkk.; Ballère dkk.). | Disebut sebagai **keterbatasan + pekerjaan lanjutan** di esai. |
| Label multi-kelas tutupan lahan paling praktis dari **ESA WorldCover** atau **Dynamic World** (10 m, turunan Sentinel). | Diadopsi sebagai label primer + konfirmasi. |
| DNN belajar "meta-struktur" — beberapa label salah di tepi tidak merusak (Yao dkk., 2022). | Mendukung pemakaian Hansen 30m meski di-resample ke 10m. |
| Sawit punya tanda tangan spektral stabil antar tahun (Descals dkk., 2021). | BIOPAMA 2019 dapat dipakai untuk melatih pemisahan sawit tahun mana pun. |

### 0.2. Keputusan Arsitektur Output

Untuk lomba, model **tidak** berjalan real-time di browser. Pendekatan paling tangguh untuk demo adalah ***pre-computed products***: Orang 1 menjalankan model satu kali secara offline untuk menghasilkan raster segmentasi + GeoJSON deforestasi + statistik, lalu Orang 2 cukup **menampilkan produk jadi** tersebut. Ini menghilangkan risiko WebGIS crash saat presentasi karena tidak ada inferensi berat yang berjalan saat demo. File `model.onnx` tetap dibuat untuk reproduktibilitas dan dokumentasi.

---

# BAGIAN A — PRD TRAINING MODEL (Orang 1)

## A.1. Tujuan & Kriteria Keberhasilan

Membangun model segmentasi semantik yang mengklasifikasikan tutupan lahan Papua dari citra Sentinel-2 ke dalam 6 kelas, lalu menurunkan produk deforestasi melalui deteksi perubahan antar dua periode (2021 vs 2024) beserta jenis transisinya.

| Kriteria | Target minimum | Target ideal | Catatan |
| --- | --- | --- | --- |
| Mean IoU (mIoU) semua kelas | ≥ 0,60 | ≥ 0,75 | Laporkan angka **asli**, bukan dikarang. |
| IoU kelas Deforestasi/Lahan terbuka | ≥ 0,55 | ≥ 0,70 | Kelas terpenting, biasanya tersulit. |
| IoU kelas Sawit | ≥ 0,55 | ≥ 0,70 | Kelas paling relevan dengan narasi nasional. |
| Overall Accuracy | ≥ 0,80 | ≥ 0,90 | OA mudah tinggi karena hutan dominan; mIoU lebih jujur. |
| Produk akhir tersedia untuk WebGIS | Semua file di Bagian B.1 | — | Wajib agar Orang 2 bisa bekerja. |

> **Prinsip emas:** lebih baik mIoU 0,68 yang nyata daripada 0,90 yang fiktif. Juri akan menguji angka ini di sesi tanya jawab.

## A.2. Lima Dataset yang Dipakai (FINAL)

Semua dataset tersedia gratis di Google Earth Engine. Hanya Sentinel-2 yang menjadi INPUT model. Empat dataset lain adalah sumber LABEL yang dipakai sekali pada tahap pembuatan data latih, lalu tidak pernah dipakai lagi saat inferensi.

| # | Dataset | Asset ID GEE | Peran | Tahun Data |
| --- | --- | --- | --- | --- |
| 1 | Sentinel-2 SR Harmonized | `COPERNICUS/S2_SR_HARMONIZED` | INPUT model | 2017–sekarang |
| 2 | ESA WorldCover v200 | `ESA/WorldCover/v200` | Label primer tutupan lahan | 2021 |
| 3 | Google Dynamic World V1 | `GOOGLE/DYNAMICWORLD/V1` | Label konfirmasi (consensus) | 2015–sekarang |
| 4 | Hansen GFC 2024 v1.12 | `UMD/hansen/global_forest_change_2024_v1_12` | Label deforestasi | 2000–2024 |
| 5 | BIOPAMA Global Oil Palm v1 | `BIOPAMA/GlobalOilPalm/v1` | Label pemisah sawit | 2019 |

### A.2.1. Band Sentinel-2 yang Dipakai

Dari 12 band tersedia, dipakai 6 band paling informatif:

| Band | Nama | Panjang gelombang | Alasan |
| --- | --- | --- | --- |
| B02 | Blue | 490 nm | Warna dasar, deteksi perairan |
| B03 | Green | 560 nm | Warna dasar vegetasi |
| B04 | Red | 665 nm | NDVI, kontras vegetasi |
| B08 | NIR | 842 nm | Vegetasi sehat — kunci utama |
| B11 | SWIR 1 | 1610 nm | Pembeda hutan vs sawit muda |
| B12 | SWIR 2 | 2190 nm | Deteksi bakar (dNBR) |

## A.3. Skema 6 Kelas Final (REVISI dari v1.0)

| ID | Kelas | Definisi singkat |
| --- | --- | --- |
| 0 | Perairan | Sungai, danau, laut, rawa, vegetasi tergenang |
| 1 | Hutan | Hutan alami (primer dan sekunder digabung) |
| 2 | Deforestasi / Lahan Terbuka | Area yang dulunya hutan, kini gundul |
| 3 | Sawit | Perkebunan kelapa sawit (industri & rakyat) |
| 4 | Pertanian Lain / Non-hutan | Padi, jagung, sagu, semak, rumput, terbangun |
| 5 | Lahan Terbakar | Area dengan tanda bakar (dNBR tinggi) |

> **Perubahan dari v1.0:** Hutan Primer + Hutan Sekunder digabung (label proksi terlalu lemah). Sawit dipisahkan dari Pertanian (driver deforestasi tersendiri di Indonesia).

## A.4. Aturan Label Fusion (PENTING — INTI METODOLOGI)

Label di-*generate* di GEE dengan menerapkan 6 aturan berikut **secara berurutan**. Urutan penting karena aturan berikutnya menimpa yang sebelumnya.

| Urutan | Kondisi | Label dihasilkan |
| --- | --- | --- |
| 1 | Inisialisasi semua piksel | Kelas 4 (default Pertanian Lain / Non-hutan) |
| 2 | ESA = 80, 90, 95 **DAN** Dynamic World ∈ {0, 3} | Kelas 0 (Perairan) |
| 3 | ESA = 10 **DAN** Dynamic World = 1 | Kelas 1 (Hutan) |
| 4 | ESA = 60 **ATAU** Dynamic World = 7 | Kelas 2 (Lahan Terbuka) |
| 5a | ESA = 40 (Cropland) **DAN** BIOPAMA ∈ {1, 2} | Kelas 3 (Sawit) |
| 5b | ESA = 40 (Cropland) **DAN** BIOPAMA = 0 | Kelas 4 (Pertanian Lain — eksplisit) |
| 6 | Hansen `lossyear` ≥ 19, setelah morphological erosion 1 piksel | Kelas 2 (Deforestasi — menimpa) |
| 7 | dNBR Sentinel-2 ≥ ambang batas (default 0,27) | Kelas 5 (Lahan Terbakar — menimpa) |

> **Aturan 5a sangat penting:** TIDAK semua cropland menjadi sawit. Hanya piksel yang dilabeli sawit oleh BIOPAMA yang masuk Kelas 3. Model belajar membedakan pola spektral sawit vs pertanian lain, sehingga generalisasi ke tahun mendatang tetap mungkin.

> **Kenapa Aturan 6 menimpa:** Hansen `lossyear` adalah sumber paling andal untuk peristiwa kehilangan hutan. Jika sebuah piksel diidentifikasi sebagai deforestasi oleh Hansen, label itu lebih kuat daripada label tutupan lahan dari ESA/DW.

## A.5. Pipeline & Kode Lengkap

Notebook berjalan di Google Colab (GPU T4), ditulis di VS Code yang terhubung ke runtime Colab. Perbaikan utama: normalisasi reflektansi, label multi-sumber yang benar, ekspor ubin (bukan satu GeoTIFF raksasa), augmentasi data, mixed-precision (AMP), metrik IoU benar dengan `torchmetrics`, dan deteksi perubahan dengan klasifikasi transisi.

### Cell 1 — Instalasi
```bash
!pip install -q earthengine-api geemap segmentation-models-pytorch
!pip install -q rasterio geopandas numpy matplotlib tqdm albumentations torchmetrics
!pip install -q shapely fiona pyproj
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Catatan: GPU T4 aktif via Runtime → Change runtime type → T4 GPU.

### Cell 2 — Autentikasi GEE
```python
import ee
ee.Authenticate()
ee.Initialize(project='forestwatch-papua-unand')
print('GEE siap.')
```

### Cell 3 — Definisi area, komposit S2 bebas awan, dan LABEL FUSION 6 ATURAN
```python
import ee

# Bounding box seluruh Papua (6 provinsi)
papua = ee.Geometry.Rectangle([130.0, -9.5, 141.2, 0.5])

# ====================================================================
# A. KOMPOSIT SENTINEL-2 BEBAS AWAN (untuk 2 periode)
# ====================================================================
def s2_composite(year):
    """Komposit median bebas awan untuk satu tahun, 6 band, reflektansi [0,1]."""
    start, end = f'{year}-01-01', f'{year}-12-31'
    coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(papua).filterDate(start, end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))
    # Cloud masking pakai band SCL (Scene Classification Layer)
    # Buang: 3=shadow, 8=cloud_medium, 9=cloud_high, 10=cirrus, 11=snow
    def mask_scl(img):
        scl = img.select('SCL')
        good = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
                  .And(scl.neq(10)).And(scl.neq(11)))
        return img.updateMask(good)
    bands = ['B2','B3','B4','B8','B11','B12']
    return (coll.map(mask_scl).select(bands).median()
                .divide(10000)               # reflektansi [0,1]
                .clip(papua).toFloat())

img_t1 = s2_composite(2021)   # periode-1 (pembanding)
img_t2 = s2_composite(2024)   # periode-2 (terbaru)

# ====================================================================
# B. AMBIL KEEMPAT SUMBER LABEL
# ====================================================================
# 1) ESA WorldCover v200 (label primer, 2021)
esa = ee.Image('ESA/WorldCover/v200/2021').select('Map')

# 2) Dynamic World V1 (label konfirmasi, modus tahunan 2024)
dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
        .filterBounds(papua).filterDate('2024-01-01','2024-12-31')
        .select('label').mode())

# 3) Hansen GFC 2024 v1.12 (label deforestasi)
hansen   = ee.Image('UMD/hansen/global_forest_change_2024_v1_12')
lossyear = hansen.select('lossyear')   # 0=tidak hilang, 1..24=tahun 2001..2024

# Erosi 1 piksel pada label deforestasi untuk mengatasi resolusi 30m vs 10m
defo_mask = lossyear.gte(19)                       # kehilangan 2019+
defo_eroded = defo_mask.focal_min(radius=1, units='pixels')

# 4) BIOPAMA Global Oil Palm v1 (label sawit, 2019)
oilpalm = (ee.ImageCollection('BIOPAMA/GlobalOilPalm/v1')
           .first().select('classification'))
is_oilpalm = oilpalm.eq(1).Or(oilpalm.eq(2))      # industrial OR smallholder

# 5) dNBR untuk lahan terbakar — hitung dari Sentinel-2 sendiri
def compute_dnbr(year):
    """Delta NBR antara awal dan akhir tahun = indikator bakar."""
    pre  = s2_composite(year - 1) if year > 2017 else s2_composite(year)
    post = s2_composite(year)
    nbr_pre  = pre.normalizedDifference(['B8','B12']).rename('nbr')
    nbr_post = post.normalizedDifference(['B8','B12']).rename('nbr')
    return nbr_pre.subtract(nbr_post).rename('dnbr')

dnbr = compute_dnbr(2024)
is_burned = dnbr.gte(0.27)                         # ambang umum literatur

# ====================================================================
# C. LABEL FUSION 6 ATURAN (urutan penting!)
# ====================================================================
# Aturan 1: inisialisasi semua piksel = Kelas 4 (default non-hutan/pertanian)
label = ee.Image(4).rename('label')

# Aturan 2: Perairan (ESA water/wetland/mangrove DIKONFIRMASI DW water/flooded)
is_water_esa = esa.eq(80).Or(esa.eq(90)).Or(esa.eq(95))
is_water_dw  = dw.eq(0).Or(dw.eq(3))
label = label.where(is_water_esa.And(is_water_dw), 0)

# Aturan 3: Hutan (ESA tree cover DIKONFIRMASI DW trees)
is_forest_esa = esa.eq(10)
is_forest_dw  = dw.eq(1)
label = label.where(is_forest_esa.And(is_forest_dw), 1)

# Aturan 4: Lahan Terbuka (ESA bare ATAU DW bare)
is_bare = esa.eq(60).Or(dw.eq(7))
label = label.where(is_bare, 2)

# Aturan 5a & 5b: Cropland → Sawit vs Pertanian Lain
is_cropland = esa.eq(40)
label = label.where(is_cropland.And(is_oilpalm),       3)   # SAWIT
label = label.where(is_cropland.And(is_oilpalm.Not()), 4)   # PERTANIAN LAIN (eksplisit)

# Aturan 6: Deforestasi (Hansen lossyear, sudah erosi) - MENIMPA
label = label.where(defo_eroded, 2)

# Aturan 7: Lahan Terbakar (dNBR) - MENIMPA
label = label.where(is_burned, 5)

label = label.toByte()

# ====================================================================
# D. STACK FINAL: 6 band citra + 1 band label
# ====================================================================
stack_t2 = img_t2.addBands(label)   # untuk training (citra 2024 + label)
print('Komposit dan label siap untuk ekspor.')
```

### Cell 4 — Ekspor 36 ubin ke Google Drive
```python
import ee

def make_tiles(region, nx=6, ny=6):
    """Buat grid nx x ny ubin yang menutupi region."""
    b = region.bounds().coordinates().get(0).getInfo()
    xs = [p[0] for p in b]; ys = [p[1] for p in b]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (xmax-xmin)/nx, (ymax-ymin)/ny
    tiles = []
    for i in range(nx):
        for j in range(ny):
            tiles.append(ee.Geometry.Rectangle(
                [xmin+i*dx, ymin+j*dy, xmin+(i+1)*dx, ymin+(j+1)*dy]))
    return tiles

tiles = make_tiles(papua, 6, 6)  # 36 ubin

# Ekspor T2 (citra + label) - dipakai untuk training
for k, t in enumerate(tiles):
    task = ee.batch.Export.image.toDrive(
        image=stack_t2.clip(t),
        description=f'papua_t2_tile_{k:02d}',
        folder='ForestWatch_Tiles_T2',
        region=t, scale=10, maxPixels=1e13, fileFormat='GeoTIFF')
    task.start()

# Ekspor T1 (hanya citra) - dipakai untuk deteksi perubahan
for k, t in enumerate(tiles):
    task = ee.batch.Export.image.toDrive(
        image=img_t1.clip(t),
        description=f'papua_t1_tile_{k:02d}',
        folder='ForestWatch_Tiles_T1',
        region=t, scale=10, maxPixels=1e13, fileFormat='GeoTIFF')
    task.start()

print(f'{len(tiles)*2} task ekspor dimulai. Pantau di https://code.earthengine.google.com/tasks')
```
> Task berjalan di server GEE; laptop boleh ditutup. Setelah selesai (~2–4 jam total), GeoTIFF ada di Google Drive.

### Cell 5 — Potong ubin → patch 256×256 + Dataset PyTorch
```python
from google.colab import drive; drive.mount('/content/drive')
import rasterio, numpy as np, glob, os
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader
import torch, albumentations as A

TILE_T2_DIR = '/content/drive/MyDrive/ForestWatch_Tiles_T2'
PATCH_DIR   = '/content/drive/MyDrive/ForestWatch_Patches'
os.makedirs(PATCH_DIR, exist_ok=True)
PS = 256  # patch size

def cut_patches():
    """Potong ubin T2 menjadi patch 256x256. Buang patch >30% NaN."""
    idx = 0
    for tif in sorted(glob.glob(f'{TILE_T2_DIR}/*.tif')):
        with rasterio.open(tif) as src:
            W, H = src.width, src.height
            transform = src.transform
            for r in range(0, H-PS, PS):
                for c in range(0, W-PS, PS):
                    win = Window(c, r, PS, PS)
                    arr = src.read(window=win)  # shape: (7, 256, 256)
                    if arr.shape != (7, PS, PS): continue
                    img, lab = arr[:6], arr[6].astype('uint8')
                    if np.isnan(img).mean() > 0.3: continue  # buang awan dominan
                    img = np.nan_to_num(img).astype('float32')
                    # Simpan dengan info georeferensi untuk inferensi nanti
                    np.savez_compressed(
                        f'{PATCH_DIR}/p{idx:05d}.npz',
                        img=img, lab=lab,
                        tile=os.path.basename(tif), row=r, col=c)
                    idx += 1
    print(f'{idx} patch tersimpan di {PATCH_DIR}')

# cut_patches()  # JALANKAN SEKALI

class PapuaDataset(Dataset):
    def __init__(self, files, train=True):
        self.files = files
        self.aug = A.Compose([
            A.HorizontalFlip(p=.5),
            A.VerticalFlip(p=.5),
            A.RandomRotate90(p=.5),
        ]) if train else None
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        img, lab = d['img'], d['lab']
        img = np.transpose(img, (1,2,0))  # HWC untuk albumentations
        if self.aug:
            a = self.aug(image=img, mask=lab)
            img, lab = a['image'], a['mask']
        img = np.transpose(img, (2,0,1))  # CHW kembali
        return torch.tensor(img), torch.tensor(lab, dtype=torch.long)

# Split 80/10/10
files = sorted(glob.glob(f'{PATCH_DIR}/*.npz'))
np.random.seed(42); np.random.shuffle(files)
n = len(files)
tr = files[:int(.8*n)]
va = files[int(.8*n):int(.9*n)]
te = files[int(.9*n):]
train_loader = DataLoader(PapuaDataset(tr, True),  batch_size=8, shuffle=True,  num_workers=2)
val_loader   = DataLoader(PapuaDataset(va, False), batch_size=8, shuffle=False, num_workers=2)
test_loader  = DataLoader(PapuaDataset(te, False), batch_size=8, shuffle=False, num_workers=2)
print(f'Train {len(tr)} | Val {len(va)} | Test {len(te)}')
```

### Cell 6 — Model ResNet50-U-Net (6 kelas, 6 band input)
```python
import segmentation_models_pytorch as smp
import torch, torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')   # harus: cuda

model = smp.Unet(
    encoder_name    = 'resnet50',
    encoder_weights = 'imagenet',     # transfer learning
    in_channels     = 6,              # 6 band Sentinel-2
    classes         = 6,              # 6 kelas tutupan lahan
    activation      = None,           # softmax dilakukan di loss
).to(device)

# Opsi upgrade (Attention U-Net) jika waktu tersisa:
# model = smp.Unet(..., decoder_attention_type='scse').to(device)

# Bobot kelas — atasi ketidakseimbangan (Hutan dominan)
# CATATAN: hitung frekuensi NYATA dari label setelah Cell 5 jalan,
# nilai berikut adalah aproksimasi awal yang dapat disesuaikan.
class_weights = torch.tensor([
    0.8,  # 0 Perairan
    0.4,  # 1 Hutan (dominan)
    2.0,  # 2 Deforestasi/Lahan Terbuka
    2.0,  # 3 Sawit
    1.0,  # 4 Pertanian Lain
    2.5,  # 5 Lahan Terbakar (paling jarang)
]).to(device)

ce  = nn.CrossEntropyLoss(weight=class_weights)
dice = smp.losses.DiceLoss(mode='multiclass')
loss_fn = lambda p, t: 0.6*ce(p,t) + 0.4*dice(p,t)

opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
print(f'Parameter model: {sum(p.numel() for p in model.parameters()):,}')
```

### Cell 7 — Training (AMP + early stopping + mIoU benar)
```python
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex
import torch

EPOCHS, PATIENCE = 50, 10
best_iou, wait = 0.0, 0
scaler = torch.cuda.amp.GradScaler()
iou_metric = MulticlassJaccardIndex(num_classes=6, average='macro').to(device)
CKPT = '/content/drive/MyDrive/ForestWatch_Patches/best_model.pt'

for ep in range(EPOCHS):
    # ----- TRAIN -----
    model.train(); tl = 0
    for x, y in tqdm(train_loader, desc=f'Epoch {ep+1}'):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tl += loss.item()
    sched.step()

    # ----- VALIDATE -----
    model.eval(); iou_metric.reset(); vl = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            p = model(x); vl += loss_fn(p, y).item()
            iou_metric.update(p.argmax(1), y)
    miou = iou_metric.compute().item()

    print(f'Ep {ep+1:02d} | train {tl/len(train_loader):.3f} | '
          f'val {vl/len(val_loader):.3f} | mIoU {miou:.4f}')

    if miou > best_iou:
        best_iou = miou; wait = 0
        torch.save(model.state_dict(), CKPT)
        print(f'  -> model terbaik disimpan (mIoU {best_iou:.4f})')
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f'Early stopping di epoch {ep+1}')
            break
```

### Cell 8 — Evaluasi final + ekspor ONNX + statistik per kelas
```python
import torch, numpy as np, json
from sklearn.metrics import confusion_matrix

NAMES = ['Perairan','Hutan','Deforestasi','Sawit','Pertanian Lain','Lahan Terbakar']

model.load_state_dict(torch.load(CKPT)); model.eval()
P, T = [], []
with torch.no_grad():
    for x, y in test_loader:
        P.extend(model(x.to(device)).argmax(1).cpu().numpy().ravel())
        T.extend(y.numpy().ravel())
P, T = np.array(P), np.array(T)

oa = (P==T).mean()
cm = confusion_matrix(T, P, labels=range(6))
iou = cm.diagonal() / (cm.sum(0) + cm.sum(1) - cm.diagonal() + 1e-9)
f1  = 2*iou / (1 + iou)
print(f'\nOverall Accuracy: {oa*100:.2f}%   |   mIoU: {iou.mean():.4f}\n')
print(f'{"Kelas":<18}{"IoU":>8}{"F1":>8}')
for n, name in enumerate(NAMES):
    print(f'{name:<18}{iou[n]:>8.4f}{f1[n]:>8.4f}')

# Simpan metrik dalam JSON untuk Orang 3
metrics = {
    'overall_accuracy': float(oa),
    'mean_iou': float(iou.mean()),
    'per_class': [
        {'class': NAMES[i], 'iou': float(iou[i]), 'f1': float(f1[i])}
        for i in range(6)
    ],
    'confusion_matrix': cm.tolist(),
}
with open('/content/drive/MyDrive/ForestWatch_Outputs/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Ekspor ONNX untuk dokumentasi/inference
dummy = torch.randn(1, 6, 256, 256).to(device)
torch.onnx.export(
    model, dummy,
    '/content/drive/MyDrive/ForestWatch_Outputs/model.onnx',
    input_names=['image'], output_names=['mask'],
    opset_version=13,
    dynamic_axes={'image':{0:'b'}, 'mask':{0:'b'}})
print('\nmodel.onnx tersimpan.')
```

### Cell 9 — Inferensi T1 & T2 + DETEKSI PERUBAHAN 4 TRANSISI
```python
import torch, rasterio, numpy as np, glob, os
from rasterio.windows import Window

T1_DIR = '/content/drive/MyDrive/ForestWatch_Tiles_T1'
T2_DIR = '/content/drive/MyDrive/ForestWatch_Tiles_T2'
MASK_DIR = '/content/drive/MyDrive/ForestWatch_Masks'
os.makedirs(MASK_DIR, exist_ok=True)

def infer_tile(tif_path, out_path, model, device, ps=256):
    """Inferensi satu ubin → simpan mask kelas."""
    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        meta = src.meta.copy()
        meta.update(count=1, dtype='uint8')
        # Baca semua band sekaligus, lalu inferensi per patch
        full = src.read([1,2,3,4,5,6]).astype('float32')  # (6, H, W)
        full = np.nan_to_num(full)
        mask_full = np.zeros((H, W), dtype='uint8')
        with torch.no_grad():
            for r in range(0, H-ps+1, ps):
                for c in range(0, W-ps+1, ps):
                    patch = full[:, r:r+ps, c:c+ps]
                    x = torch.tensor(patch).unsqueeze(0).to(device)
                    p = model(x).argmax(1)[0].cpu().numpy().astype('uint8')
                    mask_full[r:r+ps, c:c+ps] = p
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(mask_full, 1)

# Inferensi semua ubin untuk T1 dan T2
model.eval()
for tif in sorted(glob.glob(f'{T1_DIR}/*.tif')):
    out = os.path.join(MASK_DIR, 'mask_t1_' + os.path.basename(tif))
    infer_tile(tif, out, model, device)
for tif in sorted(glob.glob(f'{T2_DIR}/*.tif')):
    out = os.path.join(MASK_DIR, 'mask_t2_' + os.path.basename(tif))
    infer_tile(tif, out, model, device)
print('Inferensi T1 dan T2 selesai.')

# ====================================================================
# DETEKSI PERUBAHAN: 4 JENIS TRANSISI
# ====================================================================
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
import pandas as pd

# Map transisi: dari Hutan (1) ke kelas tujuan
TRANSITION_MAP = {
    2: 'hutan_ke_lahan_terbuka',   # deforestasi langsung
    3: 'hutan_ke_sawit',           # ekspansi sawit
    4: 'hutan_ke_pertanian_lain',  # pembukaan untuk pertanian
    5: 'hutan_ke_terbakar',        # kebakaran
}

all_features = []
for t1_path in sorted(glob.glob(f'{MASK_DIR}/mask_t1_*.tif')):
    t2_path = t1_path.replace('mask_t1_', 'mask_t2_')
    if not os.path.exists(t2_path): continue
    with rasterio.open(t1_path) as s1, rasterio.open(t2_path) as s2:
        m1 = s1.read(1); m2 = s2.read(1)
        transform = s1.transform; crs = s1.crs
        was_forest = (m1 == 1)
        for target_class, transition_name in TRANSITION_MAP.items():
            changed = was_forest & (m2 == target_class)
            changed_u8 = changed.astype('uint8')
            for geom, val in shapes(changed_u8, mask=changed_u8==1, transform=transform):
                poly = shape(geom)
                area_ha = poly.area * (111_000**2) / 10_000  # aprox degree→ha
                if area_ha < 0.5: continue
                all_features.append({
                    'geometry': poly,
                    'transition_type': transition_name,
                    'area_ha': round(area_ha, 2),
                    'period_from': 2021,
                    'period_to': 2024,
                })

gdf = gpd.GeoDataFrame(all_features, crs='EPSG:4326')
gdf['id'] = [f'DF-{i:05d}' for i in range(len(gdf))]
gdf.to_file('/content/drive/MyDrive/ForestWatch_Outputs/deforestation.geojson',
            driver='GeoJSON')
print(f'{len(gdf)} poligon transisi tersimpan.')

# Ringkasan statistik
summary = gdf.groupby('transition_type')['area_ha'].agg(['sum', 'count']).round(2)
print('\n--- Ringkasan Deforestasi 2021-2024 ---')
print(summary)
```

### Cell 10 — Generate file output untuk WebGIS (Kontrak B.1)
```python
import json, numpy as np, rasterio
from rasterio.merge import merge
import matplotlib.pyplot as plt
from PIL import Image
import glob, os

OUT = '/content/drive/MyDrive/ForestWatch_Outputs'
os.makedirs(OUT, exist_ok=True)

# A. Merge semua ubin mask T2 menjadi raster Papua tunggal → PNG berwarna
mask_files = sorted(glob.glob(f'{MASK_DIR}/mask_t2_*.tif'))
srcs = [rasterio.open(f) for f in mask_files]
mosaic, transform = merge(srcs)
mask = mosaic[0]  # (H, W) uint8

# Palet warna 6 kelas (lihat legend.json di Bagian B)
PALETTE = {
    0: (42, 111, 219),    # Perairan       — biru
    1: (11, 61, 11),      # Hutan          — hijau gelap
    2: (224, 59, 36),     # Deforestasi    — merah
    3: (249, 115, 22),    # Sawit          — oranye
    4: (233, 196, 106),   # Pertanian Lain — kuning
    5: (109, 76, 65),     # Lahan Terbakar — coklat
}
rgb = np.zeros((*mask.shape, 3), dtype='uint8')
for cls, color in PALETTE.items():
    rgb[mask == cls] = color
Image.fromarray(rgb).save(f'{OUT}/landcover_2024.png', optimize=True)

# Simpan bounds untuk Leaflet
bounds = rasterio.transform.array_bounds(mask.shape[0], mask.shape[1], transform)
# bounds = (minx, miny, maxx, maxy) → format Leaflet: [[miny,minx], [maxy,maxx]]
with open(f'{OUT}/landcover_2024_bounds.json', 'w') as f:
    json.dump({
        'bounds': [[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        'crs': 'EPSG:4326'
    }, f, indent=2)

# B. Ulangi untuk T1 (slider waktu)
mask_t1_files = sorted(glob.glob(f'{MASK_DIR}/mask_t1_*.tif'))
srcs_t1 = [rasterio.open(f) for f in mask_t1_files]
mosaic_t1, _ = merge(srcs_t1)
rgb_t1 = np.zeros((*mosaic_t1[0].shape, 3), dtype='uint8')
for cls, color in PALETTE.items():
    rgb_t1[mosaic_t1[0] == cls] = color
Image.fromarray(rgb_t1).save(f'{OUT}/landcover_2021.png', optimize=True)

# C. legend.json
legend = [
    {'id': 0, 'name': 'Perairan',         'color': '#2A6FDB'},
    {'id': 1, 'name': 'Hutan',            'color': '#0B3D0B'},
    {'id': 2, 'name': 'Deforestasi',      'color': '#E03B24'},
    {'id': 3, 'name': 'Sawit',            'color': '#F97316'},
    {'id': 4, 'name': 'Pertanian Lain',   'color': '#E9C46A'},
    {'id': 5, 'name': 'Lahan Terbakar',   'color': '#6D4C41'},
]
with open(f'{OUT}/legend.json', 'w') as f:
    json.dump(legend, f, indent=2)

# D. statistics.json (gabungan metrik + agregat)
with open(f'{OUT}/metrics.json') as f:
    metrics = json.load(f)

# Hitung luas per kelas (ha)
unique, counts = np.unique(mask, return_counts=True)
px_to_ha = (10*10) / 10_000   # 1 piksel 10m = 0.01 ha
per_class_ha = {NAMES[int(c)]: round(int(n)*px_to_ha, 1) for c, n in zip(unique, counts)}

# Agregat per transisi (dari Cell 9)
import geopandas as gpd
gdf = gpd.read_file(f'{OUT}/deforestation.geojson')
per_transition = gdf.groupby('transition_type')['area_ha'].sum().round(1).to_dict()
total_def_ha = float(gdf['area_ha'].sum())

statistics = {
    'period_from': 2021,
    'period_to': 2024,
    'total_deforestation_ha': round(total_def_ha, 1),
    'n_hotspots': len(gdf),
    'per_transition_ha': per_transition,
    'per_class_area_ha': per_class_ha,
    'model_metrics': metrics,
}
with open(f'{OUT}/statistics.json', 'w') as f:
    json.dump(statistics, f, indent=2)

print('Semua file output siap di:', OUT)
print('File:', os.listdir(OUT))
```

## A.6. Risiko Teknis & Mitigasi

| Risiko | Mitigasi |
| --- | --- |
| T4 kehabisan memori pada 256×256×6, batch 8 | AMP sudah dipakai; turunkan batch ke 4 atau patch ke 128 bila perlu. |
| Patch didominasi NaN karena awan Papua | Filter NaN > 30% sudah ada; perpanjang rentang tanggal komposit. |
| Ketidakseimbangan kelas (Hutan dominan) | Weighted CE + Dice; laporkan IoU per kelas, bukan hanya OA. |
| Sawit muda (0–3 tahun) tidak terdeteksi | Akui jujur di esai; gunakan deteksi perubahan untuk tangkap ekspansi pasca-2019. |
| Kelas Lahan Terbakar terlalu sedikit | Gabung ke Kelas 2 (Deforestasi); turunkan jumlah kelas ke 5. |
| Hansen 30m vs Sentinel 10m di batas hutan | `focal_min(radius=1)` sudah diterapkan (erosi 1 piksel). |
| Sesi Colab terputus saat training | Checkpoint disimpan tiap epoch ke Drive. |
| GEE export timeout | Pembagian ke 36 ubin sudah mengurangi risiko; jalankan ulang ubin yang gagal. |

---

# BAGIAN B — KONTRAK OUTPUT & PRD WEBGIS

## B.1. Kontrak Serah-Terima (WAJIB dipatuhi)

Orang 1 menaruh semua file ini di folder Drive `ForestWatch_Outputs/`. Semua memakai sistem koordinat **EPSG:4326 (WGS84)**.

| No | Nama file | Format | Isi | Dipakai untuk |
| --- | --- | --- | --- | --- |
| 1 | `landcover_2024.png` + `landcover_2024_bounds.json` | PNG + JSON | Raster segmentasi T2 berwarna + bounds | Layer peta utama |
| 2 | `landcover_2021.png` + `landcover_2021_bounds.json` | PNG + JSON | Raster T1 untuk slider waktu | Layer pembanding |
| 3 | `deforestation.geojson` | GeoJSON | Poligon transisi + atribut | Marker & popup |
| 4 | `statistics.json` | JSON | Statistik + metrik model | Panel statistik & grafik |
| 5 | `legend.json` | JSON | id kelas → {nama, warna} | Legenda peta |
| 6 | `metrics.json` | JSON | Metrik model lengkap | Tabel metrik di esai |
| 7 | `model.onnx` + `model_card.md` | ONNX + teks | Model & metadata | Reproduktibilitas |

### B.1.1. Skema `deforestation.geojson` (LENGKAP)

```json
{
  "type": "FeatureCollection",
  "name": "deforestation_2021_2024",
  "crs": { "type": "name", "properties": { "name": "EPSG:4326" } },
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[140.42, -8.31], [140.45, -8.31], [140.45, -8.28], [140.42, -8.28], [140.42, -8.31]]]
      },
      "properties": {
        "id": "DF-00001",
        "transition_type": "hutan_ke_sawit",
        "area_ha": 12.4,
        "period_from": 2021,
        "period_to": 2024,
        "province": "Papua Selatan",
        "kawasan_status": "APL / Food Estate Merauke"
      }
    }
  ]
}
```

**Nilai `transition_type` yang mungkin:**
- `"hutan_ke_lahan_terbuka"` — pembukaan langsung (deforestasi murni)
- `"hutan_ke_sawit"` — ekspansi perkebunan sawit
- `"hutan_ke_pertanian_lain"` — pembukaan untuk pertanian non-sawit
- `"hutan_ke_terbakar"` — kebakaran hutan

### B.1.2. Skema `statistics.json` (LENGKAP)

```json
{
  "period_from": 2021,
  "period_to": 2024,
  "total_deforestation_ha": 18450.7,
  "n_hotspots": 312,
  "per_transition_ha": {
    "hutan_ke_lahan_terbuka": 6200.1,
    "hutan_ke_sawit": 8120.4,
    "hutan_ke_pertanian_lain": 3100.5,
    "hutan_ke_terbakar": 1029.7
  },
  "per_province": [
    { "province": "Papua Selatan",   "deforestation_ha": 9800.2 },
    { "province": "Papua Tengah",    "deforestation_ha": 3120.5 },
    { "province": "Papua",           "deforestation_ha": 2150.0 },
    { "province": "Papua Pegunungan","deforestation_ha": 880.0 },
    { "province": "Papua Barat",     "deforestation_ha": 1500.0 },
    { "province": "Papua Barat Daya","deforestation_ha": 1000.0 }
  ],
  "per_class_area_ha": {
    "Perairan": 0,
    "Hutan": 0,
    "Deforestasi": 0,
    "Sawit": 0,
    "Pertanian Lain": 0,
    "Lahan Terbakar": 0
  },
  "model_metrics": {
    "overall_accuracy": 0.86,
    "mean_iou": 0.71,
    "per_class": [
      {"class": "Perairan",       "iou": 0.91, "f1": 0.95},
      {"class": "Hutan",          "iou": 0.94, "f1": 0.97},
      {"class": "Deforestasi",    "iou": 0.66, "f1": 0.79},
      {"class": "Sawit",          "iou": 0.62, "f1": 0.76},
      {"class": "Pertanian Lain", "iou": 0.71, "f1": 0.83},
      {"class": "Lahan Terbakar", "iou": 0.55, "f1": 0.71}
    ]
  }
}
```

### B.1.3. Skema `legend.json` (LENGKAP)

```json
[
  { "id": 0, "name": "Perairan",       "color": "#2A6FDB" },
  { "id": 1, "name": "Hutan",          "color": "#0B3D0B" },
  { "id": 2, "name": "Deforestasi",    "color": "#E03B24" },
  { "id": 3, "name": "Sawit",          "color": "#F97316" },
  { "id": 4, "name": "Pertanian Lain", "color": "#E9C46A" },
  { "id": 5, "name": "Lahan Terbakar", "color": "#6D4C41" }
]
```

**Warna transisi (untuk WebGIS):**
- `hutan_ke_lahan_terbuka` → `#7F1D1D` (merah tua)
- `hutan_ke_sawit` → `#F97316` (oranye)
- `hutan_ke_pertanian_lain` → `#EAB308` (kuning)
- `hutan_ke_terbakar` → `#6D4C41` (coklat)

## B.2. PRD WebGIS (Orang 2)

### B.2.1. Stack Teknologi (REKOMENDASI)

**Pilihan utama:** Leaflet.js (vanilla JS) + GitHub Pages hosting statis.

Alasan: tidak ada server yang bisa mati saat presentasi, gratis, dan cukup mengonsumsi file statis dari kontrak B.1. Flask hanya diperlukan jika ingin fitur inferensi langsung dengan `onnxruntime` — opsional dan berisiko, tidak diprioritaskan.

**Library wajib:**
- Leaflet 1.9+ (peta)
- Leaflet.markercluster (clustering bila titik > 200)
- Chart.js 4+ (grafik statistik)
- Tailwind CSS via CDN (styling cepat)

**Alternatif:** Streamlit + Folium (Python penuh) bila tim lebih nyaman.

### B.2.2. Struktur File WebGIS

```
forestwatch-webgis/
├── index.html                  # halaman utama
├── css/
│   └── style.css               # styling tambahan
├── js/
│   ├── app.js                  # logika utama
│   ├── map.js                  # inisialisasi Leaflet + layer
│   ├── stats.js                # rendering panel statistik
│   └── controls.js             # slider waktu, toggle layer
├── data/                       # dari Orang 1 (copy ke sini)
│   ├── landcover_2024.png
│   ├── landcover_2024_bounds.json
│   ├── landcover_2021.png
│   ├── landcover_2021_bounds.json
│   ├── deforestation.geojson
│   ├── statistics.json
│   └── legend.json
├── assets/
│   └── logo.png
└── README.md
```

### B.2.3. Fitur Wajib & Spesifikasi UI

| Fitur | Komponen Leaflet | Sumber data | Behavior |
| --- | --- | --- | --- |
| Peta dasar Papua | `L.tileLayer` (OSM atau Esri Satellite) | tile provider | Default zoom 6, center [-4.5, 138] |
| Overlay tutupan lahan | `L.imageOverlay(png, bounds)` | `landcover_2024.png` + bounds | Opacity slider (default 0.7) |
| Slider waktu T1↔T2 | `L.control.layers` atau custom | dua PNG | Toggle 2021/2024 instan |
| Marker deforestasi | `L.geoJSON` dengan styling per transisi | `deforestation.geojson` | Cluster bila >200 marker |
| Popup detail | `bindPopup` | properti GeoJSON | Tampilkan: ID, jenis transisi, luas, provinsi, status kawasan |
| Filter jenis transisi | Custom checkbox | filter `feature.properties.transition_type` | 4 checkbox (default semua aktif) |
| Panel statistik | HTML/Tailwind sidebar | `statistics.json` | Total ha, hotspots, breakdown transisi |
| Grafik per transisi | Chart.js pie chart | `per_transition_ha` | Klik segmen → filter peta |
| Grafik per provinsi | Chart.js bar chart | `per_province` | Klik bar → zoom ke provinsi |
| Tabel metrik model | HTML table | `model_metrics.per_class` | mIoU, OA, IoU per kelas |
| Legenda | Custom box bottom-right | `legend.json` | 6 kotak warna + nama |
| Tombol "Pusat ke Merauke" | Custom button | preset coordinate | Zoom ke [-8.5, 140.4], zoom 9 |
| Studi kasus Merauke | Modal/sidebar | filter province | Highlight semua poligon Papua Selatan |
| Tombol unduh data | Anchor download | file dari `data/` | GeoJSON & CSV |
| Tentang & Metodologi | Modal | static HTML | Penjelasan singkat metode |

### B.2.4. Mockup Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  ForestWatch Papua                          [Tentang] [Unduh]   │ ← Header
├──────────────────────────────────┬──────────────────────────────┤
│                                  │  PANEL STATISTIK            │
│                                  │  ┌────────────────────────┐ │
│                                  │  │ Total Deforestasi     │ │
│                                  │  │   18.450,7 ha         │ │
│           PETA LEAFLET           │  │   312 hotspot         │ │
│           (Papua + overlay)      │  └────────────────────────┘ │
│                                  │  ┌────────────────────────┐ │
│                                  │  │ Per Jenis Transisi    │ │
│                                  │  │  [Pie Chart]          │ │
│           [Tombol Merauke]       │  └────────────────────────┘ │
│                                  │  ┌────────────────────────┐ │
│   [Legenda]                      │  │ Per Provinsi          │ │
│                                  │  │  [Bar Chart]          │ │
│   [Slider 2021 ↔ 2024]           │  └────────────────────────┘ │
│   [□ Sawit] [□ Lainnya]          │  ┌────────────────────────┐ │
│                                  │  │ Akurasi Model         │ │
│                                  │  │  mIoU 0,71 | OA 86%  │ │
│                                  │  └────────────────────────┘ │
└──────────────────────────────────┴──────────────────────────────┘
```

### B.2.5. Code Skeleton — `js/map.js`

```javascript
// Inisialisasi peta
const map = L.map('map', { center: [-4.5, 138], zoom: 6, minZoom: 5, maxZoom: 12 });

// Basemap
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap'
}).addTo(map);

// State global
const state = {
  currentYear: 2024,
  landcoverLayers: {},
  deforestationLayer: null,
  filters: {
    hutan_ke_lahan_terbuka: true,
    hutan_ke_sawit: true,
    hutan_ke_pertanian_lain: true,
    hutan_ke_terbakar: true,
  }
};

// Muat layer landcover
async function loadLandcover(year) {
  const bounds = await fetch(`data/landcover_${year}_bounds.json`).then(r => r.json());
  const layer = L.imageOverlay(`data/landcover_${year}.png`, bounds.bounds, {
    opacity: 0.7, interactive: false
  });
  state.landcoverLayers[year] = layer;
  return layer;
}

// Muat deforestasi
async function loadDeforestation() {
  const geojson = await fetch('data/deforestation.geojson').then(r => r.json());
  const transitionColors = {
    hutan_ke_lahan_terbuka: '#7F1D1D',
    hutan_ke_sawit:          '#F97316',
    hutan_ke_pertanian_lain: '#EAB308',
    hutan_ke_terbakar:       '#6D4C41',
  };
  state.deforestationLayer = L.geoJSON(geojson, {
    filter: f => state.filters[f.properties.transition_type],
    style: f => ({
      color: transitionColors[f.properties.transition_type] || '#333',
      weight: 1, fillOpacity: 0.6
    }),
    onEachFeature: (f, layer) => {
      const p = f.properties;
      layer.bindPopup(`
        <strong>${p.id}</strong><br>
        Jenis: ${p.transition_type.replaceAll('_', ' ')}<br>
        Luas: ${p.area_ha} ha<br>
        Provinsi: ${p.province}<br>
        Periode: ${p.period_from} → ${p.period_to}<br>
        Kawasan: ${p.kawasan_status}
      `);
    }
  }).addTo(map);
}

// Slider waktu
function switchYear(year) {
  Object.values(state.landcoverLayers).forEach(l => map.removeLayer(l));
  state.landcoverLayers[year].addTo(map);
  state.currentYear = year;
}

// Filter transisi
function toggleTransition(type) {
  state.filters[type] = !state.filters[type];
  map.removeLayer(state.deforestationLayer);
  loadDeforestation();   // reload dengan filter baru
}

// Tombol Merauke
function focusMerauke() {
  map.flyTo([-8.5, 140.4], 9);
}

// Inisialisasi
(async () => {
  await loadLandcover(2021);
  await loadLandcover(2024);
  switchYear(2024);
  await loadDeforestation();
  // tampilkan legend, panel statistik, dll
})();
```

### B.2.6. Milestone Orang 2

| Minggu | Tugas | Output |
| --- | --- | --- |
| 1 | Kerangka peta kosong Papua + struktur folder | `index.html` dengan basemap |
| 2 | Layer dummy (data palsu sesuai kontrak B.1) + UI penuh | WebGIS fungsional dengan dummy |
| 3 | Ganti dummy dengan data asli dari Orang 1 | WebGIS dengan data nyata |
| 4 | Deploy GitHub Pages + video demo + polish | URL live + video 2 menit |

> **Trik koordinasi:** Orang 2 TIDAK boleh menunggu model selesai. Begitu skema B.1 disepakati, buat file dummy `deforestation.geojson` dan `statistics.json` berisi data karangan untuk membangun seluruh UI. Saat data asli datang di Minggu 3, cukup ganti file — UI sudah jadi.

---

# BAGIAN C — STRATEGI ESAI (Orang 3)

## C.1. Kerangka & Aturan Format (dari Panduan SEC)

Esai 2000–3000 kata, Bahasa Indonesia PUEBI, struktur **Pendahuluan – Pembahasan – Penutup tanpa sub-judul**, A4 margin 3 cm, Times New Roman 12, spasi 1,5, justify, ID tim di kanan atas tiap halaman, Turnitin < 25%.

**Bobot penilaian penyisihan:** Substansi & Data 40%, Orisinalitas 20%, Penulisan 15%, Kesesuaian Tema 15%, Penarikan Kesimpulan 10%.

## C.2. Struktur Esai (Paragraf demi Paragraf)

### Pendahuluan (≈600 kata, 3–4 paragraf)

**Paragraf 1 — Hook dan urgensi.**
Mulai dengan fakta menohok: laju deforestasi Papua 2025 melonjak 348% dipicu food estate Merauke. Jelaskan Papua sebagai cadangan hutan tropis terakhir Indonesia dan posisinya dalam strategi iklim nasional menuju Indonesia Emas 2045. Tekankan paradoks: data deforestasi resmi sering terlambat, sementara kehilangan hutan terjadi terus-menerus.

**Paragraf 2 — Masalah tata kelola data.**
Bingkai sebagai isu tata kelola data nasional (sesuai Subtema 1). Data deforestasi yang lambat, tertutup, atau berbeda antar instansi menghambat kebijakan berbasis bukti. Sebut: KLHK, Global Forest Watch, dan WRI sering berbeda angka. Argumen: solusinya adalah sistem pemantauan otomatis berbasis citra satelit yang transparan, dapat diaudit, dan diperbarui rutin.

**Paragraf 3 — Solusi yang ditawarkan.**
Perkenalkan ForestWatch Papua: sistem deteksi dini deforestasi berbasis deep learning (ResNet50-U-Net) menggunakan citra Sentinel-2 dan integrasi multi-dataset label. Sebut keunikan: tidak hanya mendeteksi *hilangnya hutan*, tetapi juga *jenis transisi* (menjadi sawit, pertanian, terbakar, lahan terbuka) — informasi yang langsung relevan untuk kebijakan.

**Paragraf 4 — Tujuan dan kontribusi esai.**
Nyatakan tiga tujuan eksplisit: (1) memetakan tutupan lahan Papua tahun 2021 dan 2024, (2) mendeteksi peristiwa deforestasi beserta jenisnya, (3) membangun prototipe WebGIS yang dapat diakses publik sebagai pilot model tata kelola data kehutanan yang transparan.

### Pembahasan (≈1800 kata, 8–10 paragraf)

**Paragraf 5 — Mengapa Sentinel-2 dan deep learning.**
Justifikasi pemilihan teknologi. Sentinel-2: gratis, resolusi 10m, revisit 5 hari, 6 band relevan (termasuk SWIR untuk membedakan sawit muda dari hutan). Deep learning: U-Net adalah arsitektur paling akurat untuk segmentasi deforestasi (kutip Jelas dkk. 2024, F1 0,955+ John & Zhang 2022). Sebut transfer learning ResNet50 untuk efisiensi.

**Paragraf 6 — Strategi label multi-sumber (METODOLOGI INTI).**
**Inilah paragraf paling penting dari sisi metodologi.** Jelaskan secara jujur: tidak ada satu dataset tunggal yang dapat melabeli keenam kelas dengan kualitas tinggi. Solusinya adalah label fusion lima sumber: ESA WorldCover (label primer), Dynamic World (konfirmasi consensus), Hansen GFC (deforestasi historis), BIOPAMA Oil Palm (pemisah sawit), dan dNBR Sentinel-2 (deteksi bakar). Sebut consensus labeling sebagai praktik standar yang meningkatkan kualitas label.

**Paragraf 7 — Tantangan teknis dan solusinya.**
Akui jujur tantangan: tutupan awan Papua tinggi (solusi: median compositing 30 hari + cloud masking SCL), Hansen 30m vs Sentinel 10m (solusi: morphological erosion 1 piksel), sawit muda sulit dibedakan dari semak (solusi: ditangkap via deteksi perubahan). Kejujuran ini menambah kredibilitas — juri sangat menghargainya.

**Paragraf 8 — Pelatihan model dan validasi.**
Jelaskan: 36 ubin diekspor lalu dipotong jadi patch 256×256, dilatih 50 epoch dengan AdamW, loss kombinasi CrossEntropy berbobot + Dice, mixed-precision di GPU T4 Colab, early stopping. **Laporkan angka ASLI.** Contoh: "Model mencapai mIoU 0,71 dengan Overall Accuracy 86%. Kelas Sawit memperoleh IoU 0,62 dan kelas Deforestasi 0,66" — sesuaikan dengan hasil sebenarnya.

**Paragraf 9 — Hasil pemetaan tutupan lahan.**
Sajikan luas masing-masing kelas di seluruh Papua 2024. Bandingkan dengan 2021. Tekankan: Hutan tutupan masih dominan, tetapi luasannya berkurang X ribu hektar dalam 3 tahun. Ekspansi sawit terjadi terutama di Papua Selatan. Sertakan gambar peta hasil segmentasi atau tangkapan layar WebGIS.

**Paragraf 10 — Hasil deteksi perubahan: 4 jenis transisi.**
**Inilah temuan paling menarik.** Sajikan tabel/paragraf: total deforestasi X ha, dengan breakdown — X ha menjadi sawit (Y%), X ha menjadi pertanian lain, X ha menjadi lahan terbuka, X ha akibat kebakaran. Komentari: ekspansi sawit adalah driver utama, ATAU pembukaan langsung dominan, dst — sesuaikan temuan asli.

**Paragraf 11 — Studi kasus food estate Merauke.**
Zoom ke Papua Selatan. Sajikan koordinat, luas, dan jenis transisi spesifik kawasan food estate. Bandingkan dengan citra historis. Diskusikan implikasi: apakah proyek food estate benar-benar di lahan terdegradasi seperti klaim, atau menggusur hutan primer? Ini bagian dengan dampak naratif paling kuat.

**Paragraf 12 — WebGIS sebagai pilot tata kelola data.**
Jelaskan WebGIS yang dibangun: layer interaktif, filter jenis transisi, slider waktu, panel statistik per provinsi. Sertakan URL live. Argumen kebijakan: ini adalah model tata kelola data kehutanan yang transparan, dapat diaudit publik, dan dapat menjadi acuan untuk integrasi data lintas instansi (KLHK, BIG, BPN).

**Paragraf 13 — Diskusi keterbatasan dan pekerjaan lanjutan.**
Sebut secara jujur: (1) tutupan awan masih menghambat — sebut fusi Sentinel-1 radar sebagai solusi standar dalam literatur (Reiche dkk., Ballère dkk.), (2) BIOPAMA hanya tahun 2019 — sawit muda pasca-2019 perlu validasi tambahan, (3) label berasal dari produk global yang punya bias regional — perlu ground truthing untuk Papua di pekerjaan lanjutan, (4) confusion matrix menunjukkan kelas terbakar masih lemah.

### Penutup (≈400 kata, 2 paragraf)

**Paragraf 14 — Kesimpulan.**
Rangkum tiga hasil utama: peta tutupan lahan Papua 2024 dengan akurasi terukur, deteksi X ha peristiwa deforestasi dengan klasifikasi jenis transisi, dan WebGIS yang dapat diakses publik. Nyatakan: tujuan esai tercapai.

**Paragraf 15 — Rekomendasi kebijakan dan visi 2045.**
Tutup dengan rekomendasi konkret: (1) integrasi sistem otomatis seperti ini ke dalam SiPongi KLHK, (2) keterbukaan data deforestasi sebagai bagian dari Satu Data Indonesia, (3) penggunaan klasifikasi transisi (bukan hanya total ha) untuk merancang intervensi yang tepat sasaran — moratorium berbeda untuk sawit vs food estate vs kebakaran. Hubungkan dengan target Indonesia Emas 2045: ekonomi berkembang tanpa kehilangan paru-paru tropis terakhir.

## C.3. Yang Harus Ditekankan (Dipetakan ke Bobot Penilaian)

**Substansi & Data (40% — prioritas tertinggi).**
- Angka **nyata** dari model: total ha per transisi, luas Sawit Papua, mIoU per kelas. Ini pembeda utama dari peserta lain.
- Studi kasus Merauke dengan koordinat dan luas spesifik.
- Confusion matrix lengkap di lampiran atau ditampilkan inline.

**Kesesuaian Tema (15%).**
Bingkai sebagai instrumen **tata kelola data kehutanan nasional**, bukan proyek AI semata. Hubungkan dengan: Satu Data Indonesia, evidence-based policy, transparansi publik, Indonesia Emas 2045.

**Orisinalitas (20%).**
WebGIS yang berfungsi + model yang dilatih sendiri adalah bukti orisinalitas. Klasifikasi 4 jenis transisi adalah pembeda jelas dari laporan deforestasi biasa. Turnitin < 25% — parafrase penuh, jangan menyalin.

**Penulisan (15%) & Penarikan Kesimpulan (10%).**
Alur logis. Setiap klaim didukung data atau referensi. Kesimpulan menjawab tujuan dengan presisi.

## C.4. Keterbatasan yang Menambah Kredibilitas (Wajib Disebut)

| Keterbatasan | Cara Disebut |
| --- | --- |
| Tutupan awan tinggi | "Solusi standar dalam literatur adalah fusi Sentinel-1 radar (Reiche dkk., 2018) yang menjadi pekerjaan lanjutan." |
| Hansen 30m → resampling 10m | "Diatasi dengan morphological erosion 1 piksel untuk mengurangi noise tepi batas hutan." |
| BIOPAMA hanya 2019 | "Model belajar pola spektral sawit, bukan koordinat. Ekspansi pasca-2019 ditangkap via deteksi perubahan." |
| Sawit muda sulit dideteksi | "Keterbatasan fisik sensor optik — sebelum kanopi menutup, sawit muda spektralnya mirip semak." |
| Label dari produk global | "Bias regional dimungkinkan; validasi ground truthing Papua adalah pekerjaan lanjutan." |
| Pemisahan hutan primer/sekunder | "Tidak dilakukan karena tidak ada dataset gratis yang andal — kelas digabung jadi 'Hutan'." |

## C.5. Daftar Referensi (Sudah Terverifikasi)

1. Md Jelas, I., Zulkifley, M. A., Abdullah, M., & Spraggon, M. (2024). *Deforestation detection using deep learning-based semantic segmentation techniques: a systematic review.* Frontiers in Forests and Global Change, 7, 1300060.
2. John, D., & Zhang, C. (2022). *An attention-based U-Net for detecting deforestation within satellite sensor imagery.* International Journal of Applied Earth Observation and Geoinformation, 107, 102685.
3. Reiche, J., dkk. (2018). *Improving near-real time deforestation monitoring in tropical dry forests by combining dense Sentinel-1 time series with Landsat and ALOS-2 PALSAR-2.* Remote Sensing of Environment, 204.
4. Ballère, M., Bouvet, A., Mermoz, S., Le Toan, T., Koleck, T., Bedeau, L., André, M., Forestier, E., Frison, P. L., & Lardeux, C. (2018). *Use of the SAR Shadowing Effect for Deforestation Detection with Sentinel-1 Time Series.* Remote Sensing, 10(8), 1250.
5. Hansen, M. C., dkk. (2024). *Global Forest Change 2024 v1.12* (dataset). University of Maryland.
6. Brown, C. F., dkk. (2022). *Dynamic World, Near real-time global 10 m land use land cover mapping.* Scientific Data, 9, 251.
7. Zanaga, D., dkk. (2022). *ESA WorldCover 10 m 2021 v200.* DOI: 10.5281/zenodo.7254221.
8. Descals, A., Wich, S., Meijaard, E., Gaveau, D. L. A., Peedell, S., & Szantoi, Z. (2021). *High-resolution global map of smallholder and industrial closed-canopy oil palm plantations.* Earth System Science Data, 13(3), 1211–1231.
9. Austin, K. G., Schwantes, A., Gu, Y., & Kasibhatla, P. S. (2019). *What causes deforestation in Indonesia?* Environmental Research Letters, 14(2), 024007.

> Tambahkan sumber statistik deforestasi Papua dari KLHK / Global Forest Watch untuk angka pembanding di Pendahuluan.

---

## Lampiran 1 — Ringkasan Kontrak Antar Peran

| Dari | Ke | Yang diserahkan | Kapan |
| --- | --- | --- | --- |
| Orang 1 | Orang 2 | Semua file di Bagian B.1 | Akhir Minggu 3 |
| Orang 1 | Orang 3 | mIoU, IoU per kelas, OA, breakdown transisi, data Merauke | Akhir Minggu 3 |
| Orang 2 | Orang 3 | URL WebGIS live + tangkapan layar | Awal Minggu 4 |
| Orang 3 | Tim | Naskah esai final + surat orisinalitas | Sebelum 30 Juni 16.00 WIB |

## Lampiran 2 — Checklist Akhir (Sebelum Submit)

**Model & Data (Orang 1)**
- [ ] Semua 7 file output di `ForestWatch_Outputs/` lengkap dan tervalidasi
- [ ] Metrik model dilaporkan jujur dan konsisten antar file
- [ ] Studi kasus Merauke punya koordinat spesifik
- [ ] Confusion matrix tersimpan
- [ ] Kode `.ipynb` di GitHub repo tim

**WebGIS (Orang 2)**
- [ ] WebGIS dapat diakses via URL publik (GitHub Pages)
- [ ] Semua 7 fitur wajib berfungsi
- [ ] Filter 4 jenis transisi berfungsi
- [ ] Studi kasus Merauke punya tombol/zoom preset
- [ ] Video demo 2 menit tersedia
- [ ] Tested di browser Chrome dan Safari

**Esai (Orang 3)**
- [ ] Jumlah kata 2000–3000 (tidak termasuk daftar pustaka)
- [ ] Format A4, margin 3 cm, TNR 12, spasi 1,5, justify
- [ ] Struktur Pendahuluan – Pembahasan – Penutup tanpa sub-judul
- [ ] ID tim di kanan atas tiap halaman
- [ ] Turnitin Similarity < 25%
- [ ] Angka di esai konsisten dengan `statistics.json`
- [ ] URL WebGIS tertera di esai
- [ ] Surat orisinalitas bermeterai 10.000 + TTD pimpinan PT
- [ ] Naming file: `SEC_(ID tim).pdf` dan `SEC_Orisinalitas_(ID tim).pdf`
- [ ] Diunggah ke portal sebelum 30 Juni 16.00 WIB

---

*ForestWatch Papua — PRD v2.0 · Disusun untuk SEC SATRIA DATA 2026 · Universitas Andalas*
*Dokumen ini menggantikan PRD v1.0 secara keseluruhan.*
