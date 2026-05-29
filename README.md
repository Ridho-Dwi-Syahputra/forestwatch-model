# ForestWatch Papua - Model

Sistem segmentasi tutupan lahan & deteksi deforestasi Papua berbasis **Sentinel-2** dan **ResNet50-U-Net**, untuk **Statistics Essay Competition (SEC) SATRIA DATA 2026** (Universitas Andalas).

> Codebase **role Machine Learning Lead (Orang 1)** — implementasi PRD v2.0 §A. Hasil akhir adalah **7 file kontrak** yang dipakai role WebGIS (Orang 2) sebagai input dashboard.

## Apa yang ada di sini

```
model/
├── src/forestwatch/        # Package Python modular (importable)
│   ├── gee/                # Komposit S2 + label fusion 6 aturan
│   ├── data/               # Patch extraction + PyTorch dataset
│   ├── model/              # ResNet50-U-Net + combined loss
│   ├── training/           # Training loop AMP + early stopping
│   ├── inference/          # Tile inference + 4 transisi
│   ├── outputs/            # Generate 7 file kontrak PRD §B.1
│   ├── validation/         # Schema validator
│   ├── cli/                # Entry point fw-dummy, fw-validate
│   ├── constants.py        # Single source of truth
│   ├── config.py           # YAML loader
│   └── utils/              # IO, geo, logging
├── notebooks/
│   └── forestwatch_papua_full_pipeline.ipynb   # Notebook tunggal (Colab-ready)
├── scripts/                # CLI wrappers (dummy, validate, train, infer)
├── configs/                # default.yaml, classes.yaml, papua_bbox.geojson
├── tests/                  # 78 unit tests network-free
├── outputs/                # Output 7 file kontrak (gitignored)
├── pyproject.toml          # Package metadata + extras
├── requirements.txt        # Pinned deps
├── Makefile                # make setup / test / dummy / validate
└── .env.example            # Template env var
```

## Quickstart

### Lokal (Windows / macOS / Linux)

```powershell
# 1. Buat virtual env
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows
# source .venv/bin/activate            # macOS/Linux

# 2. Install package (semua extras)
pip install -e ".[all]"

# 3. Sanity check: run tests
pytest -v

# 4. Generate 7 file dummy (untuk Orang 2 / WebGIS)
python scripts/generate_dummy_data.py --out outputs/dummy --n-polygons 60

# 5. Validasi schema dummy
python scripts/validate_outputs.py --dir outputs/dummy
```

### Google Colab

Buka [notebooks/forestwatch_papua_full_pipeline.ipynb](notebooks/forestwatch_papua_full_pipeline.ipynb) di Colab (Runtime → Change runtime type → **T4 GPU**), lalu uncomment cell pertama:

```python
!git clone https://github.com/USER/forestwatch-papua-model.git forestwatch
%cd forestwatch
!pip install -q -e ".[gee,gis,ml]"
```

Notebook tunggal ini menjalankan semua tahap (Bagian 0–10) dari setup GEE, ekspor ubin, training, inferensi, sampai generate 7 file kontrak.

## Output Akhir — 7 File Kontrak (PRD §B.1)

Semua file di folder `outputs/` (atau Drive `ForestWatch_Outputs/`), CRS **EPSG:4326**:

| # | File | Isi |
| --- | --- | --- |
| 1 | `landcover_2024.png` + `landcover_2024_bounds.json` | Raster segmentasi T2 berwarna + bounds Leaflet |
| 2 | `landcover_2021.png` + `landcover_2021_bounds.json` | Raster T1 untuk slider waktu |
| 3 | `deforestation.geojson` | Poligon transisi + atribut (4 jenis transisi) |
| 4 | `statistics.json` | Statistik agregat per provinsi/transisi + metrik model |
| 5 | `legend.json` | id kelas → {nama, warna} 6 kelas |
| 6 | `metrics.json` | Metrik model lengkap (confusion matrix) |
| 7 | `model.onnx` + `model_card.md` | Model & metadata |

Lihat skema lengkap di [docs/PRD_ForestWatch_Papua_v2.md §B.1](../docs/PRD_ForestWatch_Papua_v2.md).

## Alur Mingguan

Mengikuti [docs/model/MASTER_PLAN.md](../docs/model/MASTER_PLAN.md):

| Minggu | Bagian Notebook | Output |
| --- | --- | --- |
| 1 | Bagian 0–3 | Label fusion verified + ekspor 36 ubin dimulai + **dummy untuk Orang 2** |
| 2 | Bagian 4–5 | Patch ter-cut + `best_model.pt` dengan mIoU ≥ 0,60 |
| 3 | Bagian 6–8 | 7 file kontrak lengkap di `outputs/` |
| 4 | Polish + handoff | Sync angka ke Orang 3 (esai), backup repo |

## 6 Kelas Tutupan Lahan

| ID | Kelas | Warna |
| --- | --- | --- |
| 0 | Perairan | #2A6FDB (biru) |
| 1 | Hutan | #0B3D0B (hijau gelap) |
| 2 | Deforestasi / Lahan Terbuka | #E03B24 (merah) |
| 3 | Sawit | #F97316 (oranye) |
| 4 | Pertanian Lain | #E9C46A (kuning) |
| 5 | Lahan Terbakar | #6D4C41 (coklat) |

## 4 Jenis Transisi (Deteksi Perubahan)

- `hutan_ke_lahan_terbuka` — pembukaan hutan langsung
- `hutan_ke_sawit` — ekspansi perkebunan sawit
- `hutan_ke_pertanian_lain` — pembukaan untuk pertanian non-sawit (food estate, ladang)
- `hutan_ke_terbakar` — kebakaran hutan

## Stack Teknologi

- **Citra**: Sentinel-2 SR Harmonized (6 band: B2, B3, B4, B8, B11, B12)
- **Label sources**: ESA WorldCover v200, Dynamic World V1, Hansen GFC 2024 v1.12, BIOPAMA Oil Palm v1
- **Model**: ResNet50-U-Net (segmentation-models-pytorch), opsi upgrade Attention U-Net (`unet_scse`)
- **Loss**: 0.6 × CrossEntropy(weighted) + 0.4 × DiceLoss
- **Optimizer**: AdamW (lr=1e-3, wd=1e-4) + CosineAnnealingLR
- **Compute**: Google Earth Engine (preprocessing), Google Colab T4 (training)
- **Format output**: GeoJSON, PNG, JSON, ONNX

## CLI Reference

Setelah `pip install -e .`, tersedia 2 entry point:

```bash
fw-dummy --out outputs/dummy --n-polygons 60 --seed 42
fw-validate --dir outputs/dummy
```

Plus script wrapper di `scripts/`:

```bash
python scripts/generate_dummy_data.py --out outputs/dummy
python scripts/validate_outputs.py --dir outputs/dummy
python scripts/run_export.py --year-t1 2021 --year-t2 2024   # butuh GEE auth
python scripts/run_training.py --patch-dir data/patches --ckpt-path outputs/best_model.pt
python scripts/run_inference.py --ckpt outputs/best_model.pt --tiles-t1 ... --tiles-t2 ... --mask-dir ...
python scripts/generate_outputs.py --mask-dir data/masks --out outputs/final --metrics outputs/metrics.json
```

## Override Hyperparameter

Tiga cara:

1. **Edit YAML**: `configs/default.yaml`.
2. **Environment variable**: `FW_TRAINING__BATCH_SIZE=16`, `FW_TRAINING__AMP=false`.
3. **Argumen CLI**: `--batch-size 16 --epochs 30`.

## Troubleshooting

### `rasterio` / `geopandas` install gagal di Windows

Pakai conda:

```powershell
conda install -c conda-forge rasterio geopandas fiona pyproj
pip install -e .
```

### GPU tidak terdeteksi di Colab

Runtime → Change runtime type → **T4 GPU**, lalu restart runtime dan jalankan ulang.

### `OutOfMemoryError` saat training

Turunkan batch size:

```bash
python scripts/run_training.py --patch-dir ... --batch-size 4
```

Atau patch size lebih kecil (edit `configs/default.yaml`: `patches.size: 128`).

### `ee.Initialize()` error "Project not found"

Daftar project GEE di [console.cloud.google.com/earth-engine](https://console.cloud.google.com/earth-engine), tier **Community**. Atau set env:

```bash
export GEE_PROJECT=my-other-project
```

## Testing

```bash
make test                # pytest
make test-cov            # dengan coverage report
pytest tests/test_label_fusion.py -v
```

Tests yang butuh GPU/GEE di-mark `@pytest.mark.gpu` / `@pytest.mark.gee` dan skip default. Total 78 tests network-free, run dalam ~1.5s.

## Lisensi

MIT (lihat `pyproject.toml`).

## Referensi

- [PRD v2.0](../docs/PRD_ForestWatch_Papua_v2.md) — sumber kebenaran teknis
- [Master Plan Model](../docs/model/MASTER_PLAN.md) — eksekusi step-by-step Minggu 1-4
- [Buku Panduan v2.0](../docs/Buku_Panduan_ForestWatch_Papua_v2.docx) — konteks naratif

## Kontak Tim

ForestWatch Papua Team · Universitas Andalas · SEC SATRIA DATA 2026
