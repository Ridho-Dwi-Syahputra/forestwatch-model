"""Konstanta proyek — single source of truth.

Nilai-nilai di sini diadopsi dari PRD v2.0 (§A.3, §A.5 Cell 9–10, §B.1.3).
Jika perlu menambah/mengubah, update juga ``configs/classes.yaml`` agar konsisten.
"""

from __future__ import annotations

# ============================================================================
# 6 KELAS TUTUPAN LAHAN (PRD §A.3)
# ============================================================================
N_CLASSES: int = 7

CLASS_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)

CLASS_NAMES: tuple[str, ...] = (
    "Perairan",
    "Hutan",
    "Lahan Terbuka",
    "Sawit",
    "Pertanian Lain",
    "Tambang",
    "Permukiman",
)

CLASS_NAMES_EN: tuple[str, ...] = (
    "Water",
    "Forest",
    "Open/Bare Land",
    "Oil Palm",
    "Other Agriculture",
    "Mining",
    "Built-up/Settlement",
)

# Palette warna untuk render PNG (PRD §A.5 Cell 10)
PALETTE_RGB: dict[int, tuple[int, int, int]] = {
    0: (42, 111, 219),   # Perairan      — biru
    1: (11, 61, 11),     # Hutan          — hijau gelap
    2: (224, 59, 36),    # Lahan Terbuka  — merah
    3: (249, 115, 22),   # Sawit          — oranye
    4: (233, 196, 106),  # Pertanian Lain — kuning
    5: (142, 36, 170),   # Tambang        — ungu
    6: (117, 117, 117),  # Permukiman     — abu-abu
}

# Hex colors untuk legend.json (PRD §B.1.3)
CLASS_COLORS: dict[int, str] = {
    0: "#2A6FDB",
    1: "#0B3D0B",
    2: "#E03B24",
    3: "#F97316",
    4: "#E9C46A",
    5: "#8E24AA",
    6: "#757575",
}

# Bobot kelas awal untuk CrossEntropy berbobot (PRD §A.5 Cell 6).
# CATATAN: hitung ulang dari distribusi label sebenarnya setelah patch siap.
# Indeks 5 = Tambang, 6 = Permukiman (kelas jarang → bobot tinggi).
CLASS_WEIGHTS_DEFAULT: tuple[float, ...] = (0.8, 0.4, 2.0, 2.0, 1.0, 2.5, 2.5)

# ============================================================================
# 4 JENIS TRANSISI DEFORESTASI (PRD §A.5 Cell 9, §B.1.3)
# ============================================================================
SOURCE_CLASS_FOREST: int = 1  # transisi dimulai dari "Hutan"

# Map: target_class_id → nama transisi (snake_case untuk JSON)
TRANSITION_MAP: dict[int, str] = {
    2: "hutan_ke_lahan_terbuka",
    3: "hutan_ke_sawit",
    4: "hutan_ke_pertanian_lain",
    5: "hutan_ke_tambang",
    6: "hutan_ke_permukiman",
}

# Warna hex untuk visualisasi transisi di WebGIS
TRANSITION_COLORS: dict[str, str] = {
    "hutan_ke_lahan_terbuka": "#7F1D1D",
    "hutan_ke_sawit": "#F97316",
    "hutan_ke_pertanian_lain": "#EAB308",
    "hutan_ke_tambang": "#8E24AA",
    "hutan_ke_permukiman": "#757575",
}

TRANSITION_LABELS: dict[str, str] = {
    "hutan_ke_lahan_terbuka": "Hutan → Lahan Terbuka",
    "hutan_ke_sawit": "Hutan → Sawit",
    "hutan_ke_pertanian_lain": "Hutan → Pertanian Lain",
    "hutan_ke_tambang": "Hutan → Tambang",
    "hutan_ke_permukiman": "Hutan → Permukiman",
}

# ============================================================================
# SENTINEL-2 BAND (PRD §A.2.1)
# ============================================================================
BANDS: tuple[str, ...] = ("B2", "B3", "B4", "B8", "B11", "B12")

BAND_DESCRIPTIONS: dict[str, str] = {
    "B2": "Blue (490 nm) - warna dasar, deteksi perairan",
    "B3": "Green (560 nm) - warna dasar vegetasi",
    "B4": "Red (665 nm) - NDVI, kontras vegetasi",
    "B8": "NIR (842 nm) - vegetasi sehat, kunci utama",
    "B11": "SWIR1 (1610 nm) - pembeda hutan vs sawit muda",
    "B12": "SWIR2 (2190 nm) - pembeda lahan terbuka/tambang & mineral",
}

N_BANDS: int = len(BANDS)
REFLECTANCE_DIVISOR: int = 10000

# ============================================================================
# GEOGRAFIS (PRD §A.5 Cell 3)
# ============================================================================
PAPUA_BBOX: tuple[float, float, float, float] = (130.0, -9.5, 141.2, 0.5)
"""minLon, minLat, maxLon, maxLat untuk seluruh wilayah Papua (6 provinsi)."""

PAPUA_CENTER: tuple[float, float] = (-4.5, 138.0)
"""(lat, lon) untuk default center map Leaflet."""

MERAUKE_CENTER: tuple[float, float] = (-8.5, 140.4)
"""(lat, lon) preset zoom studi kasus food estate Merauke."""

PROVINCES: tuple[str, ...] = (
    "Papua",
    "Papua Selatan",
    "Papua Tengah",
    "Papua Pegunungan",
    "Papua Barat",
    "Papua Barat Daya",
)

CRS_DEFAULT: str = "EPSG:4326"

# ============================================================================
# DATASET ASSET ID (PRD §A.2)
# ============================================================================
GEE_ASSETS: dict[str, str] = {
    "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
    "esa_worldcover": "ESA/WorldCover/v200/2021",
    "dynamic_world": "GOOGLE/DYNAMICWORLD/V1",
    "hansen_gfc": "UMD/hansen/global_forest_change_2025_v1_13",
    # Sawit: Forest Data Partnership Palm Probability model 2025a (ImageCollection,
    # band 'probability' 0..1, 10 m, mencakup Indonesia/Papua, selaras T2=2025).
    # Menggantikan BIOPAMA/GlobalOilPalm/v1 yang hanya snapshot 2019.
    "fdp_palm": "projects/forestdatapartnership/assets/palm/model_2025a",
    "biopama_oilpalm": "BIOPAMA/GlobalOilPalm/v1",  # legacy 2019 (referensi/cross-check)
    # Tang & Werner (2023) Global Mining Footprint — FeatureCollection poligon
    # tambang (pit, tailing, dll), didelineasi dari Sentinel-2. CC BY-4.0.
    "mining": "projects/sat-io/open-datasets/global-mining/global_mining_footprints",
    # Maus dkk. (2022) global mining polygons — DIGABUNG (union) dengan Tang &
    # Werner untuk melengkapi cakupan footprint tambang di Papua (+~38% area).
    # Tetap "footprint nyata" (bukan konsesi), jadi tak ada hutan tercap tambang.
    "mining_maus": "projects/sat-io/open-datasets/global-mining/global_mining_polygons",
}

# ============================================================================
# REMAP DYNAMIC WORLD (9 kelas) → 6 KELAS PROYEK (peta dasar label fusion)
# ============================================================================
# DW 'label' band: 0 Water, 1 Trees, 2 Grass, 3 Flooded Veg, 4 Crops,
# 5 Shrub/Scrub, 6 Built, 7 Bare, 8 Snow/Ice (Brown dkk. 2022).
# DW dipakai sebagai PETA DASAR yang lengkap (tanpa default 'Pertanian Lain'),
# lalu di-overlay tematik (Hansen→2, Mining→5, Sawit→3). Lihat label_fusion.py.
#
# CATATAN BUG RAWA (DW 3 Flooded Veg): DW kerap menandai kanopi hutan rawa Papua
# (Asmat/Mamberamo, hutan rawa terluas di Asia) sebagai Flooded Veg / Water,
# sehingga jutaan piksel HUTAN salah jadi PERAIRAN. Remap awal di bawah (3→0)
# DIPERTAHANKAN sebagai default konservatif, TETAPI label_fusion menerapkan
# *healing NDVI di sumber*: piksel Perairan (0) dgn NDVI > NDVI_FOREST_THRESHOLD
# dikembalikan ke Hutan (1). Healing ini berlaku untuk SEMUA region (termasuk
# transfer-learning luar Papua), menggantikan healing pasca-hoc manual (Bagian
# 10B notebook). Lihat ``gee/label_fusion.py`` (build_label / apply_label_fusion_numpy).
DW_TO_CLASS: dict[int, int] = {
    0: 0,  # Water           → Perairan
    1: 1,  # Trees           → Hutan
    2: 4,  # Grass           → Pertanian Lain (vegetasi non-hutan)
    3: 0,  # Flooded Veg     → Perairan (di-heal via NDVI di label_fusion: rawa berkanopi → Hutan)
    4: 4,  # Crops           → Pertanian Lain
    5: 4,  # Shrub/Scrub     → Pertanian Lain (vegetasi non-hutan)
    6: 6,  # Built           → Permukiman (lahan terbangun)
    7: 2,  # Bare            → Lahan Terbuka
    8: 2,  # Snow/Ice        → Lahan Terbuka (Puncak Jaya, sangat kecil)
}

# Ambang NDVI untuk healing hutan rawa: piksel berlabel Perairan (0) dengan
# NDVI di atas nilai ini = kanopi vegetasi lebat (mustahil air murni) →
# dikembalikan ke Hutan (1). Selaras dengan healing Bagian 10B (NDVI > 0.3).
NDVI_FOREST_THRESHOLD: float = 0.3

# ============================================================================
# OUTPUT FILE NAMES (7 file kontrak — PRD §B.1)
# ============================================================================
OUTPUT_FILES: dict[str, str] = {
    "landcover_t1_png": "landcover_2021.png",
    "landcover_t1_bounds": "landcover_2021_bounds.json",
    "landcover_t2_png": "landcover_2025.png",
    "landcover_t2_bounds": "landcover_2025_bounds.json",
    "deforestation_geojson": "deforestation.geojson",
    "statistics_json": "statistics.json",
    "legend_json": "legend.json",
    "metrics_json": "metrics.json",
    "model_onnx": "model.onnx",
    "model_card": "model_card.md",
}

# ============================================================================
# KONVERSI UNIT
# ============================================================================
DEG_TO_M: float = 111_000.0
"""Aproksimasi 1 derajat ≈ 111 km (lihat ``utils.geo`` untuk presisi)."""

SQM_PER_HA: float = 10_000.0
PIXEL_AREA_HA_10M: float = (10 * 10) / SQM_PER_HA  # 0.01 ha per pixel pada resolusi 10m
