"""Konstanta proyek — single source of truth.

Nilai-nilai di sini diadopsi dari PRD v2.0 (§A.3, §A.5 Cell 9–10, §B.1.3).
Jika perlu menambah/mengubah, update juga ``configs/classes.yaml`` agar konsisten.
"""

from __future__ import annotations

# ============================================================================
# 6 KELAS TUTUPAN LAHAN (PRD §A.3)
# ============================================================================
N_CLASSES: int = 6

CLASS_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

CLASS_NAMES: tuple[str, ...] = (
    "Perairan",
    "Hutan",
    "Deforestasi",
    "Sawit",
    "Pertanian Lain",
    "Lahan Terbakar",
)

CLASS_NAMES_EN: tuple[str, ...] = (
    "Water",
    "Forest",
    "Deforestation/Bare",
    "Oil Palm",
    "Other Agriculture",
    "Burned",
)

# Palette warna untuk render PNG (PRD §A.5 Cell 10)
PALETTE_RGB: dict[int, tuple[int, int, int]] = {
    0: (42, 111, 219),   # Perairan      — biru
    1: (11, 61, 11),     # Hutan          — hijau gelap
    2: (224, 59, 36),    # Deforestasi    — merah
    3: (249, 115, 22),   # Sawit          — oranye
    4: (233, 196, 106),  # Pertanian Lain — kuning
    5: (109, 76, 65),    # Lahan Terbakar — coklat
}

# Hex colors untuk legend.json (PRD §B.1.3)
CLASS_COLORS: dict[int, str] = {
    0: "#2A6FDB",
    1: "#0B3D0B",
    2: "#E03B24",
    3: "#F97316",
    4: "#E9C46A",
    5: "#6D4C41",
}

# Bobot kelas awal untuk CrossEntropy berbobot (PRD §A.5 Cell 6).
# CATATAN: hitung ulang dari distribusi label sebenarnya setelah patch siap.
CLASS_WEIGHTS_DEFAULT: tuple[float, ...] = (0.8, 0.4, 2.0, 2.0, 1.0, 2.5)

# ============================================================================
# 4 JENIS TRANSISI DEFORESTASI (PRD §A.5 Cell 9, §B.1.3)
# ============================================================================
SOURCE_CLASS_FOREST: int = 1  # transisi dimulai dari "Hutan"

# Map: target_class_id → nama transisi (snake_case untuk JSON)
TRANSITION_MAP: dict[int, str] = {
    2: "hutan_ke_lahan_terbuka",
    3: "hutan_ke_sawit",
    4: "hutan_ke_pertanian_lain",
    5: "hutan_ke_terbakar",
}

# Warna hex untuk visualisasi transisi di WebGIS
TRANSITION_COLORS: dict[str, str] = {
    "hutan_ke_lahan_terbuka": "#7F1D1D",
    "hutan_ke_sawit": "#F97316",
    "hutan_ke_pertanian_lain": "#EAB308",
    "hutan_ke_terbakar": "#6D4C41",
}

TRANSITION_LABELS: dict[str, str] = {
    "hutan_ke_lahan_terbuka": "Hutan → Lahan Terbuka",
    "hutan_ke_sawit": "Hutan → Sawit",
    "hutan_ke_pertanian_lain": "Hutan → Pertanian Lain",
    "hutan_ke_terbakar": "Hutan → Lahan Terbakar",
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
    "B12": "SWIR2 (2190 nm) - deteksi bakar (dNBR)",
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
    "biopama_oilpalm": "BIOPAMA/GlobalOilPalm/v1",
}

# ============================================================================
# OUTPUT FILE NAMES (7 file kontrak — PRD §B.1)
# ============================================================================
OUTPUT_FILES: dict[str, str] = {
    "landcover_t1_png": "landcover_2021.png",
    "landcover_t1_bounds": "landcover_2021_bounds.json",
    "landcover_t2_png": "landcover_2024.png",
    "landcover_t2_bounds": "landcover_2024_bounds.json",
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
