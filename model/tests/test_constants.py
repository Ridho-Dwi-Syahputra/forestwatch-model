"""Test konsistensi konstanta."""

from __future__ import annotations

from forestwatch.constants import (
    BANDS,
    CLASS_COLORS,
    CLASS_IDS,
    CLASS_NAMES,
    CLASS_NAMES_EN,
    CLASS_WEIGHTS_DEFAULT,
    N_BANDS,
    N_CLASSES,
    OUTPUT_FILES,
    PALETTE_RGB,
    PAPUA_BBOX,
    PROVINCES,
    TRANSITION_COLORS,
    TRANSITION_LABELS,
    TRANSITION_MAP,
)


def test_class_count_consistent():
    assert N_CLASSES == 7
    assert len(CLASS_IDS) == N_CLASSES
    assert len(CLASS_NAMES) == N_CLASSES
    assert len(CLASS_NAMES_EN) == N_CLASSES
    assert len(CLASS_WEIGHTS_DEFAULT) == N_CLASSES
    assert len(PALETTE_RGB) == N_CLASSES
    assert len(CLASS_COLORS) == N_CLASSES


def test_class_ids_match_palette_keys():
    assert set(PALETTE_RGB.keys()) == set(CLASS_IDS)
    assert set(CLASS_COLORS.keys()) == set(CLASS_IDS)


def test_palette_rgb_values_valid():
    for cls, rgb in PALETTE_RGB.items():
        assert len(rgb) == 3
        for component in rgb:
            assert 0 <= component <= 255


def test_class_colors_hex_format():
    import re

    pat = re.compile(r"^#[0-9A-Fa-f]{6}$")
    for cls, hex_color in CLASS_COLORS.items():
        assert pat.match(hex_color), f"Invalid hex for class {cls}: {hex_color}"


def test_bands_correct():
    assert N_BANDS == 6
    assert len(BANDS) == N_BANDS
    assert set(BANDS) == {"B2", "B3", "B4", "B8", "B11", "B12"}


def test_transition_map():
    assert len(TRANSITION_MAP) == 5
    # Semua target adalah salah satu dari kelas {2, 3, 4, 5, 6}
    assert set(TRANSITION_MAP.keys()) == {2, 3, 4, 5, 6}
    # Nama transisi snake_case dimulai "hutan_ke_"
    for tname in TRANSITION_MAP.values():
        assert tname.startswith("hutan_ke_")
        assert tname in TRANSITION_COLORS
        assert tname in TRANSITION_LABELS


def test_papua_bbox_valid():
    min_lon, min_lat, max_lon, max_lat = PAPUA_BBOX
    assert min_lon < max_lon
    assert min_lat < max_lat
    # Sanity: Papua di belahan timur selatan
    assert 130 <= min_lon <= 145
    assert -10 <= min_lat <= 0


def test_six_provinces():
    assert len(PROVINCES) == 6
    assert "Papua Selatan" in PROVINCES  # food estate Merauke


def test_output_files_has_required_keys():
    required = (
        "landcover_t1_png",
        "landcover_t1_bounds",
        "landcover_t2_png",
        "landcover_t2_bounds",
        "deforestation_geojson",
        "statistics_json",
        "legend_json",
        "metrics_json",
        "model_onnx",
        "model_card",
    )
    for k in required:
        assert k in OUTPUT_FILES, f"missing key '{k}' in OUTPUT_FILES"
