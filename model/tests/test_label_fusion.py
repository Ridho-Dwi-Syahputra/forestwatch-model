"""Test label fusion DW-base (versi numpy untuk testing offline).

Pendekatan baru: Dynamic World jadi PETA DASAR (remap 9->6), lalu overlay
tematik Hansen(->2), Mining(->5), Sawit(->3, terakhir). Lihat
``forestwatch.gee.label_fusion`` & ``constants.DW_TO_CLASS``.
"""

from __future__ import annotations

import numpy as np

from forestwatch.constants import DW_TO_CLASS
from forestwatch.gee.label_fusion import apply_label_fusion_numpy


def _z(shape):
    return np.zeros(shape, dtype=np.uint8)


def test_dw_base_remap_all_nine_classes():
    """Tiap kelas DW 0..8 ter-remap sesuai DW_TO_CLASS (tanpa overlay)."""
    dw = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.uint8)
    label = apply_label_fusion_numpy(
        dw=dw,
        hansen_loss_eroded=_z(dw.shape),
        is_mining=_z(dw.shape),
        is_palm=_z(dw.shape),
    )
    expected = np.array([[DW_TO_CLASS[c] for c in range(9)]], dtype=np.uint8)
    np.testing.assert_array_equal(label, expected)


def test_dw_trees_become_forest_without_esa():
    """Bug lama hilang: DW trees -> Hutan TANPA perlu konfirmasi ESA (AND)."""
    dw = np.full((4, 4), 1, dtype=np.uint8)  # semua trees
    label = apply_label_fusion_numpy(
        dw=dw,
        hansen_loss_eroded=_z(dw.shape),
        is_mining=_z(dw.shape),
        is_palm=_z(dw.shape),
    )
    assert np.all(label == 1)  # Hutan, bukan jatuh ke 4


def test_dw_water_and_flooded_veg_become_perairan():
    """Tanpa NDVI (ndvi=None): water + flooded veg → Perairan (perilaku default)."""
    dw = np.array([[0, 3, 0, 3]], dtype=np.uint8)  # water + flooded veg
    label = apply_label_fusion_numpy(
        dw=dw,
        hansen_loss_eroded=_z(dw.shape),
        is_mining=_z(dw.shape),
        is_palm=_z(dw.shape),
    )
    assert np.all(label == 0)


def test_ndvi_healing_swamp_forest_to_hutan():
    """Healing rawa: Perairan (0) ber-NDVI tinggi → Hutan (1); NDVI rendah tetap air."""
    dw = np.array([[0, 3, 0, 3]], dtype=np.uint8)  # water + flooded veg
    # kolom 0,1 = kanopi rawa (NDVI tinggi) → Hutan; kolom 2,3 = air murni → tetap 0
    ndvi = np.array([[0.6, 0.45, 0.05, -0.1]], dtype=np.float32)
    label = apply_label_fusion_numpy(
        dw=dw,
        hansen_loss_eroded=_z(dw.shape),
        is_mining=_z(dw.shape),
        is_palm=_z(dw.shape),
        ndvi=ndvi,
    )
    np.testing.assert_array_equal(label, np.array([[1, 1, 0, 0]], dtype=np.uint8))


def test_ndvi_healing_does_not_touch_nonwater():
    """Healing hanya menyentuh Perairan (0); kelas lain dgn NDVI tinggi tak berubah."""
    dw = np.array([[1, 4, 7]], dtype=np.uint8)  # trees, crops, bare
    ndvi = np.array([[0.9, 0.9, 0.9]], dtype=np.float32)  # semua tinggi
    label = apply_label_fusion_numpy(
        dw=dw,
        hansen_loss_eroded=_z(dw.shape),
        is_mining=_z(dw.shape),
        is_palm=_z(dw.shape),
        ndvi=ndvi,
    )
    # Hutan tetap 1, crops→Pertanian Lain(4), bare→Lahan Terbuka(2) — tak di-heal.
    np.testing.assert_array_equal(label, np.array([[1, 4, 2]], dtype=np.uint8))


def test_ndvi_healing_then_mining_overlay_wins():
    """Tailing pond: air healed? Tidak (NDVI rendah). Mining overlay tetap menimpa."""
    dw = np.array([[0, 0]], dtype=np.uint8)  # dua piksel air
    ndvi = np.array([[0.7, 0.05]], dtype=np.float32)  # kiri kanopi, kanan air keruh
    mining = np.array([[0, 1]], dtype=np.uint8)  # kanan = footprint tambang
    label = apply_label_fusion_numpy(
        dw=dw,
        hansen_loss_eroded=_z(dw.shape),
        is_mining=mining,
        is_palm=_z(dw.shape),
        ndvi=ndvi,
    )
    # kiri: healed → Hutan(1); kanan: NDVI rendah tak healed, overlay mining → 5.
    np.testing.assert_array_equal(label, np.array([[1, 5]], dtype=np.uint8))


def test_dw_crops_grass_shrub_become_pertanian_lain():
    dw = np.array([[4, 2, 5]], dtype=np.uint8)  # crops, grass, shrub
    label = apply_label_fusion_numpy(
        dw=dw,
        hansen_loss_eroded=_z(dw.shape),
        is_mining=_z(dw.shape),
        is_palm=_z(dw.shape),
    )
    assert np.all(label == 4)


def test_dw_built_becomes_permukiman_bare_snow_lahan_terbuka():
    dw = np.array([[6, 7, 8]], dtype=np.uint8)  # built, bare, snow
    label = apply_label_fusion_numpy(
        dw=dw,
        hansen_loss_eroded=_z(dw.shape),
        is_mining=_z(dw.shape),
        is_palm=_z(dw.shape),
    )
    # Built → Permukiman (6); Bare & Snow → Lahan Terbuka (2).
    np.testing.assert_array_equal(label, np.array([[6, 2, 2]], dtype=np.uint8))


def test_esa_union_adds_forest_where_dw_undercalls():
    """ESA tree_cover (10) + DW grass/shrub -> Hutan (UNION, hanya menambah)."""
    dw = np.array([[1, 2, 5, 4]], dtype=np.uint8)  # trees, grass, shrub, crops
    esa = np.array([[10, 10, 10, 10]], dtype=np.uint8)  # semua tree_cover
    label = apply_label_fusion_numpy(
        dw=dw,
        hansen_loss_eroded=_z(dw.shape),
        is_mining=_z(dw.shape),
        is_palm=_z(dw.shape),
        esa=esa,
    )
    # trees/grass/shrub + ESA tree -> Hutan; crops (4) TIDAK tersentuh union.
    np.testing.assert_array_equal(label, np.array([[1, 1, 1, 4]], dtype=np.uint8))


def test_esa_union_does_not_touch_water_or_bare():
    """UNION ESA tidak boleh mengubah air/bare jadi hutan (konservatif)."""
    dw = np.array([[0, 7]], dtype=np.uint8)  # water, bare
    esa = np.array([[10, 10]], dtype=np.uint8)  # ESA salah bilang tree
    label = apply_label_fusion_numpy(
        dw=dw,
        hansen_loss_eroded=_z(dw.shape),
        is_mining=_z(dw.shape),
        is_palm=_z(dw.shape),
        esa=esa,
    )
    np.testing.assert_array_equal(label, np.array([[0, 2]], dtype=np.uint8))


def test_hansen_loss_overrides_base():
    dw = np.full((4, 4), 1, dtype=np.uint8)  # Hutan
    hansen = _z(dw.shape)
    hansen[0:2, :] = 1  # baris atas: loss sejak 2019
    label = apply_label_fusion_numpy(
        dw=dw,
        hansen_loss_eroded=hansen,
        is_mining=_z(dw.shape),
        is_palm=_z(dw.shape),
    )
    assert np.all(label[0:2, :] == 2)  # Lahan Terbuka (kelas 2)
    assert np.all(label[2:, :] == 1)   # Hutan


def test_mining_overrides_everything():
    dw = np.full((3, 3), 1, dtype=np.uint8)
    hansen = np.ones((3, 3), dtype=np.uint8)  # hansen loss di mana-mana
    mining = _z((3, 3))
    mining[1, 1] = 1
    label = apply_label_fusion_numpy(
        dw=dw,
        hansen_loss_eroded=hansen,
        is_mining=mining,
        is_palm=_z((3, 3)),
    )
    assert label[1, 1] == 5      # Tambang menimpa
    assert label[0, 0] == 2      # sekeliling tetap Lahan Terbuka


def test_palm_applied_last_overrides_all():
    """Sawit (FDP palm) diterapkan TERAKHIR → menimpa Hansen & Mining."""
    dw = np.full((1, 4), 4, dtype=np.uint8)  # crops
    hansen = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    mining = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    is_palm = np.full((1, 4), True)  # semua piksel = sawit
    label = apply_label_fusion_numpy(
        dw=dw,
        hansen_loss_eroded=hansen,
        is_mining=mining,
        is_palm=is_palm,
    )
    np.testing.assert_array_equal(label, np.array([[3, 3, 3, 3]], dtype=np.uint8))


def test_dw_to_class_covers_all_nine_dw_classes():
    """Sanity: DW_TO_CLASS lengkap (0..8) & target valid (0..5)."""
    assert sorted(DW_TO_CLASS.keys()) == list(range(9))
    assert all(0 <= v <= 6 for v in DW_TO_CLASS.values())
