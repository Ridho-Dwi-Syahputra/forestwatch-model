"""Test 6 aturan label fusion (versi numpy untuk testing offline)."""

from __future__ import annotations

import numpy as np

from forestwatch.gee.label_fusion import apply_label_fusion_numpy


def _zero_like(shape):
    return np.zeros(shape, dtype=np.uint8)


def test_default_class_is_4_when_nothing_matches():
    shape = (4, 4)
    label = apply_label_fusion_numpy(
        esa=_zero_like(shape),
        dw=_zero_like(shape),
        hansen_loss_eroded=_zero_like(shape),
        is_oilpalm=_zero_like(shape),
        is_mining=_zero_like(shape),
    )
    assert np.all(label == 4)  # Pertanian Lain default


def test_rule_water_requires_both_esa_and_dw():
    shape = (4, 4)
    esa = np.full(shape, 80, dtype=np.uint8)  # ESA water
    dw = np.full(shape, 0, dtype=np.uint8)  # DW water
    label = apply_label_fusion_numpy(
        esa=esa, dw=dw,
        hansen_loss_eroded=_zero_like(shape),
        is_oilpalm=_zero_like(shape),
        is_mining=_zero_like(shape),
    )
    assert np.all(label == 0)


def test_rule_water_skipped_if_only_one_source():
    shape = (4, 4)
    esa = np.full(shape, 80, dtype=np.uint8)  # ESA water
    dw = np.full(shape, 1, dtype=np.uint8)  # DW trees (mismatch)
    label = apply_label_fusion_numpy(
        esa=esa, dw=dw,
        hansen_loss_eroded=_zero_like(shape),
        is_oilpalm=_zero_like(shape),
        is_mining=_zero_like(shape),
    )
    # Tidak dapat label water - fallback ke default 4
    assert np.all(label == 4)


def test_rule_forest_consensus():
    shape = (4, 4)
    esa = np.full(shape, 10, dtype=np.uint8)  # tree cover
    dw = np.full(shape, 1, dtype=np.uint8)  # trees
    label = apply_label_fusion_numpy(
        esa=esa, dw=dw,
        hansen_loss_eroded=_zero_like(shape),
        is_oilpalm=_zero_like(shape),
        is_mining=_zero_like(shape),
    )
    assert np.all(label == 1)


def test_rule_bare_only_needs_one():
    shape = (4, 4)
    # ESA bare cukup, walaupun DW tidak
    esa = np.full(shape, 60, dtype=np.uint8)
    dw = np.full(shape, 5, dtype=np.uint8)  # shrub
    label = apply_label_fusion_numpy(
        esa=esa, dw=dw,
        hansen_loss_eroded=_zero_like(shape),
        is_oilpalm=_zero_like(shape),
        is_mining=_zero_like(shape),
    )
    assert np.all(label == 2)


def test_rule_cropland_splits_by_oilpalm():
    shape = (2, 4)
    esa = np.full(shape, 40, dtype=np.uint8)  # cropland
    dw = np.full(shape, 4, dtype=np.uint8)  # crops
    is_oilpalm = np.array(
        [
            [True, True, False, False],
            [False, True, False, True],
        ]
    )
    label = apply_label_fusion_numpy(
        esa=esa, dw=dw,
        hansen_loss_eroded=_zero_like(shape),
        is_oilpalm=is_oilpalm,
        is_mining=_zero_like(shape),
    )
    expected = np.array(
        [
            [3, 3, 4, 4],
            [4, 3, 4, 3],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(label, expected)


def test_rule_hansen_loss_overrides_other_classes():
    shape = (4, 4)
    # Hutan dulu (ESA=10, DW=1)
    esa = np.full(shape, 10, dtype=np.uint8)
    dw = np.full(shape, 1, dtype=np.uint8)
    # Tetapi Hansen menandai sebagian hilang sejak 2019
    hansen = np.zeros(shape, dtype=np.uint8)
    hansen[0:2, :] = 1  # baris atas: lossyear ge 19
    label = apply_label_fusion_numpy(
        esa=esa, dw=dw,
        hansen_loss_eroded=hansen,
        is_oilpalm=_zero_like(shape),
        is_mining=_zero_like(shape),
    )
    # Baris atas -> Deforestasi (2)
    assert np.all(label[0:2, :] == 2)
    # Baris bawah -> Hutan (1)
    assert np.all(label[2:, :] == 1)


def test_rule_mining_overrides_everything():
    shape = (3, 3)
    esa = np.full(shape, 10, dtype=np.uint8)
    dw = np.full(shape, 1, dtype=np.uint8)
    hansen = np.ones(shape, dtype=np.uint8)  # deforestasi
    mining = np.zeros(shape, dtype=np.uint8)
    mining[1, 1] = 1  # center tambang
    label = apply_label_fusion_numpy(
        esa=esa, dw=dw,
        hansen_loss_eroded=hansen,
        is_oilpalm=_zero_like(shape),
        is_mining=mining,
    )
    assert label[1, 1] == 5  # tambang menimpa
    # Sekeliling tetap deforestasi
    assert label[0, 0] == 2


def test_rule_order_matters():
    """Sawit (BIOPAMA) diterapkan TERAKHIR → menimpa Hansen & Burned.

    Urutan baru: cropland→4, Hansen→2, Burned→5, lalu Sawit→3 (final).
    Memastikan kelas Sawit pasti terwakili (tidak terhapus aturan lain).
    """
    shape = (1, 4)
    esa = np.full(shape, 40, dtype=np.uint8)  # cropland
    dw = np.full(shape, 4, dtype=np.uint8)
    is_oilpalm = np.full(shape, True)         # semua piksel = sawit (BIOPAMA)
    hansen = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    mining = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    label = apply_label_fusion_numpy(
        esa=esa, dw=dw,
        hansen_loss_eroded=hansen,
        is_oilpalm=is_oilpalm,
        is_mining=mining,
    )
    # is_oilpalm True di semua piksel & diterapkan terakhir → semua jadi Sawit (3)
    expected = np.array([[3, 3, 3, 3]], dtype=np.uint8)
    np.testing.assert_array_equal(label, expected)
