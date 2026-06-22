"""Test training metrics (numpy-based, no torch needed)."""

from __future__ import annotations

import numpy as np
import pytest

from forestwatch.training.metrics import (
    cohen_kappa,
    compute_confusion_matrix,
    compute_f1_per_class,
    compute_iou_per_class,
    frequency_weighted_iou,
    median_frequency_weights,
    metric_summary,
    overall_accuracy,
    precision_per_class,
    recall_per_class,
    set_seed,
)


def test_set_seed_idempotent():
    set_seed(42)
    a = np.random.rand(3)
    set_seed(42)
    b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)


def test_perfect_prediction():
    target = np.array([0, 1, 2, 0, 1, 2])
    pred = target.copy()
    cm = compute_confusion_matrix(pred, target, n_classes=3)
    # Diagonal saja
    assert cm[0, 0] == 2
    assert cm[1, 1] == 2
    assert cm[2, 2] == 2
    assert cm.sum() == 6

    iou = compute_iou_per_class(cm)
    np.testing.assert_allclose(iou, [1.0, 1.0, 1.0], atol=1e-6)
    assert overall_accuracy(cm) == 1.0


def test_all_wrong_prediction():
    target = np.array([0, 0, 0])
    pred = np.array([1, 1, 1])
    cm = compute_confusion_matrix(pred, target, n_classes=2)
    iou = compute_iou_per_class(cm)
    # Tidak ada intersection sama sekali
    np.testing.assert_allclose(iou, [0.0, 0.0], atol=1e-6)
    assert overall_accuracy(cm) == 0.0


def test_confusion_matrix_off_by_one():
    target = np.array([0, 1, 2])
    pred = np.array([1, 1, 2])
    cm = compute_confusion_matrix(pred, target, n_classes=3)
    # Kelas 0 di-misklasifikasi sebagai 1
    assert cm[0, 1] == 1
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1


def test_f1_relation_to_iou():
    target = np.array([0, 0, 1, 1, 1])
    pred = np.array([0, 1, 1, 1, 1])
    cm = compute_confusion_matrix(pred, target, n_classes=2)
    iou = compute_iou_per_class(cm)
    f1 = compute_f1_per_class(cm)
    # F1 = 2*IoU / (1+IoU)
    np.testing.assert_allclose(f1, 2 * iou / (1 + iou), atol=1e-6)


def test_metric_summary_structure():
    import pytest

    target = np.array([0, 1, 2, 0, 1])
    pred = np.array([0, 1, 2, 0, 1])
    cm = compute_confusion_matrix(pred, target, n_classes=3)
    summary = metric_summary(cm, class_names=("A", "B", "C"))
    assert summary["overall_accuracy"] == 1.0
    assert summary["mean_iou"] == pytest.approx(1.0, abs=1e-6)
    assert len(summary["per_class"]) == 3
    assert summary["per_class"][0]["class"] == "A"
    assert summary["per_class"][0]["iou"] == pytest.approx(1.0, abs=1e-6)
    assert isinstance(summary["confusion_matrix"], list)


def test_handles_out_of_range_target():
    """Target dengan label -1 atau di luar range tidak crash."""
    target = np.array([0, 1, 999], dtype=np.int64)
    pred = np.array([0, 1, 0], dtype=np.int64)
    cm = compute_confusion_matrix(pred, target, n_classes=2)
    # Hanya 2 piksel valid yang dihitung
    assert cm.sum() == 2


# ============================================================================
# Cohen's Kappa
# ============================================================================


def test_kappa_perfect_is_one():
    target = np.array([0, 1, 2, 0, 1, 2])
    cm = compute_confusion_matrix(target, target, n_classes=3)
    assert cohen_kappa(cm) == pytest.approx(1.0, abs=1e-6)


def test_kappa_in_summary():
    target = np.array([0, 1, 2, 0, 1])
    cm = compute_confusion_matrix(target, target, n_classes=3)
    summary = metric_summary(cm, class_names=("A", "B", "C"))
    assert "kappa" in summary
    assert summary["kappa"] == pytest.approx(1.0, abs=1e-6)


def test_kappa_chance_level_near_zero():
    """Prediksi acak terhadap distribusi sama → kappa mendekati 0."""
    rng = np.random.default_rng(0)
    target = rng.integers(0, 3, size=3000)
    pred = rng.integers(0, 3, size=3000)
    cm = compute_confusion_matrix(pred, target, n_classes=3)
    assert abs(cohen_kappa(cm)) < 0.1


def test_kappa_empty_matrix():
    cm = np.zeros((3, 3))
    assert cohen_kappa(cm) == 0.0


# ============================================================================
# Median-frequency class weights
# ============================================================================


def test_median_freq_weights_length():
    dist = {0: 100, 1: 5000, 2: 200, 3: 180, 4: 350, 5: 60}
    w = median_frequency_weights(dist, n_classes=6)
    assert len(w) == 6


def test_median_freq_rare_class_higher_weight():
    # Kelas 1 dominan, kelas 5 langka → bobot 5 > bobot 1
    dist = {0: 100, 1: 5000, 2: 200, 3: 180, 4: 350, 5: 60}
    w = median_frequency_weights(dist, n_classes=6, clip_min=0.1, clip_max=10.0)
    assert w[5] > w[1]


def test_median_freq_clipping():
    dist = {0: 1, 1: 1_000_000, 2: 1, 3: 1, 4: 1, 5: 1}
    w = median_frequency_weights(dist, n_classes=6, clip_min=0.3, clip_max=5.0)
    assert all(0.3 <= x <= 5.0 for x in w)


def test_median_freq_empty_distribution():
    w = median_frequency_weights({}, n_classes=6)
    assert w == [1.0] * 6


# ============================================================================
# FWIoU, precision/recall per kelas
# ============================================================================


def test_fwiou_perfect_prediction_is_one():
    target = np.array([0, 0, 1, 1, 1, 2])
    cm = compute_confusion_matrix(target, target, n_classes=3)
    assert frequency_weighted_iou(cm) == pytest.approx(1.0, abs=1e-6)


def test_fwiou_balanced_classes_equals_miou():
    # Frekuensi tiap kelas sama → FWIoU == mIoU
    target = np.array([0, 1, 2, 0, 1, 2])
    pred = np.array([1, 1, 1, 0, 1, 2])
    cm = compute_confusion_matrix(pred, target, n_classes=3)
    iou = compute_iou_per_class(cm)
    assert frequency_weighted_iou(cm) == pytest.approx(iou.mean(), abs=1e-6)


def test_fwiou_within_bounds():
    rng = np.random.default_rng(1)
    target = rng.integers(0, 4, size=500)
    pred = rng.integers(0, 4, size=500)
    cm = compute_confusion_matrix(pred, target, n_classes=4)
    fwiou = frequency_weighted_iou(cm)
    assert 0.0 <= fwiou <= 1.0


def test_fwiou_empty_matrix():
    cm = np.zeros((3, 3))
    assert frequency_weighted_iou(cm) == 0.0


def test_precision_recall_known_confusion():
    # Kelas 0: 1 TP, 1 FN (diklasifikasi sbg 1); kelas 1: 1 TP, 1 FP (dari kelas 0)
    target = np.array([0, 0, 1])
    pred = np.array([0, 1, 1])
    cm = compute_confusion_matrix(pred, target, n_classes=2)
    recall = recall_per_class(cm)
    precision = precision_per_class(cm)
    # Recall kelas 0 = 1/2 (1 benar dari 2 piksel kelas 0)
    np.testing.assert_allclose(recall, [0.5, 1.0], atol=1e-6)
    # Precision kelas 0 = 1/1 (semua prediksi 0 benar), kelas 1 = 1/2
    np.testing.assert_allclose(precision, [1.0, 0.5], atol=1e-6)


def test_precision_recall_perfect_prediction():
    target = np.array([0, 1, 2, 0, 1, 2])
    cm = compute_confusion_matrix(target, target, n_classes=3)
    np.testing.assert_allclose(recall_per_class(cm), [1.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(precision_per_class(cm), [1.0, 1.0, 1.0], atol=1e-6)


def test_metric_summary_includes_fwiou_and_precision_recall():
    target = np.array([0, 0, 1, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 1, 2])
    cm = compute_confusion_matrix(pred, target, n_classes=3)
    summary = metric_summary(cm, class_names=("A", "B", "C"))
    assert "fwiou" in summary
    assert 0.0 <= summary["fwiou"] <= 1.0
    for entry in summary["per_class"]:
        assert "precision" in entry
        assert "recall" in entry
