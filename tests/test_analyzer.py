"""Tests for analyzer module utility functions.

Tests metric computation, confusion matrices, and helper functions.
"""

import numpy as np
import pytest
from PIL import Image

from surgical_segmentation.evaluation.analyzer import (
    DEFAULT_CLASS_NAMES,
    binary_confusion_matrix,
    build_preview_image,
    colorize_mask,
    compute_multiclass_metrics,
    confusion_matrix_multiclass,
    load_mask_prediction_pairs,
    precision_recall_curve_manual,
    resolve_class_names,
)


class TestBinaryConfusionMatrix:
    """Test binary confusion matrix computation."""

    def test_perfect_prediction(self):
        """Verify perfect prediction yields diagonal matrix."""
        true_labels = np.array([0, 0, 0, 1, 1, 1])
        pred_labels = np.array([0, 0, 0, 1, 1, 1])

        cm = binary_confusion_matrix(true_labels, pred_labels)

        assert cm[0, 0] == 3  # True negatives
        assert cm[1, 1] == 3  # True positives
        assert cm[0, 1] == 0  # False positives
        assert cm[1, 0] == 0  # False negatives

    def test_all_errors(self):
        """Verify all-wrong prediction yields off-diagonal matrix."""
        true_labels = np.array([0, 0, 0, 1, 1, 1])
        pred_labels = np.array([1, 1, 1, 0, 0, 0])  # All wrong

        cm = binary_confusion_matrix(true_labels, pred_labels)

        assert cm[0, 0] == 0  # True negatives
        assert cm[1, 1] == 0  # True positives
        assert cm[0, 1] == 3  # False positives
        assert cm[1, 0] == 3  # False negatives

    def test_mixed_predictions(self):
        """Verify mixed predictions produce correct counts."""
        true_labels = np.array([0, 0, 1, 1, 0, 1])
        pred_labels = np.array([0, 1, 1, 0, 0, 1])  # 1 FP, 1 FN

        cm = binary_confusion_matrix(true_labels, pred_labels)

        assert cm[0, 0] == 2  # True negatives
        assert cm[1, 1] == 2  # True positives
        assert cm[0, 1] == 1  # False positives
        assert cm[1, 0] == 1  # False negatives

    def test_matrix_sums_to_total(self):
        """Verify matrix sums to total sample count."""
        true_labels = np.random.randint(0, 2, 100)
        pred_labels = np.random.randint(0, 2, 100)

        cm = binary_confusion_matrix(true_labels, pred_labels)

        assert cm.sum() == 100


class TestPrecisionRecallCurve:
    """Test precision-recall curve computation."""

    def test_curve_has_correct_length(self):
        """Verify curve has expected number of thresholds."""
        true_labels = np.array([0, 0, 1, 1])
        pred_probs = np.array([0.1, 0.4, 0.6, 0.9])

        precision, recall, thresholds = precision_recall_curve_manual(
            true_labels, pred_probs, num_thresholds=50
        )

        assert len(precision) == 50
        assert len(recall) == 50
        assert len(thresholds) == 50

    def test_perfect_predictions_yield_perfect_curve(self):
        """Verify perfect separation yields optimal precision/recall."""
        true_labels = np.array([0, 0, 0, 1, 1, 1])
        pred_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        precision, recall, thresholds = precision_recall_curve_manual(
            true_labels, pred_probs, num_thresholds=100
        )

        # At threshold=0.5, should have perfect precision and recall
        mid_idx = 50
        assert precision[mid_idx] >= 0.9
        assert recall[mid_idx] >= 0.9

    def test_values_in_valid_range(self):
        """Verify precision and recall are in [0, 1]."""
        true_labels = np.random.randint(0, 2, 100)
        pred_probs = np.random.rand(100)

        precision, recall, _ = precision_recall_curve_manual(true_labels, pred_probs)

        assert all(0 <= p <= 1 for p in precision)
        assert all(0 <= r <= 1 for r in recall)


class TestResolveClassNames:
    """Test class name resolution logic."""

    def test_default_names_for_binary(self):
        """Verify default names for binary classification."""
        names = resolve_class_names(2, None)

        assert names == ["background", "instrument"]

    def test_override_names(self):
        """Verify custom names override defaults."""
        names = resolve_class_names(2, "bg,tool")

        assert names == ["bg", "tool"]

    def test_names_padded_for_more_classes(self):
        """Verify names are padded when more classes than names."""
        names = resolve_class_names(4, "bg,inst")

        assert len(names) == 4
        assert names[0] == "bg"
        assert names[1] == "inst"
        assert "class_2" in names[2]

    def test_names_truncated_for_fewer_classes(self):
        """Verify names are truncated when fewer classes than names."""
        names = resolve_class_names(2, "bg,inst,extra,unused")

        assert len(names) == 2


class TestLoadMaskPredictionPairs:
    """Test mask/prediction pair loading."""

    def test_loads_matching_files(self, tmp_path):
        """Verify only matching files are loaded."""
        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        gt_dir.mkdir()
        pred_dir.mkdir()

        # Create matching pairs
        for name in ["mask_001", "mask_002"]:
            mask = np.zeros((100, 100), dtype=np.uint8)
            Image.fromarray(mask).save(gt_dir / f"{name}.png")
            Image.fromarray(mask).save(pred_dir / f"{name}.png")

        # Create non-matching file
        Image.fromarray(mask).save(gt_dir / "mask_003.png")

        pairs = load_mask_prediction_pairs(gt_dir, pred_dir, max_samples=10)

        assert len(pairs) == 2

    def test_respects_max_samples(self, tmp_path):
        """Verify max_samples limits loaded pairs."""
        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        gt_dir.mkdir()
        pred_dir.mkdir()

        # Create 10 pairs
        for i in range(10):
            mask = np.zeros((50, 50), dtype=np.uint8)
            Image.fromarray(mask).save(gt_dir / f"mask_{i:03d}.png")
            Image.fromarray(mask).save(pred_dir / f"mask_{i:03d}.png")

        pairs = load_mask_prediction_pairs(gt_dir, pred_dir, max_samples=3)

        assert len(pairs) == 3

    def test_raises_on_missing_gt_dir(self, tmp_path):
        """Verify error raised for missing ground truth directory."""
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            load_mask_prediction_pairs(tmp_path / "nonexistent", pred_dir, max_samples=10)

    def test_raises_on_missing_pred_dir(self, tmp_path):
        """Verify error raised for missing prediction directory."""
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            load_mask_prediction_pairs(gt_dir, tmp_path / "nonexistent", max_samples=10)

    def test_raises_on_no_matching_files(self, tmp_path):
        """Verify error raised when no files match."""
        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        gt_dir.mkdir()
        pred_dir.mkdir()

        # Create non-overlapping files
        mask = np.zeros((50, 50), dtype=np.uint8)
        Image.fromarray(mask).save(gt_dir / "gt_only.png")
        Image.fromarray(mask).save(pred_dir / "pred_only.png")

        with pytest.raises(FileNotFoundError):
            load_mask_prediction_pairs(gt_dir, pred_dir, max_samples=10)


class TestConfusionMatrixMulticlass:
    """Test multiclass confusion matrix computation."""

    def test_binary_case_matches_binary_function(self):
        """Verify multiclass function works for binary case."""
        true_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        pred_mask = np.array([[0, 0], [1, 1]], dtype=np.uint8)

        cm = confusion_matrix_multiclass(true_mask, pred_mask, num_classes=2)

        # Flatten for comparison
        true_flat = true_mask.ravel()
        pred_flat = pred_mask.ravel()

        assert cm[0, 0] == np.sum((true_flat == 0) & (pred_flat == 0))
        assert cm[0, 1] == np.sum((true_flat == 0) & (pred_flat == 1))
        assert cm[1, 0] == np.sum((true_flat == 1) & (pred_flat == 0))
        assert cm[1, 1] == np.sum((true_flat == 1) & (pred_flat == 1))

    def test_three_class_case(self):
        """Verify multiclass confusion matrix for 3 classes."""
        true_mask = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
        pred_mask = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint8)

        cm = confusion_matrix_multiclass(true_mask, pred_mask, num_classes=3)

        assert cm.shape == (3, 3)
        assert cm.sum() == 6  # Total pixels

    def test_ignores_invalid_values(self):
        """Verify values outside [0, num_classes) are ignored."""
        true_mask = np.array([[0, 1, 99], [0, 1, 255]], dtype=np.uint8)
        pred_mask = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.uint8)

        cm = confusion_matrix_multiclass(true_mask, pred_mask, num_classes=2)

        # Only valid pixels should be counted
        assert cm.sum() == 4


class TestComputeMulticlassMetrics:
    """Test multiclass metric computation."""

    def test_perfect_prediction_all_ones(self):
        """Verify perfect prediction yields all 1.0 metrics."""
        cm = np.array([[50, 0], [0, 50]], dtype=np.int64)

        metrics = compute_multiclass_metrics(cm)

        np.testing.assert_array_almost_equal(metrics["precision"], [1.0, 1.0])
        np.testing.assert_array_almost_equal(metrics["recall"], [1.0, 1.0])
        np.testing.assert_array_almost_equal(metrics["iou"], [1.0, 1.0])
        np.testing.assert_array_almost_equal(metrics["dice"], [1.0, 1.0])
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_precision_calculation(self):
        """Verify precision = TP / (TP + FP)."""
        # Class 0: TP=40, FP=10
        # Class 1: TP=30, FP=20
        cm = np.array([[40, 20], [10, 30]], dtype=np.int64)

        metrics = compute_multiclass_metrics(cm)

        # Precision[0] = 40 / (40 + 10) = 0.8
        # Precision[1] = 30 / (30 + 20) = 0.6
        assert metrics["precision"][0] == pytest.approx(0.8, rel=0.01)
        assert metrics["precision"][1] == pytest.approx(0.6, rel=0.01)

    def test_recall_calculation(self):
        """Verify recall = TP / (TP + FN)."""
        # Class 0: TP=40, FN=20
        # Class 1: TP=30, FN=10
        cm = np.array([[40, 20], [10, 30]], dtype=np.int64)

        metrics = compute_multiclass_metrics(cm)

        # Recall[0] = 40 / (40 + 20) = 0.667
        # Recall[1] = 30 / (30 + 10) = 0.75
        assert metrics["recall"][0] == pytest.approx(40 / 60, rel=0.01)
        assert metrics["recall"][1] == pytest.approx(0.75, rel=0.01)

    def test_iou_matches_formula(self):
        """Verify IoU = TP / (TP + FP + FN)."""
        cm = np.array([[80, 10], [15, 45]], dtype=np.int64)

        metrics = compute_multiclass_metrics(cm)

        # IoU[1] = 45 / (45 + 10 + 15) = 45/70
        expected_iou = 45 / 70
        assert metrics["iou"][1] == pytest.approx(expected_iou, rel=0.01)

    def test_dice_matches_formula(self):
        """Verify Dice = 2*TP / (2*TP + FP + FN)."""
        cm = np.array([[80, 10], [15, 45]], dtype=np.int64)

        metrics = compute_multiclass_metrics(cm)

        # Dice[1] = 2*45 / (2*45 + 10 + 15) = 90/115
        expected_dice = 90 / 115
        assert metrics["dice"][1] == pytest.approx(expected_dice, rel=0.01)


class TestColorizeMask:
    """Test mask colorization for visualization."""

    def test_output_shape(self):
        """Verify RGB output has correct shape."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50:, :] = 1

        rgb = colorize_mask(mask, num_classes=2)

        assert rgb.shape == (100, 100, 3)

    def test_output_dtype(self):
        """Verify RGB output is uint8."""
        mask = np.zeros((50, 50), dtype=np.uint8)

        rgb = colorize_mask(mask, num_classes=2)

        assert rgb.dtype == np.uint8

    def test_different_classes_get_different_colors(self):
        """Verify different classes are distinguishable."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50:, :] = 1

        rgb = colorize_mask(mask, num_classes=2)

        # Colors should differ between regions
        color_class0 = rgb[0, 0]
        color_class1 = rgb[99, 0]

        assert not np.array_equal(color_class0, color_class1)


class TestBuildPreviewImage:
    """Test preview image generation."""

    def test_output_concatenates_masks(self):
        """Verify output concatenates ground truth and prediction."""
        true_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask = np.ones((100, 100), dtype=np.uint8)

        preview = build_preview_image(true_mask, pred_mask, num_classes=2)

        # Width should be 2*100 + 5 (spacer)
        assert preview.shape[0] == 100
        assert preview.shape[1] == 205

    def test_handles_mismatched_sizes(self):
        """Verify function handles different mask sizes."""
        true_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask = np.zeros((80, 120), dtype=np.uint8)

        # Should not raise
        preview = build_preview_image(true_mask, pred_mask, num_classes=2)

        assert preview is not None

    def test_output_is_rgb(self):
        """Verify output has 3 channels."""
        true_mask = np.zeros((50, 50), dtype=np.uint8)
        pred_mask = np.zeros((50, 50), dtype=np.uint8)

        preview = build_preview_image(true_mask, pred_mask, num_classes=2)

        assert preview.shape[2] == 3


class TestDefaultClassNames:
    """Test default class names constant."""

    def test_default_names_exist(self):
        """Verify DEFAULT_CLASS_NAMES is defined."""
        assert DEFAULT_CLASS_NAMES is not None
        assert len(DEFAULT_CLASS_NAMES) >= 2

    def test_default_includes_background(self):
        """Verify background is in default names."""
        assert "background" in DEFAULT_CLASS_NAMES


class TestLoadMaskPredictionPairsExtra:
    """Additional tests for load_mask_prediction_pairs function."""

    def test_handles_mismatched_sizes(self, tmp_path):
        """Verify function handles masks of different sizes."""
        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        gt_dir.mkdir()
        pred_dir.mkdir()

        # Create masks with different sizes
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask = np.zeros((50, 50), dtype=np.uint8)
        Image.fromarray(gt_mask).save(gt_dir / "test.png")
        Image.fromarray(pred_mask).save(pred_dir / "test.png")

        pairs = load_mask_prediction_pairs(gt_dir, pred_dir, max_samples=10)

        # Should resize pred to match gt
        assert len(pairs) == 1
        assert pairs[0]["true"].shape == pairs[0]["pred"].shape


class TestAnalyzeModelPerformance:
    """Test analyze_model_performance dispatch function."""

    def test_dispatches_to_synthetic(self, tmp_path, monkeypatch):
        """Verify synthetic mode is dispatched correctly."""
        import argparse
        from unittest.mock import MagicMock

        from surgical_segmentation.evaluation.analyzer import analyze_model_performance

        mock_synthetic = MagicMock()
        monkeypatch.setattr(
            "surgical_segmentation.evaluation.analyzer.run_synthetic_analysis", mock_synthetic
        )

        args = argparse.Namespace(mode="synthetic")
        analyze_model_performance(args)

        mock_synthetic.assert_called_once_with(args)

    def test_dispatches_to_dataset(self, tmp_path, monkeypatch):
        """Verify dataset mode is dispatched correctly."""
        import argparse
        from unittest.mock import MagicMock

        from surgical_segmentation.evaluation.analyzer import analyze_model_performance

        mock_dataset = MagicMock()
        monkeypatch.setattr(
            "surgical_segmentation.evaluation.analyzer.run_dataset_analysis", mock_dataset
        )

        args = argparse.Namespace(mode="dataset")
        analyze_model_performance(args)

        mock_dataset.assert_called_once_with(args)


class TestLoadMaskPredictionPairsValidation:
    """Test load_mask_prediction_pairs validation."""

    def test_raises_on_none_dirs(self):
        """Verify ValueError raised for None directories."""
        with pytest.raises(ValueError, match="Both --mask-dir and --pred-dir"):
            load_mask_prediction_pairs(None, None, max_samples=10)
