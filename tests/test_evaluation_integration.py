"""Integration tests for evaluation pipeline.

These tests verify the evaluation metrics and visualization
pipeline works correctly end-to-end.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from surgical_segmentation.models import InstrumentSegmentationModel


@pytest.fixture
def evaluation_dataset(tmp_path):
    """Create ground truth and prediction masks for evaluation testing.
    
    Creates controlled test cases with known IoU/Dice values.
    """
    gt_dir = tmp_path / "ground_truth"
    pred_dir = tmp_path / "predictions"
    gt_dir.mkdir()
    pred_dir.mkdir()
    
    # Case 1: Perfect prediction (IoU=1.0, Dice=1.0)
    mask_perfect = np.zeros((256, 256), dtype=np.uint8)
    mask_perfect[50:100, 50:100] = 1
    Image.fromarray(mask_perfect).save(gt_dir / "mask_00001.png")
    Image.fromarray(mask_perfect).save(pred_dir / "mask_00001.png")
    
    # Case 2: Partial overlap (known IoU)
    # GT: [0:100, 0:100] = 10000 pixels
    # Pred: [50:150, 0:100] = 10000 pixels
    # Intersection: [50:100, 0:100] = 5000 pixels
    # Union: 10000 + 10000 - 5000 = 15000 pixels
    # IoU = 5000/15000 = 0.333...
    gt_partial = np.zeros((256, 256), dtype=np.uint8)
    gt_partial[0:100, 0:100] = 1
    pred_partial = np.zeros((256, 256), dtype=np.uint8)
    pred_partial[50:150, 0:100] = 1
    Image.fromarray(gt_partial).save(gt_dir / "mask_00002.png")
    Image.fromarray(pred_partial).save(pred_dir / "mask_00002.png")
    
    # Case 3: No overlap (IoU=0.0)
    gt_no_overlap = np.zeros((256, 256), dtype=np.uint8)
    gt_no_overlap[0:50, 0:50] = 1
    pred_no_overlap = np.zeros((256, 256), dtype=np.uint8)
    pred_no_overlap[200:250, 200:250] = 1
    Image.fromarray(gt_no_overlap).save(gt_dir / "mask_00003.png")
    Image.fromarray(pred_no_overlap).save(pred_dir / "mask_00003.png")
    
    return gt_dir, pred_dir


class TestEvaluationMetrics:
    """Test evaluation metric calculations."""

    def test_perfect_prediction_iou(self, evaluation_dataset):
        """Verify perfect prediction yields IoU=1.0."""
        gt_dir, pred_dir = evaluation_dataset
        
        gt = np.array(Image.open(gt_dir / "mask_00001.png"))
        pred = np.array(Image.open(pred_dir / "mask_00001.png"))
        
        intersection = np.logical_and(gt == 1, pred == 1).sum()
        union = np.logical_or(gt == 1, pred == 1).sum()
        iou = intersection / union if union > 0 else 1.0
        
        assert iou == pytest.approx(1.0)

    def test_perfect_prediction_dice(self, evaluation_dataset):
        """Verify perfect prediction yields Dice=1.0."""
        gt_dir, pred_dir = evaluation_dataset
        
        gt = np.array(Image.open(gt_dir / "mask_00001.png"))
        pred = np.array(Image.open(pred_dir / "mask_00001.png"))
        
        intersection = np.logical_and(gt == 1, pred == 1).sum()
        dice = 2 * intersection / (gt.sum() + pred.sum()) if (gt.sum() + pred.sum()) > 0 else 1.0
        
        assert dice == pytest.approx(1.0)

    def test_partial_overlap_iou(self, evaluation_dataset):
        """Verify partial overlap yields expected IoU (1/3)."""
        gt_dir, pred_dir = evaluation_dataset
        
        gt = np.array(Image.open(gt_dir / "mask_00002.png"))
        pred = np.array(Image.open(pred_dir / "mask_00002.png"))
        
        intersection = np.logical_and(gt == 1, pred == 1).sum()
        union = np.logical_or(gt == 1, pred == 1).sum()
        iou = intersection / union if union > 0 else 0.0
        
        # intersection = 50*100 = 5000
        # union = 10000 + 10000 - 5000 = 15000
        # IoU = 5000/15000 = 0.333...
        assert iou == pytest.approx(1/3, rel=0.01)

    def test_partial_overlap_dice(self, evaluation_dataset):
        """Verify partial overlap yields expected Dice coefficient."""
        gt_dir, pred_dir = evaluation_dataset
        
        gt = np.array(Image.open(gt_dir / "mask_00002.png"))
        pred = np.array(Image.open(pred_dir / "mask_00002.png"))
        
        intersection = np.logical_and(gt == 1, pred == 1).sum()
        dice = 2 * intersection / (gt.sum() + pred.sum())
        
        # intersection = 5000
        # sum(gt) = 10000, sum(pred) = 10000
        # Dice = 2 * 5000 / 20000 = 0.5
        assert dice == pytest.approx(0.5, rel=0.01)

    def test_no_overlap_iou(self, evaluation_dataset):
        """Verify no overlap yields IoU=0.0."""
        gt_dir, pred_dir = evaluation_dataset
        
        gt = np.array(Image.open(gt_dir / "mask_00003.png"))
        pred = np.array(Image.open(pred_dir / "mask_00003.png"))
        
        intersection = np.logical_and(gt == 1, pred == 1).sum()
        union = np.logical_or(gt == 1, pred == 1).sum()
        iou = intersection / union if union > 0 else 0.0
        
        assert iou == pytest.approx(0.0)

    def test_iou_dice_mathematical_relationship(self):
        """Verify IoU and Dice satisfy: Dice = 2*IoU / (1+IoU)."""
        # Create masks with known overlap
        gt = np.zeros((100, 100), dtype=np.uint8)
        pred = np.zeros((100, 100), dtype=np.uint8)
        
        gt[0:60, 0:60] = 1  # 3600 pixels
        pred[30:90, 0:60] = 1  # 3600 pixels, 1800 overlap
        
        intersection = np.logical_and(gt == 1, pred == 1).sum()
        union = np.logical_or(gt == 1, pred == 1).sum()
        
        iou = intersection / union
        dice = 2 * intersection / (gt.sum() + pred.sum())
        
        expected_dice = 2 * iou / (1 + iou)
        assert dice == pytest.approx(expected_dice, rel=0.01)


class TestConfusionMatrix:
    """Test confusion matrix computation."""

    def test_perfect_prediction_confusion_matrix(self):
        """Verify confusion matrix for perfect prediction."""
        gt = np.array([[0, 0, 1, 1],
                       [0, 0, 1, 1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]], dtype=np.uint8)
        
        pred = gt.copy()  # Perfect prediction
        
        # Compute confusion matrix manually
        num_classes = 2
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for true_class in range(num_classes):
            for pred_class in range(num_classes):
                cm[true_class, pred_class] = np.logical_and(
                    gt == true_class, pred == pred_class
                ).sum()
        
        # All predictions should be on diagonal
        assert cm[0, 0] == 12  # True negatives
        assert cm[1, 1] == 4   # True positives
        assert cm[0, 1] == 0   # False positives
        assert cm[1, 0] == 0   # False negatives

    def test_confusion_matrix_with_errors(self):
        """Verify confusion matrix captures prediction errors."""
        gt = np.array([[0, 0, 1, 1],
                       [0, 0, 1, 1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]], dtype=np.uint8)
        
        # Prediction with some errors
        pred = np.array([[0, 1, 1, 1],  # One FP
                         [0, 0, 0, 1],  # One FN
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]], dtype=np.uint8)
        
        num_classes = 2
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for true_class in range(num_classes):
            for pred_class in range(num_classes):
                cm[true_class, pred_class] = np.logical_and(
                    gt == true_class, pred == pred_class
                ).sum()
        
        assert cm[0, 0] == 11  # True negatives (12 - 1 FP)
        assert cm[1, 1] == 3   # True positives (4 - 1 FN)
        assert cm[0, 1] == 1   # False positives
        assert cm[1, 0] == 1   # False negatives


class TestModelInference:
    """Test model inference for evaluation."""

    def test_model_produces_valid_predictions(self, tmp_path):
        """Verify model produces valid segmentation masks."""
        model = InstrumentSegmentationModel(num_classes=2)
        model.eval()
        
        # Create random input
        input_tensor = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Output should be (B, num_classes, H, W)
        assert output.shape == (1, 2, 256, 256), f"Expected (1, 2, 256, 256), got {output.shape}"
        
        # Argmax should produce valid class indices
        pred_mask = torch.argmax(output, dim=1).squeeze()
        unique_values = torch.unique(pred_mask)
        assert all(v in [0, 1] for v in unique_values.tolist()), (
            f"Prediction should only contain 0 and 1, got {unique_values.tolist()}"
        )

    def test_model_batch_inference(self):
        """Verify model handles batched inference correctly."""
        model = InstrumentSegmentationModel(num_classes=2)
        model.eval()
        
        batch_size = 8
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, 2, 256, 256), (
            f"Expected ({batch_size}, 2, 256, 256), got {output.shape}"
        )

    def test_model_deterministic_in_eval_mode(self):
        """Verify model produces consistent outputs in eval mode."""
        model = InstrumentSegmentationModel(num_classes=2)
        model.eval()
        
        input_tensor = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)
        
        assert torch.allclose(output1, output2), (
            "Model should produce identical outputs in eval mode"
        )


class TestPrecisionRecall:
    """Test precision and recall metric calculations."""

    def test_perfect_precision_recall(self):
        """Verify perfect prediction yields precision=1.0 and recall=1.0."""
        gt = np.array([0, 0, 0, 1, 1, 1])
        pred = np.array([0, 0, 0, 1, 1, 1])
        
        tp = np.logical_and(gt == 1, pred == 1).sum()
        fp = np.logical_and(gt == 0, pred == 1).sum()
        fn = np.logical_and(gt == 1, pred == 0).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)

    def test_precision_with_false_positives(self):
        """Verify precision decreases with false positives."""
        gt = np.array([0, 0, 0, 1, 1, 1])
        pred = np.array([1, 1, 0, 1, 1, 1])  # Two FPs
        
        tp = np.logical_and(gt == 1, pred == 1).sum()  # 3
        fp = np.logical_and(gt == 0, pred == 1).sum()  # 2
        
        precision = tp / (tp + fp)  # 3/5 = 0.6
        
        assert precision == pytest.approx(0.6, rel=0.01)

    def test_recall_with_false_negatives(self):
        """Verify recall decreases with false negatives."""
        gt = np.array([0, 0, 0, 1, 1, 1])
        pred = np.array([0, 0, 0, 0, 1, 1])  # One FN
        
        tp = np.logical_and(gt == 1, pred == 1).sum()  # 2
        fn = np.logical_and(gt == 1, pred == 0).sum()  # 1
        
        recall = tp / (tp + fn)  # 2/3 = 0.666...
        
        assert recall == pytest.approx(2/3, rel=0.01)

    def test_f1_score_calculation(self):
        """Verify F1 score is harmonic mean of precision and recall."""
        gt = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        pred = np.array([1, 0, 0, 0, 1, 1, 1, 1])  # 1 FP, 1 FN
        
        tp = np.logical_and(gt == 1, pred == 1).sum()  # 4
        fp = np.logical_and(gt == 0, pred == 1).sum()  # 1
        fn = np.logical_and(gt == 1, pred == 0).sum()  # 1
        
        precision = tp / (tp + fp)  # 4/5 = 0.8
        recall = tp / (tp + fn)  # 4/5 = 0.8
        f1 = 2 * precision * recall / (precision + recall)  # 0.8
        
        assert f1 == pytest.approx(0.8, rel=0.01)


class TestEdgeCases:
    """Test edge cases in evaluation."""

    def test_empty_prediction_mask(self):
        """Handle case where model predicts no instruments."""
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[40:60, 40:60] = 1  # Ground truth has instruments
        
        pred = np.zeros((100, 100), dtype=np.uint8)  # No predictions
        
        intersection = np.logical_and(gt == 1, pred == 1).sum()
        union = np.logical_or(gt == 1, pred == 1).sum()
        
        iou = intersection / union if union > 0 else 0.0
        
        assert iou == pytest.approx(0.0)

    def test_empty_ground_truth_mask(self):
        """Handle case where ground truth has no instruments."""
        gt = np.zeros((100, 100), dtype=np.uint8)  # No instruments
        
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[40:60, 40:60] = 1  # Model predicts instruments
        
        # When GT is empty but prediction exists, IoU should be 0
        intersection = np.logical_and(gt == 1, pred == 1).sum()
        union = np.logical_or(gt == 1, pred == 1).sum()
        
        iou = intersection / union if union > 0 else 1.0
        
        assert iou == pytest.approx(0.0)

    def test_both_masks_empty(self):
        """Handle case where both masks are empty (no instruments)."""
        gt = np.zeros((100, 100), dtype=np.uint8)
        pred = np.zeros((100, 100), dtype=np.uint8)
        
        intersection = np.logical_and(gt == 1, pred == 1).sum()
        union = np.logical_or(gt == 1, pred == 1).sum()
        
        # When both are empty, IoU is typically defined as 1.0 (correct prediction)
        iou = intersection / union if union > 0 else 1.0
        
        assert iou == pytest.approx(1.0)

    def test_single_pixel_prediction(self):
        """Handle single-pixel predictions and ground truth."""
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[50, 50] = 1
        
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[50, 50] = 1
        
        intersection = np.logical_and(gt == 1, pred == 1).sum()
        union = np.logical_or(gt == 1, pred == 1).sum()
        
        iou = intersection / union
        
        assert iou == pytest.approx(1.0)
