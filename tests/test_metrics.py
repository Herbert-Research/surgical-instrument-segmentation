"""Tests for evaluation metrics."""

import numpy as np
import pytest


def compute_iou(true_mask: np.ndarray, pred_mask: np.ndarray, class_id: int) -> float:
    """Compute Intersection over Union for a single class."""
    true_binary = true_mask == class_id
    pred_binary = pred_mask == class_id

    intersection = np.logical_and(true_binary, pred_binary).sum()
    union = np.logical_or(true_binary, pred_binary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_dice(true_mask: np.ndarray, pred_mask: np.ndarray, class_id: int) -> float:
    """Compute Dice coefficient for a single class."""
    true_binary = true_mask == class_id
    pred_binary = pred_mask == class_id

    intersection = np.logical_and(true_binary, pred_binary).sum()
    total = true_binary.sum() + pred_binary.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2 * intersection / total


class TestIoU:
    """Test suite for IoU metric."""

    def test_perfect_prediction(self, sample_mask_numpy):
        """IoU should be 1.0 for perfect prediction."""
        iou = compute_iou(sample_mask_numpy, sample_mask_numpy, class_id=1)
        assert iou == pytest.approx(1.0)

    def test_no_overlap(self, sample_mask_numpy):
        """IoU should be 0.0 for no overlap."""
        pred_mask = np.zeros_like(sample_mask_numpy)
        pred_mask[200:250, 200:250] = 1  # Different region

        iou = compute_iou(sample_mask_numpy, pred_mask, class_id=1)
        assert iou == pytest.approx(0.0)

    def test_partial_overlap(self):
        """IoU should be correct for partial overlap."""
        true_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask = np.zeros((100, 100), dtype=np.uint8)

        true_mask[0:50, 0:50] = 1  # 2500 pixels
        pred_mask[25:75, 0:50] = 1  # 2500 pixels, 1250 overlap

        # Intersection = 25*50 = 1250
        # Union = 2500 + 2500 - 1250 = 3750
        # IoU = 1250/3750 = 0.333...

        iou = compute_iou(true_mask, pred_mask, class_id=1)
        assert iou == pytest.approx(1 / 3, rel=0.01)


class TestDice:
    """Test suite for Dice coefficient."""

    def test_perfect_prediction(self, sample_mask_numpy):
        """Dice should be 1.0 for perfect prediction."""
        dice = compute_dice(sample_mask_numpy, sample_mask_numpy, class_id=1)
        assert dice == pytest.approx(1.0)

    def test_no_overlap(self, sample_mask_numpy):
        """Dice should be 0.0 for no overlap."""
        pred_mask = np.zeros_like(sample_mask_numpy)
        pred_mask[200:250, 200:250] = 1

        dice = compute_dice(sample_mask_numpy, pred_mask, class_id=1)
        assert dice == pytest.approx(0.0)

    def test_iou_dice_relationship(self, sample_mask_numpy):
        """Verify IoU and Dice relationship: Dice = 2*IoU / (1+IoU)."""
        pred_mask = sample_mask_numpy.copy()
        pred_mask[100:125, 100:150] = 0  # Remove half the prediction

        iou = compute_iou(sample_mask_numpy, pred_mask, class_id=1)
        dice = compute_dice(sample_mask_numpy, pred_mask, class_id=1)

        expected_dice = 2 * iou / (1 + iou)
        assert dice == pytest.approx(expected_dice, rel=0.01)
