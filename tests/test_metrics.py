"""Tests for evaluation metrics using production implementations."""

import pytest
import torch

from surgical_segmentation.evaluation.metrics import (
    compute_all_metrics,
    compute_dice,
    compute_iou,
    compute_precision,
    compute_recall,
)


def test_iou_known_values() -> None:
    """IoU matches hand-calculated value for partial overlap."""
    pred = torch.tensor([[1, 1], [0, 0]])
    target = torch.tensor([[1, 0], [1, 0]])

    # intersection=1, union=3 => IoU=1/3
    iou = compute_iou(pred, target)
    assert iou == pytest.approx(1 / 3, rel=0.01)


def test_dice_known_values() -> None:
    """Dice matches hand-calculated value for partial overlap."""
    pred = torch.tensor([[1, 1], [0, 0]])
    target = torch.tensor([[1, 0], [1, 0]])

    # intersection=1, total positives=4 => Dice=2/4=0.5
    dice = compute_dice(pred, target)
    assert dice == pytest.approx(0.5, rel=0.01)


def test_precision_recall_known_values() -> None:
    """Precision and recall match hand-calculated values."""
    pred = torch.tensor([1, 1, 0, 0])
    target = torch.tensor([1, 0, 1, 0])

    # TP=1, FP=1, FN=1
    precision = compute_precision(pred, target)
    recall = compute_recall(pred, target)

    assert precision == pytest.approx(0.5, rel=0.01)
    assert recall == pytest.approx(0.5, rel=0.01)


def test_all_metrics_consistency() -> None:
    """compute_all_metrics matches individual metric functions."""
    pred = torch.tensor([[1, 0], [1, 0]])
    target = torch.tensor([[1, 1], [0, 0]])

    results = compute_all_metrics(pred, target)

    assert results["iou"] == pytest.approx(compute_iou(pred, target))
    assert results["dice"] == pytest.approx(compute_dice(pred, target))
    assert results["precision"] == pytest.approx(compute_precision(pred, target))
    assert results["recall"] == pytest.approx(compute_recall(pred, target))


def test_both_masks_empty() -> None:
    """Empty masks are treated as perfect agreement."""
    pred = torch.zeros((10, 10), dtype=torch.int64)
    target = torch.zeros((10, 10), dtype=torch.int64)

    assert compute_iou(pred, target) == pytest.approx(1.0)
    assert compute_dice(pred, target) == pytest.approx(1.0)
    assert compute_precision(pred, target) == pytest.approx(1.0)
    assert compute_recall(pred, target) == pytest.approx(1.0)


def test_empty_prediction_nonempty_target() -> None:
    """Empty prediction against non-empty target yields near-zero metrics."""
    pred = torch.zeros((10, 10), dtype=torch.int64)
    target = torch.zeros((10, 10), dtype=torch.int64)
    target[2:4, 2:4] = 1

    assert compute_iou(pred, target) == pytest.approx(0.0, abs=1e-6)
    assert compute_dice(pred, target) == pytest.approx(0.0, abs=1e-6)
    assert compute_recall(pred, target) == pytest.approx(0.0, abs=1e-6)


def test_single_pixel_match() -> None:
    """Single-pixel matches return perfect scores."""
    pred = torch.zeros((5, 5), dtype=torch.int64)
    target = torch.zeros((5, 5), dtype=torch.int64)
    pred[2, 2] = 1
    target[2, 2] = 1

    assert compute_iou(pred, target) == pytest.approx(1.0)
    assert compute_dice(pred, target) == pytest.approx(1.0)


def test_invalid_class_ids_raise() -> None:
    """Non-binary values should raise a ValueError."""
    pred = torch.tensor([[0, 2], [1, 0]])
    target = torch.tensor([[0, 1], [1, 0]])

    with pytest.raises(ValueError, match="binary"):
        compute_iou(pred, target)
