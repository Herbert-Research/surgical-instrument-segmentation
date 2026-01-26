"""
Canonical metric implementations for surgical segmentation.

All metric calculations in the pipeline MUST use these functions.
Do not reimplement metrics elsewhere.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch


class SegmentationMetrics(TypedDict):
    """Typed dictionary for metric outputs."""

    precision: np.ndarray
    recall: np.ndarray
    iou: np.ndarray
    dice: np.ndarray
    support: np.ndarray
    accuracy: float


def _validate_binary_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Validate that a tensor contains only binary values and return a bool mask.

    Args:
        tensor: Input tensor expected to contain only 0/1 values.
        name: Human-readable name used in error messages.

    Returns:
        Boolean tensor suitable for logical operations.

    Raises:
        TypeError: If the input is not a torch.Tensor.
        ValueError: If the tensor contains values other than 0 or 1.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")

    unique_values = torch.unique(tensor)
    valid_mask = (unique_values == 0) | (unique_values == 1)
    if not torch.all(valid_mask):
        raise ValueError(
            f"{name} must be binary with values in {{0, 1}}. "
            f"Found values: {unique_values.tolist()}"
        )
    return tensor.bool()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute Intersection over Union.

    Args:
        pred: Binary prediction tensor of shape (H, W) or (N, H, W).
        target: Binary ground truth tensor of shape (H, W) or (N, H, W).
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        IoU score as a float.

    Raises:
        TypeError: If inputs are not torch tensors.
        ValueError: If inputs are not binary.
    """
    pred_bool = _validate_binary_tensor(pred, "pred")
    target_bool = _validate_binary_tensor(target, "target")

    intersection = (pred_bool & target_bool).float().sum()
    union = (pred_bool | target_bool).float().sum()
    return float(((intersection + smooth) / (union + smooth)).item())


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient.

    Args:
        pred: Binary prediction tensor of shape (H, W) or (N, H, W).
        target: Binary ground truth tensor of shape (H, W) or (N, H, W).
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice coefficient as a float.

    Raises:
        TypeError: If inputs are not torch tensors.
        ValueError: If inputs are not binary.
    """
    pred_bool = _validate_binary_tensor(pred, "pred")
    target_bool = _validate_binary_tensor(target, "target")

    intersection = (pred_bool & target_bool).float().sum()
    total = pred_bool.float().sum() + target_bool.float().sum()
    return float(((2 * intersection + smooth) / (total + smooth)).item())


def compute_precision(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute precision (positive predictive value).

    Args:
        pred: Binary prediction tensor of shape (H, W) or (N, H, W).
        target: Binary ground truth tensor of shape (H, W) or (N, H, W).
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Precision as a float.

    Raises:
        TypeError: If inputs are not torch tensors.
        ValueError: If inputs are not binary.
    """
    pred_bool = _validate_binary_tensor(pred, "pred")
    target_bool = _validate_binary_tensor(target, "target")

    tp = (pred_bool & target_bool).float().sum()
    fp = (pred_bool & ~target_bool).float().sum()
    return float(((tp + smooth) / (tp + fp + smooth)).item())


def compute_recall(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute recall (sensitivity).

    Args:
        pred: Binary prediction tensor of shape (H, W) or (N, H, W).
        target: Binary ground truth tensor of shape (H, W) or (N, H, W).
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Recall as a float.

    Raises:
        TypeError: If inputs are not torch tensors.
        ValueError: If inputs are not binary.
    """
    pred_bool = _validate_binary_tensor(pred, "pred")
    target_bool = _validate_binary_tensor(target, "target")

    tp = (pred_bool & target_bool).float().sum()
    fn = (~pred_bool & target_bool).float().sum()
    return float(((tp + smooth) / (tp + fn + smooth)).item())


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute IoU, Dice, precision, and recall in one call.

    Args:
        pred: Binary prediction tensor of shape (H, W) or (N, H, W).
        target: Binary ground truth tensor of shape (H, W) or (N, H, W).

    Returns:
        Dictionary with keys: "iou", "dice", "precision", "recall".

    Raises:
        TypeError: If inputs are not torch tensors.
        ValueError: If inputs are not binary.
    """
    return {
        "iou": compute_iou(pred, target),
        "dice": compute_dice(pred, target),
        "precision": compute_precision(pred, target),
        "recall": compute_recall(pred, target),
    }


def confusion_matrix_multiclass(
    true_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute confusion matrix for multiclass segmentation.

    Args:
        true_mask: Ground truth mask with integer class labels.
        pred_mask: Predicted mask with integer class labels.
        num_classes: Total number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    true = np.asarray(true_mask, dtype=np.int64)
    pred = np.asarray(pred_mask, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (true >= 0) & (true < num_classes)
    true = true[valid].ravel()
    pred = pred[valid].ravel()
    flat_index = true * num_classes + pred
    counts = np.bincount(flat_index, minlength=num_classes**2)
    cm += counts.reshape(num_classes, num_classes)
    return cm


def compute_metrics_from_cm(cm: np.ndarray) -> SegmentationMetrics:
    """Compute precision, recall, IoU, Dice, and accuracy from a confusion matrix.

    Args:
        cm: Confusion matrix of shape (num_classes, num_classes).

    Returns:
        Dictionary containing per-class arrays for precision/recall/IoU/Dice,
        support counts, and scalar overall accuracy.
    """
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    accuracy = tp.sum() / cm.sum() if cm.sum() > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "support": support,
        "accuracy": float(accuracy),
    }
