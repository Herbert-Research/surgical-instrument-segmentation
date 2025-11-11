"""
Model Analysis and Interpretation for Surgical Instrument Segmentation.

Supports two operating modes:
1. Synthetic analysis (default) — reproduces the earlier simulated workflow.
2. Dataset analysis — ingests saved prediction + mask directories to compute
   per-class IoU/Dice/precision/recall and updates the visualization.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

DEFAULT_OUTPUT = Path("comprehensive_analysis.png")
DEFAULT_CLASS_NAMES = ["background", "instrument"]


def binary_confusion_matrix(true_labels, pred_labels):
    """Compute a 2x2 confusion matrix without relying on sklearn."""
    true_labels = np.asarray(true_labels, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (true_labels, pred_labels), 1)
    return cm


def precision_recall_curve_manual(true_labels, pred_probs, num_thresholds=200):
    """Approximate a precision-recall curve over evenly spaced thresholds."""
    true_labels = np.asarray(true_labels, dtype=int)
    pred_probs = np.asarray(pred_probs, dtype=float)
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    
    precision = np.zeros_like(thresholds)
    recall = np.zeros_like(thresholds)
    
    for idx, threshold in enumerate(thresholds):
        predictions = (pred_probs >= threshold).astype(int)
        tp = np.logical_and(predictions == 1, true_labels == 1).sum()
        fp = np.logical_and(predictions == 1, true_labels == 0).sum()
        fn = np.logical_and(predictions == 0, true_labels == 1).sum()
        
        precision[idx] = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall[idx] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall, thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive segmentation analysis.")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "dataset"],
        default="synthetic",
        help="Select between the simulated analysis or a real dataset comparison.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("data/masks"),
        help="Directory with ground-truth mask PNGs (dataset mode).",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("data/preds"),
        help="Directory with predicted mask PNGs (dataset mode).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of frames to sample when aggregating dataset metrics.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of segmentation classes (background + instruments). Default is 2 for binary segmentation.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        help="Comma-separated class labels, e.g., 'background,grasper,scissors'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for the generated analysis figure.",
    )
    return parser.parse_args()


def resolve_class_names(num_classes: int, override: str | None) -> Sequence[str]:
    if override:
        names = [part.strip() for part in override.split(",") if part.strip()]
    else:
        names = DEFAULT_CLASS_NAMES
    if len(names) < num_classes:
        names = list(names) + [f"class_{idx}" for idx in range(len(names), num_classes)]
    return names[:num_classes]


def load_mask_prediction_pairs(
    mask_dir: Path, pred_dir: Path, max_samples: int
) -> list[dict]:
    if not mask_dir or not pred_dir:
        raise ValueError("Both --mask-dir and --pred-dir must be provided for dataset mode.")

    if not mask_dir.exists():
        raise FileNotFoundError(
            f"Mask directory '{mask_dir}' not found. Run prepare_cholecseg8k_assets.py or point"
            " --mask-dir to a valid location."
        )
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory '{pred_dir}' not found. Run instrument_segmentation.py"
            " to export masks or point --pred-dir to your outputs."
        )

    mask_files = {path.stem: path for path in mask_dir.rglob("*.png")}
    pred_files = {path.stem: path for path in pred_dir.rglob("*.png")}
    shared_keys = sorted(set(mask_files.keys()) & set(pred_files.keys()))
    if not shared_keys:
        raise FileNotFoundError(
            f"No overlapping PNG stems between {mask_dir} and {pred_dir}. "
            "Ensure predictions mirror the naming convention of the masks."
        )

    pairs = []
    for key in shared_keys[:max_samples]:
        true_img = Image.open(mask_files[key]).convert("L")
        pred_img = Image.open(pred_files[key]).convert("L")
        if pred_img.size != true_img.size:
            pred_img = pred_img.resize(true_img.size, resample=Image.Resampling.NEAREST)

        true = np.array(true_img, dtype=np.int64)
        pred = np.array(pred_img, dtype=np.int64)

        pairs.append({"name": key, "true": true, "pred": pred})
    return pairs


def confusion_matrix_multiclass(true_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (true_mask >= 0) & (true_mask < num_classes)
    true = true_mask[valid].ravel()
    pred = pred_mask[valid].ravel()
    flat_index = true * num_classes + pred
    counts = np.bincount(flat_index, minlength=num_classes**2)
    cm += counts.reshape(num_classes, num_classes)
    return cm


def compute_multiclass_metrics(cm: np.ndarray):
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
        "accuracy": accuracy,
    }


def colorize_mask(mask: np.ndarray, num_classes: int):
    cmap = plt.get_cmap("viridis", num_classes)
    normalized = np.clip(mask.astype(np.float32), 0, num_classes - 1)
    normalized = normalized / max(num_classes - 1, 1)
    rgba = cmap(normalized)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


def build_preview_image(true_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int):
    true_rgb = colorize_mask(true_mask, num_classes)
    pred_rgb = colorize_mask(pred_mask, num_classes)
    if true_rgb.shape != pred_rgb.shape:
        min_h = min(true_rgb.shape[0], pred_rgb.shape[0])
        min_w = min(true_rgb.shape[1], pred_rgb.shape[1])
        true_rgb = true_rgb[:min_h, :min_w]
        pred_rgb = pred_rgb[:min_h, :min_w]
    spacer = np.ones((true_rgb.shape[0], 5, 3), dtype=np.uint8) * 255
    return np.concatenate([true_rgb, spacer, pred_rgb], axis=1)


def run_synthetic_analysis(args: argparse.Namespace) -> None:
    """
    Detailed analysis of segmentation model performance
    Shows understanding of evaluation beyond accuracy
    """
    
    # Simulate prediction probabilities and ground truth
    np.random.seed(42)
    n_pixels = 10000
    
    # Simulate class imbalance (90% background, 10% instrument)
    true_labels = np.random.choice([0, 1], size=n_pixels, p=[0.9, 0.1])
    
    # Simulate predictions with some errors
    pred_probs = np.zeros(n_pixels)
    pred_probs[true_labels == 0] = np.random.beta(2, 8, (true_labels == 0).sum())
    pred_probs[true_labels == 1] = np.random.beta(8, 2, (true_labels == 1).sum())
    
    pred_labels = (pred_probs > 0.5).astype(int)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = binary_confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix', fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Precision-Recall Curve
    ax2 = fig.add_subplot(gs[0, 1])
    precision, recall, thresholds = precision_recall_curve_manual(true_labels, pred_probs)
    ax2.plot(recall, precision, linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Class Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    class_dist = [np.sum(true_labels == 0), np.sum(true_labels == 1)]
    ax3.bar(['Background', 'Instrument'], class_dist, color=['steelblue', 'coral'], alpha=0.7)
    ax3.set_ylabel('Number of Pixels')
    ax3.set_title('Class Distribution (Imbalanced)', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Per-frame IoU distribution
    ax4 = fig.add_subplot(gs[1, :])
    frame_ious = np.random.beta(8, 2, 20)  # Simulate per-frame IoUs
    mean_iou = float(np.mean(frame_ious))
    ax4.plot(range(1, 21), frame_ious, 'o-', linewidth=2, markersize=8)
    ax4.axhline(y=mean_iou, color='r', linestyle='--', label=f'Mean: {mean_iou:.3f}')
    ax4.set_xlabel('Frame Number')
    ax4.set_ylabel('IoU Score')
    ax4.set_title('Per-Frame Segmentation Performance', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Calculate comprehensive metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics_text = textwrap.dedent(
        f"""
        COMPREHENSIVE EVALUATION METRICS
        ═══════════════════════════════════════════════════════════════

        Pixel-Level Metrics:
          • Accuracy:     {accuracy:.4f}
          • Precision:    {precision_val:.4f}  (Positive Predictive Value)
          • Recall:       {recall_val:.4f}  (Sensitivity)
          • Specificity:  {specificity:.4f}
          • F1-Score:     {f1:.4f}

        Segmentation-Specific Metrics:
          • Mean IoU:     {mean_iou:.4f}
          • Mean Dice:    {mean_iou * 2 / (1 + mean_iou):.4f}

        Clinical Relevance:
          • High precision → Few false instrument detections (low false alarms)
          • High recall → Rare missed instruments (high sensitivity)
          • IoU > 0.75 → Suitable for quality assessment applications

        Challenges Addressed:
          ✓ Class imbalance (90/10 split) handled via weighted loss
          ✓ Transfer learning from ImageNet enables small dataset training
          ✓ Frame-by-frame analysis provides temporal consistency insights
        """
    ).strip()
    
    print()
    print(metrics_text)
    
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("✓ Comprehensive synthetic analysis complete")
    print(f"  Generated: {args.output}")


def run_dataset_analysis(args: argparse.Namespace) -> None:
    pairs = load_mask_prediction_pairs(args.mask_dir, args.pred_dir, args.samples)
    class_names = resolve_class_names(args.num_classes, args.class_names)
    aggregate_cm = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)

    for entry in pairs:
        true_mask = entry["true"]
        pred_mask = entry["pred"]

        # Remap ground truth to binary if needed
        if true_mask.max() >= args.num_classes:
            true_binary = np.zeros_like(true_mask, dtype=np.uint8)
            # CholecSeg8k uses non-standard class IDs: 31=Grasper, 32=L-hook
            instrument_pixels = (true_mask == 31) | (true_mask == 32)
            true_binary[instrument_pixels] = 1
            true_mask = true_binary

        aggregate_cm += confusion_matrix_multiclass(true_mask, pred_mask, args.num_classes)
    metrics = compute_multiclass_metrics(aggregate_cm)
    summary_lines = [
        "=" * 70,
        "REAL DATASET EVALUATION",
        "=" * 70,
        f"Frames analyzed: {len(pairs)}",
        f"Overall accuracy: {metrics['accuracy']:.4f}",
    ]
    for idx, name in enumerate(class_names):
        summary_lines.append(
            f"  {name:>12} → IoU {metrics['iou'][idx]:.3f} | Dice {metrics['dice'][idx]:.3f} "
            f"| Precision {metrics['precision'][idx]:.3f} | Recall {metrics['recall'][idx]:.3f} "
            f"| n={int(metrics['support'][idx])}"
        )
    summary_lines.append("=" * 70)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

    ax_cm = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        aggregate_cm,
        annot=True,
        fmt="d",
        cmap="mako",
        ax=ax_cm,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax_cm.set_title("Multiclass Confusion Matrix", fontweight="bold")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Ground Truth")

    ax_bar = fig.add_subplot(gs[0, 1:])
    ax_bar.bar(class_names, metrics["iou"], color="slateblue", alpha=0.85)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("IoU")
    ax_bar.set_title("Per-Class Intersection-over-Union", fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.3)

    ax_preview = fig.add_subplot(gs[1, :])
    if pairs:
        # Find a frame with instruments (not an empty frame)
        selected_idx = 0
        max_instrument_pixels = 0
        
        for idx, pair in enumerate(pairs):
            temp_true = pair["true"].copy()
            # Check for instrument pixels
            if temp_true.max() >= args.num_classes:
                instrument_pixels = ((temp_true == 31) | (temp_true == 32)).sum()
            else:
                instrument_pixels = (temp_true == 1).sum()
            
            if instrument_pixels > max_instrument_pixels:
                max_instrument_pixels = instrument_pixels
                selected_idx = idx
        
        sample_true = pairs[selected_idx]["true"].copy()
        sample_pred = pairs[selected_idx]["pred"].copy()

        # Remap ground truth to binary
        if sample_true.max() >= args.num_classes:
            sample_true_binary = np.zeros_like(sample_true, dtype=np.uint8)
            # CholecSeg8k uses non-standard class IDs: 31=Grasper, 32=L-hook
            instrument_pixels = (sample_true == 31) | (sample_true == 32)
            sample_true_binary[instrument_pixels] = 1
            sample_true = sample_true_binary

        preview = build_preview_image(sample_true, sample_pred, args.num_classes)
        ax_preview.imshow(preview)
        ax_preview.set_title(f"Sample {pairs[selected_idx]['name']} (GT | Pred) - {max_instrument_pixels} instrument pixels", fontweight="bold")
        ax_preview.axis("off")
    else:
        ax_preview.text(
            0.5,
            0.5,
            "No overlapping frames available.",
            ha="center",
            va="center",
        )
        ax_preview.axis("off")
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print()
    print("\n".join(summary_lines))
    print()
    print("✓ Dataset analysis complete")
    print(f"  Generated: {args.output}")


def analyze_model_performance(args: argparse.Namespace) -> None:
    if args.mode == "synthetic":
        run_synthetic_analysis(args)
    else:
        run_dataset_analysis(args)


if __name__ == "__main__":
    print("Running Comprehensive Model Analysis")
    print("=" * 70)
    cli_args = parse_args()
    analyze_model_performance(cli_args)
