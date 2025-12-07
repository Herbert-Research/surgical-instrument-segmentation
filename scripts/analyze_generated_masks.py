"""
Analyze the quality of generated masks by comparing with ground truth.

This script helps evaluate model performance on new videos by:
1. Visual side-by-side comparison (frame, ground truth, prediction)
2. Quantitative metrics (IoU, Dice, Precision, Recall)
3. Frame-by-frame analysis to identify problematic frames
4. Statistical summary with histograms

Usage:
    # Compare generated masks with ground truth from the Full Dataset
    python scripts/analyze_generated_masks.py \
        --generated-dir "datasets/Cholec80/video01_test" \
        --ground-truth-frames-dir "datasets/Full Dataset/frames" \
        --ground-truth-masks-dir "datasets/Full Dataset/masks" \
        --video-name "video01" \
        --output "video01_quality_analysis.png"
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze generated mask quality")
    parser.add_argument(
        "--generated-dir",
        type=Path,
        required=True,
        help="Directory with generated frames and masks (video01_test)",
    )
    parser.add_argument(
        "--ground-truth-frames-dir",
        type=Path,
        help="Directory with ground truth frames (optional for metrics)",
    )
    parser.add_argument(
        "--ground-truth-masks-dir",
        type=Path,
        help="Directory with ground truth masks (optional for metrics comparison)",
    )
    parser.add_argument(
        "--video-name", type=str, default="video01", help="Video name prefix (e.g., video01)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/generated_masks_analysis.png"),
        help="Output visualization file",
    )
    parser.add_argument(
        "--num-samples", type=int, default=6, help="Number of samples to show in visualization"
    )
    return parser.parse_args()


def load_generated_pairs(generated_dir: Path) -> list[tuple[Path, Path]]:
    """Load frame-mask pairs from generated directory."""
    all_files = sorted(generated_dir.glob("*.png"))

    frames = [f for f in all_files if "_frame.png" in f.name]
    masks = [f for f in all_files if "_mask.png" in f.name]

    pairs = []
    for frame_path in frames:
        # Find corresponding mask
        mask_name = frame_path.name.replace("_frame.png", "_mask.png")
        mask_path = generated_dir / mask_name
        if mask_path.exists():
            pairs.append((frame_path, mask_path))

    return pairs


def find_ground_truth_mask(frame_path: Path, gt_masks_dir: Path, video_name: str) -> Optional[Path]:
    """Try to find corresponding ground truth mask."""
    # Extract frame number from generated filename: video01_000005_frame.png
    parts = frame_path.stem.split("_")
    if len(parts) >= 2:
        frame_num = parts[1]  # Get the frame number

        # Try different naming patterns
        patterns = [
            f"{video_name}_mask_{frame_num}.png",
            f"{video_name}_mask_{int(frame_num):06d}.png",
        ]

        for pattern in patterns:
            gt_path = gt_masks_dir / pattern
            if gt_path.exists():
                return gt_path

    return None


def calculate_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    """Calculate segmentation metrics."""
    # Binarize masks
    pred_binary = (pred_mask > 127).astype(np.uint8)

    # Handle different ground truth formats
    if true_mask.max() > 1:
        # CholecSeg8k format (class IDs 31, 32)
        true_binary = ((true_mask == 31) | (true_mask == 32) | (true_mask == 1)).astype(np.uint8)
    else:
        true_binary = true_mask.astype(np.uint8)

    # Calculate metrics
    tp = np.logical_and(pred_binary == 1, true_binary == 1).sum()
    fp = np.logical_and(pred_binary == 1, true_binary == 0).sum()
    fn = np.logical_and(pred_binary == 0, true_binary == 1).sum()
    tn = np.logical_and(pred_binary == 0, true_binary == 0).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "instrument_pixels": int(true_binary.sum()),
    }


def analyze_visual_quality(pairs: list[tuple[Path, Path]], num_samples: int = 6) -> Figure:
    """Create visual analysis of generated masks."""
    print("\n" + "=" * 70)
    print("VISUAL QUALITY ANALYSIS")
    print("=" * 70)

    # Select diverse samples
    step = max(1, len(pairs) // num_samples)
    selected = [pairs[i * step] for i in range(min(num_samples, len(pairs)))]

    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 3 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (frame_path, mask_path) in enumerate(selected):
        # Load images
        frame = Image.open(frame_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert mask to binary for visualization
        mask_array = np.array(mask)
        instrument_pixels = (mask_array > 127).sum()
        total_pixels = mask_array.size
        coverage = (instrument_pixels / total_pixels) * 100

        # Plot frame
        axes[idx, 0].imshow(frame)
        axes[idx, 0].set_title(f"Frame {idx+1}: {frame_path.name}", fontweight="bold")
        axes[idx, 0].axis("off")

        # Plot mask with stats
        axes[idx, 1].imshow(mask_array, cmap="viridis")
        axes[idx, 1].set_title(f"Predicted Mask ({coverage:.1f}% instruments)", fontweight="bold")
        axes[idx, 1].axis("off")

        print(f"  Frame {idx+1}: {instrument_pixels:,} instrument pixels ({coverage:.1f}%)")

    plt.tight_layout()
    return fig


def analyze_with_ground_truth(
    pairs: list[tuple[Path, Path]], gt_masks_dir: Path, video_name: str, num_samples: int = 6
) -> tuple[Optional[Figure], list[dict[str, float]]]:
    """Analyze quality against ground truth masks."""
    print("\n" + "=" * 70)
    print("GROUND TRUTH COMPARISON")
    print("=" * 70)

    all_metrics = []
    samples_with_gt = []

    # Find pairs with ground truth
    for frame_path, pred_mask_path in pairs:
        gt_mask_path = find_ground_truth_mask(frame_path, gt_masks_dir, video_name)
        if gt_mask_path:
            samples_with_gt.append((frame_path, pred_mask_path, gt_mask_path))

    if not samples_with_gt:
        print("  ⚠ No matching ground truth masks found")
        print(f"  Searched in: {gt_masks_dir}")
        print(f"  For video: {video_name}")
        return None, []

    print(f"  Found {len(samples_with_gt)} frames with ground truth")

    # Calculate metrics for all samples
    for frame_path, pred_mask_path, gt_mask_path in samples_with_gt:
        pred_mask = np.array(Image.open(pred_mask_path).convert("L"))
        gt_mask = np.array(Image.open(gt_mask_path).convert("L"))

        # Resize if needed
        if pred_mask.shape != gt_mask.shape:
            pred_mask_img = Image.fromarray(pred_mask)
            # Convert shape to (width, height) tuple for PIL
            new_size = (gt_mask.shape[1], gt_mask.shape[0])
            pred_mask_img = pred_mask_img.resize(new_size, Image.Resampling.NEAREST)
            pred_mask = np.array(pred_mask_img)

        metrics = calculate_metrics(pred_mask, gt_mask)
        metrics["frame"] = frame_path.name
        all_metrics.append(metrics)

    # Print summary statistics
    if all_metrics:
        avg_iou = np.mean([m["iou"] for m in all_metrics])
        avg_dice = np.mean([m["dice"] for m in all_metrics])
        avg_precision = np.mean([m["precision"] for m in all_metrics])
        avg_recall = np.mean([m["recall"] for m in all_metrics])

        print(f"\n  Average Metrics:")
        print(f"    IoU:       {avg_iou:.3f}")
        print(f"    Dice:      {avg_dice:.3f}")
        print(f"    Precision: {avg_precision:.3f}")
        print(f"    Recall:    {avg_recall:.3f}")

    # Create visualization
    step = max(1, len(samples_with_gt) // num_samples)
    selected = [samples_with_gt[i * step] for i in range(min(num_samples, len(samples_with_gt)))]

    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 3.5 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (frame_path, pred_mask_path, gt_mask_path) in enumerate(selected):
        # Load images
        frame = Image.open(frame_path).convert("RGB")
        pred_mask = np.array(Image.open(pred_mask_path).convert("L"))
        gt_mask = np.array(Image.open(gt_mask_path).convert("L"))

        # Resize prediction if needed
        if pred_mask.shape != gt_mask.shape:
            pred_mask_img = Image.fromarray(pred_mask)
            # Convert shape to (width, height) tuple for PIL
            new_size = (gt_mask.shape[1], gt_mask.shape[0])
            pred_mask_img = pred_mask_img.resize(new_size, Image.Resampling.NEAREST)
            pred_mask = np.array(pred_mask_img)

        # Calculate metrics
        metrics = calculate_metrics(pred_mask, gt_mask)

        # Plot frame
        axes[idx, 0].imshow(frame)
        axes[idx, 0].set_title(f"Frame {idx+1}", fontweight="bold")
        axes[idx, 0].axis("off")

        # Plot ground truth
        gt_binary = ((gt_mask == 31) | (gt_mask == 32) | (gt_mask == 1)).astype(np.uint8) * 255
        axes[idx, 1].imshow(gt_binary, cmap="viridis")
        axes[idx, 1].set_title("Ground Truth", fontweight="bold")
        axes[idx, 1].axis("off")

        # Plot prediction
        axes[idx, 2].imshow(pred_mask, cmap="viridis")
        axes[idx, 2].set_title(
            f'Prediction\nIoU: {metrics["iou"]:.3f} | Dice: {metrics["dice"]:.3f}',
            fontweight="bold",
        )
        axes[idx, 2].axis("off")

    plt.tight_layout()
    return fig, all_metrics


def main():
    args = parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANALYZING GENERATED MASKS")
    print("=" * 70)
    print(f"Generated directory: {args.generated_dir}")

    # Load generated pairs
    pairs = load_generated_pairs(args.generated_dir)
    print(f"Found {len(pairs)} frame-mask pairs")

    if not pairs:
        print("❌ No frame-mask pairs found!")
        return

    # Analyze with or without ground truth
    if args.ground_truth_masks_dir and args.ground_truth_masks_dir.exists():
        fig, metrics = analyze_with_ground_truth(
            pairs, args.ground_truth_masks_dir, args.video_name, args.num_samples
        )
        if fig:
            fig.savefig(args.output, dpi=300, bbox_inches="tight")
            print(f"\n✓ Saved analysis to: {args.output}")
        else:
            # Fallback to visual-only analysis
            fig = analyze_visual_quality(pairs, args.num_samples)
            fig.savefig(args.output, dpi=300, bbox_inches="tight")
            print(f"\n✓ Saved visual analysis to: {args.output}")
    else:
        # Visual-only analysis
        fig = analyze_visual_quality(pairs, args.num_samples)
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"\n✓ Saved visual analysis to: {args.output}")

    plt.close(fig)
    print("=" * 70)


if __name__ == "__main__":
    main()
