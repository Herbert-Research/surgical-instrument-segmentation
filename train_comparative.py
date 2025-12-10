"""
Comparative Training Script for Baseline Analysis
Trains both U-Net and DeepLabV3-ResNet50 on identical data splits
with statistical validation and cross-validation support.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from surgical_segmentation import SurgicalDataset
from surgical_segmentation.evaluation.statistics import interpret_effect_size, paired_comparison
from surgical_segmentation.models import InstrumentSegmentationModel, UNet
from surgical_segmentation.training.trainer import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INSTRUMENT_CLASS_WEIGHT,
    NUM_CLASSES,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparative model training")
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparative"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["unet", "deeplabv3"],
        choices=["unet", "deeplabv3"],
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def get_model(architecture: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Factory function for model instantiation."""

    if architecture == "unet":
        model = UNet(n_channels=3, n_classes=num_classes, bilinear=True)
        print(f"✓ U-Net initialized: {model.count_parameters():,} parameters")
    elif architecture == "deeplabv3":
        model = InstrumentSegmentationModel(num_classes=num_classes)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ DeepLabV3-ResNet50 initialized: {total_params:,} parameters")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    total_epochs: int,
) -> float:
    """Train for one epoch."""

    model.train()
    epoch_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
    for frames, masks in pbar:
        frames = frames.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(frames)
        if isinstance(outputs, dict):  # DeepLabV3 returns dict
            outputs = outputs["out"]

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return epoch_loss / max(1, len(dataloader))


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int = NUM_CLASSES,
) -> dict[str, float]:
    """Evaluate model and return metrics."""

    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for frames, masks in tqdm(dataloader, desc="Evaluating"):
            frames = frames.to(device, non_blocking=True)
            masks_np = masks.cpu().numpy()

            outputs = model(frames)
            if isinstance(outputs, dict):
                outputs = outputs["out"]

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for pred, true in zip(preds, masks_np):
                for c in range(num_classes):
                    true_c = true == c
                    pred_c = pred == c
                    cm[c, c] += np.logical_and(true_c, pred_c).sum()
                    for c2 in range(num_classes):
                        if c != c2:
                            cm[c, c2] += np.logical_and(true == c, pred == c2).sum()

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    dice = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=(2 * tp + fp + fn) > 0,
    )

    accuracy = tp.sum() / cm.sum() if cm.sum() > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "iou_background": float(iou[0]),
        "iou_instrument": float(iou[1]) if num_classes > 1 else float(iou[0]),
        "iou_mean": float(iou.mean()),
        "dice_background": float(dice[0]),
        "dice_instrument": float(dice[1]) if num_classes > 1 else float(dice[0]),
        "dice_mean": float(dice.mean()),
        "precision_instrument": float(precision[1]) if num_classes > 1 else float(precision[0]),
        "recall_instrument": float(recall[1]) if num_classes > 1 else float(recall[0]),
        "confusion_matrix": cm.tolist(),
    }


def train_model_complete(
    architecture: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
) -> tuple[nn.Module, dict]:
    """Complete training pipeline for one architecture."""

    print(f"\n{'=' * 70}")
    print(f"Training {architecture.upper()}")
    print(f"{'=' * 70}")

    model = get_model(architecture, NUM_CLASSES)
    model = model.to(args.device)

    class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=args.device)
    class_weights[1] = INSTRUMENT_CLASS_WEIGHT
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    training_history: dict[str, Any] = {
        "architecture": architecture,
        "train_losses": [],
        "val_metrics": [],
        "training_time_seconds": 0,
        "parameters": sum(p.numel() for p in model.parameters()),
    }

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            args.device,
            epoch,
            args.epochs,
        )
        training_history["train_losses"].append(train_loss)

        if epoch % 5 == 0 or epoch == args.epochs:
            val_metrics = evaluate_model(model, val_loader, args.device, NUM_CLASSES)
            val_metrics["epoch"] = epoch
            training_history["val_metrics"].append(val_metrics)

            print(f"\nEpoch {epoch} Validation:")
            print(f"  IoU (instrument): {val_metrics['iou_instrument']:.4f}")
            print(f"  Dice (instrument): {val_metrics['dice_instrument']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

    training_history["training_time_seconds"] = time.time() - start_time

    print("\nFinal Evaluation...")
    final_metrics = evaluate_model(model, val_loader, args.device, NUM_CLASSES)
    training_history["final_metrics"] = final_metrics

    print(f"\n{'=' * 70}")
    print(f"Final Results - {architecture.upper()}")
    print(f"{'=' * 70}")
    print(f"Training time: {training_history['training_time_seconds']:.1f}s")
    print(f"Final IoU (instrument): {final_metrics['iou_instrument']:.4f}")
    print(f"Final Dice (instrument): {final_metrics['dice_instrument']:.4f}")
    print(f"Final Accuracy: {final_metrics['accuracy']:.4f}")

    return model, training_history


def get_video_groups(frame_dir: Path) -> dict[str, list[str]]:
    """Group frames by their source video for cross-validation.

    Args:
        frame_dir: Directory containing frame PNGs

    Returns:
        Dictionary mapping video IDs to list of frame filenames
    """
    video_groups: dict[str, list[str]] = {}

    for frame_path in sorted(frame_dir.glob("*.png")):
        # Expected format: video01_frame_000000.png or video01_00080_frame.png
        parts = frame_path.stem.split("_")
        if len(parts) >= 2:
            video_id = parts[0]  # e.g., "video01"
            video_groups.setdefault(video_id, []).append(frame_path.name)

    return video_groups


def train_fold(
    architecture: str,
    train_frames: list[str],
    val_frames: list[str],
    frame_dir: Path,
    mask_dir: Path,
    args: argparse.Namespace,
    fold_id: int,
    n_folds: int,
) -> dict[str, float]:
    """Train and evaluate a single fold for cross-validation.

    Args:
        architecture: Model architecture name
        train_frames: List of training frame filenames
        val_frames: List of validation frame filenames
        frame_dir: Directory containing frames
        mask_dir: Directory containing masks
        args: Command line arguments
        fold_id: Current fold number (1-indexed)
        n_folds: Total number of folds

    Returns:
        Dictionary with evaluation metrics for this fold
    """
    print(f"\n  Fold {fold_id}/{n_folds}: {len(train_frames)} train, {len(val_frames)} val")

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_dataset = SurgicalDataset(
        str(frame_dir),
        str(mask_dir),
        transform=train_transform,
        augment=True,
        file_list=train_frames,
    )
    val_dataset = SurgicalDataset(
        str(frame_dir), str(mask_dir), transform=val_transform, augment=False, file_list=val_frames
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # Initialize model
    model = get_model(architecture, NUM_CLASSES).to(args.device)

    class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=args.device)
    class_weights[1] = INSTRUMENT_CLASS_WEIGHT
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train for specified epochs
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, args.device, epoch, args.epochs)

    # Evaluate
    metrics = evaluate_model(model, val_loader, args.device, NUM_CLASSES)

    print(f"    IoU: {metrics['iou_instrument']:.4f}, Dice: {metrics['dice_instrument']:.4f}")

    return metrics


def run_cross_validation(
    architecture: str,
    frame_dir: Path,
    mask_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run k-fold cross-validation for a single architecture.

    Args:
        architecture: Model architecture name
        frame_dir: Directory containing frames
        mask_dir: Directory containing masks
        args: Command line arguments

    Returns:
        Dictionary with per-fold results and aggregate statistics
    """
    print(f"\n{'=' * 70}")
    print(f"CROSS-VALIDATION: {architecture.upper()} ({args.n_folds} folds)")
    print(f"{'=' * 70}")

    # Group frames by video
    video_groups = get_video_groups(frame_dir)
    video_ids = sorted(video_groups.keys())

    if len(video_ids) < args.n_folds:
        print(f"  Warning: Only {len(video_ids)} videos, reducing folds to {len(video_ids)}")
        n_folds = len(video_ids)
    else:
        n_folds = args.n_folds

    # Create folds (leave-one-video-out or group videos)
    fold_results: list[dict[str, Any]] = []
    fold_ious: list[float] = []
    fold_dices: list[float] = []

    # Distribute videos across folds
    videos_per_fold = len(video_ids) // n_folds
    remainder = len(video_ids) % n_folds

    fold_video_assignments: list[list[str]] = []
    idx = 0
    for fold in range(n_folds):
        n_videos = videos_per_fold + (1 if fold < remainder else 0)
        fold_video_assignments.append(video_ids[idx : idx + n_videos])
        idx += n_videos

    for fold_id in range(n_folds):
        seed_everything(args.seed + fold_id)  # Different seed per fold for reproducibility

        val_videos = fold_video_assignments[fold_id]
        train_videos = [v for v in video_ids if v not in val_videos]

        train_frames = [f for v in train_videos for f in video_groups[v]]
        val_frames = [f for v in val_videos for f in video_groups[v]]

        metrics = train_fold(
            architecture=architecture,
            train_frames=train_frames,
            val_frames=val_frames,
            frame_dir=frame_dir,
            mask_dir=mask_dir,
            args=args,
            fold_id=fold_id + 1,
            n_folds=n_folds,
        )

        fold_results.append(
            {
                "fold_id": fold_id + 1,
                "train_videos": train_videos,
                "val_videos": val_videos,
                "num_train_frames": len(train_frames),
                "num_val_frames": len(val_frames),
                "metrics": metrics,
            }
        )
        fold_ious.append(metrics["iou_instrument"])
        fold_dices.append(metrics["dice_instrument"])

    # Compute aggregate statistics
    mean_iou = float(np.mean(fold_ious))
    std_iou = float(np.std(fold_ious, ddof=1)) if len(fold_ious) > 1 else 0.0
    mean_dice = float(np.mean(fold_dices))
    std_dice = float(np.std(fold_dices, ddof=1)) if len(fold_dices) > 1 else 0.0

    print(
        f"\n  Summary: IoU = {mean_iou:.4f} ± {std_iou:.4f}, "
        f"Dice = {mean_dice:.4f} ± {std_dice:.4f}"
    )

    return {
        "architecture": architecture,
        "n_folds": n_folds,
        "fold_results": fold_results,
        "fold_ious": fold_ious,
        "fold_dices": fold_dices,
        "summary": {
            "mean_iou": mean_iou,
            "std_iou": std_iou,
            "mean_dice": mean_dice,
            "std_dice": std_dice,
        },
    }


def generate_statistical_comparison(
    cv_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Generate statistical comparison between architectures.

    Args:
        cv_results: Cross-validation results for each architecture
        output_dir: Directory to save comparison results

    Returns:
        Dictionary with statistical comparison results
    """
    architectures = list(cv_results.keys())

    if len(architectures) < 2:
        print("\n  Skipping statistical comparison (need at least 2 architectures)")
        return {}

    print(f"\n{'=' * 70}")
    print("STATISTICAL COMPARISON")
    print(f"{'=' * 70}")

    comparisons: dict[str, Any] = {"pairwise_comparisons": []}

    # Compare all pairs
    for i, arch_a in enumerate(architectures):
        for arch_b in architectures[i + 1 :]:
            scores_a_iou = cv_results[arch_a]["fold_ious"]
            scores_b_iou = cv_results[arch_b]["fold_ious"]
            scores_a_dice = cv_results[arch_a]["fold_dices"]
            scores_b_dice = cv_results[arch_b]["fold_dices"]

            # IoU comparison
            iou_result = paired_comparison(
                scores_a=scores_a_iou,
                scores_b=scores_b_iou,
                model_a_name=arch_a.upper(),
                model_b_name=arch_b.upper(),
                metric_name="IoU",
            )

            # Dice comparison
            dice_result = paired_comparison(
                scores_a=scores_a_dice,
                scores_b=scores_b_dice,
                model_a_name=arch_a.upper(),
                model_b_name=arch_b.upper(),
                metric_name="Dice",
            )

            # Print results
            print(f"\n{arch_a.upper()} vs {arch_b.upper()}:")

            # IoU results
            print("  IoU:")
            iou_ci_a = f"[{iou_result.ci_a[0]:.4f}, {iou_result.ci_a[1]:.4f}]"
            iou_ci_b = f"[{iou_result.ci_b[0]:.4f}, {iou_result.ci_b[1]:.4f}]"
            iou_effect = interpret_effect_size(iou_result.effect_size)
            print(
                f"    {arch_a.upper()}: {iou_result.model_a_mean:.4f} "
                f"± {iou_result.model_a_std:.4f} (95% CI: {iou_ci_a})"
            )
            print(
                f"    {arch_b.upper()}: {iou_result.model_b_mean:.4f} "
                f"± {iou_result.model_b_std:.4f} (95% CI: {iou_ci_b})"
            )
            print(
                f"    t-statistic: {iou_result.t_statistic:.4f}, "
                f"p-value: {iou_result.p_value:.6f}"
            )
            print(f"    Effect size (Cohen's d): {iou_result.effect_size:.4f} " f"({iou_effect})")
            iou_sig = "YES" if iou_result.is_significant else "NO"
            print(f"    Significant: {iou_sig} (α=0.05)")

            # Dice results
            print("  Dice:")
            dice_ci_a = f"[{dice_result.ci_a[0]:.4f}, {dice_result.ci_a[1]:.4f}]"
            dice_ci_b = f"[{dice_result.ci_b[0]:.4f}, {dice_result.ci_b[1]:.4f}]"
            dice_effect = interpret_effect_size(dice_result.effect_size)
            print(
                f"    {arch_a.upper()}: {dice_result.model_a_mean:.4f} "
                f"± {dice_result.model_a_std:.4f} (95% CI: {dice_ci_a})"
            )
            print(
                f"    {arch_b.upper()}: {dice_result.model_b_mean:.4f} "
                f"± {dice_result.model_b_std:.4f} (95% CI: {dice_ci_b})"
            )
            print(
                f"    t-statistic: {dice_result.t_statistic:.4f}, "
                f"p-value: {dice_result.p_value:.6f}"
            )
            print(f"    Effect size (Cohen's d): {dice_result.effect_size:.4f} " f"({dice_effect})")
            dice_sig = "YES" if dice_result.is_significant else "NO"
            print(f"    Significant: {dice_sig} (α=0.05)")

            # Determine winner
            if iou_result.is_significant:
                winner = arch_a if iou_result.model_a_mean > iou_result.model_b_mean else arch_b
                print(f"\n  → {winner.upper()} is significantly better (p < 0.05)")
            else:
                print("\n  → No significant difference between models")

            comparisons["pairwise_comparisons"].append(
                {
                    "model_a": arch_a,
                    "model_b": arch_b,
                    "iou": iou_result.to_dict(),
                    "dice": dice_result.to_dict(),
                }
            )

    return comparisons


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPARATIVE MODEL TRAINING WITH STATISTICAL VALIDATION")
    print("=" * 70)
    print(f"Models: {', '.join(args.models)}")
    print(f"Epochs: {args.epochs}")
    print(f"Cross-validation folds: {args.n_folds if not args.skip_cv else 'Skipped'}")
    print(f"Device: {args.device}")

    # Check if we should run cross-validation
    if not args.skip_cv:
        # Run cross-validation for each architecture
        cv_results: dict[str, dict[str, Any]] = {}

        for architecture in args.models:
            cv_results[architecture] = run_cross_validation(
                architecture=architecture,
                frame_dir=args.frame_dir,
                mask_dir=args.mask_dir,
                args=args,
            )

        # Save CV results
        cv_results_path = args.output_dir / "cross_validation_results.json"
        with open(cv_results_path, "w", encoding="utf-8") as f:
            json.dump(cv_results, f, indent=2)
        print(f"\n✓ Saved cross-validation results: {cv_results_path}")

        # Generate statistical comparison
        stat_comparison = generate_statistical_comparison(cv_results, args.output_dir)

        # Build comprehensive comparison output
        comparison: dict[str, Any] = {
            "metadata": {
                "n_folds": args.n_folds,
                "epochs_per_fold": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
            "models": {},
            "statistical_comparison": stat_comparison,
        }

        for arch in args.models:
            cv = cv_results[arch]
            n_folds_sqrt = np.sqrt(cv["n_folds"])
            mean_iou = cv["summary"]["mean_iou"]
            std_iou = cv["summary"]["std_iou"]
            mean_dice = cv["summary"]["mean_dice"]
            std_dice = cv["summary"]["std_dice"]

            comparison["models"][arch] = {
                "mean_iou": mean_iou,
                "std_iou": std_iou,
                "ci_95_iou": [
                    round(mean_iou - 1.96 * std_iou / n_folds_sqrt, 4),
                    round(mean_iou + 1.96 * std_iou / n_folds_sqrt, 4),
                ],
                "mean_dice": mean_dice,
                "std_dice": std_dice,
                "ci_95_dice": [
                    round(mean_dice - 1.96 * std_dice / n_folds_sqrt, 4),
                    round(mean_dice + 1.96 * std_dice / n_folds_sqrt, 4),
                ],
                "fold_ious": [round(x, 4) for x in cv["fold_ious"]],
                "fold_dices": [round(x, 4) for x in cv["fold_dices"]],
            }

        comparison_path = args.output_dir / "comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"✓ Saved comprehensive comparison: {comparison_path}")

        # Print summary table
        print(f"\n{'=' * 70}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"\n{'Architecture':<15} {'IoU (mean±std)':<20} {'Dice (mean±std)':<20}")
        print("-" * 55)
        for arch in args.models:
            cv = cv_results[arch]
            iou_str = f"{cv['summary']['mean_iou']:.4f} ± {cv['summary']['std_iou']:.4f}"
            dice_str = f"{cv['summary']['mean_dice']:.4f} ± {cv['summary']['std_dice']:.4f}"
            print(f"{arch:<15} {iou_str:<20} {dice_str:<20}")

    else:
        # Original single-split training
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        all_frames = sorted([f.name for f in args.frame_dir.glob("*.png")])
        rng = np.random.default_rng(args.seed)
        rng.shuffle(all_frames)

        val_size = max(1, int(len(all_frames) * 0.2))
        train_frames = all_frames[:-val_size]
        val_frames = all_frames[-val_size:]

        print(f"\nDataset: {len(train_frames)} train, {len(val_frames)} val")

        train_dataset = SurgicalDataset(
            str(args.frame_dir),
            str(args.mask_dir),
            transform=train_transform,
            augment=True,
            file_list=train_frames,
        )
        val_dataset = SurgicalDataset(
            str(args.frame_dir),
            str(args.mask_dir),
            transform=val_transform,
            augment=False,
            file_list=val_frames,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        results: dict[str, dict] = {}
        for architecture in args.models:
            model, history = train_model_complete(architecture, train_loader, val_loader, args)

            model_path = args.output_dir / f"{architecture}_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"✓ Saved: {model_path}")

            history_path = args.output_dir / f"{architecture}_history.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            results[architecture] = history

        print(f"\n{'=' * 70}")
        print("COMPARATIVE ANALYSIS")
        print(f"{'=' * 70}")

        single_split_comparison: dict[str, Any] = {}
        for arch in args.models:
            metrics = results[arch]["final_metrics"]
            single_split_comparison[arch] = {
                "iou_instrument": metrics["iou_instrument"],
                "dice_instrument": metrics["dice_instrument"],
                "accuracy": metrics["accuracy"],
                "parameters": results[arch]["parameters"],
                "training_time": results[arch]["training_time_seconds"],
            }

        print(
            f"\n{'Architecture':<15} {'IoU':<8} {'Dice':<8} {'Accuracy':<10} "
            f"{'Params':<12} {'Time (s)':<10}"
        )
        print("-" * 70)
        for arch in args.models:
            c = single_split_comparison[arch]
            print(
                f"{arch:<15} {c['iou_instrument']:<8.4f} {c['dice_instrument']:<8.4f} "
                f"{c['accuracy']:<10.4f} {c['parameters']:<12,} {c['training_time']:<10.1f}"
            )

        comparison_path = args.output_dir / "comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(single_split_comparison, f, indent=2)
        print(f"\n✓ Saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
