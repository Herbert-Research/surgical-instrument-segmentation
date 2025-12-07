"""
Comparative Training Script for Baseline Analysis
Trains both U-Net and DeepLabV3-ResNet50 on identical data splits
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPARATIVE MODEL TRAINING")
    print("=" * 70)
    print(f"Models: {', '.join(args.models)}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.02,
            ),
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

    comparison = {}
    for arch in args.models:
        metrics = results[arch]["final_metrics"]
        comparison[arch] = {
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
        c = comparison[arch]
        print(
            f"{arch:<15} {c['iou_instrument']:<8.4f} {c['dice_instrument']:<8.4f} "
            f"{c['accuracy']:<10.4f} {c['parameters']:<12,} {c['training_time']:<10.1f}"
        )

    comparison_path = args.output_dir / "comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✓ Saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
