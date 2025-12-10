"""K-Fold Cross-Validation for Surgical Instrument Segmentation.

Implements leave-one-video-out cross-validation for robust evaluation
across different surgical procedures.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from surgical_segmentation.datasets import SurgicalDataset
from surgical_segmentation.models import InstrumentSegmentationModel, UNet

NUM_CLASSES = 2  # Background + instrument
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INSTRUMENT_CLASS_WEIGHT = 3.0
DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Make the run deterministic across Python, NumPy, and PyTorch."""

    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class FoldResult:
    """Results from a single cross-validation fold."""

    fold_id: int
    val_video: str
    train_videos: list[str]
    iou_instrument: float
    dice_instrument: float
    accuracy: float
    num_train_frames: int
    num_val_frames: int


@dataclass
class CrossValidationResult:
    """Aggregated cross-validation results."""

    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_iou(self) -> float:
        return float(np.mean([f.iou_instrument for f in self.folds])) if self.folds else 0.0

    @property
    def std_iou(self) -> float:
        return float(np.std([f.iou_instrument for f in self.folds])) if self.folds else 0.0

    @property
    def mean_dice(self) -> float:
        return float(np.mean([f.dice_instrument for f in self.folds])) if self.folds else 0.0

    @property
    def std_dice(self) -> float:
        return float(np.std([f.dice_instrument for f in self.folds])) if self.folds else 0.0

    def to_dict(self) -> dict:
        return {
            "summary": {
                "mean_iou": self.mean_iou,
                "std_iou": self.std_iou,
                "mean_dice": self.mean_dice,
                "std_dice": self.std_dice,
                "num_folds": len(self.folds),
            },
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "val_video": f.val_video,
                    "train_videos": f.train_videos,
                    "iou_instrument": f.iou_instrument,
                    "dice_instrument": f.dice_instrument,
                    "accuracy": f.accuracy,
                    "num_train_frames": f.num_train_frames,
                    "num_val_frames": f.num_val_frames,
                }
                for f in self.folds
            ],
        }

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def get_video_groups(frame_dir: Path) -> dict[str, list[str]]:
    """Group frames by their source video.

    Args:
        frame_dir: Directory containing frame PNGs

    Returns:
        Dictionary mapping video IDs to list of frame filenames
    """
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    video_groups: dict[str, list[str]] = {}

    for frame_path in sorted(frame_dir.glob("*.png")):
        # Expected format: video01_frame_000000.png
        parts = frame_path.stem.split("_")
        if len(parts) >= 2:
            video_id = parts[0]  # e.g., "video01"
            video_groups.setdefault(video_id, []).append(frame_path.name)

    if not video_groups:
        raise FileNotFoundError(f"No PNG frames found in {frame_dir}")

    return video_groups


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return train/validation transforms."""

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

    return train_transform, val_transform


def build_dataloaders(
    frame_dir: Path,
    mask_dir: Path,
    train_frames: Sequence[str],
    val_frames: Sequence[str],
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation dataloaders."""

    train_transform, val_transform = build_transforms()

    train_dataset = SurgicalDataset(
        frame_dir=str(frame_dir),
        mask_dir=str(mask_dir),
        transform=train_transform,
        augment=True,
        file_list=train_frames,
    )
    val_dataset = SurgicalDataset(
        frame_dir=str(frame_dir),
        mask_dir=str(mask_dir),
        transform=val_transform,
        augment=False,
        file_list=val_frames,
    )

    pin_memory = device.startswith("cuda") and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def build_model(model_class: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Instantiate model by name."""

    if model_class == "deeplabv3":
        model = InstrumentSegmentationModel(num_classes=num_classes)
    elif model_class == "unet":
        model = UNet(n_channels=3, n_classes=num_classes, bilinear=True)
    else:
        raise ValueError(f"Unsupported model_class: {model_class}")
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    desc: str,
) -> float:
    """Train for a single epoch."""

    model.train()
    epoch_loss = 0.0

    for frames, masks in tqdm(dataloader, desc=desc, leave=False):
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

    return epoch_loss / max(1, len(dataloader))


def confusion_matrix_multiclass(true_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (true_mask >= 0) & (true_mask < num_classes)
    true = true_mask[valid].ravel()
    pred = pred_mask[valid].ravel()
    flat_index = true * num_classes + pred
    counts = np.bincount(flat_index, minlength=num_classes**2)
    cm += counts.reshape(num_classes, num_classes)
    return cm


def compute_metrics_from_cm(cm: np.ndarray) -> dict[str, Any]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

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
        "accuracy": float(accuracy),
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> dict[str, Any]:
    """Evaluate model and compute segmentation metrics."""

    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for frames, masks in tqdm(dataloader, desc="Evaluating fold", leave=False):
            frames = frames.to(device, non_blocking=True)
            masks_np = masks.cpu().numpy()

            outputs = model(frames)
            if isinstance(outputs, dict):
                outputs = outputs["out"]

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for pred, true in zip(preds, masks_np):
                cm += confusion_matrix_multiclass(true, pred, num_classes)

    metrics = compute_metrics_from_cm(cm)
    metrics["iou_instrument"] = (
        float(metrics["iou"][1]) if num_classes > 1 else float(metrics["iou"][0])
    )
    metrics["dice_instrument"] = (
        float(metrics["dice"][1]) if num_classes > 1 else float(metrics["dice"][0])
    )

    return metrics


def leave_one_video_out_cv(
    frame_dir: Path,
    mask_dir: Path,
    model_class: str = "deeplabv3",
    epochs: int = 10,
    batch_size: int = 4,
    device: str = "cuda",
    output_dir: Path = Path("outputs/cross_validation"),
    num_workers: int = 4,
    learning_rate: float = 1e-3,
    seed: int = DEFAULT_SEED,
) -> CrossValidationResult:
    """Run leave-one-video-out cross-validation.

    For each video in the dataset:
    1. Hold out that video for validation
    2. Train on all other videos
    3. Evaluate on held-out video
    4. Record metrics
    """

    frame_dir = frame_dir.resolve()
    mask_dir = mask_dir.resolve()
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_groups = get_video_groups(frame_dir)
    videos = sorted(video_groups.keys())

    print(f"Found {len(videos)} videos: {videos}")
    print(f"Total frames: {sum(len(v) for v in video_groups.values())}")

    result = CrossValidationResult()

    for fold_idx, val_video in enumerate(videos):
        seed_everything(seed + fold_idx)

        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{len(videos)}: Validating on {val_video}")
        print(f"{'='*70}")

        train_frames: list[str] = []
        val_frames = video_groups[val_video]
        train_videos: list[str] = []

        for video_id, frames in video_groups.items():
            if video_id != val_video:
                train_frames.extend(frames)
                train_videos.append(video_id)

        print(f"Train: {len(train_frames)} frames from {len(train_videos)} videos")
        print(f"Val: {len(val_frames)} frames from {val_video}")

        train_loader, val_loader = build_dataloaders(
            frame_dir,
            mask_dir,
            train_frames,
            val_frames,
            batch_size=batch_size,
            num_workers=num_workers,
            device=str(device_obj),
        )

        model = build_model(model_class, num_classes=NUM_CLASSES).to(device_obj)

        class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=device_obj)
        if NUM_CLASSES > 1:
            class_weights[1] = INSTRUMENT_CLASS_WEIGHT
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):
            desc = f"Fold {fold_idx + 1} Epoch {epoch}/{epochs}"
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device_obj, desc)
            print(f"  Epoch {epoch}/{epochs} - loss: {loss:.4f}")

        metrics = evaluate_model(model, val_loader, device_obj, num_classes=NUM_CLASSES)

        fold_result = FoldResult(
            fold_id=fold_idx,
            val_video=val_video,
            train_videos=train_videos,
            iou_instrument=float(metrics["iou_instrument"]),
            dice_instrument=float(metrics["dice_instrument"]),
            accuracy=float(metrics["accuracy"]),
            num_train_frames=len(train_frames),
            num_val_frames=len(val_frames),
        )
        result.folds.append(fold_result)

        fold_metrics_path = output_dir / f"fold_{fold_idx + 1:02d}_{val_video}.json"
        with open(fold_metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "val_video": val_video,
                    "train_videos": train_videos,
                    "metrics": metrics,
                    "num_train_frames": len(train_frames),
                    "num_val_frames": len(val_frames),
                },
                f,
                indent=2,
            )
        print(f"  Saved fold metrics to {fold_metrics_path}")

        # Free GPU memory between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result.save(output_dir / "cross_validation_results.json")

    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Mean IoU: {result.mean_iou:.4f} ± {result.std_iou:.4f}")
    print(f"Mean Dice: {result.mean_dice:.4f} ± {result.std_dice:.4f}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-video-out cross-validation.")
    parser.add_argument(
        "--frame-dir",
        type=Path,
        required=True,
        help="Directory containing frame PNGs (e.g., data/full_dataset/frames)",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Directory containing mask PNGs (e.g., data/full_dataset/masks)",
    )
    parser.add_argument(
        "--model-class",
        choices=["deeplabv3", "unet"],
        default="deeplabv3",
        help="Model architecture to evaluate",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training/eval")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_validation"))
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leave_one_video_out_cv(
        frame_dir=args.frame_dir,
        mask_dir=args.mask_dir,
        model_class=args.model_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
