"""
Laparoscopic Instrument Segmentation using Deep Learning
Author: Maximilian Dressler
Purpose: Automated surgical instrument segmentation for laparoscopic surgery
         quality assessment. Technical foundation for procedure-specific
         applications including gastrectomy, cholecystectomy, and other
         minimally invasive procedures.
"""

import argparse
import importlib
import random
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _load_dependency(module_name: str) -> Any:
    """
    Dynamically import a dependency so Pylance doesn't flag missing modules.
    Raises a helpful error if the package is unavailable at runtime.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Missing required dependency '{module_name}'. "
            "Install project requirements via `pip install -r requirements.txt`."
        ) from exc


torch: Any = _load_dependency("torch")
nn: Any = torch.nn
torchvision: Any = _load_dependency("torchvision")
transforms = torchvision.transforms
cv2 = _load_dependency("cv2")
tqdm = _load_dependency("tqdm").tqdm


NUM_CLASSES = 2  # Background + all instruments (binary segmentation)
CLASS_NAMES = ["background", "instrument"]
DEFAULT_DATA_SEED = 42
TOTAL_SYNTHETIC_FRAMES = 20
INSTRUMENT_CLASS_WEIGHT = 3.0
DEFAULT_FRAME_DIR = Path("data/sample_frames")
DEFAULT_MASK_DIR = Path("data/masks")
DEFAULT_PRED_DIR = Path("data/preds")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FIGURES_DIR = Path("outputs/figures")
MODELS_DIR = Path("outputs/models")
TRAINING_LOSS_PATH = FIGURES_DIR / "training_loss.png"
SEGMENTATION_FIG_PATH = FIGURES_DIR / "segmentation_results.png"
DEFAULT_MODEL_PATH = MODELS_DIR / "instrument_segmentation_model.pth"


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Surgical instrument segmentation demo")
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=DEFAULT_FRAME_DIR,
        help="Directory containing RGB frames (default: data/sample_frames)",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=DEFAULT_MASK_DIR,
        help="Directory containing mask PNGs (default: data/masks)",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help="Directory where prediction PNGs will be written",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Do not create synthetic data even if frame/mask dirs are empty",
    )
    return parser.parse_args()


def seed_everything(seed: int = DEFAULT_DATA_SEED) -> None:
    """Make the demo deterministic across Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AdditiveGaussianNoise:
    """Inject low-amplitude Gaussian noise after tensor conversion."""

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, tensor: Any) -> Any:
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


print("Surgical Instrument Segmentation Pipeline")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("=" * 70)

# %% ============================================
# PART 1: DATA GENERATION (For Demo)
# ============================================


def create_synthetic_surgical_frames(frame_dir: Path, mask_dir: Path, force: bool = False):
    """
    Create synthetic surgical frames for demonstration
    In real application, these would be actual laparoscopic video frames
    """
    frame_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = any(frame_dir.glob("*.png"))
    existing_masks = any(mask_dir.glob("*.png"))
    if existing_frames and existing_masks and not force:
        print(
            "\nDetected existing frame/mask assets – skipping synthetic generation to"
            f" avoid overwriting approved data (delete {frame_dir} to regenerate)."
        )
        return False

    print("\nGenerating synthetic surgical frames...")

    rng = np.random.default_rng(DEFAULT_DATA_SEED)

    for i in range(TOTAL_SYNTHETIC_FRAMES):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = rng.integers(80, 120, (480, 640))
        frame[:, :, 1] = rng.integers(60, 100, (480, 640))
        frame[:, :, 2] = rng.integers(120, 180, (480, 640))

        noise = rng.normal(0, 15, (480, 640, 3))
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

        if rng.random() < 0.35:
            occ_w = int(rng.integers(60, 160))
            occ_h = int(rng.integers(40, 120))
            x = int(rng.integers(0, 640 - occ_w))
            y = int(rng.integers(0, 480 - occ_h))
            smoke_intensity = int(rng.integers(180, 230))
            alpha = rng.random() * 0.4 + 0.2
            overlay = np.full((occ_h, occ_w, 3), smoke_intensity, dtype=np.uint8)
            frame[y : y + occ_h, x : x + occ_w] = np.clip(
                alpha * overlay + (1 - alpha) * frame[y : y + occ_h, x : x + occ_w],
                0,
                255,
            ).astype(np.uint8)

        mask = np.zeros((480, 640), dtype=np.uint8)
        if i < 15:
            cv2.ellipse(frame, (200, 240), (80, 15), 45, 0, 360, (200, 200, 210), -1)
            cv2.ellipse(frame, (200, 240), (80, 15), 45, 0, 360, (180, 180, 190), 3)
            cv2.ellipse(
                mask,
                (200, 240),
                (80, 15),
                45,
                0,
                360,
                1,
                -1,
            )

        if i < 12:
            cv2.ellipse(frame, (440, 300), (60, 12), -30, 0, 360, (190, 195, 205), -1)
            cv2.ellipse(frame, (440, 300), (60, 12), -30, 0, 360, (170, 175, 185), 3)
            cv2.ellipse(
                mask,
                (440, 300),
                (60, 12),
                -30,
                0,
                360,
                1,
                -1,
            )

        if rng.random() < 0.25:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            frame[:, :, 1] = np.clip(frame[:, :, 1] * rng.uniform(0.6, 0.9), 0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)

        cv2.imwrite(str(frame_dir / f"frame_{i:03d}.png"), frame)
        cv2.imwrite(str(mask_dir / f"mask_{i:03d}.png"), mask)

    print(f"✓ Created 20 synthetic surgical frames")
    return True


# %% ============================================
# PART 2: MODEL DEFINITION
# ============================================

from surgical_segmentation.datasets import SurgicalDataset
from surgical_segmentation.models.deeplabv3 import InstrumentSegmentationModel

# %% ============================================
# PART 3: DATA LOADING
# ============================================


# %% ============================================
# PART 4: TRAINING
# ============================================


def train_model(
    model,
    train_loader,
    num_epochs=15,
    learning_rate=0.001,
    num_classes: int = NUM_CLASSES,
):
    """
    Train the segmentation model
    Note: This is a simplified training loop for demonstration
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    class_weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    class_weights[1:] = INSTRUMENT_CLASS_WEIGHT
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n{'='*70}")
    print(f"Training on device: {device}")
    print(f"{'='*70}\n")

    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for frames, masks in pbar:
                frames = frames.to(device)
                masks = masks.to(device)

                outputs = model(frames)
                loss = criterion(outputs, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

    return model, losses


# %% ============================================
# PART 5: EVALUATION AND VISUALIZATION
# ============================================


def confusion_matrix_multiclass(true_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (true_mask >= 0) & (true_mask < num_classes)
    true = true_mask[valid].ravel()
    pred = pred_mask[valid].ravel()
    indices = true * num_classes + pred
    counts = np.bincount(indices, minlength=num_classes**2)
    cm += counts.reshape(num_classes, num_classes)
    return cm


def compute_metrics_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    accuracy = tp.sum() / cm.sum() if cm.sum() else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "support": support,
        "accuracy": accuracy,
    }


def evaluate_model(
    model,
    dataset,
    num_visual_samples: int = 4,
    prediction_dir: Optional[Path] = None,
):
    """Evaluate the model, visualize samples, and optionally export masks."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    aggregate_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    num_visuals = min(num_visual_samples, len(dataset))
    fig = axes = None
    cmap = plt.get_cmap("viridis", NUM_CLASSES)

    if num_visuals > 0:
        fig, axes = plt.subplots(num_visuals, 3, figsize=(15, 4 * num_visuals))
        if num_visuals == 1:
            axes = np.expand_dims(axes, axis=0)

    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx in range(len(dataset)):
            frame, true_mask = dataset[idx]
            frame_batch = frame.unsqueeze(0).to(device)
            output = model(frame_batch)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            true_np = true_mask.numpy().astype(np.uint8)

            aggregate_cm += confusion_matrix_multiclass(true_np, pred_mask, NUM_CLASSES)

            if prediction_dir is not None:
                frame_name = dataset.frames[idx]
                pred_name = frame_name.replace("frame", "mask")
                Image.fromarray(pred_mask).save(prediction_dir / pred_name)

            if idx < num_visuals and axes is not None:
                metrics = compute_metrics_from_cm(
                    confusion_matrix_multiclass(true_np, pred_mask, NUM_CLASSES)
                )
                instrument_iou = (
                    metrics["iou"][1:].mean() if NUM_CLASSES > 1 else float(metrics["iou"][0])
                )
                instrument_dice = (
                    metrics["dice"][1:].mean() if NUM_CLASSES > 1 else float(metrics["dice"][0])
                )

                frame_np = frame.permute(1, 2, 0).cpu().numpy()
                frame_np = (frame_np * IMAGENET_STD) + IMAGENET_MEAN
                frame_np = np.clip(frame_np, 0, 1)

                axes[idx, 0].imshow(frame_np)
                axes[idx, 0].set_title(f"Frame {idx+1}", fontweight="bold")
                axes[idx, 0].axis("off")

                axes[idx, 1].imshow(true_np, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
                axes[idx, 1].set_title("Ground Truth Mask", fontweight="bold")
                axes[idx, 1].axis("off")

                axes[idx, 2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
                axes[idx, 2].set_title(
                    f"Prediction (IoU {instrument_iou:.3f}, Dice {instrument_dice:.3f})",
                    fontweight="bold",
                )
                axes[idx, 2].axis("off")

    metrics = compute_metrics_from_cm(aggregate_cm)
    mean_iou = metrics["iou"][1:].mean() if NUM_CLASSES > 1 else float(metrics["iou"][0])
    mean_dice = metrics["dice"][1:].mean() if NUM_CLASSES > 1 else float(metrics["dice"][0])

    if fig is not None:
        plt.tight_layout()
        plt.savefig(SEGMENTATION_FIG_PATH, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"\n{'='*70}")
    print("EVALUATION METRICS")
    print(f"{'='*70}")
    print(f"Overall accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean IoU (instrument classes): {mean_iou:.4f}")
    print(f"Mean Dice (instrument classes): {mean_dice:.4f}")
    for idx, label in enumerate(CLASS_NAMES):
        print(
            f"  {label:>12} → IoU {metrics['iou'][idx]:.3f} | Dice {metrics['dice'][idx]:.3f} "
            f"| Precision {metrics['precision'][idx]:.3f} | Recall {metrics['recall'][idx]:.3f} "
            f"| n={int(metrics['support'][idx])}"
        )
    print(f"{'='*70}\n")

    if prediction_dir is not None:
        print(f"Saved per-frame predictions to: {prediction_dir}")

    return metrics


# %% ============================================
# PART 6: MAIN EXECUTION
# ============================================


def main():
    """Main execution pipeline"""

    args = parse_cli_args()
    frame_dir = args.frame_dir.resolve()
    mask_dir = args.mask_dir.resolve()
    prediction_dir = args.prediction_dir.resolve()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything()
    data_created = False
    if not args.skip_synthetic:
        data_created = create_synthetic_surgical_frames(frame_dir, mask_dir)

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            AdditiveGaussianNoise(std=0.02),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    all_frames = sorted(p.name for p in frame_dir.glob("*.png"))
    if not all_frames:
        raise FileNotFoundError(
            f"No frames found in {frame_dir}. Provide --frame-dir pointing to your"
            " dataset (e.g., datasets/Cholec80/sample_frames)."
        )

    total_frames = len(all_frames)
    if total_frames < 2:
        raise RuntimeError(
            "Need at least two frames to create a train/validation split. Add more"
            " data or keep the synthetic set intact."
        )

    rng = np.random.default_rng(DEFAULT_DATA_SEED)
    rng.shuffle(all_frames)

    val_size = max(1, int(round(total_frames * 0.2)))
    train_size = total_frames - val_size
    if train_size == 0:
        train_size = total_frames - 1
        val_size = 1

    train_frames = all_frames[:train_size]
    val_frames = all_frames[train_size:]

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
        transform=eval_transform,
        augment=False,
        file_list=val_frames,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    print(f"\nDataset Summary:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Training frames: {len(train_dataset)} (augmentations enabled)")
    print(f"  - Validation frames: {len(val_dataset)}")
    if not data_created:
        print("  - Note: Existing frames detected; synthetic generation skipped")

    model = InstrumentSegmentationModel(num_classes=NUM_CLASSES)

    print("\nStarting training...")
    model, losses = train_model(
        model,
        train_loader,
        num_epochs=15,
        learning_rate=0.001,
        num_classes=NUM_CLASSES,
    )

    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, "o-", linewidth=2, markersize=8)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Over Time", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(TRAINING_LOSS_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nEvaluating model on validation set...")
    evaluate_model(
        model,
        val_dataset,
        num_visual_samples=4,
        prediction_dir=prediction_dir,
    )

    torch.save(model.state_dict(), DEFAULT_MODEL_PATH)
    print(f"✓ Model saved: {DEFAULT_MODEL_PATH}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - {SEGMENTATION_FIG_PATH} (visual comparison)")
    print(f"  - {TRAINING_LOSS_PATH} (learning curve)")
    print(f"  - {DEFAULT_MODEL_PATH} (trained weights)")
    print("\nClinical Applications:")
    print("  → Automated instrument tracking for objective skill assessment")
    print("  → Frame-by-frame segmentation enables surgical phase recognition")
    print("  → Foundation for real-time guidance systems across procedures")
    print("  → Extensible to gastrectomy, colorectal, and other laparoscopic surgeries")


if __name__ == "__main__":
    main()
