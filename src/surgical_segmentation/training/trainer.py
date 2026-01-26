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
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=None,
        help="Directory containing RGB frames (default: data/sample_frames)",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Directory containing mask PNGs (default: data/masks)",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="Directory where prediction PNGs will be written",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Do not create synthetic data even if frame/mask dirs are empty",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config file)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (overrides config file)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config file)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay for optimizer (overrides config file)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader worker processes (overrides config file)",
    )
    pin_memory_group = parser.add_mutually_exclusive_group()
    pin_memory_group.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Pin memory for DataLoader (overrides config file)",
    )
    pin_memory_group.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned memory in DataLoader (overrides config file)",
    )
    parser.set_defaults(pin_memory=None)
    parser.add_argument(
        "--train-split",
        type=float,
        default=None,
        help="Fraction of data used for training (overrides config file)",
    )
    augment_group = parser.add_mutually_exclusive_group()
    augment_group.add_argument(
        "--augment",
        dest="augment",
        action="store_true",
        help="Enable training augmentations (overrides config file)",
    )
    augment_group.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable training augmentations (overrides config file)",
    )
    parser.set_defaults(augment=None)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Square image size for resizing (overrides config file)",
    )
    return parser.parse_args()


def seed_everything(seed: int = DEFAULT_DATA_SEED) -> None:
    """
    Set random seeds for reproducibility across all random number generators.

    Ensures deterministic behavior across Python's random module, NumPy,
    and PyTorch (CPU and CUDA). Critical for scientific reproducibility
    in medical AI research where results must be verifiable.

    Args:
        seed: Integer seed value for all random number generators.
              Default is 42 (DEFAULT_DATA_SEED).

    Note:
        Sets torch.backends.cudnn.deterministic = True which may impact
        performance but ensures reproducible results across runs.

    Example:
        >>> seed_everything(42)
        >>> # All subsequent random operations will be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AdditiveGaussianNoise:
    """
    Data augmentation transform that adds Gaussian noise to image tensors.

    Simulates sensor noise and imaging artifacts common in laparoscopic
    video capture. Applied after tensor conversion but before normalization
    to maintain realistic noise characteristics.

    Args:
        std: Standard deviation of Gaussian noise. Default 0.02 provides
             subtle noise without degrading image quality significantly.
             Typical range: [0.01, 0.05].

    Attributes:
        std (float): Noise standard deviation.

    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.ToTensor(),
        ...     AdditiveGaussianNoise(std=0.02),
        ...     transforms.Normalize(mean, std)
        ... ])

    Note:
        Output is clamped to [0, 1] to maintain valid tensor value range.
        If std <= 0, the transform returns the input unchanged.
    """

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


def create_synthetic_surgical_frames(
    frame_dir: Path,
    mask_dir: Path,
    force: bool = False,
    seed: int = DEFAULT_DATA_SEED,
):
    """
    Create synthetic surgical frames for demonstration and testing.

    Generates artificial laparoscopic-like images with simulated surgical
    instruments for pipeline testing when real data is unavailable. The
    synthetic frames mimic key visual characteristics of laparoscopic surgery:
    - Pink/red tissue-like background colors
    - Metallic instrument-like elliptical shapes
    - Optional smoke/vapor occlusion artifacts
    - Realistic noise patterns

    Args:
        frame_dir: Directory path to save generated RGB frame images.
                   Created if it doesn't exist.
        mask_dir: Directory path to save corresponding binary masks.
                  Created if it doesn't exist.
        force: If True, regenerate data even if directories contain existing
               files. Default False preserves existing data.

    Returns:
        bool: True if synthetic data was generated, False if skipped
              (existing data found and force=False).

    Generated Data:
        - 20 frames (480x640 RGB PNG images)
        - 20 corresponding masks (480x640 binary PNG images)
        - Filenames: frame_XXX.png and mask_XXX.png (zero-padded index)

    Instrument Simulation:
        - Two elliptical instrument shapes per frame (positions vary)
        - Instruments rendered with metallic gray colors
        - Masks encode instrument pixels as class 1, background as class 0

    Example:
        >>> frame_dir = Path('data/sample_frames')
        >>> mask_dir = Path('data/masks')
        >>> created = create_synthetic_surgical_frames(frame_dir, mask_dir)
        >>> if created:
        ...     print('Generated new synthetic dataset')

    Note:
        Synthetic data is for development/testing only. Real clinical
        applications require actual annotated surgical video data.
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

    rng = np.random.default_rng(seed)

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

    print("✓ Created 20 synthetic surgical frames")
    return True


# %% ============================================
# PART 2: MODEL DEFINITION
# ============================================

# Deferred imports to avoid circular dependencies (these modules import from trainer)
from surgical_segmentation.datasets import SurgicalDataset  # noqa: E402, F811
from surgical_segmentation.evaluation.metrics import (  # noqa: E402, F811
    compute_metrics_from_cm,
    confusion_matrix_multiclass,
)
from surgical_segmentation.models.deeplabv3 import InstrumentSegmentationModel  # noqa: E402, F811
from surgical_segmentation.utils.config import Config, load_config  # noqa: E402, F811


def load_training_config(
    config_path: Optional[Path] = None, cli_overrides: Optional[dict] = None
) -> Config:
    """
    Load training configuration from YAML file with CLI overrides.

    Centralizes configuration management by loading from a YAML file and
    applying any command-line overrides. This enables reproducible experiments
    while still allowing quick parameter adjustments via CLI.

    Args:
        config_path: Path to YAML configuration file. If None, uses the
                     default configuration from config/default.yaml.
        cli_overrides: Dictionary of CLI arguments to override config values.
                       Keys should match config attributes (e.g., "epochs").
                       None values are ignored.

    Returns:
        Config: Validated configuration object with all parameters.

    Example:
        >>> config = load_training_config(
        ...     config_path=Path("config/experiment_01.yaml"),
        ...     cli_overrides={"epochs": 30, "batch_size": 8}
        ... )
        >>> print(config.training.epochs)  # 30 (from CLI override)
    """
    # Build override dict for config loader
    overrides = {}
    if cli_overrides:
        if cli_overrides.get("epochs") is not None:
            overrides["training.epochs"] = cli_overrides["epochs"]
        if cli_overrides.get("batch_size") is not None:
            overrides["training.batch_size"] = cli_overrides["batch_size"]
        if cli_overrides.get("learning_rate") is not None:
            overrides["training.learning_rate"] = cli_overrides["learning_rate"]
        if cli_overrides.get("weight_decay") is not None:
            overrides["training.weight_decay"] = cli_overrides["weight_decay"]
        if cli_overrides.get("num_workers") is not None:
            overrides["training.num_workers"] = cli_overrides["num_workers"]
        if cli_overrides.get("pin_memory") is not None:
            overrides["training.pin_memory"] = cli_overrides["pin_memory"]
        if cli_overrides.get("train_split") is not None:
            overrides["data.train_split"] = cli_overrides["train_split"]
        if cli_overrides.get("augment") is not None:
            overrides["data.augment"] = cli_overrides["augment"]
        if cli_overrides.get("image_size") is not None:
            overrides["data.image_size"] = cli_overrides["image_size"]

    config = load_config(config_path, override=overrides if overrides else None)
    return config


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
    weight_decay: float = 0.0,
    num_classes: int = NUM_CLASSES,
    config: Optional[Config] = None,
):
    """
    Train the segmentation model using weighted cross-entropy loss.

    Implements a standard deep learning training loop with class-weighted
    loss to handle the severe class imbalance typical in surgical instrument
    segmentation (instruments occupy ~2% of pixels).

    Args:
        model: PyTorch segmentation model (e.g., InstrumentSegmentationModel).
               Must accept (B, 3, H, W) input and produce (B, C, H, W) output.
        train_loader: DataLoader yielding (frames, masks) batches where:
                      - frames: Tensor of shape (B, 3, H, W)
                      - masks: Tensor of shape (B, H, W) with class indices
        num_epochs: Number of complete passes through the training data.
                    Default: 15 epochs. Overridden by config if provided.
        learning_rate: Initial learning rate for Adam optimizer.
                       Default: 0.001. Overridden by config if provided.
        weight_decay: L2 regularization factor applied by the optimizer.
                  Default: 0.0. Overridden by config if provided.
        num_classes: Number of output classes for loss computation.
                     Default: 2 (background + instrument).
        config: Optional Config object to override training parameters.
                If provided, extracts epochs, learning_rate, and class_weights
                from config.training.

    Returns:
        Tuple[nn.Module, List[float]]: Trained model and list of average
        loss values per epoch for learning curve visualization.

    Training Details:
        - Optimizer: Adam with default betas (0.9, 0.999)
        - Loss: CrossEntropyLoss with class weights [1.0, 3.0] to
                compensate for background-dominant class distribution
        - Device: Automatically uses CUDA if available (or from config.hardware)

    Example:
        >>> model = InstrumentSegmentationModel(num_classes=2)
        >>> train_loader = DataLoader(dataset, batch_size=4)
        >>> trained_model, losses = train_model(model, train_loader, num_epochs=10)
        >>> plt.plot(losses)  # Visualize learning curve

        >>> # Using configuration file
        >>> config = load_config("config/default.yaml")
        >>> trained_model, losses = train_model(model, train_loader, config=config)

    Note:
        This is a simplified training loop for demonstration purposes.
        Production training should include validation monitoring, early
        stopping, learning rate scheduling, and checkpoint saving.
    """
    # Apply config overrides if provided
    if config is not None:
        num_epochs = config.training.epochs
        learning_rate = config.training.learning_rate
        weight_decay = config.training.weight_decay
        num_classes = config.model.num_classes
        device = torch.device(config.hardware.get_device())
        instrument_weight = config.training.class_weights.instrument
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instrument_weight = INSTRUMENT_CLASS_WEIGHT

    model = model.to(device)

    class_weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    class_weights[1:] = instrument_weight
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"\n{'='*70}")
    print(f"Training on device: {device}")
    if config is not None:
        print(
            f"Configuration: epochs={num_epochs}, lr={learning_rate}, "
            f"instrument_weight={instrument_weight}"
        )
    print(f"{'='*70}\n")

    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for frames, masks in pbar:
                frames = frames.to(device)
                masks = masks.to(device)

                # Fail loudly if an entire batch contains no instrument pixels.
                # This guards against label mismatches in synthetic or real data.
                instrument_pixels = (masks > 0).sum().item()
                if instrument_pixels == 0:
                    raise ValueError(
                        "Instrument mask is empty for an entire batch. "
                        "Check label remapping and synthetic mask generation."
                    )

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


def evaluate_model(
    model,
    dataset,
    num_visual_samples: int = 4,
    prediction_dir: Optional[Path] = None,
):
    """
    Evaluate segmentation model and generate visual comparison outputs.

    Performs comprehensive evaluation including:
    1. Computing aggregate metrics across all samples
    2. Generating side-by-side visualizations (frame, ground truth, prediction)
    3. Optionally exporting predicted masks for downstream analysis

    Args:
        model: Trained PyTorch segmentation model in eval mode.
        dataset: SurgicalDataset instance yielding (frame, mask) pairs.
        num_visual_samples: Number of samples to include in visualization.
                           Default: 4. Set to 0 to disable visualization.
        prediction_dir: Optional path to save predicted masks as PNG files.
                       If None, predictions are not saved to disk.

    Returns:
        Dict[str, np.ndarray]: Evaluation metrics dictionary containing:
            - 'precision', 'recall', 'iou', 'dice' (per-class arrays)
            - 'support' (pixels per class)
            - 'accuracy' (overall pixel accuracy)

    Side Effects:
        - Saves visualization to outputs/figures/segmentation_results.png
        - Optionally saves prediction masks to prediction_dir
        - Prints formatted metrics table to stdout

    Visualization Output:
        Creates a figure with num_visual_samples rows and 3 columns:
        - Column 1: Original frame (denormalized from ImageNet stats)
        - Column 2: Ground truth segmentation mask
        - Column 3: Model prediction with IoU/Dice scores

    Example:
        >>> model = InstrumentSegmentationModel(num_classes=2)
        >>> model.load_state_dict(torch.load('model.pth'))
        >>> metrics = evaluate_model(
        ...     model,
        ...     val_dataset,
        ...     num_visual_samples=6,
        ...     prediction_dir=Path('outputs/predictions')
        ... )
        >>> print(f"Mean IoU: {metrics['iou'][1]:.3f}")

    Note:
        Evaluation runs in torch.no_grad() context to disable gradient
        computation for memory efficiency during inference.
    """

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
    """
    Main execution pipeline for surgical instrument segmentation.

    Orchestrates the complete training and evaluation workflow:
    1. Parse command-line arguments for data paths
    2. Generate synthetic data if needed (for demo purposes)
    3. Create train/validation split with proper transforms
    4. Train DeepLabV3 model with class-weighted loss
    5. Evaluate on validation set and generate visualizations
    6. Save trained model weights

    Command Line Arguments:
        --frame-dir: Directory containing RGB frames (default: data/sample_frames)
        --mask-dir: Directory containing mask PNGs (default: data/masks)
        --prediction-dir: Directory for prediction outputs (default: data/preds)
        --skip-synthetic: Don't create synthetic data if dirs are empty

    Output Files:
        - outputs/figures/training_loss.png: Learning curve visualization
        - outputs/figures/segmentation_results.png: Visual comparison
        - outputs/models/instrument_segmentation_model.pth: Trained weights
        - data/preds/*.png: Per-frame prediction masks

    Example:
        # Train on default synthetic data
        $ python -m surgical_segmentation.training.trainer

        # Train on custom dataset
        $ python -m surgical_segmentation.training.trainer \
            --frame-dir datasets/Cholec80/sample_frames \
            --mask-dir datasets/Cholec80/masks

    Pipeline Details:
        - Data split: 80% train, 20% validation
        - Training augmentation: ColorJitter, GaussianNoise
        - Evaluation: IoU, Dice, Precision, Recall per class
        - Training: 15 epochs, Adam optimizer, lr=0.001

    Raises:
        FileNotFoundError: If frame_dir contains no PNG files.
        RuntimeError: If fewer than 2 frames available for split.
    """

    args = parse_cli_args()
    cli_overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "train_split": args.train_split,
        "augment": args.augment,
        "image_size": args.image_size,
    }
    config = load_training_config(args.config, cli_overrides=cli_overrides)

    frame_dir = (args.frame_dir or Path(config.paths.frame_dir)).resolve()
    mask_dir = (args.mask_dir or Path(config.paths.mask_dir)).resolve()
    prediction_dir = (args.prediction_dir or Path(config.paths.predictions_dir)).resolve()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(config.training.seed)
    data_created = False
    if not args.skip_synthetic:
        data_created = create_synthetic_surgical_frames(frame_dir, mask_dir)

    image_size = config.data.image_size
    normalize_mean = config.data.normalize.mean
    normalize_std = config.data.normalize.std
    color_jitter = config.augmentation.color_jitter

    if config.data.augment:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.ColorJitter(
                    brightness=color_jitter.brightness,
                    contrast=color_jitter.contrast,
                    saturation=color_jitter.saturation,
                    hue=color_jitter.hue,
                ),
                AdditiveGaussianNoise(std=config.augmentation.gaussian_noise_std),
                transforms.Normalize(mean=normalize_mean, std=normalize_std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
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

    if not 0 < config.data.train_split < 1:
        raise ValueError(
            "train_split must be a float between 0 and 1 (exclusive). "
            f"Got {config.data.train_split}."
        )

    rng = np.random.default_rng(DEFAULT_DATA_SEED)
    rng.shuffle(all_frames)

    val_size = max(1, int(round(total_frames * (1 - config.data.train_split))))
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
        augment=config.data.augment,
        file_list=train_frames,
    )
    val_dataset = SurgicalDataset(
        frame_dir=str(frame_dir),
        mask_dir=str(mask_dir),
        transform=eval_transform,
        augment=False,
        file_list=val_frames,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    print("\nDataset Summary:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Training frames: {len(train_dataset)} (augmentations enabled)")
    print(f"  - Validation frames: {len(val_dataset)}")
    if not data_created:
        print("  - Note: Existing frames detected; synthetic generation skipped")

    model = InstrumentSegmentationModel(num_classes=config.model.num_classes)

    print("\nStarting training...")
    model, losses = train_model(
        model,
        train_loader,
        config=config,
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
