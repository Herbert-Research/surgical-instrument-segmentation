"""
Generate an unbiased random sample visualization for surgical segmentation.

Use this script for representative qualitative results. It samples frames
uniformly at random and records the seed for reproducibility.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from surgical_segmentation.evaluation.metrics import compute_all_metrics
from surgical_segmentation.models import InstrumentSegmentationModel

# Constants
IMAGENET_MEAN: Sequence[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: Sequence[float] = [0.229, 0.224, 0.225]
NUM_CLASSES: int = 2
DEFAULT_IMAGE_SIZE: int = 256
DEFAULT_MODEL_PATH: Path = Path("outputs/models/instrument_segmentation_model.pth")
DEFAULT_OUTPUT_PATH: Path = Path("outputs/figures/random_sample_segmentation_results.png")
DEFAULT_RANDOM_SEED: int = 1337


def select_random_frames(mask_dir: Path, num_samples: int, seed: int) -> list[str]:
    """Select a random sample of frames for visualization.

    Args:
        mask_dir: Directory containing mask images.
        num_samples: Number of frames to select.
        seed: Random seed for reproducibility.

    Returns:
        List of selected frame stems.
    """
    mask_files = sorted(mask_dir.glob("*.png"))
    if len(mask_files) == 0:
        return []

    rng = np.random.default_rng(seed)
    sample_count = min(num_samples, len(mask_files))
    selected = rng.choice(mask_files, size=sample_count, replace=False)
    return [mask_path.stem for mask_path in selected]


def generate_visualization(
    frame_dir: Path,
    mask_dir: Path,
    pred_dir: Path,
    model_path: Path,
    output_path: Path,
    num_samples: int,
    seed: int,
) -> None:
    """Generate random sample visualization.

    Args:
        frame_dir: Directory containing RGB frames.
        mask_dir: Directory containing ground truth masks.
        pred_dir: Directory containing predicted masks (unused, kept for parity).
        model_path: Path to trained model weights.
        output_path: Output path for visualization.
        num_samples: Number of sample frames to visualize.
        seed: Random seed for reproducibility.
    """
    _ = pred_dir
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Random sample seed: {seed}")
    selected_frames = select_random_frames(mask_dir, num_samples=num_samples, seed=seed)

    if len(selected_frames) == 0:
        print("No frames available for visualization!")
        return

    print(f"\nSelected frames: {selected_frames[:3]}... (showing first 3)")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InstrumentSegmentationModel(num_classes=NUM_CLASSES)

    # Load state dict with strict=False to handle auxiliary classifier
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Setup transform
    eval_transform = transforms.Compose(
        [
            transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Create visualization
    fig, axes = plt.subplots(len(selected_frames), 3, figsize=(15, 4 * len(selected_frames)))
    if len(selected_frames) == 1:
        axes = np.expand_dims(axes, axis=0)

    cmap = plt.get_cmap("viridis", NUM_CLASSES)

    with torch.no_grad():
        for idx, frame_stem in enumerate(selected_frames):
            # Load frame
            frame_name = frame_stem.replace("mask", "frame") + ".png"
            mask_name = frame_stem + ".png"

            frame_path = frame_dir / frame_name
            mask_path = mask_dir / mask_name
            # pred_path would be: pred_dir / mask_name (unused in visualization)

            if not frame_path.exists():
                print(f"Warning: Frame not found: {frame_path}")
                continue

            # Load and process frame
            frame_pil = Image.open(frame_path).convert("RGB")
            frame_tensor: torch.Tensor = eval_transform(frame_pil)  # type: ignore
            frame_tensor = frame_tensor.unsqueeze(0).to(device)

            # Get prediction
            output = model(frame_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # Load true mask
            true_mask_pil = Image.open(mask_path).convert("L")
            true_mask = np.array(
                true_mask_pil.resize(
                    (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), Image.Resampling.NEAREST
                ),
                dtype=np.uint8,
            )

            # Remap mask if needed (CholecSeg8k class IDs)
            remapped = np.zeros_like(true_mask, dtype=np.uint8)
            instrument_mask = (true_mask == 31) | (true_mask == 32) | (true_mask == 1)
            remapped[instrument_mask] = 1
            true_mask = remapped

            # Calculate metrics for this frame using canonical implementation
            metrics = compute_all_metrics(
                torch.from_numpy(pred_mask),
                torch.from_numpy(true_mask),
            )
            iou = metrics["iou"]
            dice = metrics["dice"]

            # Denormalize frame for display
            frame_np = frame_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            frame_np = (frame_np * IMAGENET_STD) + IMAGENET_MEAN
            frame_np = np.clip(frame_np, 0, 1)

            # Plot
            axes[idx, 0].imshow(frame_np)
            axes[idx, 0].set_title(f"Input Frame {idx + 1}", fontweight="bold", fontsize=12)
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(true_mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
            axes[idx, 1].set_title("Ground Truth Mask", fontweight="bold", fontsize=12)
            axes[idx, 1].axis("off")

            axes[idx, 2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
            axes[idx, 2].set_title(
                f"Prediction (IoU: {iou:.3f}, Dice: {dice:.3f})",
                fontweight="bold",
                fontsize=12,
            )
            axes[idx, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved visualization to: {output_path}")
    plt.close(fig)


def main() -> None:
    """CLI entry point for random sample visualizations."""
    parser = argparse.ArgumentParser(
        description="Generate random sample segmentation visualizations (representative)"
    )
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=Path("datasets/Full Dataset/frames"),
        help="Directory containing RGB frames",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("datasets/Full Dataset/masks"),
        help="Directory containing ground truth masks",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("datasets/Full Dataset/preds"),
        help="Directory containing predicted masks",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for visualization",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of sample frames to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed recorded for reproducibility",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RANDOM SAMPLE SEGMENTATION VISUALIZATION")
    print("=" * 70)

    generate_visualization(
        frame_dir=args.frame_dir,
        mask_dir=args.mask_dir,
        pred_dir=args.pred_dir,
        model_path=args.model_path,
        output_path=args.output,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
