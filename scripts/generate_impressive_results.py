"""
Generate impressive visualization results by selecting frames with instruments.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from surgical_segmentation.models import InstrumentSegmentationModel

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 2
DEFAULT_MODEL_PATH = Path("outputs/models/instrument_segmentation_model.pth")
DEFAULT_OUTPUT_PATH = Path("outputs/figures/impressive_segmentation_results.png")


def find_frames_with_instruments(mask_dir: Path, min_instrument_pixels: int = 1000):
    """Find frames that have significant instrument presence."""
    mask_files = sorted(mask_dir.glob("*.png"))
    frames_with_instruments = []

    print(f"Scanning {len(mask_files)} masks for frames with instruments...")

    for mask_path in mask_files:
        mask = np.array(Image.open(mask_path).convert("L"))

        # Check for instrument pixels (class 31 or 32, or already converted to 1)
        instrument_pixels = np.sum((mask == 1) | (mask == 31) | (mask == 32))

        if instrument_pixels >= min_instrument_pixels:
            frames_with_instruments.append((mask_path.stem, instrument_pixels))

    # Sort by number of instrument pixels (descending) to get most interesting frames
    frames_with_instruments.sort(key=lambda x: x[1], reverse=True)

    print(f"Found {len(frames_with_instruments)} frames with instruments")
    return frames_with_instruments


def generate_visualization(
    frame_dir: Path,
    mask_dir: Path,
    pred_dir: Path,
    model_path: Path,
    output_path: Path,
    num_samples: int = 6,
):
    """Generate visualization with frames that have instruments."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find frames with instruments
    frames_with_instruments = find_frames_with_instruments(mask_dir)

    if len(frames_with_instruments) == 0:
        print("No frames with instruments found!")
        return

    # Select diverse samples (spread across the list)
    step = max(1, len(frames_with_instruments) // num_samples)
    selected_frames = [
        frames_with_instruments[i * step][0]
        for i in range(min(num_samples, len(frames_with_instruments)))
    ]

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
            transforms.Resize((256, 256)),
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
            pred_path = pred_dir / mask_name

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
                true_mask_pil.resize((256, 256), Image.Resampling.NEAREST), dtype=np.uint8
            )

            # Remap mask if needed (CholecSeg8k class IDs)
            remapped = np.zeros_like(true_mask, dtype=np.uint8)
            instrument_mask = (true_mask == 31) | (true_mask == 32) | (true_mask == 1)
            remapped[instrument_mask] = 1
            true_mask = remapped

            # Calculate metrics for this frame
            tp = np.logical_and(pred_mask == 1, true_mask == 1).sum()
            fp = np.logical_and(pred_mask == 1, true_mask == 0).sum()
            fn = np.logical_and(pred_mask == 0, true_mask == 1).sum()

            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

            # Denormalize frame for display
            frame_np = frame_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            frame_np = (frame_np * IMAGENET_STD) + IMAGENET_MEAN
            frame_np = np.clip(frame_np, 0, 1)

            # Plot
            axes[idx, 0].imshow(frame_np)
            axes[idx, 0].set_title(f"Input Frame {idx+1}", fontweight="bold", fontsize=12)
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(true_mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
            axes[idx, 1].set_title("Ground Truth Mask", fontweight="bold", fontsize=12)
            axes[idx, 1].axis("off")

            axes[idx, 2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
            axes[idx, 2].set_title(
                f"Prediction (IoU: {iou:.3f}, Dice: {dice:.3f})", fontweight="bold", fontsize=12
            )
            axes[idx, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved impressive visualization to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate impressive segmentation results")
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
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to trained model weights"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output path for visualization"
    )
    parser.add_argument(
        "--num-samples", type=int, default=6, help="Number of sample frames to visualize"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GENERATING IMPRESSIVE SEGMENTATION RESULTS")
    print("=" * 70)

    generate_visualization(
        frame_dir=args.frame_dir,
        mask_dir=args.mask_dir,
        pred_dir=args.pred_dir,
        model_path=args.model_path,
        output_path=args.output,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
