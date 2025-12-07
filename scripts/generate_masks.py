"""
Generate segmentation masks from surgical videos using the trained model.

This script applies the trained model to new surgical videos:
1. Extracts frames from surgical videos
2. Uses the trained outputs/models/instrument_segmentation_model.pth to generate masks
3. Outputs frames + predicted masks in the format expected by the training pipeline

Use this for:
- Creating training data from new surgical videos
- Generating predictions for videos without manual annotations
- Bootstrapping semi-supervised annotation workflows

Example:
    # Generate masks for a new surgical video
    python scripts/generate_masks.py \
        --video-path /path/to/new_surgery.mp4 \
        --model-path outputs/models/instrument_segmentation_model.pth \
        --output-dir outputs/generated/video01 \
        --frame-step 10

    # Then use the output directly with train-segmentation for fine-tuning
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

DEFAULT_MODEL_PATH = Path("outputs/models/instrument_segmentation_model.pth")
DEFAULT_OUTPUT_DIR = Path("outputs/generated_output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate masks from surgical videos using trained model."
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="Path to input surgical video (MP4, AVI, MOV).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained segmentation model weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save both frames and masks together.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Extract every Nth frame from video.",
    )
    parser.add_argument(
        "--long-side",
        type=int,
        default=640,
        help="Resize longest side to this value (maintains aspect ratio).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum number of frames to process per video.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence for instrument pixel classification (0-1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu).",
    )
    return parser.parse_args()


def load_model(model_path: Path, device: str) -> torch.nn.Module:
    """Load the trained segmentation model."""
    print(f"[INFO] Loading model from {model_path}")
    # Load with aux_classifier disabled for inference
    model = deeplabv3_resnet50(num_classes=2, weights=None, aux_loss=False)

    # Handle both full model saves and state_dict saves
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Remove 'model.' prefix if present and filter out aux_classifier keys
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'model.' prefix
        if key.startswith("model."):
            new_key = key[6:]
        else:
            new_key = key

        # Skip aux_classifier layers (only used during training)
        if "aux_classifier" in new_key:
            continue

        new_state_dict[new_key] = value

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    if missing_keys:
        print(f"[WARN] Missing keys: {missing_keys[:5]}...")  # Show first 5
    if unexpected_keys:
        print(f"[WARN] Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5

    model = model.to(device)
    model.eval()
    print(f"[INFO] Model loaded successfully on {device}")
    return model


def resize_with_aspect(image: np.ndarray, long_side: int) -> np.ndarray:
    """Resize image maintaining aspect ratio."""
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def predict_mask(
    model: torch.nn.Module,
    frame_rgb: np.ndarray,
    device: str,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """
    Generate segmentation mask for a single frame.

    Returns binary mask where:
    - 0 = background
    - 1 = instrument (any surgical instrument)
    """
    # Prepare image for model
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Convert to PIL and apply transforms (returns tensor)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # type: ignore[union-attr]

    # Run inference
    with torch.no_grad():
        output = model(input_batch)["out"]

        # Apply softmax to get probabilities
        probabilities = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()

        # Get instrument class probability (class 1)
        instrument_prob = probabilities[1]

        # Apply confidence threshold
        predictions = (instrument_prob >= confidence_threshold).astype(np.uint8)

    # Scale to 0-255 for visibility (0=background black, 255=instrument white)
    return predictions * 255


def process_video(
    video_path: Path,
    model: torch.nn.Module,
    output_dir: Path,
    device: str,
    frame_step: int,
    long_side: int,
    max_frames: int,
    confidence_threshold: float,
) -> int:
    """Extract frames and generate masks from video."""

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video: {video_path.name}")
    print(f"       Total frames: {total_frames}, FPS: {fps:.2f}")
    print(f"       Extracting every {frame_step} frames (max: {max_frames})")

    frame_idx = 0
    saved_count = 0
    video_stem = video_path.stem

    while saved_count < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Skip frames based on frame_step
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        # Convert and resize frame
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = resize_with_aspect(frame_rgb, long_side)

        # Generate mask using model
        mask = predict_mask(model, frame_rgb, device, confidence_threshold)

        # Save frame and mask with names that sort together
        # Format: video01_000001_frame.png, video01_000001_mask.png
        base_name = f"{video_stem}_{saved_count:06d}"
        frame_name = f"{base_name}_frame.png"
        mask_name = f"{base_name}_mask.png"

        frame_path = output_dir / frame_name
        mask_path = output_dir / mask_name

        cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), mask)

        saved_count += 1
        frame_idx += 1

        # Progress update
        if saved_count % 50 == 0:
            print(f"       Processed {saved_count} frames...")

    cap.release()
    return saved_count


def main() -> None:
    args = parse_args()

    # Validate inputs
    if not args.video_path.exists():
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    # Load model
    model = load_model(args.model_path, args.device)

    # Process video
    frames_saved = process_video(
        video_path=args.video_path,
        model=model,
        output_dir=args.output_dir,
        device=args.device,
        frame_step=args.frame_step,
        long_side=args.long_side,
        max_frames=args.max_frames,
        confidence_threshold=args.confidence_threshold,
    )

    print(f"\n✓ Processing complete!")
    print(f"  Frames saved: {frames_saved}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Files are named: videoXX_NNNNNN_frame.png and videoXX_NNNNNN_mask.png")
    print(f"  (sorted alphabetically, frames and masks alternate)")
    print(f"\nThese outputs can now be used with train-segmentation for:")
    print(f"  - Fine-tuning on new surgical procedures")
    print(f"  - Bootstrapping semi-supervised annotation")
    print(f"  - Quality assessment and validation")


if __name__ == "__main__":
    main()
