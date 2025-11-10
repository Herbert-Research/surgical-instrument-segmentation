"""
Generate segmentation masks from surgical videos using the trained model.

This script bridges the gap between raw videos and the training pipeline:
1. Extracts frames from surgical videos (like prepare_cholec80.py)
2. Uses the trained instrument_segmentation_model.pth to generate masks
3. Outputs frames + predicted masks in the format expected by the training pipeline

Use this for:
- Creating training data from new surgical videos
- Generating predictions for videos without manual annotations
- Bootstrapping semi-supervised annotation workflows

Example:
    # Generate masks for a new surgical video
    python generate_masks_from_model.py \
        --video-path /path/to/new_surgery.mp4 \
        --model-path instrument_segmentation_model.pth \
        --output-frame-dir data/sample_frames \
        --output-mask-dir data/masks \
        --frame-step 10
        
    # Then use the output directly with instrument_segmentation.py for fine-tuning
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50


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
        default=Path("instrument_segmentation_model.pth"),
        help="Path to trained segmentation model weights.",
    )
    parser.add_argument(
        "--output-frame-dir",
        type=Path,
        default=Path("data/sample_frames"),
        help="Directory to save extracted frames.",
    )
    parser.add_argument(
        "--output-mask-dir",
        type=Path,
        default=Path("data/masks"),
        help="Directory to save predicted masks.",
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
    model = deeplabv3_resnet50(num_classes=3, weights=None)
    
    # Handle both full model saves and state_dict saves
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
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
    - 1 = instrument (grasper)
    - 2 = instrument (scissors/hook)
    """
    # Prepare image for model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to PIL and apply transforms (returns tensor)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # type: ignore[union-attr]
    
    # Run inference
    with torch.no_grad():
        output = model(input_batch)['out']
        # Get class predictions
        predictions = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    return predictions.astype(np.uint8)


def process_video(
    video_path: Path,
    model: torch.nn.Module,
    output_frame_dir: Path,
    output_mask_dir: Path,
    device: str,
    frame_step: int,
    long_side: int,
    max_frames: int,
    confidence_threshold: float,
) -> int:
    """Extract frames and generate masks from video."""
    
    # Ensure output directories exist
    output_frame_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        # Save frame
        frame_name = f"{video_stem}_frame_{frame_idx:06d}.png"
        frame_path = output_frame_dir / frame_name
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        # Save mask
        mask_name = f"{video_stem}_mask_{frame_idx:06d}.png"
        mask_path = output_mask_dir / mask_name
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
        output_frame_dir=args.output_frame_dir,
        output_mask_dir=args.output_mask_dir,
        device=args.device,
        frame_step=args.frame_step,
        long_side=args.long_side,
        max_frames=args.max_frames,
        confidence_threshold=args.confidence_threshold,
    )
    
    print(f"\n✓ Processing complete!")
    print(f"  Frames saved: {frames_saved}")
    print(f"  Frames directory: {args.output_frame_dir}")
    print(f"  Masks directory:  {args.output_mask_dir}")
    print(f"\nThese outputs can now be used with instrument_segmentation.py for:")
    print(f"  - Fine-tuning on new surgical procedures")
    print(f"  - Bootstrapping semi-supervised annotation")
    print(f"  - Quality assessment and validation")


if __name__ == "__main__":
    main()
