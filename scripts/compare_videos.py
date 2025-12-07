"""Compare video01 frames to video52 training frames to understand domain shift."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Load a video52 training frame
video52_frame_path = "datasets/Cholec80/sample_frames/video52_frame_000100.png"
video52_mask_path = "datasets/Cholec80/masks/video52_mask_000100.png"

# Load a generated video01 frame and mask
video01_frame_path = "datasets/Cholec80/video01_generated_frames/video01_frame_000100.png"
video01_mask_path = "datasets/Cholec80/video01_generated_masks/video01_mask_000100.png"

print("=" * 60)
print("COMPARISON: Video52 (training) vs Video01 (inference)")
print("=" * 60)

# Video 52 analysis
if Path(video52_frame_path).exists():
    v52_frame = np.array(Image.open(video52_frame_path))
    v52_mask = np.array(Image.open(video52_mask_path))
    v52_instrument_pixels = np.sum(v52_mask > 0)
    v52_total = v52_mask.size
    v52_percentage = (v52_instrument_pixels / v52_total) * 100

    print(f"\nVIDEO 52 (Training Data):")
    print(f"  Frame shape: {v52_frame.shape}")
    print(f"  Mask unique values: {np.unique(v52_mask)}")
    print(f"  Instrument pixels: {v52_instrument_pixels:,} ({v52_percentage:.2f}%)")
    print(f"  Mean brightness: {v52_frame.mean():.1f}")

# Video 01 analysis
if Path(video01_frame_path).exists():
    v01_frame = np.array(Image.open(video01_frame_path))
    v01_mask = np.array(Image.open(video01_mask_path))
    v01_instrument_pixels = np.sum(v01_mask > 0)
    v01_total = v01_mask.size
    v01_percentage = (v01_instrument_pixels / v01_total) * 100

    print(f"\nVIDEO 01 (Inference - Predicted):")
    print(f"  Frame shape: {v01_frame.shape}")
    print(f"  Mask unique values: {np.unique(v01_mask)}")
    print(f"  Instrument pixels: {v01_instrument_pixels:,} ({v01_percentage:.2f}%)")
    print(f"  Mean brightness: {v01_frame.mean():.1f}")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
if v52_percentage > 5 and v01_percentage < 1:
    print("⚠️  Domain Shift Detected!")
    print("   - Video52 has many instruments in frame")
    print("   - Video01 has few/no instruments detected")
    print("   - This suggests video01 is from a different")
    print("     surgical procedure or phase")
elif v01_percentage < 0.5:
    print("⚠️  Low instrument detection in video01")
    print("   - Either there are genuinely few instruments,")
    print("   - Or the model needs fine-tuning on video01 data")
else:
    print("✓ Model appears to be working on video01")
