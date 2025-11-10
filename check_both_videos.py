"""Check instrument detection in video52 generated masks."""
from PIL import Image
import numpy as np
from pathlib import Path

mask_dir = Path("datasets/Cholec80/video52_generated_masks")
masks_to_check = list(mask_dir.glob("*.png"))[:10]  # Check first 10

print("=" * 70)
print("VIDEO 52 - Model Performance (trained on this video)")
print("=" * 70)

for mask_path in sorted(masks_to_check):
    mask = np.array(Image.open(mask_path))
    instrument_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    percentage = (instrument_pixels / total_pixels) * 100
    print(f"{mask_path.name}: {instrument_pixels:7,} pixels ({percentage:5.2f}%)")

print("\n" + "=" * 70)
print("COMPARISON WITH VIDEO 01")
print("=" * 70)

# Check video01 for comparison
v01_mask_dir = Path("datasets/Cholec80/video01_generated_masks")
v01_masks = list(v01_mask_dir.glob("*.png"))[:10]

for mask_path in sorted(v01_masks):
    mask = np.array(Image.open(mask_path))
    instrument_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    percentage = (instrument_pixels / total_pixels) * 100
    print(f"{mask_path.name}: {instrument_pixels:7,} pixels ({percentage:5.2f}%)")

print("\n" + "=" * 70)
print("VERDICT:")
print("=" * 70)
print("If video52 shows high % and video01 shows low %:")
print("  ✓ Model IS working correctly!")
print("  ✓ Video01 simply has different content/instruments")
print("  → Solution: Fine-tune model on video01 data")
