"""Quick script to check generated mask values."""
from pathlib import Path

import numpy as np
from PIL import Image

mask_dir = Path("datasets/Cholec80/video01_generated_masks")
masks_to_check = [
    "video01_mask_000000.png",
    "video01_mask_000100.png",
    "video01_mask_000200.png",
    "video01_mask_000300.png",
    "video01_mask_000400.png",
]

for mask_name in masks_to_check:
    mask_path = mask_dir / mask_name
    if mask_path.exists():
        mask = np.array(Image.open(mask_path))
        instrument_pixels = np.sum(mask == 1)
        total_pixels = mask.size
        percentage = (instrument_pixels / total_pixels) * 100
        print(f"{mask_name}: {instrument_pixels:6d} instrument pixels ({percentage:5.2f}%)")
