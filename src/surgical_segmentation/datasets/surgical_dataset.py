"""Dataset utilities for surgical instrument segmentation."""

from __future__ import annotations

import os
import random
from collections.abc import Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance
from PIL.Image import Transpose
from torchvision import transforms


class SurgicalDataset(torch.utils.data.Dataset):
    """Dataset for surgical instrument segmentation with paired augmentations."""

    def __init__(
        self,
        frame_dir: str,
        mask_dir: str,
        transform=None,
        augment: bool = False,
        file_list: Iterable[str] | None = None,
    ):
        self.frame_dir = frame_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment = augment
        all_frames = sorted(os.listdir(frame_dir))
        if file_list is not None:
            self.frames = list(file_list)
        else:
            self.frames = all_frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_name = self.frames[idx]
        frame_path = os.path.join(self.frame_dir, frame_name)

        mask_name = frame_name.replace("frame", "mask")
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"No mask found for {frame_name}. Expected: {mask_name}\n"
                "Make sure you've run prepare_cholecseg8k_assets.py to prepare the dataset."
            )

        frame = Image.open(frame_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            frame, mask = self._apply_pair_augmentations(frame, mask)

        mask = mask.resize((256, 256), Image.Resampling.NEAREST)

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ]
            )(frame)

        mask_array = np.array(mask, dtype=np.int16)

        # CholecSeg8k uses non-standard class IDs (not matching paper Table I):
        # Class 31 = Grasper, Class 32 = L-hook Electrocautery
        # Synthetic generator uses class 1 for instruments.
        remapped = np.zeros_like(mask_array, dtype=np.uint8)
        instrument_mask = (mask_array == 1) | (mask_array == 31) | (mask_array == 32)
        remapped[instrument_mask] = 1

        mask_tensor = torch.from_numpy(remapped).long()

        return frame, mask_tensor

    @staticmethod
    def _apply_pair_augmentations(frame: Image.Image, mask: Image.Image):
        """Apply spatial transforms that keep frame/mask aligned."""
        if random.random() < 0.5:
            frame = frame.transpose(Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Transpose.FLIP_LEFT_RIGHT)

        if random.random() < 0.2:
            angle = random.uniform(-12.0, 12.0)
            frame = frame.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(90, 60, 60))
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)

        if random.random() < 0.2:
            frame = ImageEnhance.Brightness(frame).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.2:
            frame = ImageEnhance.Contrast(frame).enhance(random.uniform(0.8, 1.2))
        return frame, mask


__all__ = ["SurgicalDataset"]
