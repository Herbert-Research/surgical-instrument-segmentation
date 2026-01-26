"""DeepLabV3-ResNet50 model wrapper for surgical instrument segmentation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash for a local file.

    Args:
        file_path: Path to the file to hash.

    Returns:
        SHA256 hex digest of the file contents.
    """
    hasher = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_pretrained_weights_metadata(pretrained: bool) -> dict[str, str | bool | None]:
    """Collect metadata for pretrained weights used by DeepLabV3-ResNet50.

    Args:
        pretrained: Whether pretrained weights are enabled.

    Returns:
        Dictionary containing weight source URL and SHA256 if available.
    """
    if not pretrained:
        return {
            "pretrained": False,
            "weights_name": None,
            "source_url": None,
            "sha256": None,
            "local_path": None,
        }

    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    source_url = weights.url
    filename = source_url.split("/")[-1] if source_url else None
    local_path = None
    sha256 = None

    if filename:
        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
        candidate_path = cache_dir / filename
        if candidate_path.exists():
            local_path = str(candidate_path)
            sha256 = _compute_sha256(candidate_path)

    return {
        "pretrained": True,
        "weights_name": weights.name,
        "source_url": source_url,
        "sha256": sha256,
        "local_path": local_path,
    }


class InstrumentSegmentationModel(nn.Module):
    """
    Surgical instrument segmentation model using transfer learning.

    Based on DeepLabV3 with a ResNet50 backbone (ImageNet pre-trained).
    Set ``pretrained=False`` (or export ``SURGICAL_SEGMENTATION_DISABLE_PRETRAINED=1``)
    to avoid downloading weights in offline or CI environments.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        disable_pretrained = os.getenv("SURGICAL_SEGMENTATION_DISABLE_PRETRAINED", "0") == "1"
        if disable_pretrained:
            pretrained = False

        weights = (
            torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.pretrained = pretrained
        self.weights_metadata = get_pretrained_weights_metadata(pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


__all__ = ["InstrumentSegmentationModel", "get_pretrained_weights_metadata"]
