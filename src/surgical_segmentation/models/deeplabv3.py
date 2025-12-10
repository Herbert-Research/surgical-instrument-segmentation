"""DeepLabV3-ResNet50 model wrapper for surgical instrument segmentation."""

from __future__ import annotations

import torch.nn as nn
import torchvision


class InstrumentSegmentationModel(nn.Module):
    """
    Surgical instrument segmentation model using transfer learning.

    Based on DeepLabV3 with a ResNet50 backbone (ImageNet pre-trained).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        )
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)["out"]


__all__ = ["InstrumentSegmentationModel"]
