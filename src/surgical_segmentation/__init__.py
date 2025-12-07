"""
Surgical Instrument Segmentation Package

A deep learning pipeline for automated surgical instrument segmentation
in laparoscopic surgery videos.

Author: Maximilian Herbert Dressler
"""

__version__ = "0.1.0"
__author__ = "Maximilian Herbert Dressler"

from surgical_segmentation.datasets import SurgicalDataset
from surgical_segmentation.models import InstrumentSegmentationModel, UNet

__all__ = ["InstrumentSegmentationModel", "SurgicalDataset", "UNet"]
