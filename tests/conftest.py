"""Pytest fixtures for surgical segmentation tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_image_tensor():
    """Create a sample normalized image tensor (B, C, H, W)."""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_mask_tensor():
    """Create a sample binary mask tensor (B, H, W)."""
    mask = torch.zeros(1, 256, 256, dtype=torch.long)
    # Add some instrument pixels
    mask[0, 100:150, 100:200] = 1
    return mask


@pytest.fixture
def sample_mask_numpy():
    """Create a sample mask as numpy array."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100:150, 100:200] = 1
    return mask


@pytest.fixture
def sample_cholecseg_mask():
    """Create a mask with CholecSeg8k class IDs (31, 32)."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:100, 50:150] = 31  # Grasper
    mask[150:200, 100:200] = 32  # L-hook
    return mask
