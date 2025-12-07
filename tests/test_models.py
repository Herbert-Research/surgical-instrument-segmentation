"""Tests for model architectures."""

import pytest
import torch

from surgical_segmentation.models import InstrumentSegmentationModel, UNet


class TestUNet:
    """Test suite for U-Net architecture."""

    def test_forward_pass_shape(self, sample_image_tensor):
        """Verify output shape matches input spatial dimensions."""
        model = UNet(n_channels=3, n_classes=2, bilinear=True)
        output = model(sample_image_tensor)

        assert output.shape == (1, 2, 256, 256), f"Expected (1, 2, 256, 256), got {output.shape}"

    def test_forward_pass_no_nan(self, sample_image_tensor):
        """Verify forward pass produces no NaN values."""
        model = UNet(n_channels=3, n_classes=2)
        output = model(sample_image_tensor)

        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_parameter_count(self):
        """Verify parameter count is reasonable."""
        model = UNet(n_channels=3, n_classes=2, bilinear=True)
        param_count = model.count_parameters()

        # U-Net should have ~17M parameters with bilinear upsampling
        assert 15_000_000 < param_count < 20_000_000, f"Unexpected parameter count: {param_count:,}"

    @pytest.mark.parametrize("n_classes", [2, 5, 10])
    def test_variable_output_classes(self, sample_image_tensor, n_classes):
        """Verify model works with different numbers of output classes."""
        model = UNet(n_channels=3, n_classes=n_classes)
        output = model(sample_image_tensor)

        assert output.shape[1] == n_classes


class TestDeepLabV3:
    """Test suite for DeepLabV3-ResNet50 architecture."""

    def test_forward_pass_shape(self, sample_image_tensor):
        """Verify output shape matches input spatial dimensions."""
        model = InstrumentSegmentationModel(num_classes=2)
        model.eval()  # Required for BatchNorm with small batch sizes
        output = model(sample_image_tensor)

        assert output.shape == (1, 2, 256, 256), f"Expected (1, 2, 256, 256), got {output.shape}"

    def test_forward_pass_no_nan(self, sample_image_tensor):
        """Verify forward pass produces no NaN values."""
        model = InstrumentSegmentationModel(num_classes=2)
        model.eval()  # Required for BatchNorm with small batch sizes
        output = model(sample_image_tensor)

        assert not torch.isnan(output).any(), "Output contains NaN values"
