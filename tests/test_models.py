"""Tests for model architectures."""

import pytest
import torch

from surgical_segmentation.models import InstrumentSegmentationModel, UNet
from surgical_segmentation.models.unet import DoubleConv, Down, OutConv, Up


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

    def test_non_bilinear_upsampling(self, sample_image_tensor):
        """Verify UNet works without bilinear upsampling."""
        model = UNet(n_channels=3, n_classes=2, bilinear=False)
        output = model(sample_image_tensor)

        assert output.shape == (1, 2, 256, 256)


class TestUNetComponents:
    """Test individual UNet building blocks."""

    def test_double_conv_output_shape(self):
        """Verify DoubleConv produces correct output shape."""
        layer = DoubleConv(3, 64)
        x = torch.randn(1, 3, 256, 256)
        output = layer(x)

        assert output.shape == (1, 64, 256, 256)

    def test_down_halves_spatial_dims(self):
        """Verify Down block halves spatial dimensions."""
        layer = Down(64, 128)
        x = torch.randn(1, 64, 256, 256)
        output = layer(x)

        assert output.shape == (1, 128, 128, 128)

    def test_up_non_bilinear(self):
        """Verify Up block works without bilinear upsampling."""
        # For non-bilinear: ConvTranspose2d halves channels (1024 -> 512)
        # Then concat with skip (512) gives 1024 total -> DoubleConv to 512
        layer = Up(1024, 512, bilinear=False)
        x1 = torch.randn(1, 1024, 16, 16)
        x2 = torch.randn(1, 512, 32, 32)
        output = layer(x1, x2)

        assert output.shape == (1, 512, 32, 32)

    def test_out_conv_output_shape(self):
        """Verify OutConv produces correct output channels."""
        layer = OutConv(64, 2)
        x = torch.randn(1, 64, 256, 256)
        output = layer(x)

        assert output.shape == (1, 2, 256, 256)


class TestDeepLabV3:
    """Test suite for DeepLabV3-ResNet50 architecture."""

    def test_forward_pass_shape(self, sample_image_tensor):
        """Verify output shape matches input spatial dimensions."""
        model = InstrumentSegmentationModel(num_classes=2, pretrained=False)
        model.eval()  # Required for BatchNorm with small batch sizes
        output = model(sample_image_tensor)

        assert output.shape == (1, 2, 256, 256), f"Expected (1, 2, 256, 256), got {output.shape}"

    def test_forward_pass_no_nan(self, sample_image_tensor):
        """Verify forward pass produces no NaN values."""
        model = InstrumentSegmentationModel(num_classes=2, pretrained=False)
        model.eval()  # Required for BatchNorm with small batch sizes
        output = model(sample_image_tensor)

        assert not torch.isnan(output).any(), "Output contains NaN values"


class TestUNetTestFunction:
    """Test the built-in test_unet function."""

    def test_test_unet_runs_successfully(self):
        """Verify test_unet() function runs without errors."""
        from surgical_segmentation.models.unet import test_unet

        # Should run without raising any exceptions
        test_unet()

    def test_test_unet_output_shape_verification(self, capsys):
        """Verify test_unet prints expected output."""
        from surgical_segmentation.models.unet import test_unet

        test_unet()
        captured = capsys.readouterr()
        assert "U-Net test passed" in captured.out
        assert "Output shape: torch.Size([1, 2, 256, 256])" in captured.out
