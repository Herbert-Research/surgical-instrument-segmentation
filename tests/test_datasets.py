"""Tests for dataset loading and preprocessing."""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from surgical_segmentation.datasets import SurgicalDataset


@pytest.fixture
def dataset_directory(tmp_path):
    """Create a temporary dataset directory with frame/mask pairs."""
    frame_dir = tmp_path / "frames"
    mask_dir = tmp_path / "masks"
    frame_dir.mkdir()
    mask_dir.mkdir()

    # Create 5 sample frame/mask pairs
    for i in range(5):
        # Create RGB frame
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        Image.fromarray(frame).save(frame_dir / f"frame_{i:05d}.png")

        # Create mask with CholecSeg8k class IDs
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 100:200] = 31  # Grasper region
        if i % 2 == 0:
            mask[250:350, 300:400] = 32  # L-hook region
        Image.fromarray(mask).save(mask_dir / f"mask_{i:05d}.png")

    return frame_dir, mask_dir


class TestMaskRemapping:
    """Test CholecSeg8k mask class remapping logic."""

    def test_remap_class_31_to_instrument(self, sample_cholecseg_mask):
        """Verify class 31 (Grasper) is remapped to 1."""
        remapped = np.zeros_like(sample_cholecseg_mask, dtype=np.uint8)
        instrument_mask = (sample_cholecseg_mask == 31) | (sample_cholecseg_mask == 32)
        remapped[instrument_mask] = 1

        # Check grasper region
        assert remapped[75, 100] == 1, "Class 31 should map to 1"

    def test_remap_class_32_to_instrument(self, sample_cholecseg_mask):
        """Verify class 32 (L-hook) is remapped to 1."""
        remapped = np.zeros_like(sample_cholecseg_mask, dtype=np.uint8)
        instrument_mask = (sample_cholecseg_mask == 31) | (sample_cholecseg_mask == 32)
        remapped[instrument_mask] = 1

        # Check L-hook region
        assert remapped[175, 150] == 1, "Class 32 should map to 1"

    def test_background_remains_zero(self, sample_cholecseg_mask):
        """Verify background pixels remain 0 after remapping."""
        remapped = np.zeros_like(sample_cholecseg_mask, dtype=np.uint8)
        instrument_mask = (sample_cholecseg_mask == 31) | (sample_cholecseg_mask == 32)
        remapped[instrument_mask] = 1

        # Check background region
        assert remapped[0, 0] == 0, "Background should remain 0"
        assert remapped[255, 255] == 0, "Background should remain 0"

    def test_remapped_mask_is_binary(self, sample_cholecseg_mask):
        """Verify remapped mask only contains values 0 and 1."""
        remapped = np.zeros_like(sample_cholecseg_mask, dtype=np.uint8)
        instrument_mask = (sample_cholecseg_mask == 31) | (sample_cholecseg_mask == 32)
        remapped[instrument_mask] = 1

        unique_values = np.unique(remapped)
        assert set(unique_values).issubset({0, 1}), f"Expected only 0 and 1, got {unique_values}"


class TestSurgicalDataset:
    """Test SurgicalDataset class functionality."""

    def test_dataset_length_matches_files(self, dataset_directory):
        """Verify dataset length equals number of frame files."""
        frame_dir, mask_dir = dataset_directory

        dataset = SurgicalDataset(str(frame_dir), str(mask_dir))

        assert len(dataset) == 5, f"Expected 5 samples, got {len(dataset)}"

    def test_getitem_returns_tensor_tuple(self, dataset_directory):
        """Verify __getitem__ returns (frame_tensor, mask_tensor) tuple."""
        frame_dir, mask_dir = dataset_directory

        dataset = SurgicalDataset(str(frame_dir), str(mask_dir))

        result = dataset[0]

        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, "Should return (frame, mask)"
        assert isinstance(result[0], torch.Tensor), "Frame should be a tensor"
        assert isinstance(result[1], torch.Tensor), "Mask should be a tensor"

    def test_getitem_frame_shape(self, dataset_directory):
        """Verify frame tensor has correct shape (C, H, W)."""
        frame_dir, mask_dir = dataset_directory

        dataset = SurgicalDataset(str(frame_dir), str(mask_dir))

        frame, _ = dataset[0]

        assert frame.shape == (3, 256, 256), f"Expected (3, 256, 256), got {frame.shape}"

    def test_getitem_mask_shape(self, dataset_directory):
        """Verify mask tensor has correct shape (H, W)."""
        frame_dir, mask_dir = dataset_directory

        dataset = SurgicalDataset(str(frame_dir), str(mask_dir))

        _, mask = dataset[0]

        assert mask.shape == (256, 256), f"Expected (256, 256), got {mask.shape}"

    def test_getitem_mask_dtype(self, dataset_directory):
        """Verify mask tensor is long type for cross-entropy loss."""
        frame_dir, mask_dir = dataset_directory

        dataset = SurgicalDataset(str(frame_dir), str(mask_dir))

        _, mask = dataset[0]

        assert mask.dtype == torch.long, f"Expected torch.long, got {mask.dtype}"

    def test_getitem_mask_values_binary(self, dataset_directory):
        """Verify mask values are binary (0 or 1) after remapping."""
        frame_dir, mask_dir = dataset_directory

        dataset = SurgicalDataset(str(frame_dir), str(mask_dir))

        for idx in range(len(dataset)):
            _, mask = dataset[idx]
            unique = torch.unique(mask)
            assert all(
                v in [0, 1] for v in unique.tolist()
            ), f"Mask at idx {idx} contains invalid values: {unique.tolist()}"

    def test_custom_transform_applied(self, dataset_directory):
        """Verify custom transform is applied to frames."""
        frame_dir, mask_dir = dataset_directory

        # Custom transform that produces 128x128 images
        custom_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )

        dataset = SurgicalDataset(
            str(frame_dir),
            str(mask_dir),
            transform=custom_transform,
        )

        frame, _ = dataset[0]

        assert frame.shape == (3, 128, 128), f"Expected (3, 128, 128), got {frame.shape}"

    def test_file_list_filtering(self, dataset_directory):
        """Verify file_list parameter filters loaded files."""
        frame_dir, mask_dir = dataset_directory

        # Only load first 2 files
        file_list = [f"frame_{i:05d}.png" for i in range(2)]

        dataset = SurgicalDataset(
            str(frame_dir),
            str(mask_dir),
            file_list=file_list,
        )

        assert len(dataset) == 2, f"Expected 2 samples with file_list, got {len(dataset)}"

    def test_missing_mask_raises_error(self, tmp_path):
        """Verify FileNotFoundError raised when mask is missing."""
        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        frame_dir.mkdir()
        mask_dir.mkdir()

        # Create frame without corresponding mask
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(frame).save(frame_dir / "frame_00000.png")

        dataset = SurgicalDataset(str(frame_dir), str(mask_dir))

        with pytest.raises(FileNotFoundError):
            _ = dataset[0]

    def test_frames_sorted_alphabetically(self, dataset_directory):
        """Verify frames are loaded in sorted order."""
        frame_dir, mask_dir = dataset_directory

        dataset = SurgicalDataset(str(frame_dir), str(mask_dir))

        expected_order = [f"frame_{i:05d}.png" for i in range(5)]
        assert dataset.frames == expected_order


class TestDatasetAugmentation:
    """Test data augmentation functionality."""

    def test_augmentation_produces_valid_output(self, dataset_directory):
        """Verify augmented data produces valid tensors."""
        frame_dir, mask_dir = dataset_directory

        dataset = SurgicalDataset(
            str(frame_dir),
            str(mask_dir),
            augment=True,
        )

        frame, mask = dataset[0]

        assert frame.shape == (3, 256, 256), "Augmented frame should be correct shape"
        assert mask.shape == (256, 256), "Augmented mask should be correct shape"
        assert not torch.isnan(frame).any(), "Frame should not contain NaN"

    def test_augmentation_preserves_mask_values(self, dataset_directory):
        """Verify augmentation doesn't corrupt mask values."""
        frame_dir, mask_dir = dataset_directory

        dataset = SurgicalDataset(
            str(frame_dir),
            str(mask_dir),
            augment=True,
        )

        # Check multiple samples
        for idx in range(len(dataset)):
            _, mask = dataset[idx]
            unique = torch.unique(mask)
            assert all(
                v in [0, 1] for v in unique.tolist()
            ), f"Augmentation corrupted mask values at idx {idx}"

    def test_horizontal_flip_alignment(self, tmp_path):
        """Verify horizontal flip keeps frame and mask aligned."""
        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        frame_dir.mkdir()
        mask_dir.mkdir()

        # Create asymmetric frame and mask (left side only)
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[:, :128, 0] = 255  # Red on left half
        Image.fromarray(frame).save(frame_dir / "frame_00000.png")

        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[:, :128] = 31  # Instrument on left half
        Image.fromarray(mask).save(mask_dir / "mask_00000.png")

        # Test multiple times (flip is stochastic)
        dataset = SurgicalDataset(
            str(frame_dir),
            str(mask_dir),
            augment=True,
        )

        # Just verify no errors occur
        for _ in range(10):
            frame_tensor, mask_tensor = dataset[0]
            assert frame_tensor.shape == (3, 256, 256)
            assert mask_tensor.shape == (256, 256)


class TestDatasetEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_directory_creates_empty_dataset(self, tmp_path):
        """Verify empty directories create empty dataset."""
        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        frame_dir.mkdir()
        mask_dir.mkdir()

        dataset = SurgicalDataset(str(frame_dir), str(mask_dir))

        assert len(dataset) == 0

    def test_single_sample_dataset(self, tmp_path):
        """Verify dataset works with single sample."""
        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        frame_dir.mkdir()
        mask_dir.mkdir()

        # Create single sample
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:150, 100:150] = 31

        Image.fromarray(frame).save(frame_dir / "frame_00000.png")
        Image.fromarray(mask).save(mask_dir / "mask_00000.png")

        dataset = SurgicalDataset(str(frame_dir), str(mask_dir))

        assert len(dataset) == 1
        frame_tensor, mask_tensor = dataset[0]
        assert frame_tensor.shape == (3, 256, 256)

    def test_grayscale_mask_handling(self, tmp_path):
        """Verify grayscale masks are handled correctly."""
        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        frame_dir.mkdir()
        mask_dir.mkdir()

        # Create frame and grayscale mask
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:150, 100:150] = 31

        Image.fromarray(frame).save(frame_dir / "frame_00000.png")
        Image.fromarray(mask, mode="L").save(mask_dir / "mask_00000.png")

        dataset = SurgicalDataset(str(frame_dir), str(mask_dir))

        _, mask_tensor = dataset[0]
        assert mask_tensor.dtype == torch.long
