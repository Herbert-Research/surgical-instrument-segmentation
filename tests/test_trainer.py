"""Tests for trainer module utility functions.

Tests utility functions, configuration, and helper methods in the trainer module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from surgical_segmentation.training.trainer import (
    NUM_CLASSES,
    CLASS_NAMES,
    DEFAULT_DATA_SEED,
    INSTRUMENT_CLASS_WEIGHT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    seed_everything,
    AdditiveGaussianNoise,
    confusion_matrix_multiclass,
    compute_metrics_from_cm,
    create_synthetic_surgical_frames,
    evaluate_model,
)


class TestConstants:
    """Test module constants are correctly defined."""

    def test_num_classes_is_binary(self):
        """Verify NUM_CLASSES is 2 for binary segmentation."""
        assert NUM_CLASSES == 2

    def test_class_names_match_num_classes(self):
        """Verify CLASS_NAMES has correct number of entries."""
        assert len(CLASS_NAMES) == NUM_CLASSES

    def test_class_names_includes_background(self):
        """Verify background is first class."""
        assert CLASS_NAMES[0] == "background"

    def test_instrument_weight_greater_than_one(self):
        """Verify instrument class weight handles imbalance."""
        assert INSTRUMENT_CLASS_WEIGHT > 1.0

    def test_imagenet_mean_has_three_channels(self):
        """Verify ImageNet mean has RGB channels."""
        assert len(IMAGENET_MEAN) == 3
        assert all(0 <= v <= 1 for v in IMAGENET_MEAN)

    def test_imagenet_std_has_three_channels(self):
        """Verify ImageNet std has RGB channels."""
        assert len(IMAGENET_STD) == 3
        assert all(0 < v <= 1 for v in IMAGENET_STD)


class TestSeedEverything:
    """Test deterministic seeding function."""

    def test_seed_produces_deterministic_numpy(self):
        """Verify NumPy random is deterministic after seeding."""
        seed_everything(42)
        values1 = np.random.rand(10)
        
        seed_everything(42)
        values2 = np.random.rand(10)
        
        np.testing.assert_array_equal(values1, values2)

    def test_seed_produces_deterministic_torch(self):
        """Verify PyTorch random is deterministic after seeding."""
        seed_everything(42)
        tensor1 = torch.rand(10)
        
        seed_everything(42)
        tensor2 = torch.rand(10)
        
        assert torch.allclose(tensor1, tensor2)

    def test_different_seeds_produce_different_values(self):
        """Verify different seeds produce different outputs."""
        seed_everything(42)
        values1 = np.random.rand(10)
        
        seed_everything(123)
        values2 = np.random.rand(10)
        
        assert not np.allclose(values1, values2)


class TestAdditiveGaussianNoise:
    """Test additive Gaussian noise augmentation."""

    def test_noise_with_zero_std_returns_unchanged(self):
        """Verify zero std returns identical tensor."""
        noise_transform = AdditiveGaussianNoise(std=0.0)
        tensor = torch.rand(3, 64, 64)
        
        result = noise_transform(tensor)
        
        assert torch.allclose(tensor, result)

    def test_noise_with_positive_std_modifies_tensor(self):
        """Verify positive std modifies tensor values."""
        noise_transform = AdditiveGaussianNoise(std=0.1)
        tensor = torch.ones(3, 64, 64) * 0.5
        
        torch.manual_seed(42)
        result = noise_transform(tensor)
        
        assert not torch.allclose(tensor, result)

    def test_noise_output_clamped_to_valid_range(self):
        """Verify output is clamped to [0, 1]."""
        noise_transform = AdditiveGaussianNoise(std=0.5)
        tensor = torch.ones(3, 64, 64)  # At upper bound
        
        result = noise_transform(tensor)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_noise_preserves_tensor_shape(self):
        """Verify noise doesn't change tensor shape."""
        noise_transform = AdditiveGaussianNoise(std=0.1)
        tensor = torch.rand(3, 128, 256)
        
        result = noise_transform(tensor)
        
        assert result.shape == tensor.shape

    def test_noise_is_stochastic(self):
        """Verify noise produces different results each call."""
        noise_transform = AdditiveGaussianNoise(std=0.1)
        tensor = torch.ones(3, 64, 64) * 0.5
        
        results = [noise_transform(tensor.clone()) for _ in range(5)]
        
        # At least some results should differ
        all_same = all(torch.allclose(results[0], r) for r in results[1:])
        assert not all_same, "Noise should be stochastic"


class TestConfusionMatrixMulticlass:
    """Test multiclass confusion matrix computation."""

    def test_perfect_prediction_diagonal(self):
        """Verify perfect prediction puts all values on diagonal."""
        true_mask = np.array([[0, 0, 1, 1],
                              [0, 0, 1, 1]], dtype=np.uint8)
        pred_mask = true_mask.copy()
        
        cm = confusion_matrix_multiclass(true_mask, pred_mask, num_classes=2)
        
        # All predictions should be on diagonal
        off_diagonal = cm.sum() - np.trace(cm)
        assert off_diagonal == 0

    def test_confusion_matrix_shape(self):
        """Verify confusion matrix has correct shape."""
        true_mask = np.random.randint(0, 3, (100, 100), dtype=np.uint8)
        pred_mask = np.random.randint(0, 3, (100, 100), dtype=np.uint8)
        
        cm = confusion_matrix_multiclass(true_mask, pred_mask, num_classes=3)
        
        assert cm.shape == (3, 3)

    def test_confusion_matrix_total_equals_pixels(self):
        """Verify confusion matrix sums to total pixel count."""
        true_mask = np.random.randint(0, 2, (50, 60), dtype=np.uint8)
        pred_mask = np.random.randint(0, 2, (50, 60), dtype=np.uint8)
        
        cm = confusion_matrix_multiclass(true_mask, pred_mask, num_classes=2)
        
        assert cm.sum() == 50 * 60

    def test_confusion_matrix_binary_values(self):
        """Verify confusion matrix values for known case."""
        # GT: 4 background, 4 instrument
        # Pred: 3 correct bg, 1 FP, 3 correct inst, 1 FN
        true_mask = np.array([[0, 0, 1, 1],
                              [0, 0, 1, 1]], dtype=np.uint8)
        pred_mask = np.array([[0, 1, 1, 1],  # 1 FP
                              [0, 0, 0, 1]], dtype=np.uint8)  # 1 FN
        
        cm = confusion_matrix_multiclass(true_mask, pred_mask, num_classes=2)
        
        assert cm[0, 0] == 3  # True negatives
        assert cm[0, 1] == 1  # False positives
        assert cm[1, 0] == 1  # False negatives
        assert cm[1, 1] == 3  # True positives


class TestComputeMetricsFromCM:
    """Test metric computation from confusion matrix."""

    def test_perfect_prediction_metrics(self):
        """Verify perfect prediction yields optimal metrics."""
        cm = np.array([[100, 0],
                       [0, 50]], dtype=np.int64)
        
        metrics = compute_metrics_from_cm(cm)
        
        assert np.allclose(metrics['precision'], [1.0, 1.0])
        assert np.allclose(metrics['recall'], [1.0, 1.0])
        assert np.allclose(metrics['iou'], [1.0, 1.0])
        assert np.allclose(metrics['dice'], [1.0, 1.0])
        assert metrics['accuracy'] == pytest.approx(1.0)

    def test_metrics_with_errors(self):
        """Verify metrics correctly reflect errors."""
        # 80 TN, 10 FP, 5 FN, 45 TP
        cm = np.array([[80, 10],
                       [5, 45]], dtype=np.int64)
        
        metrics = compute_metrics_from_cm(cm)
        
        # Instrument class (index 1):
        # Precision = TP / (TP + FP) = 45 / 55 ≈ 0.818
        # Recall = TP / (TP + FN) = 45 / 50 = 0.9
        assert metrics['precision'][1] == pytest.approx(45/55, rel=0.01)
        assert metrics['recall'][1] == pytest.approx(0.9, rel=0.01)
        
        # Accuracy = (TN + TP) / Total = 125 / 140
        assert metrics['accuracy'] == pytest.approx(125/140, rel=0.01)

    def test_iou_calculation(self):
        """Verify IoU calculation matches formula."""
        # Class 1: TP=40, FP=10, FN=10
        cm = np.array([[100, 10],
                       [10, 40]], dtype=np.int64)
        
        metrics = compute_metrics_from_cm(cm)
        
        # IoU = TP / (TP + FP + FN) = 40 / 60 ≈ 0.667
        expected_iou = 40 / (40 + 10 + 10)
        assert metrics['iou'][1] == pytest.approx(expected_iou, rel=0.01)

    def test_dice_calculation(self):
        """Verify Dice calculation matches formula."""
        cm = np.array([[100, 10],
                       [10, 40]], dtype=np.int64)
        
        metrics = compute_metrics_from_cm(cm)
        
        # Dice = 2*TP / (2*TP + FP + FN) = 80 / 100 = 0.8
        expected_dice = 2 * 40 / (2 * 40 + 10 + 10)
        assert metrics['dice'][1] == pytest.approx(expected_dice, rel=0.01)

    def test_support_calculation(self):
        """Verify support (ground truth count) is correct."""
        cm = np.array([[90, 10],
                       [15, 35]], dtype=np.int64)
        
        metrics = compute_metrics_from_cm(cm)
        
        # Support for class 0 = 90 + 10 = 100
        # Support for class 1 = 15 + 35 = 50
        assert metrics['support'][0] == 100
        assert metrics['support'][1] == 50

    def test_handles_zero_predictions(self):
        """Verify metrics handle case with no predictions for a class."""
        cm = np.array([[100, 0],
                       [50, 0]], dtype=np.int64)  # No positive predictions
        
        metrics = compute_metrics_from_cm(cm)
        
        # Precision undefined (0/0), should be 0
        # Recall = 0/50 = 0
        assert metrics['recall'][1] == 0.0

    def test_handles_empty_class(self):
        """Verify metrics handle class with no ground truth."""
        cm = np.array([[0, 0],
                       [0, 150]], dtype=np.int64)  # No background pixels
        
        metrics = compute_metrics_from_cm(cm)
        
        # Background class has no support
        assert metrics['support'][0] == 0


class TestTrainModelFunction:
    """Test the train_model function behavior."""

    def test_train_model_returns_model_and_losses(self):
        """Verify train_model returns trained model and loss history."""
        from surgical_segmentation.models import InstrumentSegmentationModel
        from surgical_segmentation.datasets import SurgicalDataset
        from torch.utils.data import DataLoader
        from torchvision import transforms
        import tempfile
        from pathlib import Path
        from PIL import Image
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            frame_dir = tmp_path / "frames"
            mask_dir = tmp_path / "masks"
            frame_dir.mkdir()
            mask_dir.mkdir()
            
            # Create minimal dataset
            for i in range(4):
                frame = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
                mask = np.zeros((256, 256), dtype=np.uint8)
                mask[100:150, 100:150] = 31
                Image.fromarray(frame).save(frame_dir / f"frame_{i:05d}.png")
                Image.fromarray(mask).save(mask_dir / f"mask_{i:05d}.png")
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            
            dataset = SurgicalDataset(str(frame_dir), str(mask_dir), transform=transform)
            dataloader = DataLoader(dataset, batch_size=2)
            
            model = InstrumentSegmentationModel(num_classes=2)
            
            from surgical_segmentation.training.trainer import train_model
            trained_model, losses = train_model(
                model, dataloader, num_epochs=2, learning_rate=0.001
            )
            
            assert trained_model is not None
            assert isinstance(losses, list)
            assert len(losses) == 2
            assert all(isinstance(l, float) for l in losses)


class TestSyntheticDataGeneration:
    """Test synthetic surgical frame generation."""

    def test_creates_frames_and_masks(self, tmp_path):
        """Verify synthetic generation creates both frames and masks."""
        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        
        result = create_synthetic_surgical_frames(frame_dir, mask_dir, force=True)
        
        assert result is True
        assert frame_dir.exists()
        assert mask_dir.exists()
        assert len(list(frame_dir.glob("*.png"))) == 20
        assert len(list(mask_dir.glob("*.png"))) == 20

    def test_skips_existing_data(self, tmp_path):
        """Verify generation skips when data already exists."""
        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        frame_dir.mkdir()
        mask_dir.mkdir()
        
        # Create existing data
        (frame_dir / "frame_001.png").touch()
        (mask_dir / "mask_001.png").touch()
        
        result = create_synthetic_surgical_frames(frame_dir, mask_dir, force=False)
        
        assert result is False  # Should skip

    def test_force_regenerates_data(self, tmp_path):
        """Verify force=True regenerates even with existing data."""
        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        frame_dir.mkdir()
        mask_dir.mkdir()
        
        # Create minimal existing data
        (frame_dir / "frame_001.png").touch()
        (mask_dir / "mask_001.png").touch()
        
        result = create_synthetic_surgical_frames(frame_dir, mask_dir, force=True)
        
        assert result is True
        assert len(list(frame_dir.glob("*.png"))) == 20

    def test_frames_have_valid_dimensions(self, tmp_path):
        """Verify generated frames have correct dimensions."""
        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        
        create_synthetic_surgical_frames(frame_dir, mask_dir, force=True)
        
        # Check first frame
        from PIL import Image
        frame = Image.open(next(frame_dir.glob("*.png")))
        mask = Image.open(next(mask_dir.glob("*.png")))
        
        assert frame.size == (640, 480)
        assert mask.size == (640, 480)

    def test_masks_contain_instruments(self, tmp_path):
        """Verify some masks contain instrument pixels."""
        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        
        create_synthetic_surgical_frames(frame_dir, mask_dir, force=True)
        
        # Check that at least some masks have instrument pixels
        masks_with_instruments = 0
        for mask_path in mask_dir.glob("*.png"):
            mask = np.array(Image.open(mask_path))
            if (mask == 1).any():
                masks_with_instruments += 1
        
        assert masks_with_instruments > 0


class TestEvaluateModel:
    """Test model evaluation function."""

    def test_evaluate_returns_metrics_dict(self, tmp_path, monkeypatch):
        """Verify evaluate_model returns a metrics dictionary."""
        # Force CPU to avoid CUDA device mismatch in tests
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
        
        from surgical_segmentation.models import InstrumentSegmentationModel
        from surgical_segmentation.datasets import SurgicalDataset
        from torchvision import transforms
        
        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        frame_dir.mkdir()
        mask_dir.mkdir()
        
        # Create test dataset
        for i in range(4):
            frame = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            mask[100:150, 100:150] = 31
            Image.fromarray(frame).save(frame_dir / f"frame_{i:05d}.png")
            Image.fromarray(mask).save(mask_dir / f"mask_{i:05d}.png")
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        dataset = SurgicalDataset(str(frame_dir), str(mask_dir), transform=transform)
        model = InstrumentSegmentationModel(num_classes=2)
        
        metrics = evaluate_model(model, dataset, num_visual_samples=0)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'iou' in metrics
        assert 'dice' in metrics

    def test_evaluate_exports_predictions(self, tmp_path, monkeypatch):
        """Verify evaluate_model exports prediction masks."""
        # Force CPU to avoid CUDA device mismatch in tests
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
        
        from surgical_segmentation.models import InstrumentSegmentationModel
        from surgical_segmentation.datasets import SurgicalDataset
        from torchvision import transforms
        
        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        pred_dir = tmp_path / "preds"
        frame_dir.mkdir()
        mask_dir.mkdir()
        
        # Create test dataset
        for i in range(2):
            frame = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            mask[100:150, 100:150] = 31
            Image.fromarray(frame).save(frame_dir / f"frame_{i:05d}.png")
            Image.fromarray(mask).save(mask_dir / f"mask_{i:05d}.png")
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        dataset = SurgicalDataset(str(frame_dir), str(mask_dir), transform=transform)
        model = InstrumentSegmentationModel(num_classes=2)
        
        evaluate_model(model, dataset, num_visual_samples=0, prediction_dir=pred_dir)
        
        assert pred_dir.exists()
        assert len(list(pred_dir.glob("*.png"))) == 2
