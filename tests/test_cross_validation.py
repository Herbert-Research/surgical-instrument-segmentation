"""Tests for cross-validation module.

Tests dataclasses, configuration, and utility functions used in cross-validation.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from surgical_segmentation.training.cross_validation import (
    DEFAULT_SEED,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    CrossValidationResult,
    FoldResult,
    build_transforms,
    get_video_groups,
)


class TestFoldResult:
    """Test FoldResult dataclass."""

    def test_creates_valid_fold_result(self):
        """Verify FoldResult can be instantiated with valid data."""
        result = FoldResult(
            fold_id=1,
            val_video="video01",
            train_videos=["video02", "video03"],
            iou_instrument=0.85,
            dice_instrument=0.91,
            accuracy=0.95,
            num_train_frames=100,
            num_val_frames=20,
        )

        assert result.fold_id == 1
        assert result.val_video == "video01"
        assert len(result.train_videos) == 2
        assert result.iou_instrument == 0.85
        assert result.dice_instrument == 0.91

    def test_stores_train_videos_list(self):
        """Verify train_videos is stored as list."""
        result = FoldResult(
            fold_id=0,
            val_video="video01",
            train_videos=["video02", "video03", "video04"],
            iou_instrument=0.8,
            dice_instrument=0.85,
            accuracy=0.9,
            num_train_frames=200,
            num_val_frames=50,
        )

        assert isinstance(result.train_videos, list)
        assert "video02" in result.train_videos


class TestCrossValidationResult:
    """Test CrossValidationResult dataclass."""

    def test_empty_result_has_zero_metrics(self):
        """Verify empty result returns zero metrics."""
        result = CrossValidationResult()

        assert result.mean_iou == 0.0
        assert result.std_iou == 0.0
        assert result.mean_dice == 0.0
        assert result.std_dice == 0.0

    def test_mean_iou_calculation(self):
        """Verify mean IoU is calculated correctly."""
        fold1 = FoldResult(
            fold_id=0,
            val_video="v1",
            train_videos=["v2"],
            iou_instrument=0.8,
            dice_instrument=0.85,
            accuracy=0.9,
            num_train_frames=100,
            num_val_frames=20,
        )
        fold2 = FoldResult(
            fold_id=1,
            val_video="v2",
            train_videos=["v1"],
            iou_instrument=0.9,
            dice_instrument=0.95,
            accuracy=0.95,
            num_train_frames=100,
            num_val_frames=20,
        )

        result = CrossValidationResult(folds=[fold1, fold2])

        assert result.mean_iou == pytest.approx(0.85, rel=0.01)

    def test_std_iou_calculation(self):
        """Verify std IoU is calculated correctly."""
        fold1 = FoldResult(
            fold_id=0,
            val_video="v1",
            train_videos=["v2"],
            iou_instrument=0.8,
            dice_instrument=0.85,
            accuracy=0.9,
            num_train_frames=100,
            num_val_frames=20,
        )
        fold2 = FoldResult(
            fold_id=1,
            val_video="v2",
            train_videos=["v1"],
            iou_instrument=0.9,
            dice_instrument=0.95,
            accuracy=0.95,
            num_train_frames=100,
            num_val_frames=20,
        )

        result = CrossValidationResult(folds=[fold1, fold2])

        # std of [0.8, 0.9] = 0.05
        assert result.std_iou == pytest.approx(0.05, rel=0.01)

    def test_mean_dice_calculation(self):
        """Verify mean Dice is calculated correctly."""
        fold1 = FoldResult(
            fold_id=0,
            val_video="v1",
            train_videos=["v2"],
            iou_instrument=0.8,
            dice_instrument=0.80,
            accuracy=0.9,
            num_train_frames=100,
            num_val_frames=20,
        )
        fold2 = FoldResult(
            fold_id=1,
            val_video="v2",
            train_videos=["v1"],
            iou_instrument=0.9,
            dice_instrument=0.90,
            accuracy=0.95,
            num_train_frames=100,
            num_val_frames=20,
        )

        result = CrossValidationResult(folds=[fold1, fold2])

        assert result.mean_dice == pytest.approx(0.85, rel=0.01)

    def test_to_dict_structure(self):
        """Verify to_dict returns correct structure."""
        fold = FoldResult(
            fold_id=0,
            val_video="video01",
            train_videos=["video02"],
            iou_instrument=0.85,
            dice_instrument=0.90,
            accuracy=0.95,
            num_train_frames=100,
            num_val_frames=25,
        )
        result = CrossValidationResult(folds=[fold])

        d = result.to_dict()

        assert "summary" in d
        assert "folds" in d
        assert d["summary"]["mean_iou"] == pytest.approx(0.85)
        assert d["summary"]["num_folds"] == 1
        assert len(d["folds"]) == 1

    def test_save_creates_valid_json(self, tmp_path):
        """Verify save creates valid JSON file."""
        fold = FoldResult(
            fold_id=0,
            val_video="video01",
            train_videos=["video02"],
            iou_instrument=0.85,
            dice_instrument=0.90,
            accuracy=0.95,
            num_train_frames=100,
            num_val_frames=25,
        )
        result = CrossValidationResult(folds=[fold])

        save_path = tmp_path / "cv_results.json"
        result.save(save_path)

        # Verify file exists and contains valid JSON
        assert save_path.exists()
        with open(save_path) as f:
            loaded = json.load(f)

        assert loaded["summary"]["mean_iou"] == pytest.approx(0.85)


class TestGetVideoGroups:
    """Test video grouping function."""

    def test_groups_frames_by_video(self, tmp_path):
        """Verify frames are grouped by video ID."""
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        # Create frames for two videos
        for i in range(3):
            (frame_dir / f"video01_frame_{i:06d}.png").touch()
        for i in range(2):
            (frame_dir / f"video02_frame_{i:06d}.png").touch()

        groups = get_video_groups(frame_dir)

        assert "video01" in groups
        assert "video02" in groups
        assert len(groups["video01"]) == 3
        assert len(groups["video02"]) == 2

    def test_raises_on_missing_directory(self, tmp_path):
        """Verify FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            get_video_groups(tmp_path / "nonexistent")

    def test_raises_on_empty_directory(self, tmp_path):
        """Verify FileNotFoundError for empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            get_video_groups(empty_dir)

    def test_returns_sorted_filenames(self, tmp_path):
        """Verify filenames are sorted within each group."""
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        # Create frames out of order
        (frame_dir / "video01_frame_000003.png").touch()
        (frame_dir / "video01_frame_000001.png").touch()
        (frame_dir / "video01_frame_000002.png").touch()

        groups = get_video_groups(frame_dir)

        # Frames should be sorted
        filenames = groups["video01"]
        assert filenames == sorted(filenames)


class TestBuildTransforms:
    """Test transform building function."""

    def test_returns_two_transforms(self):
        """Verify function returns tuple of two transforms."""
        train_tf, val_tf = build_transforms()

        assert train_tf is not None
        assert val_tf is not None

    def test_transforms_are_callable(self):
        """Verify transforms can be called."""
        train_tf, val_tf = build_transforms()

        # Create dummy image
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

        # Should not raise
        train_result = train_tf(img)
        val_result = val_tf(img)

        assert train_result is not None
        assert val_result is not None

    def test_transforms_produce_tensors(self):
        """Verify transforms produce tensors."""
        train_tf, val_tf = build_transforms()
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

        train_result = train_tf(img)
        val_result = val_tf(img)

        assert isinstance(train_result, torch.Tensor)
        assert isinstance(val_result, torch.Tensor)

    def test_transforms_produce_correct_shape(self):
        """Verify transforms produce (3, 256, 256) tensors."""
        train_tf, val_tf = build_transforms()
        # Input different size
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

        train_result = train_tf(img)
        val_result = val_tf(img)

        assert train_result.shape == (3, 256, 256)
        assert val_result.shape == (3, 256, 256)


class TestConstants:
    """Test module constants."""

    def test_num_classes_binary(self):
        """Verify NUM_CLASSES is 2 for binary segmentation."""
        assert NUM_CLASSES == 2

    def test_imagenet_mean_valid(self):
        """Verify ImageNet mean values are valid."""
        assert len(IMAGENET_MEAN) == 3
        assert all(0 <= v <= 1 for v in IMAGENET_MEAN)

    def test_imagenet_std_valid(self):
        """Verify ImageNet std values are valid."""
        assert len(IMAGENET_STD) == 3
        assert all(0 < v <= 1 for v in IMAGENET_STD)

    def test_default_seed_is_integer(self):
        """Verify DEFAULT_SEED is an integer."""
        assert isinstance(DEFAULT_SEED, int)


class TestCrossValidationResultSave:
    """Test CrossValidationResult.save method."""

    def test_save_creates_json_file(self, tmp_path):
        """Verify save() creates a valid JSON file."""
        result = CrossValidationResult()
        result.folds.append(
            FoldResult(
                fold_id=0,
                val_video="video01",
                train_videos=["video02", "video03"],
                iou_instrument=0.85,
                dice_instrument=0.91,
                accuracy=0.95,
                num_train_frames=100,
                num_val_frames=20,
            )
        )

        output_path = tmp_path / "cv_results.json"
        result.save(output_path)

        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert "summary" in loaded
        assert "folds" in loaded
        assert loaded["summary"]["mean_iou"] == 0.85

    def test_save_empty_result(self, tmp_path):
        """Verify save() handles empty results."""
        result = CrossValidationResult()
        output_path = tmp_path / "empty_results.json"
        result.save(output_path)

        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert loaded["summary"]["mean_iou"] == 0.0
        assert loaded["summary"]["num_folds"] == 0


class TestGetVideoGroupsEdgeCases:
    """Test edge cases for get_video_groups function."""

    def test_nonexistent_directory_raises(self, tmp_path):
        """Verify FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            get_video_groups(tmp_path / "nonexistent")

    def test_empty_directory_raises(self, tmp_path):
        """Verify FileNotFoundError for directory with no PNGs."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No PNG frames found"):
            get_video_groups(empty_dir)

    def test_groups_frames_correctly(self, tmp_path):
        """Verify frames are grouped by video ID."""
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        # Create frames from multiple videos
        (frame_dir / "video01_frame_000001.png").touch()
        (frame_dir / "video01_frame_000002.png").touch()
        (frame_dir / "video02_frame_000001.png").touch()

        groups = get_video_groups(frame_dir)

        assert "video01" in groups
        assert "video02" in groups
        assert len(groups["video01"]) == 2
        assert len(groups["video02"]) == 1


class TestSeedEverythingCrossValidation:
    """Test seed_everything function from cross_validation module."""

    def test_seed_makes_numpy_deterministic(self):
        """Verify seeding makes numpy random deterministic."""
        from surgical_segmentation.training.cross_validation import seed_everything

        seed_everything(123)
        first = np.random.rand(10).tolist()

        seed_everything(123)
        second = np.random.rand(10).tolist()

        assert first == second

    def test_seed_makes_torch_deterministic(self):
        """Verify seeding makes torch random deterministic."""
        from surgical_segmentation.training.cross_validation import seed_everything

        seed_everything(456)
        first = torch.rand(10).tolist()

        seed_everything(456)
        second = torch.rand(10).tolist()

        assert first == second


class TestBuildModel:
    """Test build_model function."""

    def test_build_deeplabv3(self):
        """Verify building DeepLabV3 model."""
        from surgical_segmentation.training.cross_validation import build_model

        model = build_model("deeplabv3", num_classes=2)
        assert model is not None

    def test_build_unet(self):
        """Verify building UNet model."""
        from surgical_segmentation.training.cross_validation import build_model

        model = build_model("unet", num_classes=2)
        assert model is not None

    def test_invalid_model_class_raises(self):
        """Verify invalid model class raises ValueError."""
        from surgical_segmentation.training.cross_validation import build_model

        with pytest.raises(ValueError, match="Unsupported"):
            build_model("invalid_model")


class TestBuildDataloaders:
    """Test build_dataloaders function."""

    def test_returns_two_dataloaders(self, tmp_path):
        """Verify function returns train and val dataloaders."""
        from surgical_segmentation.training.cross_validation import build_dataloaders

        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        frame_dir.mkdir()
        mask_dir.mkdir()

        # Create sample data
        for i in range(4):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            mask[100:150, 100:150] = 31
            Image.fromarray(frame).save(frame_dir / f"video01_frame_{i:06d}.png")
            Image.fromarray(mask).save(mask_dir / f"video01_mask_{i:06d}.png")

        train_frames = [f"video01_frame_{i:06d}.png" for i in range(2)]
        val_frames = [f"video01_frame_{i:06d}.png" for i in range(2, 4)]

        train_loader, val_loader = build_dataloaders(
            frame_dir=frame_dir,
            mask_dir=mask_dir,
            train_frames=train_frames,
            val_frames=val_frames,
            batch_size=2,
            num_workers=0,
            device="cpu",
        )

        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader.dataset) == 2
        assert len(val_loader.dataset) == 2


class TestTrainOneEpoch:
    """Test train_one_epoch function."""

    def test_returns_float_loss(self, tmp_path):
        """Verify train_one_epoch returns float loss value."""
        from surgical_segmentation.training.cross_validation import (
            build_dataloaders,
            build_model,
            train_one_epoch,
        )

        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        frame_dir.mkdir()
        mask_dir.mkdir()

        for i in range(4):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            mask[100:150, 100:150] = 31
            Image.fromarray(frame).save(frame_dir / f"video01_frame_{i:06d}.png")
            Image.fromarray(mask).save(mask_dir / f"video01_mask_{i:06d}.png")

        train_frames = [f"video01_frame_{i:06d}.png" for i in range(4)]

        train_loader, _ = build_dataloaders(
            frame_dir=frame_dir,
            mask_dir=mask_dir,
            train_frames=train_frames,
            val_frames=train_frames[:1],
            batch_size=2,
            num_workers=0,
            device="cpu",
        )

        model = build_model("unet", num_classes=2)
        device = torch.device("cpu")
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            desc="Test",
        )

        assert isinstance(loss, float)
        assert loss >= 0

    def test_handles_deeplabv3_dict_output(self, tmp_path):
        """Verify train_one_epoch handles DeepLabV3 dict output."""
        from surgical_segmentation.training.cross_validation import (
            build_dataloaders,
            build_model,
            train_one_epoch,
        )

        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        frame_dir.mkdir()
        mask_dir.mkdir()

        for i in range(4):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            mask[100:150, 100:150] = 1
            Image.fromarray(frame).save(frame_dir / f"video01_frame_{i:06d}.png")
            Image.fromarray(mask).save(mask_dir / f"video01_mask_{i:06d}.png")

        train_frames = [f"video01_frame_{i:06d}.png" for i in range(4)]

        train_loader, _ = build_dataloaders(
            frame_dir=frame_dir,
            mask_dir=mask_dir,
            train_frames=train_frames,
            val_frames=train_frames[:1],
            batch_size=2,
            num_workers=0,
            device="cpu",
        )

        # Use DeepLabV3 which returns dict output
        model = build_model("deeplabv3", num_classes=2)
        device = torch.device("cpu")
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            desc="Test",
        )

        assert isinstance(loss, float)
        assert loss >= 0


class TestEvaluateModelCV:
    """Test evaluate_model function from cross_validation module."""

    def test_returns_metrics_dict(self, tmp_path):
        """Verify evaluate_model returns metrics dictionary."""
        from surgical_segmentation.training.cross_validation import (
            build_dataloaders,
            build_model,
            evaluate_model,
        )

        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        frame_dir.mkdir()
        mask_dir.mkdir()

        for i in range(4):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            mask[100:150, 100:150] = 31
            Image.fromarray(frame).save(frame_dir / f"video01_frame_{i:06d}.png")
            Image.fromarray(mask).save(mask_dir / f"video01_mask_{i:06d}.png")

        val_frames = [f"video01_frame_{i:06d}.png" for i in range(4)]

        _, val_loader = build_dataloaders(
            frame_dir=frame_dir,
            mask_dir=mask_dir,
            train_frames=val_frames[:2],
            val_frames=val_frames[2:],
            batch_size=2,
            num_workers=0,
            device="cpu",
        )

        model = build_model("unet", num_classes=2)
        device = torch.device("cpu")
        model = model.to(device)

        metrics = evaluate_model(model, val_loader, device, num_classes=2)

        assert "accuracy" in metrics
        assert "iou" in metrics
        assert "dice" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_with_deeplabv3(self, tmp_path):
        """Verify evaluate_model handles DeepLabV3 dict output."""
        from surgical_segmentation.training.cross_validation import (
            build_dataloaders,
            build_model,
            evaluate_model,
        )

        frame_dir = tmp_path / "frames"
        mask_dir = tmp_path / "masks"
        frame_dir.mkdir()
        mask_dir.mkdir()

        for i in range(4):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            mask[100:150, 100:150] = 1
            Image.fromarray(frame).save(frame_dir / f"video01_frame_{i:06d}.png")
            Image.fromarray(mask).save(mask_dir / f"video01_mask_{i:06d}.png")

        val_frames = [f"video01_frame_{i:06d}.png" for i in range(4)]

        _, val_loader = build_dataloaders(
            frame_dir=frame_dir,
            mask_dir=mask_dir,
            train_frames=val_frames[:2],
            val_frames=val_frames[2:],
            batch_size=2,
            num_workers=0,
            device="cpu",
        )

        # Use DeepLabV3 which returns dict output
        model = build_model("deeplabv3", num_classes=2)
        device = torch.device("cpu")
        model = model.to(device)

        metrics = evaluate_model(model, val_loader, device, num_classes=2)

        assert "iou_instrument" in metrics
        assert "dice_instrument" in metrics


class TestConfusionMatrixMulticlassCV:
    """Test confusion_matrix_multiclass from cross_validation module."""

    def test_perfect_prediction(self):
        """Verify perfect prediction has diagonal matrix."""
        from surgical_segmentation.training.cross_validation import confusion_matrix_multiclass

        true = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        pred = np.array([[0, 0], [1, 1]], dtype=np.uint8)

        cm = confusion_matrix_multiclass(true, pred, num_classes=2)

        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[0, 1] == 0
        assert cm[1, 0] == 0


class TestComputeMetricsFromCMCV:
    """Test compute_metrics_from_cm from cross_validation module."""

    def test_returns_all_metrics(self):
        """Verify function returns all expected metrics."""
        from surgical_segmentation.training.cross_validation import compute_metrics_from_cm

        cm = np.array([[50, 10], [5, 35]], dtype=np.int64)

        metrics = compute_metrics_from_cm(cm)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "iou" in metrics
        assert "dice" in metrics
