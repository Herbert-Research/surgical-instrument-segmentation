"""Tests for cross-validation module.

Tests dataclasses, configuration, and utility functions used in cross-validation.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from surgical_segmentation.training.cross_validation import (
    FoldResult,
    CrossValidationResult,
    get_video_groups,
    build_transforms,
    NUM_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_SEED,
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
            fold_id=0, val_video="v1", train_videos=["v2"],
            iou_instrument=0.8, dice_instrument=0.85,
            accuracy=0.9, num_train_frames=100, num_val_frames=20
        )
        fold2 = FoldResult(
            fold_id=1, val_video="v2", train_videos=["v1"],
            iou_instrument=0.9, dice_instrument=0.95,
            accuracy=0.95, num_train_frames=100, num_val_frames=20
        )
        
        result = CrossValidationResult(folds=[fold1, fold2])
        
        assert result.mean_iou == pytest.approx(0.85, rel=0.01)

    def test_std_iou_calculation(self):
        """Verify std IoU is calculated correctly."""
        fold1 = FoldResult(
            fold_id=0, val_video="v1", train_videos=["v2"],
            iou_instrument=0.8, dice_instrument=0.85,
            accuracy=0.9, num_train_frames=100, num_val_frames=20
        )
        fold2 = FoldResult(
            fold_id=1, val_video="v2", train_videos=["v1"],
            iou_instrument=0.9, dice_instrument=0.95,
            accuracy=0.95, num_train_frames=100, num_val_frames=20
        )
        
        result = CrossValidationResult(folds=[fold1, fold2])
        
        # std of [0.8, 0.9] = 0.05
        assert result.std_iou == pytest.approx(0.05, rel=0.01)

    def test_mean_dice_calculation(self):
        """Verify mean Dice is calculated correctly."""
        fold1 = FoldResult(
            fold_id=0, val_video="v1", train_videos=["v2"],
            iou_instrument=0.8, dice_instrument=0.80,
            accuracy=0.9, num_train_frames=100, num_val_frames=20
        )
        fold2 = FoldResult(
            fold_id=1, val_video="v2", train_videos=["v1"],
            iou_instrument=0.9, dice_instrument=0.90,
            accuracy=0.95, num_train_frames=100, num_val_frames=20
        )
        
        result = CrossValidationResult(folds=[fold1, fold2])
        
        assert result.mean_dice == pytest.approx(0.85, rel=0.01)

    def test_to_dict_structure(self):
        """Verify to_dict returns correct structure."""
        fold = FoldResult(
            fold_id=0, val_video="video01", train_videos=["video02"],
            iou_instrument=0.85, dice_instrument=0.90,
            accuracy=0.95, num_train_frames=100, num_val_frames=25
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
            fold_id=0, val_video="video01", train_videos=["video02"],
            iou_instrument=0.85, dice_instrument=0.90,
            accuracy=0.95, num_train_frames=100, num_val_frames=25
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
        from PIL import Image
        import numpy as np
        
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
        import torch
        from PIL import Image
        import numpy as np
        
        train_tf, val_tf = build_transforms()
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        
        train_result = train_tf(img)
        val_result = val_tf(img)
        
        assert isinstance(train_result, torch.Tensor)
        assert isinstance(val_result, torch.Tensor)

    def test_transforms_produce_correct_shape(self):
        """Verify transforms produce (3, 256, 256) tensors."""
        from PIL import Image
        import numpy as np
        
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
