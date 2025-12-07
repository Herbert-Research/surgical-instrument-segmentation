"""Training utilities for the surgical segmentation package."""

from surgical_segmentation.training.cross_validation import (
    CrossValidationResult,
    FoldResult,
    leave_one_video_out_cv,
)

__all__ = ["CrossValidationResult", "FoldResult", "leave_one_video_out_cv"]
