"""Evaluation utilities for the surgical segmentation package."""

from surgical_segmentation.evaluation.statistics import (
    ComparisonResult,
    bootstrap_ci,
    compute_cohens_d,
    interpret_effect_size,
    paired_comparison,
    paired_ttest,
    print_significance_report,
)

__all__ = [
    "ComparisonResult",
    "bootstrap_ci",
    "compute_cohens_d",
    "interpret_effect_size",
    "paired_comparison",
    "paired_ttest",
    "print_significance_report",
]
