"""Statistical significance testing for model comparison.

This module provides comprehensive statistical analysis tools for comparing
machine learning model performance, including:
- Paired t-tests for model comparison
- Bootstrap confidence intervals
- Effect size calculations (Cohen's d)
- Comprehensive comparison result dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    """Results of statistical comparison between two models.

    This dataclass encapsulates all statistical metrics needed for
    rigorous model comparison in scientific publications.

    Attributes:
        model_a_name: Display name for model A
        model_b_name: Display name for model B
        metric_name: Name of the metric being compared (e.g., "IoU", "Dice")
        model_a_mean: Mean score for model A
        model_a_std: Standard deviation for model A
        model_a_scores: Raw per-fold scores for model A
        model_b_mean: Mean score for model B
        model_b_std: Standard deviation for model B
        model_b_scores: Raw per-fold scores for model B
        t_statistic: t-statistic from paired t-test
        p_value: p-value from paired t-test
        effect_size: Cohen's d effect size
        is_significant: Whether difference is statistically significant
        alpha: Significance level used (default 0.05)
        ci_a: 95% confidence interval for model A (lower, upper)
        ci_b: 95% confidence interval for model B (lower, upper)
    """

    model_a_name: str
    model_b_name: str
    metric_name: str
    model_a_mean: float
    model_a_std: float
    model_a_scores: list[float] = field(default_factory=list)
    model_b_mean: float = 0.0
    model_b_std: float = 0.0
    model_b_scores: list[float] = field(default_factory=list)
    t_statistic: float = 0.0
    p_value: float = 1.0
    effect_size: float = 0.0
    is_significant: bool = False
    alpha: float = 0.05
    ci_a: tuple[float, float] = (0.0, 0.0)
    ci_b: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "comparison": f"{self.model_a_name} vs {self.model_b_name}",
            "metric": self.metric_name,
            "model_a": {
                "name": self.model_a_name,
                "mean": round(self.model_a_mean, 4),
                "std": round(self.model_a_std, 4),
                "ci_95": [round(self.ci_a[0], 4), round(self.ci_a[1], 4)],
                "scores": [round(s, 4) for s in self.model_a_scores],
            },
            "model_b": {
                "name": self.model_b_name,
                "mean": round(self.model_b_mean, 4),
                "std": round(self.model_b_std, 4),
                "ci_95": [round(self.ci_b[0], 4), round(self.ci_b[1], 4)],
                "scores": [round(s, 4) for s in self.model_b_scores],
            },
            "statistical_test": {
                "test_type": "paired_t_test",
                "t_statistic": round(self.t_statistic, 4),
                "p_value": round(self.p_value, 6),
                "alpha": self.alpha,
                "is_significant": self.is_significant,
            },
            "effect_size": {
                "cohens_d": round(self.effect_size, 4),
                "interpretation": interpret_effect_size(self.effect_size),
            },
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        if self.model_a_mean > self.model_b_mean:
            better = self.model_a_name
        else:
            better = self.model_b_name

        if self.is_significant:
            sig_text = "statistically significant"
        else:
            sig_text = "not statistically significant"

        ci_a_str = f"[{self.ci_a[0]:.4f}, {self.ci_a[1]:.4f}]"
        ci_b_str = f"[{self.ci_b[0]:.4f}, {self.ci_b[1]:.4f}]"
        effect_interp = interpret_effect_size(self.effect_size)

        lines = [
            f"{self.metric_name} Comparison:",
            f"  {self.model_a_name}: {self.model_a_mean:.4f} ± {self.model_a_std:.4f} "
            f"(95% CI: {ci_a_str})",
            f"  {self.model_b_name}: {self.model_b_mean:.4f} ± {self.model_b_std:.4f} "
            f"(95% CI: {ci_b_str})",
            f"  Difference is {sig_text} (p={self.p_value:.6f}, α={self.alpha})",
            f"  Effect size: {self.effect_size:.4f} ({effect_interp})",
        ]

        if self.is_significant:
            lines.append(f"  → {better} performs better")
        else:
            lines.append("  → No conclusive winner")

        return "\n".join(lines)


def interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size using standard thresholds.

    Args:
        cohens_d: Cohen's d value (can be negative)

    Returns:
        Human-readable interpretation of the effect size

    References:
        Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    """
    d = abs(cohens_d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def compute_cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """Compute Cohen's d effect size for paired samples.

    For paired samples, we compute Cohen's d as the mean of differences
    divided by the standard deviation of differences.

    Args:
        scores_a: Scores from model A (per-fold or per-sample)
        scores_b: Scores from model B (matched with scores_a)

    Returns:
        Cohen's d effect size value

    Raises:
        ValueError: If score lists have different lengths
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have equal length for paired comparison")

    scores_a_arr = np.array(scores_a)
    scores_b_arr = np.array(scores_b)
    diff = scores_a_arr - scores_b_arr

    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0

    return float(np.mean(diff) / std_diff)


def paired_ttest(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """Perform a paired t-test between two matched score lists.

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B (matched by fold/sample)
        alpha: Significance level (default 0.05)

    Returns:
        Tuple of (t_statistic, p_value, is_significant)

    Raises:
        ValueError: If score lists have different lengths
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have equal length")

    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    is_significant = bool(p_value < alpha)

    return float(t_stat), float(p_value), is_significant


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for the mean.

    Args:
        scores: List of scores to compute CI for
        n_bootstrap: Number of bootstrap samples (default 10,000)
        confidence: Confidence level (default 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (mean, lower_bound, upper_bound)

    Raises:
        ValueError: If scores list is empty
    """
    if not scores:
        raise ValueError("Scores list is empty")

    if random_state is not None:
        np.random.seed(random_state)

    scores_arr = np.array(scores, dtype=float)
    n = len(scores_arr)

    bootstrap_means = np.array(
        [np.mean(np.random.choice(scores_arr, size=n, replace=True)) for _ in range(n_bootstrap)]
    )

    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    mean = float(np.mean(scores_arr))

    return mean, lower, upper


def paired_comparison(
    scores_a: list[float],
    scores_b: list[float],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    metric_name: str = "IoU",
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
) -> ComparisonResult:
    """Perform comprehensive paired comparison between two models.

    This is the main entry point for model comparison. It computes:
    - Mean and standard deviation for both models
    - 95% confidence intervals via bootstrap
    - Paired t-test for statistical significance
    - Cohen's d effect size

    Args:
        scores_a: Per-fold/per-sample scores for model A
        scores_b: Per-fold/per-sample scores for model B (matched)
        model_a_name: Display name for model A
        model_b_name: Display name for model B
        metric_name: Name of the metric (e.g., "IoU", "Dice")
        alpha: Significance level (default 0.05)
        n_bootstrap: Number of bootstrap samples for CI

    Returns:
        ComparisonResult with all statistical metrics

    Raises:
        ValueError: If score arrays have different lengths

    Example:
        >>> unet_ious = [0.78, 0.82, 0.79, 0.81, 0.80]
        >>> deeplab_ious = [0.87, 0.88, 0.86, 0.89, 0.87]
        >>> result = paired_comparison(
        ...     deeplab_ious, unet_ious,
        ...     model_a_name="DeepLabV3",
        ...     model_b_name="U-Net",
        ...     metric_name="IoU"
        ... )
        >>> print(result.summary())
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length for paired test")

    scores_a_arr = np.array(scores_a)
    scores_b_arr = np.array(scores_b)

    # Compute means and stds
    mean_a = float(np.mean(scores_a_arr))
    std_a = float(np.std(scores_a_arr, ddof=1))
    mean_b = float(np.mean(scores_b_arr))
    std_b = float(np.std(scores_b_arr, ddof=1))

    # Bootstrap confidence intervals
    _, lower_a, upper_a = bootstrap_ci(scores_a, n_bootstrap=n_bootstrap)
    _, lower_b, upper_b = bootstrap_ci(scores_b, n_bootstrap=n_bootstrap)

    # Paired t-test
    t_stat, p_value, is_sig = paired_ttest(scores_a, scores_b, alpha=alpha)

    # Effect size
    effect_size = compute_cohens_d(scores_a, scores_b)

    return ComparisonResult(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        metric_name=metric_name,
        model_a_mean=mean_a,
        model_a_std=std_a,
        model_a_scores=list(scores_a),
        model_b_mean=mean_b,
        model_b_std=std_b,
        model_b_scores=list(scores_b),
        t_statistic=t_stat,
        p_value=p_value,
        effect_size=effect_size,
        is_significant=is_sig,
        alpha=alpha,
        ci_a=(lower_a, upper_a),
        ci_b=(lower_b, upper_b),
    )


def print_significance_report(
    model_a_name: str,
    model_b_name: str,
    scores_a: list[float],
    scores_b: list[float],
    metric_name: str = "IoU",
) -> None:
    """Print a formatted statistical significance report.

    Args:
        model_a_name: Display name for model A
        model_b_name: Display name for model B
        scores_a: Per-fold scores for model A
        scores_b: Per-fold scores for model B
        metric_name: Name of the metric being compared
    """
    result = paired_comparison(
        scores_a=scores_a,
        scores_b=scores_b,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        metric_name=metric_name,
    )

    print("\n" + "=" * 60)
    print(f"STATISTICAL COMPARISON: {metric_name}")
    print("=" * 60)
    print(f"\n{model_a_name}:")
    print(f"  Mean {metric_name}: {result.model_a_mean:.4f}")
    print(f"  Std Dev: {result.model_a_std:.4f}")
    print(f"  95% CI: [{result.ci_a[0]:.4f}, {result.ci_a[1]:.4f}]")
    print(f"\n{model_b_name}:")
    print(f"  Mean {metric_name}: {result.model_b_mean:.4f}")
    print(f"  Std Dev: {result.model_b_std:.4f}")
    print(f"  95% CI: [{result.ci_b[0]:.4f}, {result.ci_b[1]:.4f}]")
    print("\nPaired t-test:")
    print(f"  t-statistic: {result.t_statistic:.4f}")
    print(f"  p-value: {result.p_value:.6f}")
    print(f"  Significant at α=0.05: {'YES' if result.is_significant else 'NO'}")
    effect_interp = interpret_effect_size(result.effect_size)
    print(f"\nEffect Size (Cohen's d): {result.effect_size:.4f} ({effect_interp})")

    if result.is_significant:
        better = model_a_name if result.model_a_mean > result.model_b_mean else model_b_name
        print(f"\n  → {better} is significantly better (p < 0.05)")
    else:
        print("\n  → No significant difference between models")
