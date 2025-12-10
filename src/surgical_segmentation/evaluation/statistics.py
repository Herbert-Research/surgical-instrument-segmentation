"""Statistical significance testing for model comparison."""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_ttest(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """Perform a paired t-test between two matched score lists."""

    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have equal length")

    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    is_significant = bool(p_value < alpha)

    return float(t_stat), float(p_value), is_significant


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for the mean."""

    if not scores:
        raise ValueError("Scores list is empty")

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


def print_significance_report(
    model_a_name: str,
    model_b_name: str,
    scores_a: list[float],
    scores_b: list[float],
    metric_name: str = "IoU",
) -> None:
    """Print a formatted statistical significance report."""

    mean_a, lower_a, upper_a = bootstrap_ci(scores_a)
    mean_b, lower_b, upper_b = bootstrap_ci(scores_b)
    t_stat, p_value, is_sig = paired_ttest(scores_a, scores_b)

    print("\n" + "=" * 60)
    print(f"STATISTICAL COMPARISON: {metric_name}")
    print("=" * 60)
    print(f"\n{model_a_name}:")
    print(f"  Mean {metric_name}: {mean_a:.4f}")
    print(f"  95% CI: [{lower_a:.4f}, {upper_a:.4f}]")
    print(f"\n{model_b_name}:")
    print(f"  Mean {metric_name}: {mean_b:.4f}")
    print(f"  95% CI: [{lower_b:.4f}, {upper_b:.4f}]")
    print("\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant at α=0.05: {'YES' if is_sig else 'NO'}")

    if is_sig:
        better = model_a_name if mean_a > mean_b else model_b_name
        print(f"\n  → {better} is significantly better (p < 0.05)")
    else:
        print("\n  → No significant difference between models")
