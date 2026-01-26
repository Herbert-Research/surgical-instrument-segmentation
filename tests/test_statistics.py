"""Tests for statistical significance testing functions."""

import json

import numpy as np
import pytest

from surgical_segmentation.evaluation.statistics import (
    ComparisonResult,
    bootstrap_ci,
    compute_cohens_d,
    interpret_effect_size,
    paired_comparison,
    paired_ttest,
    print_significance_report,
)


class TestPairedTTest:
    """Test paired t-test function."""

    def test_very_similar_scores_not_significant(self):
        """Verify very similar scores produce non-significant result."""
        # Slightly different scores to avoid NaN from zero variance
        scores_a = [0.80, 0.85, 0.90, 0.82, 0.88]
        scores_b = [0.81, 0.84, 0.91, 0.83, 0.87]

        t_stat, p_value, is_sig = paired_ttest(scores_a, scores_b)

        assert p_value > 0.05
        assert is_sig is False

    def test_clearly_different_scores_significant(self):
        """Verify clearly different scores produce significant result."""
        scores_a = [0.9, 0.91, 0.92, 0.89, 0.93]
        scores_b = [0.5, 0.51, 0.49, 0.52, 0.48]

        t_stat, p_value, is_sig = paired_ttest(scores_a, scores_b)

        assert p_value < 0.05
        assert is_sig is True

    def test_returns_float_types(self):
        """Verify return types are Python floats."""
        scores_a = [0.8, 0.85, 0.9]
        scores_b = [0.74, 0.8, 0.86]

        t_stat, p_value, is_sig = paired_ttest(scores_a, scores_b)

        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert isinstance(is_sig, bool)

    def test_unequal_length_raises_error(self):
        """Verify unequal score lists raise ValueError."""
        scores_a = [0.8, 0.85, 0.9]
        scores_b = [0.75, 0.8]

        with pytest.raises(ValueError, match="equal length"):
            paired_ttest(scores_a, scores_b)

    def test_custom_alpha_level(self):
        """Verify custom alpha level changes significance."""
        # Scores with p-value around 0.08
        np.random.seed(42)
        scores_a = np.random.normal(0.8, 0.05, 10).tolist()
        deltas = np.linspace(0.015, 0.025, num=10)
        scores_b = (np.array(scores_a) - deltas).tolist()

        # At alpha=0.05, may not be significant
        _, p_value, _ = paired_ttest(scores_a, scores_b, alpha=0.05)

        # At alpha=0.10, same p-value but different significance
        _, _, is_sig_01 = paired_ttest(scores_a, scores_b, alpha=0.10)
        _, _, is_sig_001 = paired_ttest(scores_a, scores_b, alpha=0.001)

        # Same p-value, different alpha changes decision
        if p_value < 0.10:
            assert is_sig_01 is True
        if p_value > 0.001:
            assert is_sig_001 is False

    def test_positive_t_stat_when_a_greater(self):
        """Verify positive t-stat when scores_a > scores_b."""
        scores_a = [0.9, 0.92, 0.91, 0.93, 0.88]
        scores_b = [0.7, 0.73, 0.69, 0.74, 0.68]

        t_stat, _, _ = paired_ttest(scores_a, scores_b)

        assert t_stat > 0

    def test_negative_t_stat_when_b_greater(self):
        """Verify negative t-stat when scores_b > scores_a."""
        scores_a = [0.7, 0.73, 0.69, 0.74, 0.68]
        scores_b = [0.9, 0.92, 0.91, 0.93, 0.88]

        t_stat, _, _ = paired_ttest(scores_a, scores_b)

        assert t_stat < 0


class TestBootstrapCI:
    """Test bootstrap confidence interval function."""

    def test_mean_within_ci(self):
        """Verify mean is within confidence interval."""
        scores = [0.8, 0.82, 0.85, 0.83, 0.81, 0.84]

        mean, lower, upper = bootstrap_ci(scores)

        assert lower <= mean <= upper

    def test_ci_contains_true_mean(self):
        """Verify CI usually contains true mean for normal data."""
        np.random.seed(42)
        true_mean = 0.8
        scores = np.random.normal(true_mean, 0.05, 30).tolist()

        mean, lower, upper = bootstrap_ci(scores, confidence=0.95)

        # 95% CI should contain true mean (this may fail 5% of time)
        assert lower < true_mean < upper

    def test_returns_float_types(self):
        """Verify return types are Python floats."""
        scores = [0.8, 0.82, 0.85]

        mean, lower, upper = bootstrap_ci(scores)

        assert isinstance(mean, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_empty_scores_raises_error(self):
        """Verify empty scores list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci([])

    def test_higher_confidence_wider_ci(self):
        """Verify higher confidence produces wider CI."""
        scores = [0.8, 0.82, 0.85, 0.83, 0.81, 0.79, 0.84, 0.86]

        _, lower_90, upper_90 = bootstrap_ci(scores, confidence=0.90)
        _, lower_99, upper_99 = bootstrap_ci(scores, confidence=0.99)

        width_90 = upper_90 - lower_90
        width_99 = upper_99 - lower_99

        assert width_99 > width_90

    def test_more_samples_tighter_ci(self):
        """Verify larger samples produce tighter CI."""
        np.random.seed(42)
        small_sample = np.random.normal(0.8, 0.05, 5).tolist()
        large_sample = np.random.normal(0.8, 0.05, 100).tolist()

        _, lower_small, upper_small = bootstrap_ci(small_sample)
        _, lower_large, upper_large = bootstrap_ci(large_sample)

        width_small = upper_small - lower_small
        width_large = upper_large - lower_large

        assert width_large < width_small

    def test_single_value_returns_same_bounds(self):
        """Verify single value produces same lower and upper bounds."""
        scores = [0.85]

        mean, lower, upper = bootstrap_ci(scores)

        assert mean == pytest.approx(0.85)
        assert lower == pytest.approx(0.85)
        assert upper == pytest.approx(0.85)

    def test_random_state_reproducibility(self):
        """Verify random_state produces reproducible results."""
        scores = [0.8, 0.82, 0.85, 0.83, 0.81]

        mean1, lower1, upper1 = bootstrap_ci(scores, random_state=42)
        mean2, lower2, upper2 = bootstrap_ci(scores, random_state=42)

        assert mean1 == mean2
        assert lower1 == lower2
        assert upper1 == upper2


class TestComputeCohensD:
    """Test Cohen's d effect size computation."""

    def test_identical_scores_zero_effect(self):
        """Verify identical scores produce zero effect size."""
        scores_a = [0.8, 0.8, 0.8, 0.8, 0.8]
        scores_b = [0.8, 0.8, 0.8, 0.8, 0.8]

        d = compute_cohens_d(scores_a, scores_b)

        assert d == 0.0

    def test_large_difference_large_effect(self):
        """Verify large differences produce large effect sizes."""
        # Use scores with variance in differences to get meaningful Cohen's d
        scores_a = [0.90, 0.92, 0.88, 0.91, 0.89]
        scores_b = [0.50, 0.48, 0.52, 0.49, 0.51]

        d = compute_cohens_d(scores_a, scores_b)

        assert abs(d) > 0.8  # Large effect

    def test_positive_when_a_greater(self):
        """Verify positive Cohen's d when scores_a > scores_b."""
        scores_a = [0.9, 0.92, 0.91, 0.93, 0.89]
        scores_b = [0.7, 0.72, 0.71, 0.73, 0.69]

        d = compute_cohens_d(scores_a, scores_b)

        assert d > 0

    def test_negative_when_b_greater(self):
        """Verify negative Cohen's d when scores_b > scores_a."""
        scores_a = [0.7, 0.72, 0.71, 0.73, 0.69]
        scores_b = [0.9, 0.92, 0.91, 0.93, 0.89]

        d = compute_cohens_d(scores_a, scores_b)

        assert d < 0

    def test_unequal_length_raises_error(self):
        """Verify unequal score lists raise ValueError."""
        scores_a = [0.8, 0.85, 0.9]
        scores_b = [0.75, 0.8]

        with pytest.raises(ValueError, match="equal length"):
            compute_cohens_d(scores_a, scores_b)

    def test_returns_float(self):
        """Verify return type is Python float."""
        scores_a = [0.8, 0.85, 0.9]
        scores_b = [0.75, 0.8, 0.85]

        d = compute_cohens_d(scores_a, scores_b)

        assert isinstance(d, float)


class TestInterpretEffectSize:
    """Test effect size interpretation function."""

    def test_negligible_effect(self):
        """Verify negligible effect interpretation."""
        assert interpret_effect_size(0.1) == "negligible"
        assert interpret_effect_size(-0.1) == "negligible"
        assert interpret_effect_size(0.19) == "negligible"

    def test_small_effect(self):
        """Verify small effect interpretation."""
        assert interpret_effect_size(0.2) == "small"
        assert interpret_effect_size(-0.3) == "small"
        assert interpret_effect_size(0.49) == "small"

    def test_medium_effect(self):
        """Verify medium effect interpretation."""
        assert interpret_effect_size(0.5) == "medium"
        assert interpret_effect_size(-0.6) == "medium"
        assert interpret_effect_size(0.79) == "medium"

    def test_large_effect(self):
        """Verify large effect interpretation."""
        assert interpret_effect_size(0.8) == "large"
        assert interpret_effect_size(-1.0) == "large"
        assert interpret_effect_size(2.5) == "large"

    def test_zero_effect(self):
        """Verify zero effect is negligible."""
        assert interpret_effect_size(0.0) == "negligible"


class TestComparisonResult:
    """Test ComparisonResult dataclass."""

    def test_to_dict_structure(self):
        """Verify to_dict returns expected structure."""
        result = ComparisonResult(
            model_a_name="DeepLabV3",
            model_b_name="U-Net",
            metric_name="IoU",
            model_a_mean=0.87,
            model_a_std=0.02,
            model_a_scores=[0.85, 0.87, 0.89],
            model_b_mean=0.78,
            model_b_std=0.03,
            model_b_scores=[0.75, 0.78, 0.81],
            t_statistic=3.5,
            p_value=0.01,
            effect_size=1.2,
            is_significant=True,
            alpha=0.05,
            ci_a=(0.85, 0.89),
            ci_b=(0.75, 0.81),
        )

        d = result.to_dict()

        assert d["comparison"] == "DeepLabV3 vs U-Net"
        assert d["metric"] == "IoU"
        assert d["model_a"]["name"] == "DeepLabV3"
        assert d["model_a"]["mean"] == 0.87
        assert d["model_b"]["name"] == "U-Net"
        assert d["statistical_test"]["test_type"] == "paired_t_test"
        assert d["statistical_test"]["is_significant"] is True
        assert d["effect_size"]["cohens_d"] == 1.2
        assert d["effect_size"]["interpretation"] == "large"

    def test_to_dict_json_serializable(self):
        """Verify to_dict output is JSON serializable."""
        result = ComparisonResult(
            model_a_name="ModelA",
            model_b_name="ModelB",
            metric_name="Dice",
            model_a_mean=0.9,
            model_a_std=0.01,
            model_a_scores=[0.89, 0.90, 0.91],
            model_b_mean=0.85,
            model_b_std=0.02,
            model_b_scores=[0.83, 0.85, 0.87],
            t_statistic=2.5,
            p_value=0.03,
            effect_size=0.7,
            is_significant=True,
            ci_a=(0.88, 0.92),
            ci_b=(0.82, 0.88),
        )

        # Should not raise
        json_str = json.dumps(result.to_dict())
        assert isinstance(json_str, str)

        # Should roundtrip
        loaded = json.loads(json_str)
        assert loaded["metric"] == "Dice"

    def test_summary_contains_key_info(self):
        """Verify summary contains key information."""
        result = ComparisonResult(
            model_a_name="DeepLabV3",
            model_b_name="U-Net",
            metric_name="IoU",
            model_a_mean=0.87,
            model_a_std=0.02,
            model_b_mean=0.78,
            model_b_std=0.03,
            t_statistic=3.5,
            p_value=0.01,
            effect_size=1.2,
            is_significant=True,
            ci_a=(0.85, 0.89),
            ci_b=(0.75, 0.81),
        )

        summary = result.summary()

        assert "DeepLabV3" in summary
        assert "U-Net" in summary
        assert "IoU" in summary
        assert "0.87" in summary
        assert "0.78" in summary
        assert "statistically significant" in summary
        assert "DeepLabV3 performs better" in summary

    def test_summary_no_winner_when_not_significant(self):
        """Verify summary shows no winner when not significant."""
        result = ComparisonResult(
            model_a_name="ModelA",
            model_b_name="ModelB",
            metric_name="IoU",
            model_a_mean=0.80,
            model_a_std=0.05,
            model_b_mean=0.79,
            model_b_std=0.05,
            t_statistic=0.5,
            p_value=0.6,
            effect_size=0.1,
            is_significant=False,
            ci_a=(0.75, 0.85),
            ci_b=(0.74, 0.84),
        )

        summary = result.summary()

        assert "not statistically significant" in summary
        assert "No conclusive winner" in summary


class TestPairedComparison:
    """Test paired_comparison comprehensive function."""

    def test_returns_comparison_result(self):
        """Verify function returns ComparisonResult."""
        scores_a = [0.85, 0.87, 0.89, 0.86, 0.88]
        scores_b = [0.75, 0.78, 0.80, 0.76, 0.79]

        result = paired_comparison(scores_a, scores_b, model_a_name="ModelA", model_b_name="ModelB")

        assert isinstance(result, ComparisonResult)

    def test_computes_all_statistics(self):
        """Verify all statistics are computed."""
        scores_a = [0.85, 0.87, 0.89, 0.86, 0.88]
        scores_b = [0.75, 0.78, 0.80, 0.76, 0.79]

        result = paired_comparison(scores_a, scores_b)

        # Check means
        assert result.model_a_mean == pytest.approx(np.mean(scores_a), rel=1e-4)
        assert result.model_b_mean == pytest.approx(np.mean(scores_b), rel=1e-4)

        # Check stds
        assert result.model_a_std == pytest.approx(np.std(scores_a, ddof=1), rel=1e-4)
        assert result.model_b_std == pytest.approx(np.std(scores_b, ddof=1), rel=1e-4)

        # Check CIs are reasonable
        assert result.ci_a[0] < result.model_a_mean < result.ci_a[1]
        assert result.ci_b[0] < result.model_b_mean < result.ci_b[1]

        # Check significance computed
        assert isinstance(result.is_significant, bool)
        assert isinstance(result.p_value, float)
        assert isinstance(result.t_statistic, float)
        assert isinstance(result.effect_size, float)

    def test_stores_raw_scores(self):
        """Verify raw scores are stored."""
        scores_a = [0.85, 0.87, 0.89]
        scores_b = [0.75, 0.78, 0.80]

        result = paired_comparison(scores_a, scores_b)

        assert result.model_a_scores == scores_a
        assert result.model_b_scores == scores_b

    def test_unequal_length_raises_error(self):
        """Verify unequal score lists raise ValueError."""
        scores_a = [0.85, 0.87, 0.89]
        scores_b = [0.75, 0.78]

        with pytest.raises(ValueError, match="same length"):
            paired_comparison(scores_a, scores_b)

    def test_custom_names_preserved(self):
        """Verify custom model names are preserved."""
        result = paired_comparison(
            [0.85, 0.87, 0.89],
            [0.75, 0.78, 0.80],
            model_a_name="DeepLabV3-ResNet50",
            model_b_name="U-Net",
            metric_name="Dice Score",
        )

        assert result.model_a_name == "DeepLabV3-ResNet50"
        assert result.model_b_name == "U-Net"
        assert result.metric_name == "Dice Score"

    def test_custom_alpha_level(self):
        """Verify custom alpha level is used."""
        result = paired_comparison([0.85, 0.87, 0.89], [0.75, 0.78, 0.80], alpha=0.01)

        assert result.alpha == 0.01


class TestPrintSignificanceReport:
    """Test significance report printing."""

    def test_does_not_raise(self, capsys):
        """Verify function executes without errors."""
        scores_a = [0.8, 0.82, 0.85, 0.83, 0.81]
        scores_b = [0.75, 0.78, 0.80, 0.77, 0.76]

        # Should not raise
        print_significance_report("Model A", "Model B", scores_a, scores_b, metric_name="IoU")

        captured = capsys.readouterr()
        assert "Model A" in captured.out
        assert "Model B" in captured.out

    def test_outputs_metric_name(self, capsys):
        """Verify metric name appears in output."""
        scores_a = [0.8, 0.82, 0.85]
        scores_b = [0.75, 0.78, 0.80]

        print_significance_report("UNet", "DeepLabV3", scores_a, scores_b, metric_name="Dice")

        captured = capsys.readouterr()
        assert "Dice" in captured.out

    def test_outputs_significance_result(self, capsys):
        """Verify significance decision appears in output."""
        scores_a = [0.8, 0.82, 0.85]
        scores_b = [0.75, 0.78, 0.80]

        print_significance_report("UNet", "DeepLabV3", scores_a, scores_b)

        captured = capsys.readouterr()
        assert "Significant" in captured.out or "significant" in captured.out.lower()

    def test_outputs_confidence_interval(self, capsys):
        """Verify 95% CI appears in output."""
        scores_a = [0.8, 0.82, 0.85]
        scores_b = [0.75, 0.78, 0.80]

        print_significance_report("UNet", "DeepLabV3", scores_a, scores_b)

        captured = capsys.readouterr()
        assert "95% CI" in captured.out

    def test_outputs_effect_size(self, capsys):
        """Verify Cohen's d effect size appears in output."""
        scores_a = [0.8, 0.82, 0.85]
        scores_b = [0.75, 0.78, 0.80]

        print_significance_report("UNet", "DeepLabV3", scores_a, scores_b)

        captured = capsys.readouterr()
        assert "Cohen's d" in captured.out

    def test_outputs_std_dev(self, capsys):
        """Verify standard deviation appears in output."""
        scores_a = [0.8, 0.82, 0.85]
        scores_b = [0.75, 0.78, 0.80]

        print_significance_report("UNet", "DeepLabV3", scores_a, scores_b)

        captured = capsys.readouterr()
        assert "Std Dev" in captured.out

    def test_identifies_better_model_when_significant(self, capsys):
        """Verify better model is identified when significant."""
        # Clearly different scores
        scores_a = [0.9, 0.91, 0.92, 0.89, 0.93]
        scores_b = [0.5, 0.51, 0.49, 0.52, 0.48]

        print_significance_report("BetterModel", "WorseModel", scores_a, scores_b)

        captured = capsys.readouterr()
        assert "BetterModel" in captured.out
        assert "significantly better" in captured.out

    def test_no_significant_difference_message(self, capsys):
        """Verify 'no significant difference' message when p > 0.05."""
        # Very similar scores (no significant difference)
        scores_a = [0.80, 0.80, 0.80, 0.80, 0.80]
        scores_b = [0.80, 0.80, 0.80, 0.80, 0.80]

        print_significance_report("ModelA", "ModelB", scores_a, scores_b)

        captured = capsys.readouterr()
        assert "No significant difference" in captured.out
