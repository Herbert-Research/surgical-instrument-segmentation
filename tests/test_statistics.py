"""Tests for statistical significance testing functions."""

import numpy as np
import pytest

from surgical_segmentation.evaluation.statistics import (
    bootstrap_ci,
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
        scores_b = [0.75, 0.8, 0.85]

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
        scores_b = (np.array(scores_a) - 0.02).tolist()

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
        scores_a = [0.9, 0.92, 0.91, 0.93, 0.89]
        scores_b = [0.7, 0.72, 0.71, 0.73, 0.69]

        t_stat, _, _ = paired_ttest(scores_a, scores_b)

        assert t_stat > 0

    def test_negative_t_stat_when_b_greater(self):
        """Verify negative t-stat when scores_b > scores_a."""
        scores_a = [0.7, 0.72, 0.71, 0.73, 0.69]
        scores_b = [0.9, 0.92, 0.91, 0.93, 0.89]

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
