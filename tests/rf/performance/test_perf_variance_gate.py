"""Unit tests for the variance-aware performance gate statistics.

Gate class: ``perf-statistical`` (non-numerical performance gate; see
``docs/reference/gate-classification.md`` §5).

These exercise the S2.3 gate logic with no GPU and no wall-clock timing: every
input is a hand-chosen list of floats so the pass/fail behavior and the
confidence-interval arithmetic are deterministic and independently checkable.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path
import sys

import pytest
from scipy import stats


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "support"))

from perf_variance_gate import (  # noqa: E402
    RegressionGateResult,
    ci95_upper_bound,
    evaluate_regression_gate,
    median_absolute_deviation,
    paired_ratios,
    student_t_95_one_sided,
    summarize_samples,
)


def test_student_t_matches_known_table_values():
    # Classic textbook one-sided t_{0.95} values.
    assert student_t_95_one_sided(1) == pytest.approx(6.313752, abs=1e-6)
    assert student_t_95_one_sided(10) == pytest.approx(1.812461, abs=1e-6)
    assert student_t_95_one_sided(30) == pytest.approx(1.697261, abs=1e-6)


def test_student_t_is_exact_vs_scipy_for_all_df():
    # The quantile is the exact scipy Student-t ppf for every df, including the
    # large-df regime where the previous normal-quantile fallback (1.6449) was
    # anti-conservative: t_{0.95, 31} = 1.6955 > 1.6449.
    for df in (1, 2, 5, 30, 31, 60, 200, 500):
        assert student_t_95_one_sided(df) == pytest.approx(
            float(stats.t.ppf(0.95, df)), rel=1e-12
        )
    # The old fallback would have returned 1.644854 for df > 30; the exact value
    # is strictly larger, confirming the fix widens (not narrows) the interval.
    assert student_t_95_one_sided(31) == pytest.approx(1.695519, abs=1e-6)
    assert student_t_95_one_sided(31) > 1.644854


def test_student_t_decreases_monotonically_in_df():
    # t_{0.95, df} shrinks monotonically toward the normal limit 1.6449 as df
    # grows, always approaching from above.
    dfs = [1, 2, 3, 5, 10, 20, 30, 31, 60, 120, 500, 1000]
    values = [student_t_95_one_sided(df) for df in dfs]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert values[-1] > 1.644854


def test_student_t_rejects_zero_degrees_of_freedom():
    with pytest.raises(ValueError):
        student_t_95_one_sided(0)


def test_median_absolute_deviation_is_robust():
    assert median_absolute_deviation([1.0, 1.0, 1.0]) == 0.0
    # deviations from median 3.0 are [2,1,0,1,2] -> MAD = 1.0; the outlier does
    # not inflate it the way a standard deviation would.
    assert median_absolute_deviation([1.0, 2.0, 3.0, 4.0, 100.0]) == 1.0


def test_ci95_upper_bound_matches_the_closed_form():
    values = [1.00, 1.02, 0.99, 1.01, 1.03]
    n = len(values)
    expected = (
        statistics.fmean(values)
        + student_t_95_one_sided(n - 1) * statistics.stdev(values) / math.sqrt(n)
    )
    assert ci95_upper_bound(values) == pytest.approx(expected, rel=1e-12)
    # The upper bound must sit strictly above the sample mean for a spread sample.
    assert ci95_upper_bound(values) > statistics.fmean(values)


def test_ci95_upper_bound_requires_two_values():
    with pytest.raises(ValueError):
        ci95_upper_bound([1.0])


def test_paired_ratios_preserve_round_pairing():
    ratios = paired_ratios([10.0, 20.0, 40.0], [11.0, 21.0, 44.0])
    assert ratios == pytest.approx([1.1, 1.05, 1.1])


def test_paired_ratios_reject_mismatched_or_nonpositive_input():
    with pytest.raises(ValueError):
        paired_ratios([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_ratios([0.0], [1.0])


def test_summarize_samples_drops_warmups():
    summary = summarize_samples([99.0, 99.0, 1.0, 2.0, 3.0], warmups=2)
    assert summary.count == 3
    assert summary.median == 2.0
    assert summary.minimum == 1.0
    assert summary.maximum == 3.0
    assert summary.mean == pytest.approx(2.0)


def test_summarize_samples_requires_two_timed_points():
    with pytest.raises(ValueError):
        summarize_samples([1.0, 2.0], warmups=1)


def test_gate_passes_when_whole_interval_clears_target():
    # Tight sample centered at 1.006 with tiny spread: the 95% upper bound stays
    # well under the 2% target.
    ratios = [1.005, 1.006, 1.007, 1.006, 1.005]
    result = evaluate_regression_gate(ratios, target_ratio=1.02)
    assert isinstance(result, RegressionGateResult)
    assert result.passed is True
    assert result.ci95_upper_ratio < 1.02
    assert result.mean_ratio == pytest.approx(statistics.fmean(ratios))


def test_gate_fails_a_noisy_sample_whose_mean_still_looks_fine():
    # Mean ~1.01 (under the 2% target) but a wide spread pushes the 95% upper
    # bound past target: a single-point median gate would wrongly pass this.
    ratios = [0.95, 1.07, 0.94, 1.08, 1.01]
    assert statistics.fmean(ratios) < 1.02  # point estimate looks acceptable
    result = evaluate_regression_gate(ratios, target_ratio=1.02)
    assert result.ci95_upper_ratio > 1.02
    assert result.passed is False


def test_gate_fails_a_clear_regression():
    ratios = [1.20, 1.22, 1.19, 1.21, 1.20]
    result = evaluate_regression_gate(ratios, target_ratio=1.05)
    assert result.passed is False
    assert result.ci95_upper_regression_pct > 5.0


def test_gate_rejects_degenerate_configuration():
    with pytest.raises(ValueError):
        evaluate_regression_gate([1.0, 1.0], target_ratio=1.0)
    with pytest.raises(ValueError):
        evaluate_regression_gate([1.0], target_ratio=1.05)


def test_gate_reports_target_regression_percentages():
    result = evaluate_regression_gate([1.0, 1.0, 1.0], target_ratio=1.05)
    assert result.target_regression_pct == pytest.approx(5.0)
    # A perfectly flat sample has a zero-width interval at the mean.
    assert result.ci95_upper_ratio == pytest.approx(1.0)
    assert result.passed is True
