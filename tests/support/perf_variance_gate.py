"""Variance-aware performance gate statistics for the port hot path.

Gate class: ``perf-statistical`` (non-numerical performance gate; see
``docs/reference/gate-classification.md`` §5). Wall-clock regression judged by a
one-sided 95% CI upper bound, never a single-point min/median.

The audit (``docs/assessments/next-functional-audit-2026-07-18.md`` §2.6 and
step S2.3) rejects single-point min/median performance gates: a lone median that
lands inside its own noise band cannot certify a ``< 5%`` / ``< 2%`` target.
This module supplies the statistics the exclusive-window measurement (S2b) uses
to make that call defensibly.

The contract is a **one-sided upper confidence bound**: a regression is only
accepted when the *upper* bound of the 95% confidence interval on the mean
paired baseline/candidate ratio is still below the target ratio.  A wide, noisy
sample therefore fails even if its point estimate looks fine, which is exactly
the protection the audit demands.

Everything here is pure arithmetic on lists of floats, so the gate logic is unit
testable with no GPU and no wall-clock timing.  Callers (the ABBA orchestrator)
supply already-measured samples; this module never touches CUDA.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from scipy import stats


def student_t_95_one_sided(degrees_of_freedom: int) -> float:
    """Return the exact one-sided 95% Student-t critical value ``t_{0.95, df}``.

    Computed with :func:`scipy.stats.t.ppf` for *every* ``df`` (SciPy is a
    primary dependency).  An earlier revision used a small table for ``df <= 30``
    and fell back to the standard-normal quantile ``1.6449`` for larger ``df``,
    calling it "conservative"; it is not.  Because ``t_{0.95, df}`` decreases
    monotonically toward ``1.6449`` from *above*, the normal quantile is always
    *smaller* than the true critical value (e.g. ``t_{0.95, 31} = 1.6955``), so
    that fallback under-inflated the confidence interval and was
    *anti*-conservative.  Using the exact quantile removes the bias for all df.
    """

    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be at least 1.")
    return float(stats.t.ppf(0.95, degrees_of_freedom))


def median_absolute_deviation(values: list[float]) -> float:
    """Median of absolute deviations from the median (a robust spread)."""

    if not values:
        raise ValueError("median_absolute_deviation requires at least one value.")
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


@dataclass(frozen=True)
class SampleSummary:
    """Robust and parametric summary of one measurement block."""

    count: int
    median: float
    mad: float
    mean: float
    stdev: float
    minimum: float
    maximum: float


def summarize_samples(samples: list[float], *, warmups: int = 0) -> SampleSummary:
    """Drop ``warmups`` leading samples and summarize the remainder.

    ``warmups`` models untimed-then-timed acquisition where the caller still
    handed over the warmup measurements; the gate discards them so cold-cache
    or boost-clock transients never enter the statistic.
    """

    if warmups < 0:
        raise ValueError("warmups must be non-negative.")
    timed = samples[warmups:]
    if len(timed) < 2:
        raise ValueError(
            "At least two timed samples are required after dropping warmups; "
            f"got {len(timed)} from {len(samples)} samples with warmups={warmups}."
        )
    return SampleSummary(
        count=len(timed),
        median=float(statistics.median(timed)),
        mad=float(median_absolute_deviation(timed)),
        mean=float(statistics.fmean(timed)),
        stdev=float(statistics.stdev(timed)),
        minimum=float(min(timed)),
        maximum=float(max(timed)),
    )


def ci95_upper_bound(values: list[float]) -> float:
    """One-sided upper bound of the 95% confidence interval on the mean.

    ``mean + t_{0.95, n-1} * s / sqrt(n)``.  With a single value the interval is
    unbounded, so this raises rather than certifying an unmeasurable quantity.
    """

    if len(values) < 2:
        raise ValueError("ci95_upper_bound requires at least two values.")
    n = len(values)
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(n)
    return mean + student_t_95_one_sided(n - 1) * standard_error


def paired_ratios(
    baseline: list[float],
    candidate: list[float],
) -> list[float]:
    """Elementwise ``candidate / baseline`` ratios from paired rounds.

    Pairing (as opposed to pooling then dividing medians) preserves the ABBA
    round structure so shared per-round thermal/clock drift cancels inside each
    ratio before the statistic is taken.
    """

    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must have equal length.")
    if not baseline:
        raise ValueError("paired_ratios requires at least one paired round.")
    ratios = []
    for base, cand in zip(baseline, candidate, strict=True):
        if base <= 0.0:
            raise ValueError("baseline measurements must be positive.")
        ratios.append(cand / base)
    return ratios


@dataclass(frozen=True)
class RegressionGateResult:
    """Verdict of a variance-aware regression gate."""

    target_ratio: float
    rounds: int
    mean_ratio: float
    median_ratio: float
    mad_ratio: float
    ci95_upper_ratio: float
    passed: bool

    @property
    def target_regression_pct(self) -> float:
        return 100.0 * (self.target_ratio - 1.0)

    @property
    def ci95_upper_regression_pct(self) -> float:
        return 100.0 * (self.ci95_upper_ratio - 1.0)


def evaluate_regression_gate(
    ratios: list[float],
    *,
    target_ratio: float,
) -> RegressionGateResult:
    """Pass only if the 95% CI upper bound of the mean ratio is below target.

    ``target_ratio`` is expressed as a multiplier (1.05 encodes the ``< 5%``
    single-port target; 1.02 the ``< 2%`` per-additional-port target).  The
    point estimate being under target is *not* sufficient -- the whole interval
    must clear it.
    """

    if target_ratio <= 1.0:
        raise ValueError("target_ratio must exceed 1.0 (a positive regression budget).")
    if len(ratios) < 2:
        raise ValueError("A variance-aware gate requires at least two paired rounds.")
    upper = ci95_upper_bound(ratios)
    return RegressionGateResult(
        target_ratio=float(target_ratio),
        rounds=len(ratios),
        mean_ratio=float(statistics.fmean(ratios)),
        median_ratio=float(statistics.median(ratios)),
        mad_ratio=float(median_absolute_deviation(ratios)),
        ci95_upper_ratio=float(upper),
        passed=bool(upper < target_ratio),
    )
