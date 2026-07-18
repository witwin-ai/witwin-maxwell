"""Variance-aware performance gate statistics for the port hot path.

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


# One-sided Student-t 95% critical values, t_{0.95, df}, for df = 1..30.  A
# small explicit table keeps the gate dependency-free and deterministic; beyond
# df = 30 the normal approximation (1.645) is used, which is conservative here
# because t_{0.95, df} decreases monotonically toward 1.645 as df grows.
_T_95_ONE_SIDED = {
    1: 6.313752,
    2: 2.919986,
    3: 2.353363,
    4: 2.131847,
    5: 2.015048,
    6: 1.943180,
    7: 1.894579,
    8: 1.859548,
    9: 1.833113,
    10: 1.812461,
    11: 1.795885,
    12: 1.782288,
    13: 1.770933,
    14: 1.761310,
    15: 1.753050,
    16: 1.745884,
    17: 1.739607,
    18: 1.734064,
    19: 1.729133,
    20: 1.724718,
    21: 1.720743,
    22: 1.717144,
    23: 1.713872,
    24: 1.710882,
    25: 1.708141,
    26: 1.705618,
    27: 1.703288,
    28: 1.701131,
    29: 1.699127,
    30: 1.697261,
}
_T_95_LARGE_DF = 1.644854  # standard normal 95th percentile


def student_t_95_one_sided(degrees_of_freedom: int) -> float:
    """Return the one-sided 95% Student-t critical value for ``df``."""

    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be at least 1.")
    return _T_95_ONE_SIDED.get(degrees_of_freedom, _T_95_LARGE_DF)


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
