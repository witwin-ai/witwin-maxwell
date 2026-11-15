import importlib.util
from pathlib import Path

import numpy as np
import pytest

from benchmark import status
from benchmark.metrics import windowed_resonance_frequency
from benchmark.runner import SCENARIOS, _box_resonance_frequency, _quadratic_peak_frequency, build_scene


C0 = 299_792_458.0
COMB = np.linspace(1.1e8, 2.5e8, 29)
RECORD = 80.0 / C0
DT = 6.8e-11


def _cavity_spectrum(f0, convention):
    """Pulse-driven undamped mode plus an out-of-band mode, divided by the source spectrum."""
    t = np.arange(int(np.ceil(RECORD / DT))) * DT
    sigma = 1.0 / (2.0 * np.pi * 1.0e8)
    delay = 6.0 * sigma
    ramp = 0.5 * (1.0 + np.tanh((t - delay) / sigma))
    signal = ramp * (np.cos(2 * np.pi * f0 * (t - delay) + 0.7) + 0.5 * np.cos(2 * np.pi * 259.6e6 * (t - delay)))
    pulse = np.exp(-0.5 * ((t - delay) / sigma) ** 2) * np.cos(2 * np.pi * 1.9e8 * (t - delay))
    kernel = np.exp(convention * 2j * np.pi * np.outer(COMB, t))
    return (kernel @ signal) / (kernel @ pulse)


@pytest.mark.parametrize("convention", [1, -1])
@pytest.mark.parametrize("f0", [208.0e6, 210.0e6, 211.9e6, 213.5e6, 216.0e6])
def test_kernel_fit_recovers_an_undersampled_lossless_resonance(f0, convention):
    spectrum = _cavity_spectrum(f0, convention)
    estimate, sigma = windowed_resonance_frequency(COMB, spectrum, record_time=RECORD)
    # The 5 MHz comb is coarser than 1/T = 3.75 MHz: peak interpolation is off by up to 1%.
    assert abs(estimate - f0) / f0 < 1.2e-3
    assert abs(estimate - f0) < 4.0 * sigma
    assert 0 < sigma / f0 < 2.0e-3


def test_peak_interpolation_is_biased_where_the_fit_is_not():
    spectrum = _cavity_spectrum(211.0e6, 1)
    quadratic = _quadratic_peak_frequency(tuple(COMB), np.abs(spectrum))
    fitted, _ = windowed_resonance_frequency(COMB, spectrum, record_time=RECORD)
    assert abs(quadratic - 211.0e6) / 211.0e6 > 5.0e-3 > 5 * abs(fitted - 211.0e6) / 211.0e6


def test_box_resonance_is_the_discrete_yee_eigenfrequency():
    scene = build_scene("pec_cavity")
    dx, side = 0.04, 1.0
    f = _box_resonance_frequency(scene, DT)
    expected = np.arcsin(C0 * DT * np.sqrt(2.0) * np.sin(np.pi * dx / (2 * side)) / dx) / (np.pi * DT)
    assert f == pytest.approx(expected, rel=1e-12)
    assert f == pytest.approx(C0 / 2 * np.sqrt(2.0), rel=5e-4)  # continuum TM110 of the 1 m cube


def test_cavity_rows_are_scored_on_resonance_frequency(tmp_path):
    assert {name for name, s in SCENARIOS.items() if not s.score_fields} == {"pec_cavity", "pmc_cavity"}
    limits = status.load_suite_limits()
    assert set(limits.scalar_limits) == {"pec_cavity", "pmc_cavity"}
    row = ["pec_cavity", *("-" for _ in range(17))]
    row[4], row[6], row[7], row[11] = "2.2e+00 (not scored)", "2.2e+00 (not scored)", "0.95 (not scored)", "1"
    scalars = [["pec_cavity", "1.8e8", name, "1", "1", value, value, "-"] for name, value in (
        ("resonance_frequency", "3.0e-04"), ("resonance_frequency_discrete_analytic", "2.0e-04"),
        ("resonance_frequency_runtime", "1.0e-04"))]
    assert status.classify_row(row, limits, scalars)[:2] == (status.PASS_GLOBAL, "PASS")
    failing = [*scalars[:2], [*scalars[2][:5], "2.0e-03", "2.0e-03", "-"]]
    assert status.classify_row(row, limits, failing)[:2] == (status.FAIL, "FAIL")

    spec = importlib.util.spec_from_file_location(
        "integration_batch", Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py")
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    per_frequency = ["pec_cavity", "1.1e8", "2.2 (not scored)", "-", "2.2 (not scored)", "0.95 (not scored)", "-"]
    results = tmp_path / "RESULTS.md"
    results.write_text("\n".join("| " + " | ".join(r) + " |" for r in [row, per_frequency, *failing]))
    unit = dict(nodes=["benchmark:pec_cavity"], metric_limits=limits.global_limits,
                scenario_scalar_limits=limits.scalar_limits)
    cases = {case["nodeid"]: case["status"] for case in batch.benchmark_cases(results, unit)}
    assert cases == {
        "benchmark:pec_cavity": "FAIL",
        "benchmark:pec_cavity[resonance_frequency]": "PASS",
        "benchmark:pec_cavity[resonance_frequency_discrete_analytic]": "PASS",
        "benchmark:pec_cavity[resonance_frequency_runtime]": "FAIL",
    }
