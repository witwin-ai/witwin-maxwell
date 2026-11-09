from dataclasses import replace

import numpy as np
import pytest

from benchmark import report
from benchmark.metrics import field_runtime_convergence, flux_runtime_convergence
from benchmark.models import ScenarioMetrics


def test_flux_convergence_classification_and_report(tmp_path, monkeypatch):
    power = np.array([[100.0, -100.0], [50.0, -200.0]])
    doubled = np.array([[100.5, -100.5], [50.25, -240.0]])
    change, converged = flux_runtime_convergence(power, doubled)
    assert change == pytest.approx(0.2)
    assert converged is False
    small_change, small_converged = flux_runtime_convergence(power, power * 1.005)
    assert small_change == pytest.approx(0.005)
    assert small_converged is True
    assert flux_runtime_convergence(power, power * 1.01) == (0.01, True)

    field = np.array([[3.0, 4.0j], [0.0, 0.0]])
    doubled_field = field + np.array([[0.0, 0.0], [1.0j, 0.0]])
    field_change, field_converged = field_runtime_convergence(field, doubled_field)
    assert field_change == pytest.approx(0.2)
    assert field_converged is False
    small_field_change, small_field_converged = field_runtime_convergence(field, field * 1.005)
    assert small_field_change == pytest.approx(0.005)
    assert small_field_converged is True
    assert field_runtime_convergence(np.array([100.0j]), np.array([101.0j])) == (0.01, True)

    output = tmp_path / "RESULTS.md"
    monkeypatch.setattr(report, "RESULTS_MD", output)
    result = ScenarioMetrics(
        name="high_eps_box",
        description="runtime convergence",
        frequencies=(1.0, 2.0),
        maxwell_time_s=1.0,
        tidy3d_cache_hit=True,
        field_l2=0.1,
        field_shape_l2=0.01,
        field_linf=0.2,
        field_corr=0.99,
        flux_error=0.02,
        flux_runtime_change=change,
        flux_converged=converged,
        field_runtime_change=field_change,
        field_converged=field_converged,
        compared_monitor="field",
        compared_component="Ez",
        material_source_plot=tmp_path / "material.png",
        field_plot=tmp_path / "field.png",
        updated_at="2026-10-30",
        per_frequency=[{
            "frequency": 1.0,
            "field_l2": 0.1,
            "field_shape_l2": 0.01,
            "field_linf": 0.2,
            "field_corr": 0.99,
        }],
    )
    report.write_results_markdown([
        result,
        replace(result, name="dipole_vacuum", field_runtime_change=small_field_change, field_converged=True),
        replace(result, name="dipole_ey", flux_runtime_change=small_change, flux_converged=True),
        replace(result, name="dipole_two_freq", flux_runtime_change=None, flux_converged=None,
                field_runtime_change=None, field_converged=None),
    ])
    report.write_results_markdown([])
    content = output.read_text(encoding="utf-8")
    rows = {
        parts[0]: parts
        for line in content.splitlines()
        if line.startswith("| ")
        for parts in [[part.strip() for part in line.strip("|").split("|")]]
        if len(parts) == 18
    }
    assert "Flux 2x-runtime change" in content
    assert "Field 2x-runtime change" in content
    assert "excluded from flux pass/fail" in content
    assert "excluded from field pass/fail counts and summaries" in content
    assert "WITWIN_BENCHMARK_SKIP_CONVERGENCE=1" in content
    assert rows["high_eps_box"][4:11] == [
        "1.0000e-01 (unconverged)", "1.0000e-02 (unconverged)",
        "2.0000e-01 (unconverged)", "0.9900 (unconverged)", "2.0000e-01",
        "2.0000e-02 (unconverged)", "2.0000e-01",
    ]
    assert rows["dipole_vacuum"][4:11] == [
        "1.0000e-01", "1.0000e-02", "2.0000e-01", "0.9900", "5.0000e-03",
        "2.0000e-02 (unconverged)", "2.0000e-01",
    ]
    assert rows["dipole_ey"][4].endswith(" (unconverged)")
    assert rows["dipole_ey"][9:11] == ["2.0000e-02", "5.0000e-03"]
    assert rows["dipole_two_freq"][4:11] == [
        "1.0000e-01", "1.0000e-02", "2.0000e-01", "0.9900", "-", "2.0000e-02", "-",
    ]
    assert (
        "| high_eps_box | 1.00000000e+00 | 1.0000e-01 (unconverged) | "
        "1.0000e-02 (unconverged) | 2.0000e-01 (unconverged) | 0.9900 (unconverged) |"
    ) in content
