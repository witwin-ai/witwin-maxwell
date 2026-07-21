"""Structural gates for the SAR phantom validation harness (benchmark.sar_validation).

Checks the harness produces well-formed reports with the verbatim gate-classification
taxonomy, that the RESULTS.md section writer is idempotent, and that the two
solver-free runners (analytic 1 g cube, blocked antenna) produce the expected
gate/status without needing a GPU. The wave-level layered-slab convergence itself
is covered by tests/sar/test_phantom_convergence.py.
"""

import pytest

from benchmark import sar_validation as sv

_TAXONOMY = {"analytic-identity", "tautology", "symmetric", "postprocess-only", "wave-level"}


def test_scene_runners_cover_the_family():
    assert set(sv.SCENE_RUNNERS) == {
        "one_gram_cube",
        "uniform_lossy_cube",
        "layered_slab",
        "antenna_near_phantom",
    }


def test_one_gram_cube_is_analytic_identity_and_matches_hand_computed_value():
    report = sv.run_one_gram_cube()
    assert report.gate_class == sv.ANALYTIC_IDENTITY
    assert report.gate_class in _TAXONOMY
    assert report.status == "pass"
    headline = report.metrics[0]
    assert headline["quantity"] == "peak_1g_sar"
    # Hand-computed 1 g average equals the analytic point SAR to machine tolerance.
    assert headline["rel_error"] < 1e-4
    assert report.external_reference == sv.ANALYTIC_ONLY


def test_antenna_near_phantom_is_reported_blocked():
    report = sv.run_antenna_near_phantom()
    assert report.status == "blocked"
    assert report.gate_class in _TAXONOMY
    assert "conductance-aware" in " ".join(report.notes)
    # A blocked scene carries no measured headline metric.
    assert report.metrics == []


def test_layered_slab_headline_is_wave_level():
    # The binding phantom gate must be the verbatim wave-level class.
    report = sv.SarReport(
        name="sar/layered_slab", description="", gate_class=sv.WAVE_LEVEL, status="pass",
        reference="", external_reference=sv.ANALYTIC_ONLY, target="",
    )
    assert report.gate_class == "wave-level"
    assert report.gate_class in _TAXONOMY


def test_results_section_writer_is_idempotent():
    reports = [sv.run_one_gram_cube(), sv.run_antenna_near_phantom()]
    section = sv._results_section(reports)
    assert sv.SECTION_HEADER in section
    assert "sar/one_gram_cube" in section
    assert "analytic-only" in section

    base = "# Benchmark Results\n\nintro\n\n## dipole\n\n| a |\n"
    once = sv._replace_or_append_section(base, sv.SECTION_HEADER, section)
    twice = sv._replace_or_append_section(once, sv.SECTION_HEADER, section)
    # Re-running the writer replaces the section in place rather than duplicating it.
    assert once.count(sv.SECTION_HEADER) == 1
    assert twice.count(sv.SECTION_HEADER) == 1
    # A pre-existing following section is preserved.
    assert "## dipole" in once and "## dipole" in twice
