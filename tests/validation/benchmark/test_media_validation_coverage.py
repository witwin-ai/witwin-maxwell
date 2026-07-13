"""P5.6 coverage gate: every public medium capability has a validation path.

Plan §5.4 / P5.6 acceptance: the benchmark suite must fail loudly if a shipped
public medium has no validation path (a Tidy3D benchmark scenario, an FDFD
cross-check, or a documented analytic-reference test). This test discovers the
capability *flags* on ``media.py`` dynamically (not from a hardcoded class list)
and fails if any flag is missing a ``benchmark.media_coverage.MEDIA_VALIDATION``
entry, then verifies every entry's claim so a declared path cannot be fictional.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from benchmark.media_coverage import (
    CONVENTION_FREQUENCY,
    FDFD,
    FDTD_ANALYTIC,
    MEDIA_VALIDATION,
    TIDY3D,
    discover_capability_flags,
    validation_coverage_markdown_lines,
)
from benchmark.scenes import SCENARIOS, build_scene
from benchmark.scenes.media import MEDIA_EXPORT_SCENARIOS

REPO_ROOT = Path(__file__).resolve().parents[3]
_HAS_TIDY3D = importlib.util.find_spec("tidy3d") is not None


def test_every_public_capability_flag_has_a_validation_entry():
    """A newly-shipped capability flag with no declared validation path fails here."""
    flags = discover_capability_flags()
    uncovered = sorted(flag for flag in flags if flag not in MEDIA_VALIDATION)
    assert not uncovered, (
        "Public medium capability flags with no validation path in "
        "benchmark/media_coverage.py::MEDIA_VALIDATION: "
        f"{uncovered}. Add a Tidy3D scenario, an FDFD cross-check, or a documented "
        "analytic-reference test and register it there."
    )


def test_validation_entries_declare_a_known_path():
    for key, entry in MEDIA_VALIDATION.items():
        assert entry.capability == key
        assert entry.path in (TIDY3D, FDTD_ANALYTIC, FDFD), (key, entry.path)


def test_every_reference_resolves_to_a_scenario_or_a_test_file():
    """Each entry must name a real validation artifact: a registered benchmark
    scenario or an existing repository test file."""
    for key, entry in MEDIA_VALIDATION.items():
        reference = entry.reference
        if reference in SCENARIOS:
            continue
        candidate = REPO_ROOT / reference
        assert candidate.is_file(), (
            f"Validation entry {key!r} references {reference!r}, which is neither a "
            "registered benchmark scenario nor an existing test file."
        )


def test_analytic_and_fdfd_references_point_at_test_files():
    for key, entry in MEDIA_VALIDATION.items():
        if entry.path in (FDTD_ANALYTIC, FDFD):
            candidate = REPO_ROOT / entry.reference
            assert candidate.is_file(), (key, entry.reference)


def test_tidy3d_scenarios_are_registered_and_have_a_source_and_monitors():
    for scenario in MEDIA_EXPORT_SCENARIOS:
        assert scenario.name in SCENARIOS
        scene = build_scene(scenario.name)
        assert len(scene.sources) == 1
        assert len(scene.monitors) >= 5


@pytest.mark.skipif(not _HAS_TIDY3D, reason="requires tidy3d to verify export claims")
def test_tidy3d_entries_export_and_fallbacks_genuinely_raise():
    """Verify every declared path is real: Tidy3D rows export, and every
    fallback row that claims no Tidy3D equivalent genuinely raises on export."""
    import tidy3d as td

    from witwin.maxwell.adapters.tidy3d import _convert_material

    for key, entry in MEDIA_VALIDATION.items():
        material = entry.probe()
        if entry.path == TIDY3D:
            medium = _convert_material(td=td, material=material, frequencies=entry.export_frequencies)
            assert medium is not None, key
        elif not entry.tidy3d_equivalent:
            with pytest.raises(NotImplementedError):
                _convert_material(td=td, material=material, frequencies=entry.export_frequencies)


@pytest.mark.skipif(not _HAS_TIDY3D, reason="requires tidy3d to check export convention")
def test_convention_checks_match_relative_permittivity():
    """"Correct, not merely accepted": for the clean-identity media the exported
    medium's eps_model must reproduce Material.relative_permittivity at a test
    frequency, not merely construct an object."""
    import tidy3d as td

    from witwin.maxwell.adapters.tidy3d import _convert_material

    checked = 0
    for key, entry in MEDIA_VALIDATION.items():
        if not entry.convention_check:
            continue
        material = entry.probe()
        medium = _convert_material(td=td, material=material, frequencies=entry.export_frequencies)
        exported = medium.eps_model(CONVENTION_FREQUENCY)
        analytic = material.relative_permittivity(CONVENTION_FREQUENCY)
        assert abs(exported - analytic) < 1e-9, (key, exported, analytic)
        checked += 1
    assert checked >= 1


@pytest.mark.skipif(not _HAS_TIDY3D, reason="requires tidy3d to export benchmark scenes")
def test_media_export_scenarios_export_as_full_scenes():
    for scenario in MEDIA_EXPORT_SCENARIOS:
        scene = build_scene(scenario.name)
        td_sim = scene.to_tidy3d(frequencies=scenario.frequencies, run_time=1.0e-13)
        assert td_sim.structures, scenario.name


def test_validation_coverage_markdown_lists_every_capability():
    lines = "\n".join(validation_coverage_markdown_lines())
    assert "## Validation coverage" in lines
    for key in MEDIA_VALIDATION:
        assert f"`{key}`" in lines
    # Group 1 is current, while the untouched downstream groups remain stale.
    assert "STALE" in lines
    assert "Group 1" in lines
