"""P5.8 capture-coverage report.

Machine-checked version of the P5.8 plan criterion: for every scene class the
field-update CUDA graph is expected to cover (or explicitly decline), assert that
``Result.stats()["cuda_graph_active"]`` matches the documented expectation. The
bit-exactness of each captured class is proven by its dedicated parity test
(``test_cuda_graph*.py``); this file is the single consolidated coverage matrix so
a regression that silently drops a captured class -- or silently captures a class
that carries per-step host input -- fails here immediately.

Declined classes additionally assert that the human-readable decline reason still
lives in the live graphable predicate (``_make_field_update_runner`` source), so
the "false with a documented reason" half of the criterion is tied to the actual
code rather than to a stale copy in this test.
"""
from __future__ import annotations

import inspect
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime import stepping

_FREQ = 2.0e9
_TFSF_FREQ = 1.0e9
_TFSF_BOUNDS = ((-0.32, 0.32), (-0.32, 0.32), (-0.32, 0.32))


def _pml_box_domain():
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))),
        grid=mw.GridSpec.uniform(0.008),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        device="cuda",
    )


def _dipole():
    return mw.PointDipole(
        position=(-0.02, 0.0, -0.02),
        polarization="Ez",
        source_time=mw.GaussianPulse(frequency=_FREQ, fwidth=0.7e9),
    )


def _plain_scene():
    scene = _pml_box_domain()
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.04, 0.04, 0.04)),
            material=mw.Material(eps_r=6.0),
        )
    )
    scene.add_source(_dipole())
    return scene


def _dispersive_scene():
    scene = _pml_box_domain()
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.04, 0.04, 0.04)),
            material=mw.Material.drude(eps_inf=1.0, plasma_frequency=2.0e9, gamma=1.0e8),
        )
    )
    scene.add_source(_dipole())
    return scene


def _kerr_scene():
    scene = _pml_box_domain()
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.04, 0.04, 0.04)),
            material=mw.Material(eps_r=4.0, kerr_chi3=1.0e-18),
        )
    )
    scene.add_source(_dipole())
    return scene


def _bloch_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))),
        grid=mw.GridSpec.uniform(0.008),
        boundary=mw.BoundarySpec.bloch((math.pi / 1.2, math.pi / 2.4, math.pi / 3.6)),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.04, 0.04, 0.04)),
            material=mw.Material(eps_r=6.0),
        )
    )
    scene.add_source(_dipole())
    return scene


def _tfsf_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.96, 0.96), (-0.96, 0.96), (-0.96, 0.96))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.GaussianPulse(frequency=_TFSF_FREQ, fwidth=0.35e9, amplitude=80.0),
            injection=mw.TFSF(bounds=_TFSF_BOUNDS),
            name="tfsf_axis_aligned",
        )
    )
    return scene


def _magnetic_source_scene():
    # A soft plane wave injects a magnetic surface source inside the update block;
    # the block therefore carries a host-evaluated per-step signal scalar.
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.1, 0.1), (-0.1, 0.1), (-0.12, 0.12))),
        grid=mw.GridSpec.uniform(0.008),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.GaussianPulse(frequency=_FREQ, fwidth=0.7e9),
        )
    )
    return scene


def _modulated_scene():
    # A modulated dielectric with a point-dipole (electric) source: the only guard
    # that can decline capture here is modulation_enabled, so the class isolates the
    # per-step host phase-scalar decline (no magnetic surface source is present).
    scene = _pml_box_domain()
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.04, 0.04, 0.04)),
            material=mw.Material(
                eps_r=4.0,
                modulation=mw.ModulationSpec(frequency=5.0e8, amplitude=0.2),
            ),
        )
    )
    scene.add_source(_dipole())
    return scene


# (id, scene_factory, run_frequency, expected_cuda_graph_active, decline_reason)
# decline_reason is None for captured classes; for declined classes it is a
# substring that must appear in the live graphable predicate source.
_COVERAGE = [
    ("plain", _plain_scene, _FREQ, True, None),
    ("dispersive", _dispersive_scene, _FREQ, True, None),
    ("kerr", _kerr_scene, _FREQ, True, None),
    ("bloch_complex", _bloch_scene, _FREQ, True, None),
    ("tfsf_reference", _tfsf_scene, _TFSF_FREQ, True, None),
    ("magnetic_source", _magnetic_source_scene, _FREQ, False, "host-evaluated signal scalar"),
    ("modulated", _modulated_scene, _FREQ, False, "per-step host phase scalars"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD CUDA graph requires CUDA")
@pytest.mark.parametrize(
    ("scene_factory", "frequency", "expected_active", "decline_reason"),
    [(f, freq, active, reason) for _, f, freq, active, reason in _COVERAGE],
    ids=[case[0] for case in _COVERAGE],
)
def test_cuda_graph_capture_coverage(scene_factory, frequency, expected_active, decline_reason):
    result = mw.Simulation.fdtd(
        scene_factory(),
        frequencies=(frequency,),
        full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=40),
        cuda_graph=True,
    ).run()
    stats = result.stats()
    assert stats["cuda_graph_active"] is expected_active
    if decline_reason is not None:
        # The declined class must be documented by the live predicate, not just by
        # this table: keep the "false with a documented reason" criterion honest.
        predicate_source = inspect.getsource(stepping._make_field_update_runner)
        assert decline_reason in predicate_source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD CUDA graph requires CUDA")
def test_cuda_graph_disabled_never_captures():
    # Sanity floor for the coverage matrix: with cuda_graph=False every class stays
    # eager, so a True reading in the table above always means capture genuinely
    # engaged rather than a stuck flag.
    result = mw.Simulation.fdtd(
        _plain_scene(),
        frequencies=(_FREQ,),
        full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=40),
        cuda_graph=False,
    ).run()
    assert result.stats()["cuda_graph_active"] is False
