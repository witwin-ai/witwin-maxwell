from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation import tfsf_incident_is_gpu_driven
from witwin.maxwell.fdtd.runtime.stepping import _make_field_update_runner

_FREQ = 1.0e9
_TFSF_BOUNDS = ((-0.32, 0.32), (-0.32, 0.32), (-0.32, 0.32))


def _tfsf_scene(source):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.96, 0.96), (-0.96, 0.96), (-0.96, 0.96))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(source)
    return scene


def _pulse():
    return mw.GaussianPulse(frequency=_FREQ, fwidth=0.35e9, amplitude=80.0)


def _axis_aligned_scene():
    # Normal +z incidence, Ex polarization -> the axis-aligned reference provider
    # (plane_wave_axis_aligned). The in-block H correction reads the auxiliary
    # electric line through the integer-indexed batched reference kernel.
    return _tfsf_scene(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=_pulse(),
            injection=mw.TFSF(bounds=_TFSF_BOUNDS),
            name="tfsf_axis_aligned",
        )
    )


def _reference_x_ez_scene():
    # +x incidence, Ez polarization -> the dedicated x/Ez reference provider
    # (plane_wave_ref_x_ez).
    return _tfsf_scene(
        mw.PlaneWave(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            source_time=_pulse(),
            injection=mw.TFSF(bounds=_TFSF_BOUNDS),
            name="tfsf_ref_x_ez",
        )
    )


def _oblique_pulsed_scene():
    # Oblique incidence, pulsed -> the interpolated auxiliary-line provider
    # (plane_wave_aux). Its batched interpolated injection kernel accumulates
    # through non-deterministic cross-warp atomic adds, so it is declined for
    # capture (bit-identical parity is not achievable for that class).
    return _tfsf_scene(
        mw.PlaneWave(
            direction=(1.0, 0.25, 0.15),
            polarization=(0.0, 0.514495755, -0.857492925),
            source_time=_pulse(),
            injection=mw.TFSF(bounds=_TFSF_BOUNDS),
            name="tfsf_oblique_pulsed",
        )
    )


def _oblique_cw_scene():
    # Oblique incidence, CW -> the discrete-CW provider (plane_wave_discrete_cw),
    # whose in-block H correction is applied through generic host-evaluated
    # CW-phased patches. Declined for capture.
    return _tfsf_scene(
        mw.PlaneWave(
            direction=(1.0, 0.25, 0.15),
            polarization=(0.0, 0.514495755, -0.857492925),
            source_time=mw.CW(frequency=_FREQ, amplitude=80.0),
            injection=mw.TFSF(bounds=_TFSF_BOUNDS),
            name="tfsf_oblique_cw",
        )
    )


def _run(scene_factory, *, cuda_graph):
    result = mw.Simulation.fdtd(
        scene_factory(),
        frequencies=(_FREQ,),
        full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=250),
        cuda_graph=cuda_graph,
    ).run()
    view = result.at(frequency=_FREQ)
    fields = {a: np.asarray(getattr(view.E, a).detach().cpu()) for a in ("x", "y", "z")}
    provider = result.solver._tfsf_state["provider"]
    return fields, result.stats(), provider


# The reference-provider TFSF scenes are now captured into the field-update CUDA
# graph. Their in-block H correction and auxiliary magnetic advance read the
# incident field from the device-resident auxiliary 1D grid at fixed integer
# sample indices, so the block carries no per-step host input; the 1D source
# waveform is evaluated on the host only in the eager E-side auxiliary advance,
# outside the captured region. The integer-indexed reference injection kernel is
# collision-free for the axis-aligned face layout (bit-reproducible), so capture
# must reproduce the eager host path bit-for-bit.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
@pytest.mark.parametrize(
    ("scene_factory", "expected_provider"),
    [
        (_axis_aligned_scene, "plane_wave_axis_aligned"),
        (_reference_x_ez_scene, "plane_wave_ref_x_ez"),
    ],
    ids=["axis_aligned", "ref_x_ez"],
)
def test_cuda_graph_tfsf_matches_non_graph_bitexact(scene_factory, expected_provider):
    ref, ref_stats, ref_provider = _run(scene_factory, cuda_graph=False)
    got, got_stats, got_provider = _run(scene_factory, cuda_graph=True)
    assert ref_provider == expected_provider
    assert got_provider == expected_provider
    assert ref_stats["cuda_graph_active"] is False
    # The reference-provider TFSF field-update block is captured.
    assert got_stats["cuda_graph_active"] is True
    # Real-field TFSF scene with a running DFT: the GPU-driven DFT tail captures as
    # it does for the base real-field path.
    assert got_stats["tail_graph_active"] is True
    for component, field in ref.items():
        assert np.array_equal(field, got[component]), component


# Providers that cannot satisfy bit-identical capture parity must stay on the
# eager path: the oblique auxiliary-line provider (plane_wave_aux) accumulates
# through non-deterministic atomic adds, and the oblique-CW provider
# (plane_wave_discrete_cw) evaluates the incident phasor on the host each step and
# injects it as a kernel launch scalar a captured graph would freeze.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
@pytest.mark.parametrize(
    ("scene_factory", "expected_provider"),
    [
        (_oblique_pulsed_scene, "plane_wave_aux"),
        (_oblique_cw_scene, "plane_wave_discrete_cw"),
    ],
    ids=["oblique_pulsed_aux", "oblique_cw_discrete"],
)
def test_cuda_graph_tfsf_declines_non_reference_providers(scene_factory, expected_provider):
    solver = mw.Simulation.fdtd(
        scene_factory(),
        frequency=_FREQ,
        full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=60),
        cuda_graph=True,
    ).prepare().solver
    assert solver._tfsf_state["provider"] == expected_provider
    # The graphable predicate declines these providers ...
    assert tfsf_incident_is_gpu_driven(solver) is False
    # ... and the field-update runner keeps them on the eager path.
    _make_field_update_runner(solver, use_cuda_graph=True)
    assert solver._cuda_graph_active is False
