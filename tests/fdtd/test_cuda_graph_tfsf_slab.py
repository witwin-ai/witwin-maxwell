from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation import tfsf_incident_is_gpu_driven
from witwin.maxwell.fdtd.runtime.stepping import _make_field_update_runner

_FREQ = 1.0e9
_SLAB_BOUNDS = (-0.32, 0.32)


def _pulse():
    return mw.GaussianPulse(frequency=_FREQ, fwidth=0.35e9, amplitude=80.0)


def _nonperiodic_slab_scene():
    # Normal +z incidence, Ex polarization, two-face (slab) TFSF injection with
    # non-periodic (PML) transverse boundaries. This resolves to the axis-aligned
    # reference provider (plane_wave_axis_aligned) exactly as the box TFSF does:
    # the in-block H correction reads the device-resident auxiliary electric line
    # through the integer-indexed batched reference kernel, so the field-update
    # block carries no per-step host input and is CUDA-graph capturable. A finite
    # dielectric slab sits strictly inside the total-field region (its interfaces
    # stay clear of the two injection faces) so the captured block exercises the
    # static material curl/decay coefficients, not just vacuum.
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.96, 0.96), (-0.96, 0.96), (-0.96, 0.96))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.2, 1.2, 0.16)),
            material=mw.Material(eps_r=4.0),
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=_pulse(),
            injection=mw.TFSF.slab(axis="z", bounds=_SLAB_BOUNDS),
            name="tfsf_slab_axis_aligned",
        )
    )
    return scene


def _grating_slab_scene():
    # Oblique incidence under a Bloch transverse lattice (grating slab). The
    # incident coupling is applied through generic host-evaluated CW-phased patches
    # (apply_generic_source_terms samples the source_time on the host each step and
    # injects it as a kernel launch scalar), so the provider is
    # plane_wave_grating_slab_cw and the block carries per-step host input: a
    # captured graph would freeze the incident phasor, so this class stays eager.
    boundary = mw.BoundarySpec.faces(
        default="pml",
        num_layers=4,
        strength=1.0,
        x="bloch",
        y="bloch",
        z="pml",
        bloch_wavevector=(math.pi, 0.5 * math.pi, 0.0),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=boundary,
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.2, 0.1, 0.9746794344808963),
            polarization=(1.0, 0.0, -0.20519567041703082),
            source_time=mw.CW(frequency=_FREQ, amplitude=20.0),
            injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)),
            name="grating_tfsf_slab",
        )
    )
    scene.add_monitor(mw.PointMonitor("center", (0.0, 0.0, 0.0), fields=("Ex", "Ez")))
    return scene


def _run(scene_factory, *, cuda_graph):
    result = mw.Simulation.fdtd(
        scene_factory(),
        frequencies=(_FREQ,),
        full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=220),
        cuda_graph=cuda_graph,
    ).run()
    view = result.at(frequency=_FREQ)
    fields = {a: np.asarray(getattr(view.E, a).detach().cpu()) for a in ("x", "y", "z")}
    provider = result.solver._tfsf_state["provider"]
    return fields, result.stats(), provider


# The non-periodic (PML transverse) normal-incidence TFSF slab shares the
# axis-aligned reference provider with the box TFSF, so its field-update block is
# host-input-free and captures into the CUDA graph. With a dielectric slab inside
# the total-field region the captured block also carries the static material
# coefficients. Capture must reproduce the eager host path bit-for-bit.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_cuda_graph_tfsf_slab_matches_non_graph_bitexact():
    ref, ref_stats, ref_provider = _run(_nonperiodic_slab_scene, cuda_graph=False)
    got, got_stats, got_provider = _run(_nonperiodic_slab_scene, cuda_graph=True)
    assert ref_provider == "plane_wave_axis_aligned"
    assert got_provider == "plane_wave_axis_aligned"
    assert ref_stats["cuda_graph_active"] is False
    # The slab-mode reference-provider TFSF field-update block is captured.
    assert got_stats["cuda_graph_active"] is True
    # Real-field scene with a running DFT: the GPU-driven DFT tail captures too.
    assert got_stats["tail_graph_active"] is True
    for component, field in ref.items():
        assert np.array_equal(field, got[component]), component


# The grating (Bloch transverse) TFSF slab injects through host-evaluated CW-phased
# patches, so bit-identical capture parity is impossible without a GPU-driven
# generic source. It must stay on the eager path with a documented decline.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_cuda_graph_tfsf_grating_slab_declines():
    solver = mw.Simulation.fdtd(
        _grating_slab_scene(),
        frequency=_FREQ,
        full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=60),
        cuda_graph=True,
    ).prepare().solver
    assert solver._tfsf_state["provider"] == "plane_wave_grating_slab_cw"
    # The graphable predicate declines the grating-slab-CW provider ...
    assert tfsf_incident_is_gpu_driven(solver) is False
    # ... and the field-update runner keeps it on the eager path.
    _make_field_update_runner(solver, use_cuda_graph=True)
    assert solver._cuda_graph_active is False
